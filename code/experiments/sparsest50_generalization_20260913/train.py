"""Single-GPU prefix-GPT training with held-out stopping and reproducible resume."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.llm_main.model import GPT, Model_args
from experiments.sparsest50_generalization_20260913 import core, data


@dataclass
class Config:
    dataset: str
    split_dir: str
    output_dir: str
    device: str = 'cuda'
    epochs: int = 45
    batch_size: int = 64
    microbatch_size: int = 32
    n_layer: int = 6
    n_head: int = 6
    n_embed: int = 384
    condition_len: int = 64
    block_size: int = 5521
    dropout: float = 0.0
    label_smoothing: float = 0.0
    weight_decay: float = .1
    learning_rate: float = 3e-4
    warmup_steps: int = 1000
    min_lr: float = 1e-6
    lr_patience: int = 3
    stop_patience: int = 10
    min_delta: float = 1e-4
    seed: int = 42
    num_workers: int = 0

    def validate(self):
        if min(self.epochs,self.batch_size,self.microbatch_size,self.stop_patience) <= 0:
            raise ValueError('Positive epochs, batches and stopping patience required')
        if self.warmup_steps < 0 or not 0 < self.min_lr <= self.learning_rate:
            raise ValueError('Invalid learning-rate schedule')
        if not 0 <= self.dropout < 1 or not 0 <= self.label_smoothing < 1:
            raise ValueError('Invalid regularization')


def source_identity():
    files = [Path(__file__), Path(core.__file__), Path(data.__file__),
             ROOT/'src/llm_main/model.py', ROOT/'src/llm_main/transformer_unet_encoder.py']
    return {str(p.relative_to(ROOT)):data.sha256_file(p) for p in files}


def autocast(device):
    if device.type == 'cuda':
        return torch.autocast('cuda',dtype=torch.bfloat16)
    return nullcontext()


@torch.no_grad()
def evaluate_model(model, dataset, batch_size, device):
    was_training = model.training
    model.eval()
    batches = []
    loader = DataLoader(dataset,batch_size=batch_size,collate_fn=data.collate_sequences,
                        generator=torch.Generator().manual_seed(1901),num_workers=0)
    try:
        for x,y,c in loader:
            x,y,c = x.to(device),y.to(device),c.to(device)
            with autocast(device):
                logits,_,_ = model(x,condition=c)
            statistics = core.batch_statistics(logits[:,model.args.condition_len:,:],y)
            if not math.isfinite(statistics['ce_sum']):
                raise ValueError('Nonfinite evaluation loss')
            batches.append(statistics)
        return core.merge_statistics(batches)
    finally:
        model.train(was_training)


def atomic_checkpoint(path, payload):
    path = Path(path)
    temp = path.with_suffix('.pt.tmp')
    torch.save(payload,temp)
    temp.replace(path)


def capture_rng():
    return dict(torch=torch.get_rng_state(),numpy=np.random.get_state(),python=random.getstate(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    torch.set_rng_state(state['torch'].cpu())
    np.random.set_state(state['numpy']);random.setstate(state['python'])
    if state['cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def checkpoint_summary(result):
    return {k:v for k,v in result.items() if not k.startswith('sequence_')}


def recover_history(path, saved):
    lines=path.read_text().splitlines() if path.exists() else []
    records=[]
    for index,line in enumerate(lines):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines)-1:
                raise ValueError('History has corruption before its last record')
    epoch=saved['epoch']
    records=[r for r in records if r['epoch'] < epoch]+[saved['metrics']]
    if [r['epoch'] for r in records] != list(range(1,epoch+1)):
        raise ValueError('History is missing committed earlier epochs')
    temporary=path.with_suffix('.jsonl.tmp')
    temporary.write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in records),encoding='utf-8')
    temporary.replace(path)


def run(cfg: Config, resume=False, stop_after_epoch=None):
    cfg.validate()
    torch.set_num_threads(1)
    torch.manual_seed(cfg.seed);np.random.seed(cfg.seed);random.seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    device = torch.device(cfg.device)
    if device.type == 'cuda' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('This experiment requires CUDA bf16 or CPU')
    output = Path(cfg.output_dir)
    output.mkdir(parents=True,exist_ok=True)
    history_path = output/'history.jsonl'
    latest_path = output/'checkpoint_latest.pt'
    if (history_path.exists() or latest_path.exists()) and not resume:
        raise ValueError('Run exists; explicit resume is required')
    manifest = data.prepare(cfg.dataset,cfg.split_dir)
    config_identity = {k:v for k,v in asdict(cfg).items() if k not in ('dataset','split_dir','output_dir','device')}
    contract = dict(dataset_sha256=manifest['dataset_sha256'],
                    split_indices_sha256=manifest['split_indices_sha256'],
                    config=config_identity,source_sha256=source_identity(),
                    runtime=dict(backend=device.type,precision='bfloat16' if device.type=='cuda' else 'float32',
                                 torch=str(torch.__version__),cuda=torch.version.cuda,
                                 cudnn=torch.backends.cudnn.version() if device.type=='cuda' else None))
    indices = dict(np.load(Path(cfg.split_dir)/'split_indices.npz'))
    corpus = data.Corpus(cfg.dataset)
    if int(corpus.lengths.max())-1 > cfg.block_size:
        raise ValueError('Block size would truncate a structure')
    mean,std = data.fit_normalizer(corpus.raw_conditions,indices['train_indices'])
    partitions = {k:data.SequenceDataset(corpus,indices[k],mean,std)
                  for k in ('train_indices','val_indices','train_monitor_indices')}
    model_args = dict(n_layer=cfg.n_layer,n_head=cfg.n_head,n_embed=cfg.n_embed,
                      block_size=cfg.block_size,bias=False,vocab_size=516,
                      dropout=cfg.dropout,label_smoothing=0.0,
                      condition_encoder_type='mlp',condition_dim=4,condition_len=cfg.condition_len,
                      loss_ignore_index=515)
    model = GPT(Model_args(**model_args)).to(device)
    optimizer = model.configure_optimizers(cfg.weight_decay,cfg.learning_rate,(.9,.95),device.type)
    scheduler = core.make_scheduler(optimizer,cfg.lr_patience,cfg.min_lr)
    early = core.EarlyStop(cfg.stop_patience,cfg.min_delta)
    epoch,step = 0,0
    if resume:
        saved = torch.load(latest_path,map_location='cpu',weights_only=False)
        core.validate_resume(saved['contract'],contract)
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler']);early=core.EarlyStop(**saved['early_stop'])
        epoch,step = saved['epoch'],saved['iter_num']
        restore_rng(saved['rng'])
        # A crash can occur between durable checkpoint and history write.
        recover_history(history_path,saved)
    data.atomic_json(output/'run_manifest.json',dict(contract=contract,config=asdict(cfg),
                     parameter_count=sum(p.numel() for p in model.parameters()),
                     counts=manifest['counts'],normalizer_train_only=True,
                     metric='unsmoothed CE / total non-PAD tokens, eval mode, after each completed epoch',
                     condition_names=manifest['condition_names'],
                     environment=dict(torch=torch.__version__,cuda=torch.version.cuda,python=sys.version),
                     started_or_resumed_at=time.time()))
    starting_epoch=epoch
    reason = 'epoch_budget'
    for epoch in range(starting_epoch+1,cfg.epochs+1):
        if early.bad_epochs >= early.patience:
            reason='early_stopping';epoch-=1;break
        started=time.monotonic()
        model.train()
        loader=DataLoader(partitions['train_indices'],batch_size=cfg.batch_size,shuffle=True,
                          generator=torch.Generator().manual_seed(cfg.seed+epoch),
                          collate_fn=data.collate_sequences,num_workers=cfg.num_workers,
                          pin_memory=device.type=='cuda')
        objective_sum,token_count=0.0,0
        first_lr=None
        max_grad_norm=0.0
        for batch_index,(x,y,c) in enumerate(loader):
            denominator=int(y.ne(515).sum())
            if step < cfg.warmup_steps:
                lr=cfg.learning_rate*(step+1)/cfg.warmup_steps
                for group in optimizer.param_groups:group['lr']=max(cfg.min_lr,lr)
            used_lr=optimizer.param_groups[0]['lr']
            if used_lr <= 0:raise RuntimeError('Zero learning rate is prohibited')
            if first_lr is None:first_lr=used_lr
            optimizer.zero_grad(set_to_none=True)
            batch_objective=0.0
            for start in range(0,len(x),cfg.microbatch_size):
                xx,yy,cc=(v[start:start+cfg.microbatch_size].to(device,non_blocking=True) for v in (x,y,c))
                with autocast(device):
                    logits,_,_=model(xx,condition=cc)
                    logits=logits[:,cfg.condition_len:,:]
                    loss_sum=F.cross_entropy(logits.float().reshape(-1,516),yy.reshape(-1),
                                             ignore_index=515,label_smoothing=cfg.label_smoothing,reduction='sum')
                    loss=loss_sum/denominator
                if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite training loss')
                loss.backward()
                batch_objective+=float(loss_sum.detach())
                del logits,loss_sum,loss
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
            max_grad_norm=max(max_grad_norm,float(norm))
            optimizer.step();step+=1
            objective_sum+=batch_objective;token_count+=denominator
            if batch_index % 50 == 0:
                data.atomic_json(output/'status.json',dict(state='training',epoch=epoch,
                    completed_epoch=epoch-1,optimizer_step=step,batch=batch_index+1,batches=len(loader),
                    learning_rate=used_lr,train_objective=objective_sum/token_count,
                    elapsed_epoch_seconds=time.monotonic()-started,pid=os.getpid(),updated=time.time()))
        train_seconds=time.monotonic()-started
        monitor=evaluate_model(model,partitions['train_monitor_indices'],cfg.microbatch_size,device)
        validation=evaluate_model(model,partitions['val_indices'],cfg.microbatch_size,device)
        improved,stop=early.update(validation['ce'],epoch)
        if step >= cfg.warmup_steps:scheduler.step(validation['ce'])
        next_lr=optimizer.param_groups[0]['lr']
        metrics=dict(epoch=epoch,optimizer_step=step,train_objective=objective_sum/token_count,
                     train_monitor_ce=monitor['ce'],val_ce=validation['ce'],
                     generalization_gap=validation['ce']-monitor['ce'],
                     train_monitor_accuracy=monitor['accuracy'],val_accuracy=validation['accuracy'],
                     learning_rate_first=first_lr,learning_rate_last=used_lr,learning_rate_next=next_lr,
                     gradient_norm_max=max_grad_norm,best_val_ce=early.best,best_epoch=early.best_epoch,
                     bad_epochs=early.bad_epochs,train_seconds=train_seconds,
                     epoch_seconds=time.monotonic()-started,updated=time.time())
        detail=output/'metrics';detail.mkdir(exist_ok=True)
        np.savez_compressed(detail/f'epoch_{epoch:03d}.npz',
             val_indices=indices['val_indices'],val_nll_sum=validation['sequence_nll_sum'],
             val_token_count=validation['sequence_tokens'],val_sequence_ce=validation['sequence_ce'],
             train_monitor_indices=indices['train_monitor_indices'],train_monitor_sequence_ce=monitor['sequence_ce'])
        data.atomic_json(detail/f'epoch_{epoch:03d}.json',dict(train_monitor=checkpoint_summary(monitor),validation=checkpoint_summary(validation)))
        payload=dict(model=model.state_dict(),model_args=model_args,optimizer=optimizer.state_dict(),
                     scheduler=scheduler.state_dict(),early_stop=early.state_dict(),epoch=epoch,iter_num=step,
                     rng=capture_rng(),contract=contract,condition_mean=mean,condition_std=std,
                     condition_names=manifest['condition_names'],train_args=asdict(cfg),
                     best_val_loss=early.best,metrics=metrics)
        if improved:atomic_checkpoint(output/'checkpoint_best.pt',payload)
        atomic_checkpoint(latest_path,payload)
        with history_path.open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(metrics,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
        print(json.dumps(metrics),flush=True)
        if stop:
            reason='early_stopping';break
        if stop_after_epoch is not None and epoch >= stop_after_epoch:
            reason='verification_pause';break
    result=dict(state='training_complete' if reason != 'verification_pause' else 'paused',
                completed_epoch=epoch,best_epoch=early.best_epoch,best_val_ce=early.best,
                stop_reason=reason,optimizer_step=step,learning_rate=optimizer.param_groups[0]['lr'],
                checkpoint_best=str(output/'checkpoint_best.pt'),physical_validation='pending',updated=time.time())
    data.atomic_json(output/'status.json',result)
    return result


def evaluate_checkpoint(checkpoint_path,dataset_path,split_dir,partition,output,device='cuda',selection=None):
    if partition not in ('val','test'):
        raise ValueError('Evaluation partition must be val or test')
    manifest=data.prepare(dataset_path,split_dir)
    saved=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    for key in ('dataset_sha256','split_indices_sha256'):
        if saved['contract'][key]!=manifest[key]:raise ValueError('Checkpoint data identity mismatch')
    if partition=='test':
        if selection is None:raise ValueError('Test evaluation requires locked selection')
        chosen=json.loads(Path(selection).read_text())
        for key in ('dataset_sha256','split_indices_sha256'):
            if chosen[key]!=manifest[key]:raise ValueError('Selection data mismatch')
        if chosen['selected_checkpoint_sha256']!=data.sha256_file(checkpoint_path):
            raise ValueError('Selection checkpoint mismatch')
    corpus=data.Corpus(dataset_path)
    ids=np.load(Path(split_dir)/'split_indices.npz')[partition+'_indices']
    ds=data.SequenceDataset(corpus,ids,saved['condition_mean'],saved['condition_std'])
    model=GPT(Model_args(**saved['model_args'])).to(device);model.load_state_dict(saved['model'])
    result=evaluate_model(model,ds,16,torch.device(device))
    result.update(partition=partition,checkpoint_sha256=data.sha256_file(checkpoint_path),samples=len(ids))
    data.atomic_json(output,result)
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--stop-after-epoch',type=int)
    args=parser.parse_args()
    config=Config(**json.loads(Path(args.config).read_text()))
    run_lock=None
    if sys.platform=='linux':
        import fcntl
        Path(config.output_dir).mkdir(parents=True,exist_ok=True)
        run_lock=(Path(config.output_dir)/'run.lock').open('a')
        fcntl.flock(run_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        run(config,args.resume,args.stop_after_epoch)
    except Exception as exc:
        data.atomic_json(Path(config.output_dir)/'failure.json',dict(state='failed',error=repr(exc),updated=time.time()))
        raise
