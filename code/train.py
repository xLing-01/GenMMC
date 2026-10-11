"""Fresh GPT training with complete, same-checkpoint train/validation/test CE at epochs 0–200."""
from dataclasses import asdict,dataclass
import argparse
import csv
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from experiments.sparsest50_generalization_20260913 import train as base,core,data
from src.llm_main.model import GPT,Model_args


@dataclass
class Config:
    dataset:str
    split_dir:str
    output_dir:str
    optimizer_name:str="adamw"
    epochs:int=200
    device:str='cuda'
    seed:int=42
    batch_size:int=64
    microbatch_size:int=32
    eval_batch_size:int=32
    n_layer:int=6
    n_head:int=6
    n_embed:int=384
    condition_len:int=64
    block_size:int=5521
    dropout:float=.20
    label_smoothing:float=0.
    weight_decay:float=.1
    learning_rate:float=3e-4
    warmup_steps:int=1000
    min_lr:float=1e-6
    lr_patience:int=3
    num_workers:int=0
    expected_dataset_sha256:str|None=None
    expected_split_sha256:str|None=None


def validate_partitions(indices,size):
    arrays=[]
    for name in ('train_indices','val_indices','test_indices'):
        a=np.asarray(indices[name])
        if a.ndim!=1 or a.dtype.kind not in 'iu' or not len(a) or np.any(a<0) or np.any(a>=size):
            raise ValueError('Invalid partition: '+name)
        arrays.append(a)
    merged=np.concatenate(arrays)
    if len(merged)!=size or not np.array_equal(np.sort(merged),np.arange(size)):
        raise ValueError('Partitions must be disjoint and cover every sample exactly once')


def runtime(device):
    return dict(backend=device.type,precision='bfloat16' if device.type=='cuda' else 'float32',
                torch=str(torch.__version__),cuda=torch.version.cuda,
                cudnn=torch.backends.cudnn.version() if device.type=='cuda' else None)


def next_lr(cfg,step,optimizer):
    if step<cfg.warmup_steps:return max(cfg.min_lr,cfg.learning_rate*(step+1)/cfg.warmup_steps)
    return optimizer.param_groups[0]['lr']


def train_epoch(model,optimizer,cfg,dataset,epoch,step,device,progress):
    model.train();started=time.monotonic()
    loader=DataLoader(dataset,batch_size=cfg.batch_size,shuffle=True,
                      generator=torch.Generator().manual_seed(cfg.seed+epoch),
                      collate_fn=data.collate_sequences,num_workers=cfg.num_workers,
                      pin_memory=device.type=='cuda')
    total,ntokens,max_norm=0.,0,0.;first_lr=None
    for batch,(x,y,c) in enumerate(loader):
        if step<cfg.warmup_steps:
            learning_rate=next_lr(cfg,step,optimizer)
            for group in optimizer.param_groups:group['lr']=learning_rate
        learning_rate=optimizer.param_groups[0]['lr']
        if learning_rate<=0 or not math.isfinite(learning_rate):raise ValueError('Invalid learning rate')
        if first_lr is None:first_lr=learning_rate
        count=int(y.ne(515).sum());optimizer.zero_grad(set_to_none=True)
        for start in range(0,len(x),cfg.microbatch_size):
            xx,yy,cc=(v[start:start+cfg.microbatch_size].to(device,non_blocking=True) for v in (x,y,c))
            with base.autocast(device):
                logits,_,_=model(xx,condition=cc)
                summed=F.cross_entropy(logits[:,cfg.condition_len:,:].float().reshape(-1,516),yy.reshape(-1),
                                      ignore_index=515,label_smoothing=cfg.label_smoothing,reduction='sum')
                loss=summed/count
            if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite training loss')
            loss.backward();total+=float(summed.detach());del logits,summed,loss
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        max_norm=max(max_norm,float(norm));optimizer.step();step+=1;ntokens+=count
        if batch%50==0:
            progress(dict(state='training',epoch=epoch,completed_epoch=epoch-1,batch=batch+1,batches=len(loader),
                          optimizer_step=step,learning_rate=learning_rate,train_objective=total/ntokens,
                          elapsed_epoch_seconds=time.monotonic()-started))
    return step,dict(train_objective=total/ntokens,learning_rate_first=first_lr,learning_rate_last=learning_rate,
                     gradient_norm_max=max_norm,train_seconds=time.monotonic()-started)


def evaluate_all(model,partitions,cfg,device,epoch,progress):
    # Explicit RNG isolation ensures even diagnostic test evaluation cannot change future dropout/shuffles.
    rng=base.capture_rng();was_training=model.training;model.eval();results={};started=time.monotonic()
    try:
        for name in ('train','val','test'):
            dataset=partitions[name]
            progress(dict(state='evaluating_'+name,epoch=epoch,completed_epoch=epoch-1 if epoch else None,
                          samples=len(dataset)))
            results[name]=base.evaluate_model(model,dataset,cfg.eval_batch_size,device)
            if not math.isfinite(results[name]['ce']):raise ValueError('Nonfinite '+name+' CE')
    finally:
        model.train(was_training);base.restore_rng(rng)
    return results,time.monotonic()-started


def export_history(output,history):
    for filename in ('history.jsonl','loss_history.csv'):
        target=output/filename;temp=target.with_suffix(target.suffix+'.tmp')
        with temp.open('w',encoding='utf-8',newline='') as stream:
            if filename.endswith('csv'):
                writer=csv.DictWriter(stream,fieldnames=list(history[0]));writer.writeheader();writer.writerows(history)
            else:
                for row in history:stream.write(json.dumps(row,allow_nan=False)+'\n')
            stream.flush();os.fsync(stream.fileno())
        temp.replace(target)


def sync_best(output,history,best_epoch):
    row=history[best_epoch];source=output/'epochs'/f'epoch_{best_epoch:03d}.pt'
    if data.sha256_file(source)!=row['checkpoint_sha256']:raise ValueError('Committed best checkpoint changed')
    target=output/'checkpoint_best_val.pt'
    if not target.exists() or data.sha256_file(target)!=row['checkpoint_sha256']:
        temporary=target.with_suffix('.pt.tmp');shutil.copyfile(source,temporary);temporary.replace(target)
    data.atomic_json(output/'best_selection.json',dict(epoch=best_epoch,val_ce=row['val_ce'],
        checkpoint_sha256=row['checkpoint_sha256'],selection_metric='validation_ce_only',test_used=False))


def run(cfg:Config,resume=False,stop_after_epoch=None):
    if min(cfg.epochs,cfg.batch_size,cfg.microbatch_size,cfg.eval_batch_size)<1 or cfg.warmup_steps<0:
        raise ValueError('Invalid epochs or batch sizes')
    if not 0<cfg.min_lr<=cfg.learning_rate or not 0<=cfg.dropout<1 or not 0<=cfg.label_smoothing<1:
        raise ValueError('Invalid learning rate or dropout/smoothing')
    torch.set_num_threads(1);torch.manual_seed(cfg.seed);np.random.seed(cfg.seed);random.seed(cfg.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(cfg.seed)
    device=torch.device(cfg.device)
    if device.type=='cuda' and not torch.cuda.is_bf16_supported():raise ValueError('CUDA bf16 required')
    manifest=data.prepare(cfg.dataset,cfg.split_dir)
    for expected,key in [(cfg.expected_dataset_sha256,'dataset_sha256'),(cfg.expected_split_sha256,'split_indices_sha256')]:
        if expected is not None and manifest[key]!=expected:raise ValueError('Unexpected '+key)
    ids=dict(np.load(Path(cfg.split_dir)/'split_indices.npz'));corpus=data.Corpus(cfg.dataset)
    validate_partitions(ids,len(corpus.lengths))
    if int(corpus.lengths.max())-1>cfg.block_size:raise ValueError('Block size would truncate structures')
    mean,std=data.fit_normalizer(corpus.raw_conditions,ids['train_indices'])
    partitions={name:data.SequenceDataset(corpus,ids[name+'_indices'],mean,std) for name in ('train','val','test')}
    output=Path(cfg.output_dir);output.mkdir(parents=True,exist_ok=True)
    latest=output/'checkpoint_latest.pt'
    if (latest.exists() or (output/'history.jsonl').exists()) and not resume:raise ValueError('Existing run requires explicit own-checkpoint resume')
    split_copy=output/'split_indices.npz'
    if not split_copy.exists():shutil.copyfile(Path(cfg.split_dir)/'split_indices.npz',split_copy)
    if data.sha256_file(split_copy)!=manifest['split_indices_sha256']:raise ValueError('Local split copy differs')
    identity={k:v for k,v in asdict(cfg).items() if k not in ('dataset','split_dir','output_dir','device')}
    sources=dict(base.source_identity());sources[str(Path(__file__).relative_to(ROOT))]=data.sha256_file(__file__)
    contract=dict(configuration=identity,runtime=runtime(device),source_sha256=sources,
                  dataset_sha256=manifest['dataset_sha256'],split_indices_sha256=manifest['split_indices_sha256'],
                  training_origin='scratch',test_policy='record_only_not_used_for_scheduler_or_selection',
                  evaluation_policy='all_three_complete_sets_same_frozen_weights_eval_plain_nonpad_token_weighted_ce')
    run_manifest=output/'run_manifest.json'
    if run_manifest.exists() and json.loads(run_manifest.read_text()).get('contract')!=contract:
        raise ValueError('Existing run manifest has a different contract; use a new directory')
    model_args=dict(n_layer=cfg.n_layer,n_head=cfg.n_head,n_embed=cfg.n_embed,block_size=cfg.block_size,
                    bias=False,vocab_size=516,dropout=cfg.dropout,label_smoothing=0.,
                    condition_encoder_type='mlp',condition_dim=4,condition_len=cfg.condition_len,loss_ignore_index=515)
    model=GPT(Model_args(**model_args)).to(device)
    if cfg.optimizer_name == 'adamw':
        optimizer=model.configure_optimizers(cfg.weight_decay,cfg.learning_rate,(.9,.95),device.type)
    elif cfg.optimizer_name == 'sgd':
        optimizer=torch.optim.SGD(model.parameters(),lr=cfg.learning_rate,momentum=.9,
                                  dampening=0,weight_decay=cfg.weight_decay,nesterov=False)
    else:
        raise ValueError('Unknown optimizer')
    print(json.dumps(dict(event='model_ready',parameters=sum(p.numel() for p in model.parameters()),
                          optimizer=cfg.optimizer_name,config=asdict(cfg))),flush=True)
    scheduler=core.make_scheduler(optimizer,cfg.lr_patience,cfg.min_lr)
    epoch,step=-1,0;history=[];best_val=math.inf;best_epoch=None
    def progress(value):data.atomic_json(output/'status.json',dict(value,pid=os.getpid(),updated=time.time(),target_epoch=cfg.epochs,physical_validation='disabled'))
    if resume:
        saved=torch.load(latest,map_location='cpu',weights_only=False);core.validate_resume(saved['contract'],contract)
        model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer']);scheduler.load_state_dict(saved['scheduler'])
        epoch,step=saved['epoch'],saved['iter_num'];history=list(saved['history'])
        best_val,best_epoch=saved['best_val_loss'],saved['best_val_epoch'];base.restore_rng(saved['rng'])
        if [r['epoch'] for r in history]!=list(range(epoch+1)):raise ValueError('Incomplete checkpoint history')
        if any(any(r[k] is None or not math.isfinite(r[k]) for k in ('train_full_ce','val_ce','test_ce')) for r in history):
            raise ValueError('Missing set evaluation in checkpoint history')
        export_history(output,history);sync_best(output,history,best_epoch);del saved
    if not run_manifest.exists():
        data.atomic_json(run_manifest,dict(contract=contract,config=asdict(cfg),created_at=time.time(),
            counts={name:len(ds) for name,ds in partitions.items()},normalizer_fit='train_only',
            condition_names=['vf','C1111','C1122','C1212'],condition_mean=mean.tolist(),condition_std=std.tolist(),
            early_stopping=False,epoch0_measured_before_optimizer_steps=True,previous_model_loaded=False))
    if resume:data.atomic_json(output/'last_resume.json',dict(epoch=epoch,optimizer_step=step,time=time.time()))
    def commit(row,statistics):
        nonlocal best_val,best_epoch
        improved=row['val_ce']<best_val
        if improved:best_val,best_epoch=row['val_ce'],row['epoch']
        row.update(best_val_ce=best_val,best_val_epoch=best_epoch)
        epochs=output/'epochs';epochs.mkdir(exist_ok=True)
        path=epochs/f'epoch_{row["epoch"]:03d}.pt'
        snapshot=dict(model=model.state_dict(),model_args=model_args,epoch=row['epoch'],iter_num=step,
                      metrics=dict(row),contract=contract,condition_names=['vf','C1111','C1122','C1212'],
                      condition_mean=mean,condition_std=std,train_args=asdict(cfg))
        base.atomic_checkpoint(path,snapshot);row['checkpoint_sha256']=data.sha256_file(path)
        metrics=output/'metrics';metrics.mkdir(exist_ok=True);arrays={};summaries={}
        for name,stats in statistics.items():
            arrays.update({name+'_indices':ids[name+'_indices'],name+'_nll_sum':stats['sequence_nll_sum'],
                           name+'_token_count':stats['sequence_tokens']})
            summaries[name]=base.checkpoint_summary(stats)
        point=metrics/f'epoch_{row["epoch"]:03d}.npz';temp=point.with_suffix('.npz.tmp')
        with temp.open('wb') as stream:np.savez_compressed(stream,**arrays)
        temp.replace(point)
        data.atomic_json(metrics/f'epoch_{row["epoch"]:03d}.json',dict(epoch=row['epoch'],checkpoint_sha256=row['checkpoint_sha256'],sets=summaries))
        history.append(row)
        state=dict(snapshot,metrics=row,optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                   rng=base.capture_rng(),history=history,best_val_loss=best_val,best_val_epoch=best_epoch)
        base.atomic_checkpoint(latest,state);export_history(output,history)
        if improved:sync_best(output,history,best_epoch)
        progress(dict(state='epoch_complete',completed_epoch=row['epoch'],optimizer_step=step,
                      train_full_ce=row['train_full_ce'],val_ce=row['val_ce'],test_ce=row['test_ce']))
        print(json.dumps(row),flush=True)
        # Only this run's redundant epoch snapshots; best and current remain recoverable.
        for old in epochs.glob('epoch_*.pt'):
            if old.name not in {f'epoch_{best_epoch:03d}.pt',path.name}:
                old.unlink()
    while epoch<cfg.epochs:
        if stop_after_epoch is not None and epoch>=stop_after_epoch:break
        epoch+=1;started=time.monotonic()
        training=dict(train_objective=None,learning_rate_first=None,learning_rate_last=None,gradient_norm_max=None,train_seconds=0.)
        if epoch>0:step,training=train_epoch(model,optimizer,cfg,partitions['train'],epoch,step,device,progress)
        evaluated,eval_seconds=evaluate_all(model,partitions,cfg,device,epoch,progress)
        # Test CE is deliberately not read by either control decision below.
        if epoch>0 and step>=cfg.warmup_steps:scheduler.step(evaluated['val']['ce'])
        row=dict(epoch=epoch,optimizer_step=step,train_full_ce=evaluated['train']['ce'],
                 val_ce=evaluated['val']['ce'],test_ce=evaluated['test']['ce'],
                 **{name+'_samples':len(ds) for name,ds in partitions.items()},
                 **{name+'_tokens':value['token_count'] for name,value in evaluated.items()},
                 **training,learning_rate_next_step=next_lr(cfg,step,optimizer),
                 evaluation_mode='eval_full_sets',eval_seconds=eval_seconds,epoch_seconds=time.monotonic()-started)
        commit(row,evaluated)
    result=dict(state='complete' if epoch==cfg.epochs else 'verification_paused',completed_epoch=epoch,
                optimizer_step=step,rows=len(history),train_full_ce=history[-1]['train_full_ce'],
                val_ce=history[-1]['val_ce'],test_ce=history[-1]['test_ce'],best_val_epoch=best_epoch,
                best_val_ce=best_val,learning_rate_next_step=next_lr(cfg,step,optimizer),
                all_three_sets_logged_each_epoch=True,origin='scratch')
    progress(result);return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--resume',action='store_true')
    parser.add_argument('--stop-after-epoch',type=int);args=parser.parse_args()
    cfg=Config(**json.loads(Path(args.config).read_text()))
    if sys.platform=='linux':
        import fcntl
        Path(cfg.output_dir).mkdir(parents=True,exist_ok=True)
        lock=(Path(cfg.output_dir)/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        result=run(cfg,args.resume,args.stop_after_epoch)
        if result['state']=='complete':
            from experiments.sparsest50_full_metrics_20260915.report import build_report
            build_report(Path(cfg.output_dir))
    except Exception as exc:
        data.atomic_json(Path(cfg.output_dir)/'status.json',dict(state='failed',error=repr(exc),updated=time.time()));raise


if __name__=='__main__':main()
