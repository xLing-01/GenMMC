"""Comparable CE metrics, checkpoint identity, and stopping policy."""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def batch_statistics(logits, targets):
    losses = F.cross_entropy(logits.float().reshape(-1,logits.size(-1)), targets.reshape(-1),
                             ignore_index=515,reduction='none').reshape_as(targets)
    valid = targets.ne(515)
    if not bool(valid.any()):
        raise ValueError('Evaluation batch has no valid tokens')
    correct = logits.argmax(-1).eq(targets) & valid
    counts = valid.sum(1)
    if bool((counts == 0).any()):
        raise ValueError('Empty sequence in evaluation')
    sequence_sums = losses.double().sum(1)
    phase = torch.arange(targets.size(1),device=targets.device)[None,:] % 10
    masks = [(valid & targets.lt(512) & phase.eq(i)) for i in range(9)]
    masks.append(valid & targets.ge(512))
    return dict(ce_sum=float(sequence_sums.sum()),token_count=int(valid.sum()),
                correct=int(correct.sum()), sequence_ce=(sequence_sums/counts).cpu().tolist(),
                sequence_nll_sum=sequence_sums.cpu().tolist(), sequence_tokens=counts.cpu().tolist(),
                category_ce_sum=[float(losses[m].double().sum()) for m in masks],
                category_counts=[int(m.sum()) for m in masks],
                category_correct=[int(correct[m].sum()) for m in masks])


def merge_statistics(batches):
    count = sum(b['token_count'] for b in batches)
    if count <= 0:
        raise ValueError('Empty evaluation')
    sums = np.array([b['category_ce_sum'] for b in batches]).sum(0)
    counts = np.array([b['category_counts'] for b in batches]).sum(0)
    correct = np.array([b['category_correct'] for b in batches]).sum(0)
    return dict(ce=sum(b['ce_sum'] for b in batches)/count,token_count=count,
                accuracy=sum(b['correct'] for b in batches)/count,
                category_ce=[float(s/n) if n else None for s,n in zip(sums,counts)],
                category_accuracy=[float(s/n) if n else None for s,n in zip(correct,counts)],
                category_counts=counts.tolist(),
                sequence_ce=[v for b in batches for v in b['sequence_ce']],
                sequence_nll_sum=[v for b in batches for v in b['sequence_nll_sum']],
                sequence_tokens=[v for b in batches for v in b['sequence_tokens']])


@dataclass
class EarlyStop:
    patience: int = 10
    min_delta: float = 1e-4
    best: float = math.inf
    best_epoch: int = 0
    reference: float = math.inf
    bad_epochs: int = 0

    def update(self, value, epoch):
        if not math.isfinite(value):
            raise ValueError('Nonfinite validation CE')
        improved = value < self.best
        if improved:
            self.best, self.best_epoch = float(value), int(epoch)
        if value < self.reference - self.min_delta:
            self.reference, self.bad_epochs = float(value), 0
        else:
            self.bad_epochs += 1
        return improved, self.bad_epochs >= self.patience

    def state_dict(self):
        return asdict(self)


def make_scheduler(optimizer, patience=3, min_lr=1e-6):
    if min_lr <= 0:
        raise ValueError('Learning-rate floor must remain positive')
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,mode='min',factor=.5,patience=patience,threshold=1e-4,
        threshold_mode='abs',min_lr=min_lr)


def validate_resume(saved, current):
    if saved != current:
        changed = [k for k in set(saved)|set(current) if saved.get(k) != current.get(k)]
        raise ValueError('Resume contract changed: ' + ', '.join(sorted(changed)))
