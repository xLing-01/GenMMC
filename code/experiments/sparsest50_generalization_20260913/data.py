"""Immutable, grouped train/validation/test data for the sparsest50 experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

PAD = 515


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def structure_group(tokens, length):
    sequence = np.asarray(tokens[:int(length)], dtype=np.uint16)
    if len(sequence) < 2 or sequence[0] != 513 or sequence[-1] != 514:
        raise ValueError('Every structure must start with BOS and end with EOS')
    body = sequence[1:-1]
    if len(body) % 10:
        raise ValueError('Invalid rod token length')
    rods = body.reshape(-1, 10)
    if not np.all(rods[:, 9] == 512) or np.any(rods[:, :9] >= 512):
        raise ValueError('Invalid parameter/SEP token grammar')
    parameters = rods[:, :9]
    order = np.lexsort(tuple(parameters[:, k] for k in reversed(range(9))))
    return hashlib.sha256(parameters[order].astype('<u2').tobytes()).hexdigest()


def grouped_split(groups, vf, rods, seed=42):
    groups, vf, rods = map(np.asarray, (groups, vf, rods))
    if not len(groups) == len(vf) == len(rods) or len(groups) < 10:
        raise ValueError('Need at least ten aligned sample rows')
    if not np.isfinite(vf).all() or not np.isfinite(rods).all():
        raise ValueError('Nonfinite stratification data')
    unique, representatives, inverse = np.unique(groups, return_index=True, return_inverse=True)
    if len(unique) < 10:
        raise ValueError('Too few independent structure groups')
    rng = np.random.default_rng(seed)
    vf_edges = np.unique(np.quantile(vf[representatives], np.linspace(0, 1, 11)))
    rod_edges = np.unique(np.quantile(rods[representatives], np.linspace(0, 1, 6)))
    bins = np.stack((np.searchsorted(vf_edges[1:-1], vf[representatives]),
                     np.searchsorted(rod_edges[1:-1], rods[representatives])), axis=1)
    pools, sparse = [], []
    for cell in np.unique(bins, axis=0):
        group_ids = np.flatnonzero(np.all(bins == cell, axis=1))
        if len(group_ids) < 10:
            sparse.extend(group_ids.tolist())
        else:
            pools.append(group_ids)
    if sparse:
        pools.append(np.asarray(sparse))
    assignments = np.full(len(unique), -1, dtype=np.int64)
    for pool in pools:
        pool = rng.permutation(pool)
        n_val = max(1, round(len(pool) * .1)) if len(pool) >= 3 else 0
        n_test = n_val
        assignments[pool[:n_val]] = 1
        assignments[pool[n_val:n_val+n_test]] = 2
        assignments[pool[n_val+n_test:]] = 0
    result = {name: np.flatnonzero(assignments[inverse] == part)
              for part, name in enumerate(('train_indices', 'val_indices', 'test_indices'))}
    if any(len(v) == 0 for v in result.values()):
        raise ValueError('Empty split')
    return result


def fit_normalizer(raw, train_indices):
    values = np.asarray(raw[train_indices], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite training conditions')
    mean, std = values.mean(axis=0), values.std(axis=0)
    std[std < 1e-12] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


class Corpus:
    def __init__(self, path):
        with h5py.File(path, 'r') as handle:
            self.tokens = handle['tokens'][:]
            self.lengths = handle['token_lengths'][:].astype(np.int64)
            c = handle['C_eff'][:]
            self.raw_conditions = np.stack([handle['volume_fraction'][:], c[:,0,0], c[:,0,1], c[:,3,3]], axis=1).astype(np.float32)
            self.rods = handle['num_rods'][:]
        if not np.isfinite(self.raw_conditions).all():
            raise ValueError('Dataset contains nonfinite conditions')
        if len(self.lengths) != len(self.tokens) or np.any(self.lengths > self.tokens.shape[1]):
            raise ValueError('Invalid token lengths')


def prepare(dataset_path, output_dir, seed=42):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fingerprint = sha256_file(dataset_path)
    manifest_path = output_dir / 'manifest.json'
    indices_path = output_dir / 'split_indices.npz'
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text())
        if saved['dataset_sha256'] != fingerprint or saved['seed'] != seed:
            raise ValueError('Existing split uses different dataset or seed')
        if saved['split_indices_sha256'] != sha256_file(indices_path):
            raise ValueError('Split file changed')
        return saved
    if indices_path.exists():
        raise ValueError('Uncommitted split exists; use a new output directory')
    corpus = Corpus(dataset_path)
    groups = np.array([structure_group(t, n) for t,n in zip(corpus.tokens, corpus.lengths)])
    split = grouped_split(groups, corpus.raw_conditions[:,0], corpus.rods, seed)
    rng = np.random.default_rng(seed + 10000)
    split['train_monitor_indices'] = rng.permutation(split['train_indices'])[:2048]
    split['val_generation_indices'] = rng.permutation(split['val_indices'])[:64]
    split['test_generation_indices'] = rng.permutation(split['test_indices'])[:64]
    mean, std = fit_normalizer(corpus.raw_conditions, split['train_indices'])
    np.savez_compressed(indices_path, **split)
    np.savez_compressed(output_dir/'structure_groups.npz', group_hashes=groups)
    manifest = dict(schema_version=1, dataset_path=str(Path(dataset_path).resolve()),
                    dataset_sha256=fingerprint, split_indices_sha256=sha256_file(indices_path), seed=seed,
                    group_method='SHA256 of lexicographically sorted quantized rod-parameter rows',
                    stratification='group representatives: vf deciles x rod-count quintiles',
                    samples=len(groups), unique_structure_groups=len(np.unique(groups)),
                    duplicate_rows=len(groups)-len(np.unique(groups)),
                    counts={k:len(v) for k,v in split.items()},
                    condition_names=['vf','C1111','C1122','C1212'],
                    condition_mean=mean.tolist(), condition_std=std.tolist(),
                    tokenizer_note='Existing 512-bin tokenizer retained; its bin bounds were historically fitted before this split.',
                    test_policy='Locked until validation-only model selection; all new runs start from scratch.')
    atomic_json(manifest_path, manifest)
    return manifest


class SequenceDataset(Dataset):
    def __init__(self, corpus, indices, mean, std):
        self.corpus, self.indices = corpus, np.asarray(indices)
        self.conditions = (corpus.raw_conditions - np.asarray(mean)) / np.asarray(std)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        index = self.indices[item]
        return self.corpus.tokens[index,:self.corpus.lengths[index]], self.conditions[index]


def collate_sequences(batch):
    maximum = max(len(sequence)-1 for sequence, _ in batch)
    x = torch.full((len(batch),maximum), PAD, dtype=torch.long)
    y = x.clone()
    conditions = []
    for row,(sequence,condition) in enumerate(batch):
        n = len(sequence)-1
        x[row,:n] = torch.as_tensor(np.asarray(sequence[:-1], dtype=np.int64))
        y[row,:n] = torch.as_tensor(np.asarray(sequence[1:], dtype=np.int64))
        conditions.append(condition)
    return x, y, torch.as_tensor(np.asarray(conditions),dtype=torch.float32)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',required=True)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--seed',type=int,default=42)
    args = parser.parse_args()
    print(json.dumps(prepare(args.dataset,args.output_dir,args.seed),indent=2))
