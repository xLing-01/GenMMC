# Selected best ablation model

This folder stores the selected model from the 2026-09-16 ablation runs.

## Selection result

- Selected configuration: token/hidden dimension (embedding) = 192
- Best validation checkpoint: epoch 162
- Best validation CE: 0.061060373593247584
- Seed: 42
- Checkpoint selection: validation CE only; test set was not used for selection
- Best checkpoint SHA-256: `986f30202e79a7273410ccb63debc09bbb2716c21931d3333e608193cf8d98e2`

## Model configuration

- Transformer blocks: 6
- Attention heads: 6
- Condition prefix length: 64 tokens
- Dropout: 0.20
- Effective batch size: 64; microbatch size: 8
- Optimizer: AdamW, learning rate 3e-4, weight decay 0.1
- Precision: bfloat16

## Contents

- `weights/checkpoint_best_val.pt`: selected best-validation weights
- `config/`: exact run configuration, manifest, selection record, and loss history
- `code/`: source files used by the run; their SHA-256 values are recorded in `config/run_manifest.json`

The dataset and split files are not duplicated here. Their expected SHA-256 values and original remote paths are recorded in `config/run_manifest.json`.
