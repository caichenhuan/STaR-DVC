# STaR: Multi-Granular Spatio-Temporal Reasoning for Long-Form Dense Video Captioning

Official code release for **STaR: Multi-Granular Spatio-Temporal Reasoning for Long-Form Dense Video Captioning (ECAI 2025)**.

This repository contains the training and evaluation code for long-form dense video captioning on soccer videos with:

- multi-stage training
- global and local temporal reasoning
- 2D position-aware video encoding
- dense caption generation from spotting outputs

## Overview

The official ECAI 2025 release uses a two-module pipeline:

1. `spotting`: event spotting over full matches
2. `caption`: caption training / dense caption generation in `captioning.py`

In the default `full` run, the execution order is:

1. `spotting`
2. `caption`
3. `dvc`

where `dvc` is the final inference step implemented inside `captioning.py`.

The main shared components are:

- `VideoEncoder` in [model.py](model.py)
- GPT-based caption decoder in [model_gpt.py](model_gpt.py)
- Perceiver-based temporal fusion modules in [temporal_model.py](temporal_model.py)

## Repository Structure

```text
.
├── main.py               # unified pipeline entry point
├── classifying.py        # legacy pretraining script, not used in the official release pipeline
├── captioning.py         # caption training and DVC inference
├── spotting.py           # event spotting
├── dataset.py            # SoccerNet data loading
├── model.py              # shared encoder + task heads
├── model_gpt.py          # GPT caption model
├── temporal_model.py     # temporal fusion modules
├── train.py              # train / validate / test loops
├── run_base.sh           # full pipeline example
└── run_debug.sh          # stage-wise debug example
```

For a Chinese code walkthrough, see [README_流程阅读指南.md](README_流程阅读指南.md).

## Installation

Recommended environment:

- Python 3.10+
- CUDA-enabled PyTorch
- access to SoccerNet caption annotations and extracted video features
- normalized 2D position files

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Notes:

- `tinycudann` depends on the local CUDA and PyTorch stack. Install it with the version matching your machine.
- `wandb` is optional. If you do not need experiment tracking, run without `--wandb`.

## Data Layout

`SOCCERNET_PATH` should contain per-game features and SoccerNet labels:

```text
${SOCCERNET_PATH}/
  game_x/
    1_224p_5fps.npy
    2_224p_5fps.npy
    <SoccerNet caption annotation file>
```

`POSITION_PATH` should contain normalized 2D player positions:

```text
${POSITION_PATH}/
  game_x/
    1_sn_position.json
    2_sn_position.json
```

The code expects feature files named as `1_<features>` and `2_<features>`.

## Training and Evaluation

Run the official pipeline:

```bash
SOCCERNET_PATH=/path/to/soccernet \
POSITION_PATH=/path/to/positions \
GPT_PATH=/path/to/gpt2 \
MODEL_NAME=star-ecai2025 \
bash run_base.sh
```

Run a single stage:

```bash
python main.py \
  --SoccerNet_path /path/to/soccernet \
  --Position_path /path/to/positions \
  --gpt_path /path/to/gpt2 \
  --model_name star-ecai2025 \
  --features 224p_5fps.npy \
  --model_type gpt \
  --gpt_type gpt2 \
  --pool PerceiverResamplerGlobalPosition \
  --stage caption
```

Debug a specific stage:

```bash
STAGE=caption bash run_debug.sh
STAGE=spotting bash run_debug.sh
STAGE=dvc bash run_debug.sh
```

Available stages:

- `full`
- `caption`
- `spotting`
- `dvc`

## Outputs

Training artifacts are saved under:

```text
models/<model_name>/
```

This directory is excluded from version control together with logs and `wandb/` outputs.

## Citation

If you find this repository useful, please cite our ECAI 2025 paper:

```text
STaR: Multi-Granular Spatio-Temporal Reasoning for Long-Form Dense Video Captioning
ECAI 2025
```

The formatted BibTeX entry can be added once the final publication metadata is fixed for public release.

## Notes

- This public release does not include datasets, pretrained GPT checkpoints, trained weights, or private experiment logs.
- The experimental `efficient_memory` path is kept in the codebase but disabled by default in the public release.
- `classifying.py` is retained for archival purposes, but the official released pipeline only uses `spotting.py` and `captioning.py`.
