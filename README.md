# doScenes Challenge Starter

This repo is a simple, end-to-end starter for the doScenes Instructed Driving Challenge. It is designed to go from a minimal baseline (history-only) to an instruction-conditioned model, then to map/vision ablations.

Quick links:
- Challenge overview: https://mi3-lab.github.io/doScenes_challenge
- Annotations repo: https://github.com/rossgreer/doScenes

## What the task is (plain English)

You are given:
- A short history of the ego vehicle's motion (past ~2 seconds).
- Scene context (map + cameras) in the full multimodal track.
- A natural-language instruction describing intended behavior.

You must predict the ego vehicle's future trajectory for the next ~6 seconds. The key question is whether language helps compared to a history-only baseline.

**Metrics**
- ADE: mean L2 distance across all future time steps.
- FDE: L2 distance at the final time step.
- Delta ADE: ADE_baseline - ADE_instruction (positive means language helps).
- Q97.5-filtered ADE: the mean after dropping the top 2.5% errors.

Source: https://mi3-lab.github.io/doScenes_challenge

## Repository layout

- `doscenes/`: minimal modeling + data utilities
- `scripts/`: data download + inspection helpers
- `configs/`: training configs
- `data/`: local data cache (not committed)

## Setup

1. Create a Python environment (3.10+).
2. Install dependencies with uv:

```bash
uv sync
```

## Step 1 — Download instructions

```bash
python scripts/download_instructions.py --out_dir data/instructions
python scripts/inspect_instructions.py --csv_dir data/instructions
```

This pulls the public annotations (instructions) and summarizes instruction types and common verbs.

## Step 2 — Prepare a toy dataset (no nuScenes required)

```bash
python scripts/make_toy_dataset.py --out_path data/toy_dataset.parquet
```

This creates synthetic trajectories with instructions so the baseline and language-conditioned pipeline can be tested end to end.

## Step 3 — Train a history-only baseline

```bash
python scripts/train.py \
  --dataset data/toy_dataset.parquet \
  --model history_mlp \
  --run_name baseline_history
```

## Step 4 — Add instruction conditioning

```bash
python scripts/train.py \
  --dataset data/toy_dataset.parquet \
  --model history_text_fusion \
  --run_name baseline_language
```

Compare ADE/FDE and compute Delta ADE (language vs baseline).

```bash
python scripts/evaluate.py --dataset data/toy_dataset.parquet --run_dir runs/baseline_history --out_json runs/baseline_history/metrics.json
python scripts/evaluate.py --dataset data/toy_dataset.parquet --run_dir runs/baseline_language --out_json runs/baseline_language/metrics.json
python scripts/compare_runs.py \
  --baseline_metrics runs/baseline_history/metrics.json \
  --instruction_metrics runs/baseline_language/metrics.json
```

## Step 5 — Add map/vision features (toy stub)

```bash
python scripts/train.py \
  --dataset data/toy_dataset.parquet \
  --model history_text_map_fusion \
  --run_name baseline_map_language
```

This uses synthetic `map_features` in the toy dataset to validate the fusion path. Replace `map_features` with real map/vision embeddings once you build them from nuScenes.

## Step 6 — Plug in real data

The real doScenes task uses nuScenes scenes and a separate instruction CSV. This repo expects you to create a parquet file with:

- `scene_id`: string or int
- `history`: list of (x, y) pairs for past steps
- `future`: list of (x, y) pairs for future steps
- `instruction`: text string
- `instruction_type`: optional single-letter code (s/d/sd)
- `map_features`: optional fixed-length vector for map/vision features

After you export that parquet, run the same `scripts/train.py`.

### Local tiny subset (20 scenes)

1. Download nuScenes to `data/nuscenes` (manual via nuScenes site).
2. Build a small parquet subset:

```bash
python scripts/build_nuscenes_subset.py \
  --nuscenes_root data/nuscenes \
  --instructions_dir data/instructions \
  --scene_count 20 \
  --out_path data/nuscenes_subset.parquet
```

## Step 7 — Submission prep

Use `scripts/predict.py` to create a CSV with predictions. Update the output format once you confirm the official submission schema.

```bash
python scripts/predict.py --dataset data/toy_dataset.parquet --run_dir runs/baseline_language --out_csv data/preds.csv
python scripts/validate_submission.py --csv data/preds.csv
```
