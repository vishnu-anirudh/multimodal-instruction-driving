from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

import open_clip
from nuscenes.nuscenes import NuScenes

DEFAULT_PROMPTS = [
    "a hazard on the road",
    "a pedestrian crossing the road",
    "a vehicle stopped ahead",
    "an obstacle in the lane",
    "construction work on the road",
    "an emergency vehicle ahead",
]


def sample_token_at_index(nusc: NuScenes, scene: dict, index: int) -> str:
    token = scene["first_sample_token"]
    for _ in range(index):
        token = nusc.get("sample", token)["next"]
        if not token:
            break
    if not token:
        raise ValueError(f"Scene {scene['name']} has fewer than {index + 1} samples")
    return token


def load_prompts(prompt_file: str | None) -> list[str]:
    if not prompt_file:
        return DEFAULT_PROMPTS
    path = Path(prompt_file)
    prompts = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not prompts:
        raise ValueError(f"No prompts found in {path}")
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--out_path", type=str, required=True)
    parser.add_argument("--nuscenes_root", type=str, required=True)
    parser.add_argument("--history_steps", type=int, default=4)
    parser.add_argument("--camera", type=str, default="CAM_FRONT")
    parser.add_argument("--model", type=str, default="ViT-B-32")
    parser.add_argument("--pretrained", type=str, default="laion2b_s34b_b79k")
    parser.add_argument("--prompt_file", type=str, default="")
    parser.add_argument("--device", type=str, default="")
    parser.add_argument(
        "--missing_policy",
        type=str,
        choices=["drop", "zero", "nan"],
        default="drop",
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.dataset)
    if "scene_id" not in df.columns:
        raise ValueError("Dataset must include scene_id to map to nuScenes scenes")

    nusc = NuScenes(version="v1.0-trainval", dataroot=args.nuscenes_root, verbose=False)
    scene_by_name = {scene["name"]: scene for scene in nusc.scene}

    device = args.device
    if not device:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    model = model.to(device)
    model.eval()
    tokenizer = open_clip.get_tokenizer(args.model)

    prompts = load_prompts(args.prompt_file or None)
    with torch.no_grad():
        text_tokens = tokenizer(prompts).to(device)
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    hazard_scores: list[float | None] = []
    for scene_id in tqdm(df["scene_id"].tolist(), desc="Scoring hazards"):
        scene = scene_by_name.get(scene_id)
        if scene is None:
            raise ValueError(f"Scene not found in nuScenes metadata: {scene_id}")
        sample_token = sample_token_at_index(nusc, scene, args.history_steps)
        sample = nusc.get("sample", sample_token)
        if args.camera not in sample["data"]:
            raise ValueError(f"Camera '{args.camera}' not found for scene {scene_id}")
        sample_data_token = sample["data"][args.camera]
        image_path = nusc.get_sample_data_path(sample_data_token)
        if not Path(image_path).exists():
            hazard_scores.append(None)
            continue
        image = Image.open(image_path).convert("RGB")
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            sims = (image_features @ text_features.T).squeeze(0)
            hazard_score = float(torch.max(sims).item())
        hazard_scores.append(hazard_score)

    df = df.copy()
    df["hazard_score"] = np.asarray(
        [score if score is not None else np.nan for score in hazard_scores], dtype=np.float32
    )
    if args.missing_policy == "drop":
        before = len(df)
        df = df.dropna(subset=["hazard_score"]).reset_index(drop=True)
        print(f"Dropped {before - len(df)} rows with missing images")
    elif args.missing_policy == "zero":
        df["hazard_score"] = df["hazard_score"].fillna(0.0)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Wrote hazard-augmented dataset to {out_path}")


if __name__ == "__main__":
    main()
