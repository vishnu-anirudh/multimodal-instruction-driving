from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from doscenes.data.trajectory_dataset import TrajectoryDataset
from doscenes.models.history_gru_text_fusion import HistoryGRUTextFusion
from doscenes.models.history_gru_text_map_fusion import HistoryGRUTextMapFusion
from doscenes.models.history_mlp import HistoryMLP
from doscenes.models.history_text_fusion import HistoryTextFusion
from doscenes.models.history_text_map_fusion import HistoryTextMapFusion


def parse_allowed_types(value: str) -> tuple[str, ...] | None:
    cleaned = value.strip()
    if not cleaned or cleaned.lower() == "all":
        return None
    return tuple(part.strip() for part in cleaned.split(",") if part.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--out_csv", type=str, required=True)
    parser.add_argument("--min_instruction_words", type=int, default=5)
    parser.add_argument("--allowed_instruction_types", type=str, default="")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--text_cache_path", type=str, default="")
    parser.add_argument("--use_hazard", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    with open(run_dir / "config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    allowed_types = parse_allowed_types(args.allowed_instruction_types)
    dataset = TrajectoryDataset(
        args.dataset,
        min_instruction_words=args.min_instruction_words,
        allowed_instruction_types=allowed_types,
        normalize=args.normalize,
    )
    use_hazard = bool(config.get("use_hazard", False) or args.use_hazard)
    if use_hazard and "hazard_score" not in dataset.frame.columns:
        raise ValueError("use_hazard is set but dataset is missing hazard_score column")
    history_steps = config["history_steps"]
    future_steps = config["future_steps"]
    model_type = config["model"]
    text_encoder = config.get("text_encoder", "tfidf")
    minilm_model = config.get("minilm_model", "sentence-transformers/all-MiniLM-L6-v2")
    text_batch_size = config.get("text_batch_size", 64)

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    def build_text_encoder() -> "SentenceTransformer":
        from sentence_transformers import SentenceTransformer

        target_device = device
        try:
            return SentenceTransformer(minilm_model, device=target_device)
        except Exception:
            return SentenceTransformer(minilm_model, device="cpu")

    text_cache = None
    hazard_dim = 1 if use_hazard else 0
    if model_type == "history_mlp":
        model = HistoryMLP(history_steps, future_steps, hazard_dim=hazard_dim)
        vectorizer = None
    else:
        vectorizer = None
        st_model = None
        if text_encoder == "tfidf":
            import joblib

            vectorizer = joblib.load(run_dir / "vectorizer.joblib")
            text_dim = len(vectorizer.get_feature_names_out())
        else:
            st_model = build_text_encoder()
            text_dim = st_model.get_sentence_embedding_dimension()
            if args.text_cache_path:
                cache_path = Path(args.text_cache_path)
                if cache_path.exists():
                    data = np.load(cache_path, allow_pickle=True)
                    texts = data["texts"].tolist()
                    embeddings = data["embeddings"]
                    text_cache = {text: embeddings[i] for i, text in enumerate(texts)}
                else:
                    unique_texts = (
                        dataset.frame["instruction"]
                        .fillna("")
                        .astype(str)
                        .unique()
                        .tolist()
                    )
                    embeddings = st_model.encode(
                        unique_texts,
                        batch_size=text_batch_size,
                        convert_to_numpy=True,
                        normalize_embeddings=False,
                    ).astype(np.float32)
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez(cache_path, texts=np.array(unique_texts, dtype=object), embeddings=embeddings)
                    text_cache = {text: embeddings[i] for i, text in enumerate(unique_texts)}

        if model_type == "history_text_fusion":
            model = HistoryTextFusion(history_steps, future_steps, text_dim=text_dim, hazard_dim=hazard_dim)
        elif model_type == "history_gru_text_fusion":
            model = HistoryGRUTextFusion(history_steps, future_steps, text_dim=text_dim, hazard_dim=hazard_dim)
        else:
            sample = dataset[0]
            if sample.map_features is None:
                raise ValueError("Dataset is missing map_features required for map fusion")
            map_dim = int(np.asarray(sample.map_features).shape[0])
            if model_type == "history_text_map_fusion":
                model = HistoryTextMapFusion(
                    history_steps,
                    future_steps,
                    text_dim=text_dim,
                    map_dim=map_dim,
                    hazard_dim=hazard_dim,
                )
            else:
                model = HistoryGRUTextMapFusion(
                    history_steps,
                    future_steps,
                    text_dim=text_dim,
                    map_dim=map_dim,
                    hazard_dim=hazard_dim,
                )

    model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device))
    model.to(device)
    model.eval()

    rows = []
    for item in dataset:
        history = torch.tensor(item.history, dtype=torch.float32, device=device).unsqueeze(0)
        hazard = None
        if use_hazard:
            if item.hazard_score is None:
                raise ValueError("Hazard score missing for at least one sample")
            hazard = torch.tensor([[item.hazard_score]], dtype=torch.float32, device=device)
        if model_type == "history_mlp":
            pred = model(history, hazard)
        else:
            if text_encoder == "tfidf":
                feats = vectorizer.transform([item.instruction]).toarray().astype(np.float32)
            else:
                if text_cache is not None and item.instruction in text_cache:
                    feats = np.asarray([text_cache[item.instruction]], dtype=np.float32)
                else:
                    feats = st_model.encode(
                        [item.instruction],
                        batch_size=text_batch_size,
                        convert_to_numpy=True,
                        normalize_embeddings=False,
                    ).astype(np.float32)
            text_feats = torch.tensor(feats, dtype=torch.float32, device=device)
            if model_type in {"history_text_fusion", "history_gru_text_fusion"}:
                pred = model(history, text_feats, hazard)
            else:
                map_feats = torch.tensor(item.map_features, dtype=torch.float32, device=device).unsqueeze(0)
                pred = model(history, text_feats, map_feats, hazard)
        pred_np = pred.squeeze(0).detach().cpu().numpy()
        rows.append({"scene_id": item.scene_id, "future": pred_np.tolist()})

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Wrote predictions to {out_path}")
    print("Note: update the CSV format to match the official submission schema.")


if __name__ == "__main__":
    main()
