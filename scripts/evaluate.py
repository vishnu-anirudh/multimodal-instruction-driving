from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from doscenes.data.trajectory_dataset import TrajectoryDataset
from doscenes.metrics import batch_metrics
from doscenes.models.history_gru_text_fusion import HistoryGRUTextFusion
from doscenes.models.history_gru_text_map_fusion import HistoryGRUTextMapFusion
from doscenes.models.history_mlp import HistoryMLP
from doscenes.models.history_text_fusion import HistoryTextFusion
from doscenes.models.history_text_map_fusion import HistoryTextMapFusion


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--out_json", type=str, default="")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    with open(run_dir / "config.json", "r", encoding="utf-8") as f:
        config = json.load(f)

    dataset = TrajectoryDataset(args.dataset)
    history_steps = config["history_steps"]
    future_steps = config["future_steps"]
    model_type = config["model"]
    text_encoder = config.get("text_encoder", "tfidf")
    minilm_model = config.get("minilm_model", "sentence-transformers/all-MiniLM-L6-v2")
    text_batch_size = config.get("text_batch_size", 64)

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    def build_text_encoder() -> "SentenceTransformer":
        from sentence_transformers import SentenceTransformer

        target_device = device
        try:
            return SentenceTransformer(minilm_model, device=target_device)
        except Exception:
            return SentenceTransformer(minilm_model, device="cpu")

    if model_type == "history_mlp":
        model = HistoryMLP(history_steps, future_steps)
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

        if model_type == "history_text_fusion":
            model = HistoryTextFusion(history_steps, future_steps, text_dim=text_dim)
        elif model_type == "history_gru_text_fusion":
            model = HistoryGRUTextFusion(history_steps, future_steps, text_dim=text_dim)
        else:
            sample = dataset[0]
            if sample.map_features is None:
                raise ValueError("Dataset is missing map_features required for map fusion")
            map_dim = int(np.asarray(sample.map_features).shape[0])
            if model_type == "history_text_map_fusion":
                model = HistoryTextMapFusion(history_steps, future_steps, text_dim=text_dim, map_dim=map_dim)
            else:
                model = HistoryGRUTextMapFusion(history_steps, future_steps, text_dim=text_dim, map_dim=map_dim)

    model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device))
    model.to(device)
    model.eval()

    preds = []
    gts = []
    with torch.no_grad():
        for item in dataset:
            history = torch.tensor(item.history, dtype=torch.float32, device=device).unsqueeze(0)
            future = torch.tensor(item.future, dtype=torch.float32, device=device).unsqueeze(0)
            if model_type == "history_mlp":
                pred = model(history)
            else:
                if text_encoder == "tfidf":
                    feats = vectorizer.transform([item.instruction]).toarray().astype(np.float32)
                else:
                    feats = st_model.encode(
                        [item.instruction],
                        batch_size=text_batch_size,
                        convert_to_numpy=True,
                        normalize_embeddings=False,
                    ).astype(np.float32)
                text_feats = torch.tensor(feats, dtype=torch.float32, device=device)
                if model_type in {"history_text_fusion", "history_gru_text_fusion"}:
                    pred = model(history, text_feats)
                else:
                    map_feats = torch.tensor(item.map_features, dtype=torch.float32, device=device).unsqueeze(0)
                    pred = model(history, text_feats, map_feats)
            preds.append(pred.squeeze(0).cpu().numpy())
            gts.append(future.squeeze(0).cpu().numpy())

    metrics = batch_metrics(np.stack(preds, axis=0), np.stack(gts, axis=0))
    print(json.dumps(metrics, indent=2))

    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print(f"Wrote metrics to {out_path}")


if __name__ == "__main__":
    main()
