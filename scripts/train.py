from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from torch import nn
from torch.utils.data import DataLoader, Subset

from doscenes.data.trajectory_dataset import TrajectoryDataset
from doscenes.metrics import batch_metrics
from doscenes.models.history_gru_text_fusion import HistoryGRUTextFusion
from doscenes.models.history_gru_text_map_fusion import HistoryGRUTextMapFusion
from doscenes.models.history_mlp import HistoryMLP
from doscenes.models.history_text_fusion import HistoryTextFusion
from doscenes.models.history_text_map_fusion import HistoryTextMapFusion


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def split_indices(n: int, val_ratio: float, seed: int) -> tuple[list[int], list[int]]:
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_size = int(n * val_ratio)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    return train_idx, val_idx


def collate_batch(batch):
    history = torch.tensor([b.history for b in batch], dtype=torch.float32)
    future = torch.tensor([b.future for b in batch], dtype=torch.float32)
    texts = [b.instruction for b in batch]
    map_features = [b.map_features for b in batch]
    return history, future, texts, map_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument(
        "--model",
        type=str,
        choices=[
            "history_mlp",
            "history_text_fusion",
            "history_text_map_fusion",
            "history_gru_text_fusion",
            "history_gru_text_map_fusion",
        ],
        required=True,
    )
    parser.add_argument(
        "--text_encoder",
        type=str,
        choices=["tfidf", "minilm"],
        default="tfidf",
    )
    parser.add_argument(
        "--minilm_model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument("--text_batch_size", type=int, default=64)
    parser.add_argument("--run_name", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    set_seed(args.seed)
    dataset = TrajectoryDataset(args.dataset)
    train_idx, val_idx = split_indices(len(dataset), args.val_ratio, args.seed)
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)

    sample = dataset[0]
    history_steps = sample.history.shape[0]
    future_steps = sample.future.shape[0]

    device = "mps" if torch.backends.mps.is_available() else "cpu"

    vectorizer = None
    st_model = None
    text_encoder = args.text_encoder

    def build_text_encoder() -> "SentenceTransformer":
        from sentence_transformers import SentenceTransformer

        target_device = device
        try:
            return SentenceTransformer(args.minilm_model, device=target_device)
        except Exception:
            return SentenceTransformer(args.minilm_model, device="cpu")

    if args.model == "history_mlp":
        model = HistoryMLP(history_steps=history_steps, future_steps=future_steps)
    else:
        if text_encoder == "tfidf":
            vectorizer = TfidfVectorizer(max_features=256)
            train_texts = [dataset[i].instruction for i in train_idx]
            vectorizer.fit(train_texts)
            text_dim = len(vectorizer.get_feature_names_out())
        else:
            st_model = build_text_encoder()
            text_dim = st_model.get_sentence_embedding_dimension()

        if args.model == "history_text_fusion":
            model = HistoryTextFusion(
                history_steps=history_steps,
                future_steps=future_steps,
                text_dim=text_dim,
            )
        elif args.model == "history_gru_text_fusion":
            model = HistoryGRUTextFusion(
                history_steps=history_steps,
                future_steps=future_steps,
                text_dim=text_dim,
            )
        else:
            if sample.map_features is None:
                raise ValueError("Dataset is missing map_features required for map fusion")
            map_dim = int(np.asarray(sample.map_features).shape[0])
            if args.model == "history_text_map_fusion":
                model = HistoryTextMapFusion(
                    history_steps=history_steps,
                    future_steps=future_steps,
                    text_dim=text_dim,
                    map_dim=map_dim,
                )
            else:
                model = HistoryGRUTextMapFusion(
                    history_steps=history_steps,
                    future_steps=future_steps,
                    text_dim=text_dim,
                    map_dim=map_dim,
                )

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    def encode_text(texts: list[str]) -> torch.Tensor:
        if text_encoder == "tfidf":
            feats = vectorizer.transform(texts).toarray().astype(np.float32)
        else:
            feats = st_model.encode(
                texts,
                batch_size=args.text_batch_size,
                convert_to_numpy=True,
                normalize_embeddings=False,
            ).astype(np.float32)
        return torch.tensor(feats, dtype=torch.float32, device=device)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for history, future, texts, map_features in train_loader:
            history = history.to(device)
            future = future.to(device)
            optimizer.zero_grad()
            if args.model == "history_mlp":
                pred = model(history)
            else:
                text_feats = encode_text(texts)
                if args.model in {"history_text_fusion", "history_gru_text_fusion"}:
                    pred = model(history, text_feats)
                else:
                    map_feats = torch.tensor(map_features, dtype=torch.float32, device=device)
                    pred = model(history, text_feats, map_feats)
            loss = loss_fn(pred, future)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * history.size(0)
        train_loss /= len(train_set)

        model.eval()
        preds = []
        gts = []
        with torch.no_grad():
            for history, future, texts, map_features in val_loader:
                history = history.to(device)
                future = future.to(device)
                if args.model == "history_mlp":
                    pred = model(history)
                else:
                    text_feats = encode_text(texts)
                    if args.model in {"history_text_fusion", "history_gru_text_fusion"}:
                        pred = model(history, text_feats)
                    else:
                        map_feats = torch.tensor(map_features, dtype=torch.float32, device=device)
                        pred = model(history, text_feats, map_feats)
                preds.append(pred.cpu().numpy())
                gts.append(future.cpu().numpy())
        metrics = batch_metrics(np.concatenate(preds, axis=0), np.concatenate(gts, axis=0))
        print(
            f"Epoch {epoch:02d} | loss {train_loss:.4f} | "
            f"ADE {metrics['ade']:.3f} | FDE {metrics['fde']:.3f} | Q97.5 ADE {metrics['q975_ade']:.3f}"
        )

    run_dir = Path("runs") / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), run_dir / "model.pt")
    if vectorizer is not None:
        import joblib

        joblib.dump(vectorizer, run_dir / "vectorizer.joblib")

    config = vars(args)
    config.update(
        {
            "history_steps": history_steps,
            "future_steps": future_steps,
            "device": device,
            "text_encoder": text_encoder,
            "minilm_model": args.minilm_model,
            "text_batch_size": args.text_batch_size,
        }
    )
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)

    print(f"Saved run to {run_dir}")


if __name__ == "__main__":
    main()
