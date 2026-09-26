"""Command-line training entry point for the minimal SFL trajectory experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from olmo.sfl.trajectory import DeltaTransitionModel, SuccessiveClauseDataset, load_trajectory_arrays, trajectory_loss


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the minimal 9D SFL delta-transition model.")
    parser.add_argument("dataset", type=Path, help="Path to uam_meaning_trajectories.npz")
    parser.add_argument("--output-dir", type=Path, default=Path("sfl_runs"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    arrays = load_trajectory_arrays(args.dataset)
    dataset = SuccessiveClauseDataset(arrays)
    if not len(dataset):
        raise ValueError("dataset contains no within-document successive-clause transitions")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    model = DeltaTransitionModel(args.hidden_dim)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    losses: list[float] = []
    for epoch in range(args.epochs):
        model.train()
        total = 0.0
        for current, change, next_state in loader:
            predicted_change = model(current)
            loss = trajectory_loss(current, change, next_state, predicted_change)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * current.shape[0]
        epoch_loss = total / len(dataset)
        losses.append(epoch_loss)
        print(f"epoch={epoch + 1} trajectory_loss={epoch_loss:.6f}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "sfl_model_3x3.pt"
    torch.save({"model_state_dict": model.state_dict(), "hidden_dim": args.hidden_dim, "state_dim": 9}, checkpoint)
    record = {
        "dataset": str(args.dataset),
        "dataset_sha256": file_hash(args.dataset),
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "hidden_dim": args.hidden_dim,
        "transition_count": len(dataset),
        "losses": losses,
        "checkpoint": str(checkpoint),
    }
    (args.output_dir / "run_record.json").write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
