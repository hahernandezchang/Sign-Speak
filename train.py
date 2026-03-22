"""
train.py — ASL Classifier Trainer
─────────────────────────────────
Reads asl_data.csv (dynamic feature size from collector.py),
trains a neural net, and saves the model.

Install deps:
  pip install pandas scikit-learn torch

Run:
  python train.py
"""

import json
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from model_utils import ASLNet

DEFAULT_HIDDEN_SIZES = [256, 128]
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = (
    os.path.dirname(SCRIPT_DIR)
    if os.path.basename(SCRIPT_DIR).lower() == "hand_tracking"
    else SCRIPT_DIR
)
DATA_DIR = os.path.join(ROOT_DIR, "data")
MODEL_DIR = os.path.join(ROOT_DIR, "models")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ASL classifier from collected CSV features.")
    parser.add_argument("--csv-path", default=os.path.join(DATA_DIR, "asl_data.csv"))
    parser.add_argument("--model-out", default=os.path.join(MODEL_DIR, "asl_classifier.pt"))
    parser.add_argument("--label-map-out", default=os.path.join(MODEL_DIR, "asl_label_map.json"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--test-split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-class-weights", action="store_true", 
                        help="Use class weights to handle imbalanced data")
    parser.add_argument("--augment", action="store_true",
                        help="Use data augmentation (Gaussian noise) during training")
    parser.add_argument(
        "--hidden-sizes",
        default="256,128",
        help="Comma-separated hidden layer sizes, e.g. 256,128",
    )
    return parser.parse_args()


def parse_hidden_sizes(value: str) -> list[int]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        return DEFAULT_HIDDEN_SIZES
    return [int(p) for p in parts]


def augment_data(X: np.ndarray, noise_std: float = 0.01) -> np.ndarray:
    """Add small Gaussian noise to features for augmentation."""
    noise = np.random.normal(0, noise_std, X.shape).astype(np.float32)
    return X + noise


def make_train_val_split(
    X: np.ndarray,
    y: np.ndarray,
    test_split: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create a robust train/val split with graceful fallback for tiny datasets."""
    if len(X) < 2 or test_split <= 0:
        print("Skipping validation split due to small dataset or non-positive test split.")
        return X, X, y, y

    class_counts = np.bincount(y)
    can_stratify = len(class_counts) > 0 and int(class_counts.min()) >= 2
    stratify_target = y if can_stratify else None

    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=test_split,
            random_state=seed,
            stratify=stratify_target,
        )
    except ValueError as exc:
        print(f"Validation split fallback triggered: {exc}")
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=min(max(test_split, 0.1), 0.5),
            random_state=seed,
            stratify=None,
        )

    if len(np.unique(y_train)) < 2:
        print("Training split has fewer than 2 classes; using full dataset for both train/val.")
        return X, X, y, y

    if len(X_val) == 0:
        print("Validation split is empty; using training split as validation fallback.")
        X_val, y_val = X_train, y_train

    return X_train, X_val, y_train, y_val


def main() -> None:
    args = parse_args()
    hidden_sizes = parse_hidden_sizes(args.hidden_sizes)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading {args.csv_path}...")
    df = pd.read_csv(args.csv_path)

    if df.empty:
        raise ValueError("CSV is empty - collect some data first with collector.py")

    feature_cols = [c for c in df.columns if c != "label"]
    if len(feature_cols) == 0:
        raise ValueError(
            "No feature columns found in CSV. Ensure collector.py wrote numeric feature data."
        )

    input_size = len(feature_cols)

    X = df[feature_cols].values.astype(np.float32)
    y_raw = np.asarray(df["label"].astype(str).values)

    le = LabelEncoder()
    y = np.asarray(le.fit_transform(y_raw), dtype=np.int64)
    num_classes = len(le.classes_)

    print(f"  {len(X)} samples  |  {num_classes} classes: {list(le.classes_)}")
    print(f"  Feature shape: {X.shape}")

    # Compute class weights if requested
    class_weights = None
    if args.use_class_weights:
        # Count samples per class
        unique, counts = np.unique(y, return_counts=True)
        class_weights_dict = {c: len(y) / (num_classes * count) for c, count in zip(unique, counts)}
        class_weights = torch.tensor([class_weights_dict[i] for i in range(num_classes)], dtype=torch.float32)
        print(f"\n  Using class weights to handle imbalance:")
        for i, label in enumerate(le.classes_):
            print(f"    {label}: {class_weights[i]:.3f}")
        print()

    X_train, X_val, y_train, y_val = make_train_val_split(
        X, y, args.test_split, args.seed
    )

    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size)

    model = ASLNet(input_size, hidden_sizes, num_classes, args.dropout)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"\nTraining for {args.epochs} epochs...\n")
    best_val_acc = 0.0

    model_out_dir = os.path.dirname(args.model_out)
    if model_out_dir:
        os.makedirs(model_out_dir, exist_ok=True)
    label_map_out_dir = os.path.dirname(args.label_map_out)
    if label_map_out_dir:
        os.makedirs(label_map_out_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, correct = 0.0, 0

        for xb, yb in train_dl:
            # Apply augmentation to training data
            if args.augment:
                xb_np = xb.cpu().numpy() if xb.is_cuda else xb.numpy()
                xb_aug = augment_data(xb_np, noise_std=0.01)
                xb = torch.from_numpy(xb_aug).to(xb.device).to(xb.dtype)
            
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
            correct += (logits.argmax(1) == yb).sum().item()

        scheduler.step()
        train_loss /= len(train_ds)
        train_acc = correct / len(train_ds)

        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_dl:
                logits = model(xb)
                val_loss += criterion(logits, yb).item() * len(xb)
                val_correct += (logits.argmax(1) == yb).sum().item()

        val_loss /= len(val_ds)
        val_acc = val_correct / len(val_ds)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "input_size": input_size,
                    "hidden_sizes": hidden_sizes,
                    "num_classes": num_classes,
                    "dropout": args.dropout,
                },
                args.model_out,
            )

        if epoch % 5 == 0 or epoch == args.epochs:
            best_flag = "  <- best" if val_acc == best_val_acc else ""
            print(
                f"Epoch {epoch:3d}/{args.epochs}"
                f"  train loss {train_loss:.4f}  acc {train_acc:.3f}"
                f"  |  val loss {val_loss:.4f}  acc {val_acc:.3f}{best_flag}"
            )

    label_map = {str(i): label for i, label in enumerate(le.classes_)}
    with open(args.label_map_out, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)

    print(f"\nBest val accuracy : {best_val_acc:.3f}")
    print(f"Model saved       -> {args.model_out}")
    print(f"Label map saved   -> {args.label_map_out}")
    print("\nUse extractor.py or dual_runtime.py for live predictions.")


if __name__ == "__main__":
    main()