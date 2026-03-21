"""
train.py — ASL Classifier Trainer (126-feature version)
─────────────────────────────────────────────────────────
Reads asl_data.csv (126 features: 63 position + 63 velocity),
trains a neural net, saves the model.

Install deps:
  pip install pandas scikit-learn torch

Run:
  python train.py
"""

import json
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

CSV_PATH       = "asl_data.csv"
MODEL_OUT      = "asl_classifier.pt"
LABEL_MAP_OUT  = "asl_label_map.json"

# ── hyper-parameters ──────────────────────────────────────────────────────────
EPOCHS         = 60
BATCH_SIZE     = 32
LEARNING_RATE  = 1e-3
HIDDEN_SIZES   = [256, 128]  # slightly wider than before to handle 126 inputs
DROPOUT        = 0.3
TEST_SPLIT     = 0.15
SEED           = 42

INPUT_SIZE     = 126         # 63 position + 63 velocity

torch.manual_seed(SEED)
np.random.seed(SEED)

# ── load data ─────────────────────────────────────────────────────────────────
print(f"Loading {CSV_PATH}...")
df = pd.read_csv(CSV_PATH)

if df.empty:
    raise ValueError("CSV is empty — collect some data first with collector.py")

# sanity check — make sure this is the 126-feature format
feature_cols = [c for c in df.columns if c != "label"]
if len(feature_cols) != INPUT_SIZE:
    raise ValueError(
        f"Expected {INPUT_SIZE} feature columns, got {len(feature_cols)}.\n"
        "If you have an old 63-column CSV, delete it and re-collect with collector.py"
    )

X     = df[feature_cols].values.astype(np.float32)   # (N, 126)
y_raw = df["label"].values

le = LabelEncoder()
y  = le.fit_transform(y_raw).astype(np.int64)
num_classes = len(le.classes_)

print(f"  {len(X)} samples  |  {num_classes} classes: {list(le.classes_)}")
print(f"  Feature shape: {X.shape}  (63 position + 63 velocity)")

# ── train / val split ─────────────────────────────────────────────────────────
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=TEST_SPLIT, random_state=SEED, stratify=y
)

train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
val_ds   = TensorDataset(torch.from_numpy(X_val),   torch.from_numpy(y_val))
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE)


# ── model ─────────────────────────────────────────────────────────────────────
class ASLNet(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, dropout):
        super().__init__()
        layers = []
        prev = input_size
        for h in hidden_sizes:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


model     = ASLNet(INPUT_SIZE, HIDDEN_SIZES, num_classes, DROPOUT)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.CrossEntropyLoss()
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ── training loop ─────────────────────────────────────────────────────────────
print(f"\nTraining for {EPOCHS} epochs...\n")
best_val_acc = 0.0

for epoch in range(1, EPOCHS + 1):
    model.train()
    train_loss, correct = 0.0, 0

    for xb, yb in train_dl:
        optimizer.zero_grad()
        logits = model(xb)
        loss   = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(xb)
        correct    += (logits.argmax(1) == yb).sum().item()

    scheduler.step()
    train_loss /= len(train_ds)
    train_acc   = correct / len(train_ds)

    model.eval()
    val_loss, val_correct = 0.0, 0
    with torch.no_grad():
        for xb, yb in val_dl:
            logits       = model(xb)
            val_loss    += criterion(logits, yb).item() * len(xb)
            val_correct += (logits.argmax(1) == yb).sum().item()

    val_loss /= len(val_ds)
    val_acc   = val_correct / len(val_ds)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            "model_state":  model.state_dict(),
            "input_size":   INPUT_SIZE,
            "hidden_sizes": HIDDEN_SIZES,
            "num_classes":  num_classes,
            "dropout":      DROPOUT,
        }, MODEL_OUT)

    if epoch % 5 == 0 or epoch == EPOCHS:
        best_flag = "  ← best" if val_acc == best_val_acc else ""
        print(
            f"Epoch {epoch:3d}/{EPOCHS}"
            f"  train loss {train_loss:.4f}  acc {train_acc:.3f}"
            f"  |  val loss {val_loss:.4f}  acc {val_acc:.3f}{best_flag}"
        )

# ── save label map ────────────────────────────────────────────────────────────
label_map = {str(i): label for i, label in enumerate(le.classes_)}
with open(LABEL_MAP_OUT, "w") as f:
    json.dump(label_map, f, indent=2)

print(f"\nBest val accuracy : {best_val_acc:.3f}")
print(f"Model saved       → {MODEL_OUT}")
print(f"Label map saved   → {LABEL_MAP_OUT}")
print("\nRun python asl_landmark_extractor.py to use live predictions.")