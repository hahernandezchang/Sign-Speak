from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, cast

import torch


class ASLNet(torch.nn.Module):
    def __init__(self, input_size: int, hidden_sizes: list[int], num_classes: int, dropout: float):
        super().__init__()
        layers = []
        prev = input_size
        for h in hidden_sizes:
            layers += [
                torch.nn.Linear(prev, h),
                torch.nn.BatchNorm1d(h),
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
            ]
            prev = h
        layers.append(torch.nn.Linear(prev, num_classes))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


@dataclass
class LoadedClassifier:
    model: ASLNet
    label_map: dict[str, str]
    input_size: int


def load_classifier(model_path: str, label_map_path: str) -> LoadedClassifier:
    ckpt_obj = torch.load(model_path, map_location="cpu")
    if not isinstance(ckpt_obj, dict):
        raise RuntimeError(f"Invalid classifier checkpoint format: {model_path}")
    ckpt = cast(dict[str, Any], ckpt_obj)

    with open(label_map_path, "r", encoding="utf-8") as f:
        label_map_obj = json.load(f)
    if not isinstance(label_map_obj, dict):
        raise RuntimeError(f"Invalid label map format: {label_map_path}")
    label_map = {str(k): str(v) for k, v in label_map_obj.items()}

    model = ASLNet(
        int(ckpt["input_size"]),
        list(ckpt["hidden_sizes"]),
        int(ckpt["num_classes"]),
        float(ckpt["dropout"]),
    )
    model.load_state_dict(cast(dict[str, Any], ckpt["model_state"]))
    model.eval()

    return LoadedClassifier(model=model, label_map=label_map, input_size=int(ckpt["input_size"]))


def predict_label(classifier: LoadedClassifier, features: list[float]) -> tuple[str, float] | None:
    if len(features) != classifier.input_size:
        return None

    x = torch.tensor([features], dtype=torch.float32)
    with torch.no_grad():
        logits = classifier.model(x)
        probs = torch.softmax(logits, dim=1)
        idx = int(probs.argmax(1).item())
        confidence = float(probs[0, idx].item())

    predicted = classifier.label_map.get(str(idx))
    if predicted is None:
        return None
    return predicted, confidence
