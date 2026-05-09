#!/usr/bin/env python3
"""Fungi image attack comparison with lazy loading and transfer learning.

The target direction defaults to poisonous -> edible, because that is the
safety-critical mistake for a mushroom classifier. Training uses a lazy
PyTorch Dataset/DataLoader pipeline and a lightweight pretrained backbone by
default. Attack success for evasion methods is conditional on clean correctness.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision.models import MobileNet_V3_Small_Weights, ResNet18_Weights


ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("TORCH_HOME", str(ROOT / ".cache" / "torch"))
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class FungiImageDataset(Dataset):
    def __init__(self, root: Path, split: str, class_names: list[str], image_size: int) -> None:
        self.root = root
        self.split = split
        self.class_names = class_names
        self.image_size = image_size
        self.samples: list[tuple[Path, int]] = []
        for label, class_name in enumerate(class_names):
            class_dir = root / split / class_name
            files = sorted(p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
            self.samples.extend((path, label) for path in files)
        if not self.samples:
            raise RuntimeError(f"No image samples found in {root / split}")
        self.labels = [label for _, label in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[index]
        return image_to_tensor(path, self.image_size), label


class LabelOverrideDataset(Dataset):
    def __init__(self, base: Dataset, labels: Sequence[int], overrides: dict[int, int]) -> None:
        self.base = base
        self.labels = list(labels)
        for index, label in overrides.items():
            self.labels[index] = label

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        x, _ = self.base[index]
        return x, self.labels[index]


class AppendTriggerDataset(Dataset):
    def __init__(
        self,
        base: Dataset,
        labels: Sequence[int],
        append_indices: Sequence[int],
        append_labels: Sequence[int],
        trigger_size: int,
    ) -> None:
        self.base = base
        self.base_labels = list(labels)
        self.append_indices = list(append_indices)
        self.append_labels = list(append_labels)
        self.trigger_size = trigger_size
        self.labels = self.base_labels + self.append_labels

    def __len__(self) -> int:
        return len(self.base_labels) + len(self.append_indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        if index < len(self.base_labels):
            return self.base[index]
        append_pos = index - len(self.base_labels)
        x, _ = self.base[self.append_indices[append_pos]]
        return apply_trigger(x.unsqueeze(0), self.trigger_size).squeeze(0), self.append_labels[append_pos]


class TensorAppendDataset(Dataset):
    def __init__(self, base: Dataset, labels: Sequence[int], extra_x: torch.Tensor, extra_y: torch.Tensor) -> None:
        self.base = base
        self.base_labels = list(labels)
        self.extra_x = extra_x.detach().cpu()
        self.extra_y = extra_y.detach().cpu().long()
        self.labels = self.base_labels + [int(v) for v in self.extra_y.tolist()]

    def __len__(self) -> int:
        return len(self.base_labels) + len(self.extra_y)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        if index < len(self.base_labels):
            return self.base[index]
        pos = index - len(self.base_labels)
        return self.extra_x[pos], int(self.extra_y[pos])


class TransferFungiModel(nn.Module):
    def __init__(self, num_classes: int, model_name: str, pretrained: bool, freeze_features: bool) -> None:
        super().__init__()
        self.model_name = model_name
        self.register_buffer("mean", IMAGENET_MEAN.clone())
        self.register_buffer("std", IMAGENET_STD.clone())

        if model_name == "mobilenet_v3_small":
            weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            base = models.mobilenet_v3_small(weights=weights)
            self.feature_net = base.features
            self.feature_dim = 576
        elif model_name == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            base = models.resnet18(weights=weights)
            self.feature_net = nn.Sequential(*list(base.children())[:-2])
            self.feature_dim = 512
        else:
            raise ValueError(f"Unsupported model_name={model_name}")

        if freeze_features:
            for param in self.feature_net.parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Dropout(0.20),
            nn.Linear(self.feature_dim, num_classes),
        )

    def normalized(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        z = self.feature_net(self.normalized(x))
        return F.adaptive_avg_pool2d(z, (1, 1)).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_features(x))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fungi dataset attack comparison.")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "fungi")
    parser.add_argument("--image-size", type=int, default=160)
    parser.add_argument("--model", choices=["mobilenet_v3_small", "resnet18"], default="mobilenet_v3_small")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--freeze-features", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--target-class", default="edible")
    parser.add_argument("--source-class", default="poisonous")
    parser.add_argument("--eval-samples", type=int, default=40)
    parser.add_argument("--zoo-samples", type=int, default=8)
    parser.add_argument("--trigger-size", type=int, default=18)
    parser.add_argument("--adversarial-patch-size", type=int, default=32)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))


def image_to_tensor(path: Path, image_size: int) -> torch.Tensor:
    with Image.open(path) as img:
        img = ImageOps.exif_transpose(img).convert("RGB")
        img = img.resize((image_size, image_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def discover_classes(root: Path) -> list[str]:
    class_names = sorted(p.name for p in (root / "train").iterdir() if p.is_dir() and not p.name.startswith("."))
    if len(class_names) < 2:
        raise RuntimeError(f"Expected at least two classes under {root / 'train'}")
    return class_names


def labels_of(dataset: Dataset) -> list[int]:
    labels = getattr(dataset, "labels", None)
    if labels is None:
        raise RuntimeError(f"Dataset {type(dataset).__name__} does not expose labels")
    return [int(v) for v in labels]


def class_counts(labels: Sequence[int], num_classes: int) -> torch.Tensor:
    return torch.bincount(torch.tensor(labels, dtype=torch.long), minlength=num_classes).float()


def materialize_dataset(dataset: Dataset) -> tuple[torch.Tensor, torch.Tensor]:
    xs: list[torch.Tensor] = []
    ys: list[int] = []
    for i in range(len(dataset)):
        x, y = dataset[i]
        xs.append(x)
        ys.append(int(y))
    return torch.stack(xs), torch.tensor(ys, dtype=torch.long)


def materialize_indices(dataset: Dataset, indices: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
    xs: list[torch.Tensor] = []
    ys: list[int] = []
    for index in indices:
        x, y = dataset[int(index)]
        xs.append(x)
        ys.append(int(y))
    return torch.stack(xs), torch.tensor(ys, dtype=torch.long)


def augment_batch(x: torch.Tensor) -> torch.Tensor:
    out = x.clone()
    flip_mask = torch.rand(len(out), device=out.device) < 0.5
    out[flip_mask] = torch.flip(out[flip_mask], dims=[3])
    brightness = 0.85 + 0.30 * torch.rand(len(out), 1, 1, 1, device=out.device)
    return torch.clamp(out * brightness, 0.0, 1.0)


def train_model(
    dataset: Dataset,
    num_classes: int,
    seed: int,
    epochs: int,
    batch_size: int,
    model_name: str,
    pretrained: bool,
    freeze_features: bool,
    augment: bool = True,
) -> TransferFungiModel:
    torch.manual_seed(seed)
    model = TransferFungiModel(num_classes, model_name, pretrained, freeze_features)
    counts = class_counts(labels_of(dataset), num_classes)
    weights = counts.sum() / (num_classes * torch.clamp(counts, min=1.0))
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    for _ in range(epochs):
        model.train()
        if freeze_features:
            model.feature_net.eval()
        for xb, yb in loader:
            if augment:
                xb = augment_batch(xb)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb), yb, weight=weights)
            loss.backward()
            optimizer.step()
    return model.eval()


@torch.no_grad()
def predict(model: nn.Module, x: torch.Tensor, batch_size: int = 64) -> torch.Tensor:
    preds: list[torch.Tensor] = []
    model.eval()
    for start in range(0, len(x), batch_size):
        preds.append(model(x[start : start + batch_size]).argmax(dim=1))
    return torch.cat(preds)


@torch.no_grad()
def accuracy_on_dataset(model: nn.Module, dataset: Dataset, batch_size: int = 64) -> float:
    correct = 0
    total = 0
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    for xb, yb in loader:
        correct += int((model(xb).argmax(dim=1) == yb).sum())
        total += len(yb)
    return correct / max(total, 1)


def apply_trigger(x: torch.Tensor, trigger_size: int, value: float = 1.0) -> torch.Tensor:
    patched = x.clone()
    patched[:, :, -trigger_size:, -trigger_size:] = value
    return patched


def source_to_target_rate(model: nn.Module, x: torch.Tensor, y: torch.Tensor, source: int, target: int) -> float:
    mask = y == source
    if int(mask.sum()) == 0:
        return 0.0
    return float((predict(model, x[mask]) == target).float().mean().item())


def trigger_asr(model: nn.Module, x: torch.Tensor, y: torch.Tensor, source: int, target: int, trigger_size: int) -> float:
    mask = y == source
    if int(mask.sum()) == 0:
        return 0.0
    clean_x = x[mask]
    clean_y = y[mask]
    clean_pred = predict(model, clean_x)
    eligible = clean_pred == clean_y
    if int(eligible.sum()) == 0:
        return 0.0
    triggered_pred = predict(model, apply_trigger(clean_x[eligible], trigger_size))
    return float((triggered_pred == target).float().mean().item())


def additive_trigger_asr(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    source: int,
    target: int,
    trigger: torch.Tensor,
) -> float:
    mask = y == source
    if int(mask.sum()) == 0:
        return 0.0
    clean_x = x[mask]
    clean_y = y[mask]
    clean_pred = predict(model, clean_x)
    eligible = clean_pred == clean_y
    if int(eligible.sum()) == 0:
        return 0.0
    triggered_pred = predict(model, add_stealth_trigger(clean_x[eligible], trigger))
    return float((triggered_pred == target).float().mean().item())


def random_label_flip(labels: Sequence[int], rate: float, seed: int, num_classes: int) -> tuple[dict[int, int], int]:
    rng = np.random.default_rng(seed)
    count = int(round(len(labels) * rate))
    indices = rng.choice(len(labels), size=count, replace=False)
    overrides: dict[int, int] = {}
    for idx in indices:
        choices = [c for c in range(num_classes) if c != int(labels[idx])]
        overrides[int(idx)] = int(rng.choice(choices))
    return overrides, count


def targeted_label_flip(labels: Sequence[int], source: int, target: int, source_rate: float, seed: int) -> tuple[dict[int, int], int]:
    rng = np.random.default_rng(seed)
    source_indices = [i for i, y in enumerate(labels) if int(y) == source]
    count = int(round(len(source_indices) * source_rate))
    chosen = rng.choice(source_indices, size=count, replace=False)
    return {int(idx): target for idx in chosen}, count


def subpopulation_indices(dataset: Dataset, source: int, quantile: float) -> list[int]:
    labels = labels_of(dataset)
    source_indices = [i for i, y in enumerate(labels) if y == source]
    x_source, _ = materialize_indices(dataset, source_indices)
    green_minus_red = x_source[:, 1].mean(dim=(1, 2)) - x_source[:, 0].mean(dim=(1, 2))
    h, w = x_source.shape[2], x_source.shape[3]
    center = x_source[:, :, h // 4 : 3 * h // 4, w // 4 : 3 * w // 4].mean(dim=(1, 2, 3))
    score = green_minus_red + 0.25 * center
    threshold = torch.quantile(score, quantile)
    selected = [source_indices[i] for i, value in enumerate(score) if float(value) >= float(threshold)]
    return selected


def subpopulation_label_flip(
    dataset: Dataset,
    source: int,
    target: int,
    quantile: float,
    flip_rate: float,
    seed: int,
) -> tuple[dict[int, int], int, list[int]]:
    rng = np.random.default_rng(seed)
    selected = subpopulation_indices(dataset, source, quantile)
    count = max(1, int(round(len(selected) * flip_rate)))
    chosen = rng.choice(selected, size=count, replace=False)
    return {int(idx): target for idx in chosen}, count, selected


def subpopulation_to_target_rate(
    model: nn.Module,
    val_dataset: Dataset,
    source: int,
    target: int,
    quantile: float,
) -> float:
    selected = subpopulation_indices(val_dataset, source, quantile)
    if not selected:
        return 0.0
    x, y = materialize_indices(val_dataset, selected)
    pred = predict(model, x)
    return float((pred == target).float().mean().item())


def backdoor_dataset(train_dataset: Dataset, source: int, target: int, rate: float, trigger_size: int, seed: int) -> tuple[AppendTriggerDataset, int]:
    labels = labels_of(train_dataset)
    rng = np.random.default_rng(seed)
    source_indices = [i for i, y in enumerate(labels) if y == source]
    count = min(len(source_indices), int(round(len(labels) * rate)))
    chosen = [int(v) for v in rng.choice(source_indices, size=count, replace=False)]
    return AppendTriggerDataset(train_dataset, labels, chosen, [target] * count, trigger_size), count


def clean_label_backdoor_dataset(train_dataset: Dataset, target: int, rate: float, trigger_size: int, seed: int) -> tuple[AppendTriggerDataset, int]:
    labels = labels_of(train_dataset)
    rng = np.random.default_rng(seed)
    target_indices = [i for i, y in enumerate(labels) if y == target]
    count = min(len(target_indices), int(round(len(labels) * rate)))
    chosen = [int(v) for v in rng.choice(target_indices, size=count, replace=False)]
    return AppendTriggerDataset(train_dataset, labels, chosen, [target] * count, trigger_size), count


def true_feature_collision(
    model: TransferFungiModel,
    dataset: Dataset,
    base_class: int,
    target_feature_class: int,
    count: int,
    eps: float = 0.08,
    iters: int = 150,
    seed: int = 2027,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    rng = np.random.default_rng(seed)
    labels = labels_of(dataset)
    base_idx = [i for i, y in enumerate(labels) if y == base_class]
    target_idx = [i for i, y in enumerate(labels) if y == target_feature_class]
    count = min(count, len(base_idx))
    chosen = [int(v) for v in rng.choice(base_idx, size=count, replace=False)]
    x_base, _ = materialize_indices(dataset, chosen)
    x_target, _ = materialize_indices(dataset, target_idx)

    model.eval()
    with torch.no_grad():
        target_features = model.forward_features(x_target).mean(dim=0, keepdim=True).detach()

    x_poison = x_base.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([x_poison], lr=0.01)
    for _ in range(iters):
        optimizer.zero_grad(set_to_none=True)
        current_features = model.forward_features(x_poison)
        loss = F.mse_loss(current_features, target_features.expand_as(current_features))
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            delta = torch.clamp(x_poison - x_base, min=-eps, max=eps)
            x_poison.copy_(torch.clamp(x_base + delta, 0.0, 1.0))
    poison_y = torch.full((count,), base_class, dtype=torch.long)
    return x_poison.detach(), poison_y, count


def classifier_params(model: TransferFungiModel) -> list[torch.nn.Parameter]:
    return [p for p in model.classifier.parameters() if p.requires_grad]


def flat_parameter_gradient(
    loss: torch.Tensor,
    params: Sequence[torch.nn.Parameter],
    create_graph: bool,
) -> torch.Tensor:
    grads = torch.autograd.grad(loss, params, create_graph=create_graph, retain_graph=True, allow_unused=True)
    flat: list[torch.Tensor] = []
    for param, grad in zip(params, grads):
        if grad is None:
            flat.append(torch.zeros_like(param).flatten())
        else:
            flat.append(grad.flatten())
    return torch.cat(flat)


def gradient_matching_poison(
    model: TransferFungiModel,
    dataset: Dataset,
    source: int,
    target: int,
    count: int,
    eps: float,
    iters: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Witches' Brew-style clean-label poison via first-order gradient alignment."""
    rng = np.random.default_rng(seed)
    labels = labels_of(dataset)
    poison_idx = [i for i, y in enumerate(labels) if y == target]
    objective_idx = [i for i, y in enumerate(labels) if y == source]
    count = min(count, len(poison_idx), len(objective_idx))
    chosen_poison = [int(v) for v in rng.choice(poison_idx, size=count, replace=False)]
    chosen_objective = [int(v) for v in rng.choice(objective_idx, size=count, replace=False)]
    x_base, _ = materialize_indices(dataset, chosen_poison)
    x_objective, _ = materialize_indices(dataset, chosen_objective)
    y_target = torch.full((count,), target, dtype=torch.long)
    params = classifier_params(model)

    model.eval()
    model.zero_grad(set_to_none=True)
    target_loss = F.cross_entropy(model(x_objective), y_target)
    g_target = flat_parameter_gradient(target_loss, params, create_graph=False).detach()

    x_poison = x_base.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([x_poison], lr=0.01)
    for _ in range(iters):
        optimizer.zero_grad(set_to_none=True)
        model.zero_grad(set_to_none=True)
        poison_loss = F.cross_entropy(model(x_poison), y_target)
        g_poison = flat_parameter_gradient(poison_loss, params, create_graph=True)
        align_loss = 1.0 - F.cosine_similarity(g_poison.unsqueeze(0), g_target.unsqueeze(0), dim=1, eps=1e-8).mean()
        visual_loss = 0.01 * F.mse_loss(x_poison, x_base)
        (align_loss + visual_loss).backward()
        optimizer.step()
        with torch.no_grad():
            delta = torch.clamp(x_poison - x_base, min=-eps, max=eps)
            x_poison.copy_(torch.clamp(x_base + delta, 0.0, 1.0))
    return x_poison.detach(), y_target, count


def stealth_trigger_pattern(channels: int, height: int, width: int, eps: float, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    yy, xx = torch.meshgrid(torch.linspace(0, 1, height), torch.linspace(0, 1, width), indexing="ij")
    pattern_channels: list[torch.Tensor] = []
    for channel in range(channels):
        phase = float(rng.uniform(0.0, 2.0 * np.pi))
        freq_x = 7.0 + channel
        freq_y = 5.0 + 0.5 * channel
        wave = torch.sin(2.0 * np.pi * (freq_x * xx + freq_y * yy) + phase)
        pattern_channels.append(wave)
    pattern = torch.stack(pattern_channels).unsqueeze(0)
    pattern = pattern / torch.clamp(pattern.abs().amax(), min=1e-8)
    return eps * pattern


def add_stealth_trigger(x: torch.Tensor, pattern: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x + pattern.to(x.device), 0.0, 1.0)


def sleeper_agent_poison(
    model: TransferFungiModel,
    dataset: Dataset,
    source: int,
    target: int,
    count: int,
    trigger_eps: float,
    poison_eps: float,
    iters: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
    """Clean-label gradient-aligned backdoor inspired by Sleeper Agent training."""
    rng = np.random.default_rng(seed)
    labels = labels_of(dataset)
    target_idx = [i for i, y in enumerate(labels) if y == target]
    source_idx = [i for i, y in enumerate(labels) if y == source]
    count = min(count, len(target_idx), len(source_idx))
    chosen_target = [int(v) for v in rng.choice(target_idx, size=count, replace=False)]
    chosen_source = [int(v) for v in rng.choice(source_idx, size=count, replace=False)]
    x_target, _ = materialize_indices(dataset, chosen_target)
    x_source, _ = materialize_indices(dataset, chosen_source)
    trigger = stealth_trigger_pattern(x_target.shape[1], x_target.shape[2], x_target.shape[3], trigger_eps, seed)
    x_base = add_stealth_trigger(x_target, trigger)
    x_triggered_source = add_stealth_trigger(x_source, trigger)
    y_target = torch.full((count,), target, dtype=torch.long)
    params = classifier_params(model)

    model.eval()
    model.zero_grad(set_to_none=True)
    target_loss = F.cross_entropy(model(x_triggered_source), y_target)
    g_target = flat_parameter_gradient(target_loss, params, create_graph=False).detach()

    x_poison = x_base.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([x_poison], lr=0.01)
    for _ in range(iters):
        optimizer.zero_grad(set_to_none=True)
        model.zero_grad(set_to_none=True)
        poison_loss = F.cross_entropy(model(x_poison), y_target)
        g_poison = flat_parameter_gradient(poison_loss, params, create_graph=True)
        align_loss = 1.0 - F.cosine_similarity(g_poison.unsqueeze(0), g_target.unsqueeze(0), dim=1, eps=1e-8).mean()
        visual_loss = 0.01 * F.mse_loss(x_poison, x_base)
        (align_loss + visual_loss).backward()
        optimizer.step()
        with torch.no_grad():
            delta = torch.clamp(x_poison - x_base, min=-poison_eps, max=poison_eps)
            x_poison.copy_(torch.clamp(x_base + delta, 0.0, 1.0))
    return x_poison.detach(), y_target, count, trigger


def input_gradient(model: nn.Module, x: torch.Tensor, labels: torch.Tensor, targeted: bool = False) -> torch.Tensor:
    model.eval()
    x_var = x.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(x_var), labels)
    if targeted:
        loss = -loss
    loss.backward()
    return x_var.grad.detach()


def fgsm(model: nn.Module, x: torch.Tensor, y: torch.Tensor, eps: float) -> torch.Tensor:
    grad = input_gradient(model, x, y, targeted=False)
    return torch.clamp(x + eps * grad.sign(), 0.0, 1.0)


def pgd(model: nn.Module, x: torch.Tensor, y: torch.Tensor, eps: float, step: float, iters: int) -> torch.Tensor:
    adv = x.clone()
    for _ in range(iters):
        grad = input_gradient(model, adv, y, targeted=False)
        adv = adv + step * grad.sign()
        adv = torch.max(torch.min(adv, x + eps), x - eps)
        adv = torch.clamp(adv, 0.0, 1.0).detach()
    return adv


def ead_target(
    model: nn.Module,
    x: torch.Tensor,
    target: int,
    eps: float,
    step: float,
    l1_weight: float,
    l2_weight: float,
    iters: int,
) -> torch.Tensor:
    adv = x.clone()
    target_y = torch.full((len(x),), target, dtype=torch.long)
    for _ in range(iters):
        grad = input_gradient(model, adv, target_y, targeted=True) + l2_weight * (adv - x)
        z = (adv - x) + step * grad
        delta = z.sign() * torch.clamp(z.abs() - step * l1_weight, min=0.0)
        delta = torch.clamp(delta, -eps, eps)
        adv = torch.clamp(x + delta, 0.0, 1.0).detach()
    return adv


def vectorized_jsma_saliency(model: nn.Module, x: torch.Tensor, target: int, max_features: int) -> torch.Tensor:
    adv = x.clone().detach()
    batch_size = adv.size(0)
    _, channels, height, width = adv.shape
    target_y = torch.full((batch_size,), target, dtype=torch.long)
    search_mask = torch.ones_like(adv, dtype=torch.bool)
    active_mask = torch.ones(batch_size, dtype=torch.bool)

    model.eval()
    for _ in range(max_features):
        if not active_mask.any():
            break
        adv.requires_grad_(True)
        logits = model(adv)
        preds = logits.argmax(dim=1)
        active_mask = preds != target
        if not active_mask.any():
            adv = adv.detach()
            break
        loss = F.cross_entropy(logits, target_y, reduction="none")
        grad = torch.autograd.grad(loss.sum(), adv)[0]

        with torch.no_grad():
            saliency = grad.abs() * search_mask.float() * active_mask.view(-1, 1, 1, 1).float()
            best_idx = saliency.view(batch_size, -1).argmax(dim=1)
            c = best_idx // (height * width)
            rem = best_idx % (height * width)
            r = rem // width
            col = rem % width
            b_idx = torch.arange(batch_size)
            grad_val = grad[b_idx, c, r, col]
            target_val = torch.where(grad_val > 0, torch.tensor(0.0), torch.tensor(1.0))
            update_mask = active_mask
            adv[b_idx[update_mask], c[update_mask], r[update_mask], col[update_mask]] = target_val[update_mask]
            search_mask[b_idx[update_mask], c[update_mask], r[update_mask], col[update_mask]] = False
        adv = adv.detach()
    return adv


def sparse_boundary(model: nn.Module, x: torch.Tensor, max_features: int) -> torch.Tensor:
    adv = x.clone()
    for _ in range(max_features):
        adv.requires_grad_(True)
        scores = model(adv)
        y = scores.argmax(dim=1)
        sorted_scores = torch.argsort(scores, dim=1, descending=True)
        rival = sorted_scores[:, 1]
        margin = scores[torch.arange(len(adv)), y] - scores[torch.arange(len(adv)), rival]
        active = margin > 0
        if not active.any():
            adv = adv.detach()
            break
        loss = margin[active].sum()
        grad = torch.autograd.grad(loss, adv)[0]
        with torch.no_grad():
            flat_grad = grad.abs().flatten(1)
            coord = flat_grad.argmax(dim=1)
            c = coord // (adv.shape[2] * adv.shape[3])
            rem = coord % (adv.shape[2] * adv.shape[3])
            r = rem // adv.shape[3]
            col = rem % adv.shape[3]
            b_idx = torch.arange(len(adv))
            target_val = torch.where(grad[b_idx, c, r, col] > 0, torch.tensor(0.0), torch.tensor(1.0))
            adv[b_idx[active], c[active], r[active], col[active]] = target_val[active]
        adv = adv.detach()
    return adv


def apply_patch(x: torch.Tensor, patch: torch.Tensor, top: int, left: int) -> torch.Tensor:
    out = x.clone()
    size = patch.shape[-1]
    out[:, :, top : top + size, left : left + size] = patch
    return out


def adversarial_patch_eot(
    model: nn.Module,
    dataset: Dataset,
    source: int,
    target: int,
    patch_size: int,
    iters: int,
    batch_size: int,
    seed: int,
) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    labels = labels_of(dataset)
    source_idx = [i for i, y in enumerate(labels) if y == source]
    patch = torch.rand(1, 3, patch_size, patch_size, requires_grad=True)
    optimizer = torch.optim.Adam([patch], lr=0.05)
    sample_x, _ = dataset[0]
    max_top = sample_x.shape[1] - patch_size
    max_left = sample_x.shape[2] - patch_size
    for _ in range(iters):
        chosen = rng.choice(source_idx, size=min(batch_size, len(source_idx)), replace=False)
        xb, _ = materialize_indices(dataset, chosen)
        top = int(rng.integers(0, max_top + 1))
        left = int(rng.integers(0, max_left + 1))
        patched = apply_patch(xb, torch.clamp(patch, 0.0, 1.0), top, left)
        y_target = torch.full((len(patched),), target, dtype=torch.long)
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(patched), y_target)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            patch.clamp_(0.0, 1.0)
    return patch.detach()


def zoo_target(
    model: nn.Module,
    x: torch.Tensor,
    target: int,
    eps: float,
    step: float,
    iters: int,
    coords_per_iter: int,
    seed: int,
) -> tuple[torch.Tensor, int]:
    rng = np.random.default_rng(seed)
    adv = x.clone()
    queries = 0
    feature_count = int(np.prod(x.shape[1:]))
    y_target = torch.full((1,), target, dtype=torch.long)

    for i in range(len(adv)):
        base = x[i].clone()
        cur = adv[i].clone()
        for _ in range(iters):
            coords = rng.choice(feature_count, size=min(coords_per_iter, feature_count), replace=False)
            samples = cur.flatten().repeat(2 * len(coords), 1)
            for row, coord in enumerate(coords):
                samples[2 * row, coord] = min(1.0, float(samples[2 * row, coord]) + 1e-2)
                samples[2 * row + 1, coord] = max(0.0, float(samples[2 * row + 1, coord]) - 1e-2)
            batch = samples.reshape(-1, *cur.shape)
            with torch.no_grad():
                logits = model(batch)
                losses = F.cross_entropy(logits, y_target.repeat(len(batch)), reduction="none")
            queries += len(batch)
            grad_flat = torch.zeros(feature_count)
            grad_flat[torch.tensor(coords, dtype=torch.long)] = (losses[0::2] - losses[1::2]) / 2e-2
            cur = cur - step * grad_flat.reshape_as(cur).sign()
            cur = torch.max(torch.min(cur, base + eps), base - eps)
            cur = torch.clamp(cur, 0.0, 1.0)
            queries += 1
            if int(predict(model, cur.unsqueeze(0))[0]) == target:
                break
        adv[i] = cur
    return adv, queries


def boundary_search(model: nn.Module, x: torch.Tensor, guide: torch.Tensor, target: int, steps: int) -> torch.Tensor:
    adv = x.clone()
    for i in range(len(x)):
        target_guide = guide[i % len(guide)]
        lo, hi = 0.0, 1.0
        for _ in range(steps):
            mid = (lo + hi) / 2.0
            candidate = torch.clamp((1.0 - mid) * x[i] + mid * target_guide, 0.0, 1.0)
            pred = int(predict(model, candidate.unsqueeze(0))[0])
            if pred == target:
                hi = mid
            else:
                lo = mid
        adv[i] = torch.clamp((1.0 - hi) * x[i] + hi * target_guide, 0.0, 1.0)
    return adv


def perturbation_stats(clean: torch.Tensor, attacked: torch.Tensor) -> tuple[float, float, float]:
    delta = (attacked - clean).abs()
    l0 = float((delta > 1e-6).flatten(1).sum(dim=1).float().mean().item())
    linf = float(delta.flatten(1).max(dim=1).values.mean().item())
    l2 = float(torch.sqrt((delta.flatten(1) ** 2).sum(dim=1)).mean().item())
    return l0, linf, l2


def conditional_untargeted_success(model: nn.Module, clean_x: torch.Tensor, adv_x: torch.Tensor, y: torch.Tensor) -> float:
    clean_pred = predict(model, clean_x)
    eligible = clean_pred == y
    if int(eligible.sum()) == 0:
        return 0.0
    adv_pred = predict(model, adv_x[eligible])
    return float((adv_pred != y[eligible]).float().mean().item())


def conditional_targeted_success(model: nn.Module, clean_x: torch.Tensor, adv_x: torch.Tensor, y: torch.Tensor, target: int) -> float:
    clean_pred = predict(model, clean_x)
    eligible = clean_pred == y
    if int(eligible.sum()) == 0:
        return 0.0
    adv_pred = predict(model, adv_x[eligible])
    return float((adv_pred == target).float().mean().item())


def metric_row(
    method: str,
    stage: str,
    mechanism: str,
    risk: str,
    poison_rate: str,
    train_size: int,
    clean_accuracy: float,
    attack_metric: str,
    attack_success: float,
    notes: str,
    l0: float | str = "",
    linf: float | str = "",
    l2: float | str = "",
) -> dict[str, str | int | float]:
    return {
        "method": method,
        "stage": stage,
        "mechanism": mechanism,
        "risk_level": risk,
        "poison_rate": poison_rate,
        "train_size": train_size,
        "clean_accuracy": round(float(clean_accuracy), 4),
        "attack_metric": attack_metric,
        "attack_success": round(float(attack_success), 4),
        "avg_l0_features": round(float(l0), 2) if l0 != "" else "",
        "avg_linf": round(float(linf), 4) if linf != "" else "",
        "avg_l2": round(float(l2), 4) if l2 != "" else "",
        "notes": notes,
    }


def write_results(rows: list[dict[str, str | int | float]]) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    fieldnames = list(rows[0].keys())
    csv_path = RESULTS_DIR / "fungi_attack_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with (RESULTS_DIR / "fungi_attack_comparison.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    with (RESULTS_DIR / "fungi_attack_comparison.md").open("w", encoding="utf-8") as f:
        f.write("# Fungi Attack Comparison\n\n")
        f.write("| Method | Stage | Mechanism | Risk | Poison rate | Clean acc. | Metric | Success | L0 | Linf | Notes |\n")
        f.write("|---|---|---|---|---:|---:|---|---:|---:|---:|---|\n")
        for row in rows:
            f.write(
                f"| {row['method']} | {row['stage']} | {row['mechanism']} | {row['risk_level']} | "
                f"{row['poison_rate']} | {row['clean_accuracy']} | {row['attack_metric']} | "
                f"{row['attack_success']} | {row['avg_l0_features']} | {row['avg_linf']} | {row['notes']} |\n"
            )


def save_bar_chart(rows: list[dict[str, str | int | float]]) -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    plot_rows = [r for r in rows if r["method"] != "Clean baseline"]
    labels = [str(r["method"]).replace(" ", "\n") for r in plot_rows]
    clean = [float(r["clean_accuracy"]) for r in plot_rows]
    attack = [float(r["attack_success"]) for r in plot_rows]
    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(x - width / 2, clean, width, label="Clean accuracy")
    ax.bar(x + width / 2, attack, width, label="Conditional attack success / damage")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate")
    ax.set_title("Fungi attack comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fungi_attack_comparison_bars.png", dpi=220)
    plt.close(fig)


def save_examples(clean: torch.Tensor, attacked: list[tuple[str, torch.Tensor]], path: Path) -> None:
    cols = min(6, len(clean))
    rows = 1 + len(attacked)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.6))
    axes = np.atleast_2d(axes)
    for c in range(cols):
        axes[0, c].imshow(clean[c].permute(1, 2, 0).numpy())
        axes[0, c].set_title("clean", fontsize=8)
    for r, (name, images) in enumerate(attacked, start=1):
        for c in range(cols):
            axes[r, c].imshow(images[c].detach().permute(1, 2, 0).numpy())
            axes[r, c].set_title(name, fontsize=7)
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle("Fungi attack examples", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    class_names = discover_classes(args.dataset_root)
    train_ds = FungiImageDataset(args.dataset_root, "train", class_names, args.image_size)
    val_ds = FungiImageDataset(args.dataset_root, "val", class_names, args.image_size)
    val_x, val_y = materialize_dataset(val_ds)
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    if args.target_class not in class_to_idx or args.source_class not in class_to_idx:
        raise RuntimeError(f"Classes are {class_names}; requested source={args.source_class}, target={args.target_class}")
    target = class_to_idx[args.target_class]
    source = class_to_idx[args.source_class]
    num_classes = len(class_names)

    train_kwargs = dict(
        num_classes=num_classes,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_name=args.model,
        pretrained=args.pretrained,
        freeze_features=args.freeze_features,
    )

    baseline = train_model(train_ds, seed=args.seed, **train_kwargs)
    baseline_acc = accuracy_on_dataset(baseline, val_ds)

    rows: list[dict[str, str | int | float]] = [
        metric_row(
            "Clean baseline",
            "reference",
            "none",
            "none",
            "0",
            len(train_ds),
            baseline_acc,
            f"{args.source_class}_to_{args.target_class}_confusion",
            source_to_target_rate(baseline, val_x, val_y, source, target),
            f"Model={args.model}, pretrained={args.pretrained}, freeze_features={args.freeze_features}; classes={class_names}.",
        )
    ]

    base_labels = labels_of(train_ds)
    overrides, count = random_label_flip(base_labels, 0.10, args.seed + 1, num_classes)
    ds = LabelOverrideDataset(train_ds, base_labels, overrides)
    model = train_model(ds, seed=args.seed + 1, **train_kwargs)
    acc = accuracy_on_dataset(model, val_ds)
    rows.append(metric_row("Random label flip", "poisoning", "M", "R3", "10.00%", len(ds), acc, "clean_accuracy_drop", max(0.0, baseline_acc - acc), f"{count} labels changed."))

    overrides, count = targeted_label_flip(base_labels, source, target, 0.35, args.seed + 2)
    ds = LabelOverrideDataset(train_ds, base_labels, overrides)
    model = train_model(ds, seed=args.seed + 2, **train_kwargs)
    acc = accuracy_on_dataset(model, val_ds)
    rows.append(metric_row("Targeted label flip", "poisoning", "M", "R2", f"35% of {args.source_class}", len(ds), acc, f"{args.source_class}_to_{args.target_class}_confusion", source_to_target_rate(model, val_x, val_y, source, target), f"{count} source labels changed."))

    overrides, count, _ = subpopulation_label_flip(train_ds, source, target, 0.60, 0.80, args.seed + 3)
    ds = LabelOverrideDataset(train_ds, base_labels, overrides)
    model = train_model(ds, seed=args.seed + 3, **train_kwargs)
    acc = accuracy_on_dataset(model, val_ds)
    rows.append(metric_row("Subpopulation label poison", "poisoning", "M", "R2", "80% of selected subgroup", len(ds), acc, "subpopulation_to_target_confusion", subpopulation_to_target_rate(model, val_ds, source, target, 0.60), f"{count} source-subgroup labels changed."))

    ds, count = backdoor_dataset(train_ds, source, target, 0.25, args.trigger_size, args.seed + 4)
    model = train_model(ds, seed=args.seed + 4, **train_kwargs)
    acc = accuracy_on_dataset(model, val_ds)
    rows.append(metric_row("BadNets patch backdoor", "poisoning", "B", "R1", "25.00% appended", len(ds), acc, f"trigger_ASR_to_{args.target_class}", trigger_asr(model, val_x, val_y, source, target, args.trigger_size), f"{count} triggered source copies."))

    ds, count = clean_label_backdoor_dataset(train_ds, target, 0.25, args.trigger_size, args.seed + 5)
    model = train_model(ds, seed=args.seed + 5, **train_kwargs)
    acc = accuracy_on_dataset(model, val_ds)
    rows.append(metric_row("Clean-label patch backdoor", "poisoning", "B", "R1", "25.00% target-class copies", len(ds), acc, f"trigger_ASR_to_{args.target_class}", trigger_asr(model, val_x, val_y, source, target, args.trigger_size), f"{count} correctly labeled target images patched."))

    poison_x, poison_y, count, stealth_trigger = sleeper_agent_poison(
        baseline,
        train_ds,
        source,
        target,
        40,
        trigger_eps=0.055,
        poison_eps=0.05,
        iters=70,
        seed=args.seed + 6,
    )
    ds = TensorAppendDataset(train_ds, base_labels, poison_x, poison_y)
    model = train_model(ds, seed=args.seed + 6, **train_kwargs)
    acc = accuracy_on_dataset(model, val_ds)
    rows.append(metric_row("Sleeper Agent-style backdoor", "poisoning", "B", "R1", f"{count} appended", len(ds), acc, f"stealth_trigger_ASR_to_{args.target_class}", additive_trigger_asr(model, val_x, val_y, source, target, stealth_trigger), "Clean-label target poisons optimized by gradient alignment to a low-amplitude trigger objective."))

    poison_x, poison_y, count = true_feature_collision(baseline, train_ds, target, source, 60, eps=0.08, iters=150, seed=args.seed + 7)
    ds = TensorAppendDataset(train_ds, base_labels, poison_x, poison_y)
    model = train_model(ds, seed=args.seed + 7, **train_kwargs)
    acc = accuracy_on_dataset(model, val_ds)
    rows.append(metric_row("Clean-label Poison Frogs", "poisoning", "O", "R1", f"{count} appended", len(ds), acc, f"{args.source_class}_to_{args.target_class}_confusion", source_to_target_rate(model, val_x, val_y, source, target), "Optimized edible clean-label poisons toward poisonous latent features with L-infinity projection."))

    poison_x, poison_y, count = gradient_matching_poison(baseline, train_ds, source, target, 40, eps=0.08, iters=70, seed=args.seed + 8)
    ds = TensorAppendDataset(train_ds, base_labels, poison_x, poison_y)
    model = train_model(ds, seed=args.seed + 8, **train_kwargs)
    acc = accuracy_on_dataset(model, val_ds)
    rows.append(metric_row("Witches' Brew gradient match", "poisoning", "G", "R2", f"{count} appended", len(ds), acc, f"{args.source_class}_to_{args.target_class}_confusion", source_to_target_rate(model, val_x, val_y, source, target), "Clean-label poisons optimized so classifier-parameter gradients align with a poisonous-to-edible target objective."))

    baseline_pred = predict(baseline, val_x)
    source_correct = torch.nonzero((val_y == source) & (baseline_pred == val_y), as_tuple=False).flatten()
    if len(source_correct) == 0:
        source_correct = torch.nonzero(val_y == source, as_tuple=False).flatten()
    source_correct = source_correct[: min(args.eval_samples, len(source_correct))]
    attack_x = val_x[source_correct]
    attack_y = val_y[source_correct]
    target_guides = val_x[val_y == target]

    adv = fgsm(baseline, attack_x, attack_y, eps=0.08)
    l0, linf, l2 = perturbation_stats(attack_x, adv)
    rows.append(metric_row("FGSM", "evasion", "G", "R3", "0", len(train_ds), baseline_acc, "conditional_untargeted_misclassification", conditional_untargeted_success(baseline, attack_x, adv, attack_y), "Single-step gradient sign.", l0, linf, l2))

    adv = pgd(baseline, attack_x, attack_y, eps=0.10, step=0.025, iters=7)
    l0, linf, l2 = perturbation_stats(attack_x, adv)
    rows.append(metric_row("PGD", "evasion", "G", "R2", "0", len(train_ds), baseline_acc, "conditional_untargeted_misclassification", conditional_untargeted_success(baseline, attack_x, adv, attack_y), "Projected gradient ascent.", l0, linf, l2))

    adv = ead_target(baseline, attack_x, target, eps=0.16, step=0.025, l1_weight=0.0015, l2_weight=0.01, iters=24)
    l0, linf, l2 = perturbation_stats(attack_x, adv)
    rows.append(metric_row("Elastic Net EAD", "evasion", "O", "R2", "0", len(train_ds), baseline_acc, f"conditional_targeted_ASR_to_{args.target_class}", conditional_targeted_success(baseline, attack_x, adv, attack_y, target), "Targeted ISTA-style elastic-net update.", l0, linf, l2))

    small_x = attack_x[: min(16, len(attack_x))]
    small_y = attack_y[: len(small_x)]
    adv = vectorized_jsma_saliency(baseline, small_x, target, max_features=280)
    l0, linf, l2 = perturbation_stats(small_x, adv)
    rows.append(metric_row("JSMA saliency", "evasion", "M", "R3", "0", len(train_ds), baseline_acc, f"conditional_targeted_ASR_to_{args.target_class}", conditional_targeted_success(baseline, small_x, adv, small_y, target), "Vectorized sparse-feature saliency using input Jacobian.", l0, linf, l2))

    adv = sparse_boundary(baseline, small_x, max_features=280)
    l0, linf, l2 = perturbation_stats(small_x, adv)
    rows.append(metric_row("SparseFool-style boundary", "evasion", "M", "R4", "0", len(train_ds), baseline_acc, "conditional_untargeted_misclassification", conditional_untargeted_success(baseline, small_x, adv, small_y), "Vectorized sparse local-boundary crossing approximation.", l0, linf, l2))

    patch = adversarial_patch_eot(baseline, train_ds, source, target, args.adversarial_patch_size, iters=100, batch_size=min(24, len(train_ds)), seed=args.seed + 9)
    patched = apply_patch(attack_x, patch, attack_x.shape[2] - args.adversarial_patch_size, attack_x.shape[3] - args.adversarial_patch_size)
    l0, linf, l2 = perturbation_stats(attack_x, patched)
    rows.append(metric_row("Adversarial patch", "evasion", "U", "R2", "0", len(train_ds), baseline_acc, f"conditional_targeted_ASR_to_{args.target_class}", conditional_targeted_success(baseline, attack_x, patched, attack_y, target), "EOT-trained patch with random training locations; evaluated lower-right.", l0, linf, l2))

    zoo_x = attack_x[: min(args.zoo_samples, len(attack_x))]
    zoo_y = attack_y[: len(zoo_x)]
    adv, queries = zoo_target(baseline, zoo_x, target, eps=0.16, step=0.04, iters=10, coords_per_iter=64, seed=args.seed + 10)
    l0, linf, l2 = perturbation_stats(zoo_x, adv)
    rows.append(metric_row("ZOO finite difference", "evasion", "O", "R3", "0", len(train_ds), baseline_acc, f"conditional_targeted_ASR_to_{args.target_class}", conditional_targeted_success(baseline, zoo_x, adv, zoo_y, target), f"Batched black-box finite differences; {queries} model queries.", l0, linf, l2))

    adv = boundary_search(baseline, small_x, target_guides, target, steps=12)
    l0, linf, l2 = perturbation_stats(small_x, adv)
    rows.append(metric_row("HopSkipJump-style boundary", "evasion", "O", "R3", "0", len(train_ds), baseline_acc, f"conditional_targeted_ASR_to_{args.target_class}", conditional_targeted_success(baseline, small_x, adv, small_y, target), "Decision-only binary search to target-class guide images.", l0, linf, l2))

    write_results(rows)
    save_bar_chart(rows)
    sample_count = min(6, len(attack_x))
    save_examples(
        attack_x[:sample_count],
        [
            ("FGSM", fgsm(baseline, attack_x[:sample_count], attack_y[:sample_count], eps=0.08)),
            ("PGD", pgd(baseline, attack_x[:sample_count], attack_y[:sample_count], eps=0.10, step=0.025, iters=7)),
            ("Patch", apply_patch(attack_x[:sample_count], patch, attack_x.shape[2] - args.adversarial_patch_size, attack_x.shape[3] - args.adversarial_patch_size)),
        ],
        FIGURES_DIR / "fungi_attack_examples.png",
    )

    for row in rows:
        print(f"{row['method']}: clean={row['clean_accuracy']} {row['attack_metric']}={row['attack_success']}")
    print(f"\nWrote {RESULTS_DIR / 'fungi_attack_comparison.md'}")
    print(f"Wrote {FIGURES_DIR / 'fungi_attack_comparison_bars.png'}")


if __name__ == "__main__":
    main()
