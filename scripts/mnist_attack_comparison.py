#!/usr/bin/env python3
"""Compare MNIST poisoning, backdoor, and evasion attacks with a PyTorch CNN.

This script intentionally keeps MNIST as a controlled visual audit benchmark,
but the model is now a small convolutional network rather than a linear
classifier. That makes feature-collision and backdoor evidence
mathematically consistent with the deep-feature taxonomy used in the paper.
FGSM and PGD use torchattacks when it is installed; otherwise the script falls
back to equivalent local PyTorch implementations and records that fallback in
the result notes.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision.datasets import MNIST

try:  # Optional community-vetted robustness backend.
    import torchattacks  # type: ignore
except ImportError:  # pragma: no cover - environment dependent
    torchattacks = None


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
N_PIXELS = 28 * 28
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MNISTCnn(nn.Module):
    """A compact 2-convolution CNN with an explicit latent feature layer."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.dropout = nn.Dropout(0.20)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        z = F.relu(self.conv1(x))
        z = F.max_pool2d(F.relu(self.conv2(z)), 2)
        z = F.max_pool2d(z, 2)
        z = z.flatten(1)
        return F.relu(self.fc1(z))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.dropout(self.features(x))
        return self.fc2(z)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare real MNIST attack implementations with a CNN.")
    parser.add_argument("--download", action="store_true", help="Download MNIST if missing.")
    parser.add_argument("--target-digit", type=int, default=1)
    parser.add_argument("--source-digit", type=int, default=7)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--eval-samples", type=int, default=1000)
    parser.add_argument("--zoo-samples", type=int, default=40)
    parser.add_argument("--patch-train-samples", type=int, default=1500)
    parser.add_argument("--trigger-size", type=int, default=5)
    parser.add_argument("--adversarial-patch-size", type=int, default=10)
    parser.add_argument("--train-limit", type=int, default=0, help="Optional stratified train subset for smoke tests.")
    parser.add_argument("--test-limit", type=int, default=0, help="Optional stratified test subset for smoke tests.")
    parser.add_argument("--defense", choices=["none", "activation_clustering"], default="none")
    parser.add_argument("--defense-warmup-epochs", type=int, default=1)
    parser.add_argument("--defense-max-cluster-fraction", type=float, default=0.40)
    parser.add_argument("--fgsm-eps", type=float, default=0.22)
    parser.add_argument("--pgd-eps", type=float, default=0.25)
    parser.add_argument("--pgd-step", type=float, default=0.05)
    parser.add_argument("--pgd-iters", type=int, default=10)
    parser.add_argument("--sweep-eps", action="store_true", help="Grid-search the smallest eps reaching threshold ASR.")
    parser.add_argument("--sweep-grid", default="0.02,0.04,0.06,0.08,0.10,0.12,0.16,0.20,0.25")
    parser.add_argument("--sweep-threshold", type=float, default=0.80)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))


def load_mnist(download: bool) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    train = MNIST(root=str(DATA_DIR), train=True, download=download)
    test = MNIST(root=str(DATA_DIR), train=False, download=download)
    x_train = train.data.float().unsqueeze(1) / 255.0
    y_train = train.targets.long()
    x_test = test.data.float().unsqueeze(1) / 255.0
    y_test = test.targets.long()
    return x_train, y_train, x_test, y_test


def stratified_limit(x: torch.Tensor, y: torch.Tensor, limit: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    if limit <= 0 or limit >= len(y):
        return x, y
    rng = np.random.default_rng(seed)
    per_class = max(1, limit // 10)
    chosen: list[int] = []
    for digit in range(10):
        idx = torch.nonzero(y == digit, as_tuple=False).flatten().cpu().numpy()
        take = min(per_class, len(idx))
        chosen.extend(int(i) for i in rng.choice(idx, size=take, replace=False))
    if len(chosen) < limit:
        remaining = np.setdiff1d(np.arange(len(y)), np.array(chosen), assume_unique=False)
        chosen.extend(int(i) for i in rng.choice(remaining, size=min(limit - len(chosen), len(remaining)), replace=False))
    rng.shuffle(chosen)
    idx_t = torch.tensor(chosen[:limit], dtype=torch.long)
    return x[idx_t], y[idx_t]


def loader_for(x: torch.Tensor, y: torch.Tensor, batch_size: int, shuffle: bool = True) -> DataLoader:
    return DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=shuffle, num_workers=0)


def predict(model: nn.Module, x: torch.Tensor, batch_size: int = 512) -> torch.Tensor:
    model.eval()
    preds: list[torch.Tensor] = []
    with torch.no_grad():
        for (xb,) in DataLoader(TensorDataset(x), batch_size=batch_size, shuffle=False, num_workers=0):
            preds.append(model(xb.to(DEVICE)).argmax(1).cpu())
    return torch.cat(preds)


def accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    return float((predict(model, x) == y).float().mean().item())


def train_plain_model(
    x: torch.Tensor,
    y: torch.Tensor,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
) -> MNISTCnn:
    torch.manual_seed(seed)
    model = MNISTCnn().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loader = loader_for(x, y, batch_size, shuffle=True)
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            optimizer.step()
    return model.eval()


def two_means(features: torch.Tensor, seed: int, iters: int = 10) -> torch.Tensor:
    if len(features) < 2:
        return torch.zeros(len(features), dtype=torch.long)
    gen = torch.Generator().manual_seed(seed)
    centers = features[torch.randperm(len(features), generator=gen)[:2]].clone()
    assign = torch.zeros(len(features), dtype=torch.long)
    for _ in range(iters):
        dist = torch.cdist(features, centers)
        assign = dist.argmin(1)
        for k in range(2):
            if torch.any(assign == k):
                centers[k] = features[assign == k].mean(0)
    return assign


def activation_cluster_filter(
    model: MNISTCnn,
    x: torch.Tensor,
    y: torch.Tensor,
    seed: int,
    max_cluster_fraction: float,
) -> torch.Tensor:
    """Remove small same-label activation clusters, a lightweight AC defense."""
    model.eval()
    keep = torch.ones(len(y), dtype=torch.bool)
    with torch.no_grad():
        feats = []
        for (xb,) in DataLoader(TensorDataset(x), batch_size=512, shuffle=False, num_workers=0):
            feats.append(model.features(xb.to(DEVICE)).cpu())
        features = F.normalize(torch.cat(feats), dim=1)
    for digit in range(10):
        idx = torch.nonzero(y == digit, as_tuple=False).flatten()
        if len(idx) < 20:
            continue
        assign = two_means(features[idx], seed + digit)
        counts = torch.bincount(assign, minlength=2)
        small = int(counts.argmin().item())
        small_fraction = float(counts[small].item() / max(1, len(idx)))
        if 0.0 < small_fraction <= max_cluster_fraction:
            keep[idx[assign == small]] = False
    return keep


def train_model(
    x: torch.Tensor,
    y: torch.Tensor,
    seed: int,
    epochs: int,
    batch_size: int,
    lr: float,
    defense: str = "none",
    defense_warmup_epochs: int = 1,
    defense_max_cluster_fraction: float = 0.40,
) -> tuple[MNISTCnn, str]:
    if defense == "activation_clustering":
        warmup = train_plain_model(x, y, seed, max(1, defense_warmup_epochs), batch_size, lr)
        keep = activation_cluster_filter(warmup, x, y, seed, defense_max_cluster_fraction)
        removed = int((~keep).sum().item())
        model = train_plain_model(x[keep], y[keep], seed + 1000, epochs, batch_size, lr)
        return model, f"activation_clustering removed {removed}/{len(y)} training samples before final training"
    model = train_plain_model(x, y, seed, epochs, batch_size, lr)
    return model, "undefended training"


def apply_trigger(images: torch.Tensor, trigger_size: int, value: float = 1.0) -> torch.Tensor:
    patched = images.clone()
    patched[:, :, -trigger_size:, -trigger_size:] = value
    return patched


def random_label_flip(labels: torch.Tensor, rate: float, seed: int) -> tuple[torch.Tensor, int]:
    rng = np.random.default_rng(seed)
    poisoned = labels.clone()
    count = int(round(len(labels) * rate))
    idx = rng.choice(len(labels), size=count, replace=False)
    for i in idx:
        choices = [digit for digit in range(10) if digit != int(labels[int(i)])]
        poisoned[int(i)] = int(rng.choice(choices))
    return poisoned, count


def targeted_label_flip(labels: torch.Tensor, source_digit: int, target_digit: int, rate: float, seed: int) -> tuple[torch.Tensor, int]:
    rng = np.random.default_rng(seed)
    poisoned = labels.clone()
    source_idx = torch.nonzero(labels == source_digit, as_tuple=False).flatten().cpu().numpy()
    count = int(round(len(source_idx) * rate))
    chosen = rng.choice(source_idx, size=count, replace=False)
    poisoned[torch.tensor(chosen, dtype=torch.long)] = target_digit
    return poisoned, count


def subpopulation_indices(images: torch.Tensor, labels: torch.Tensor, source_digit: int, quantile: float) -> torch.Tensor:
    source_idx = torch.nonzero(labels == source_digit, as_tuple=False).flatten()
    source_images = images[source_idx]
    upper_right = source_images[:, :, :14, 14:].mean(dim=(1, 2, 3))
    lower_left = source_images[:, :, 14:, :14].mean(dim=(1, 2, 3))
    score = upper_right - lower_left
    threshold = torch.quantile(score, quantile)
    return source_idx[score >= threshold]


def subpopulation_label_flip(
    images: torch.Tensor,
    labels: torch.Tensor,
    source_digit: int,
    target_digit: int,
    quantile: float,
    flip_rate: float,
    seed: int,
) -> tuple[torch.Tensor, int, torch.Tensor]:
    rng = np.random.default_rng(seed)
    poisoned = labels.clone()
    sub_idx = subpopulation_indices(images, labels, source_digit, quantile)
    count = max(1, int(round(len(sub_idx) * flip_rate)))
    chosen = rng.choice(sub_idx.cpu().numpy(), size=count, replace=False)
    poisoned[torch.tensor(chosen, dtype=torch.long)] = target_digit
    return poisoned, count, sub_idx


def backdoor_poison(
    images: torch.Tensor,
    labels: torch.Tensor,
    target_digit: int,
    rate: float,
    trigger_size: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    rng = np.random.default_rng(seed)
    non_target = torch.nonzero(labels != target_digit, as_tuple=False).flatten().cpu().numpy()
    count = int(round(len(labels) * rate))
    chosen = rng.choice(non_target, size=count, replace=False)
    chosen_t = torch.tensor(chosen, dtype=torch.long)
    poisoned_images = torch.cat([images, apply_trigger(images[chosen_t], trigger_size)], dim=0)
    poisoned_labels = torch.cat([labels, torch.full((count,), target_digit, dtype=torch.long)], dim=0)
    return poisoned_images, poisoned_labels, count


def clean_label_backdoor_poison(
    images: torch.Tensor,
    labels: torch.Tensor,
    target_digit: int,
    rate: float,
    trigger_size: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    rng = np.random.default_rng(seed)
    target_idx = torch.nonzero(labels == target_digit, as_tuple=False).flatten().cpu().numpy()
    count = min(len(target_idx), int(round(len(labels) * rate)))
    chosen = rng.choice(target_idx, size=count, replace=False)
    chosen_t = torch.tensor(chosen, dtype=torch.long)
    poisoned_images = torch.cat([images, apply_trigger(images[chosen_t], trigger_size)], dim=0)
    poisoned_labels = torch.cat([labels, labels[chosen_t]], dim=0)
    return poisoned_images, poisoned_labels, count


def true_feature_collision_poison(
    feature_model: MNISTCnn,
    images: torch.Tensor,
    labels: torch.Tensor,
    target_digit: int,
    source_digit: int,
    count: int,
    eps: float,
    iters: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    rng = np.random.default_rng(seed)
    source_idx = torch.nonzero(labels == source_digit, as_tuple=False).flatten().cpu().numpy()
    count = min(count, len(source_idx))
    chosen = rng.choice(source_idx, size=count, replace=False)
    chosen_t = torch.tensor(chosen, dtype=torch.long)
    x_base = images[chosen_t].clone()
    feature_model.eval()
    with torch.no_grad():
        target_images = images[labels == target_digit][:512].to(DEVICE)
        target_feat = feature_model.features(target_images).mean(0, keepdim=True).detach()
    x_poison = x_base.clone().to(DEVICE).requires_grad_(True)
    base_dev = x_base.to(DEVICE)
    optimizer = torch.optim.Adam([x_poison], lr=0.01)
    for _ in range(iters):
        optimizer.zero_grad(set_to_none=True)
        feat = feature_model.features(x_poison)
        loss = F.mse_loss(feat, target_feat.expand_as(feat))
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            delta = torch.clamp(x_poison - base_dev, -eps, eps)
            x_poison.copy_(torch.clamp(base_dev + delta, 0.0, 1.0))
    poisoned_images = torch.cat([images, x_poison.detach().cpu()], dim=0)
    poisoned_labels = torch.cat([labels, torch.full((count,), source_digit, dtype=torch.long)], dim=0)
    return poisoned_images, poisoned_labels, count


def source_to_target_rate(model: nn.Module, images: torch.Tensor, labels: torch.Tensor, source_digit: int, target_digit: int) -> float:
    mask = labels == source_digit
    if not torch.any(mask):
        return 0.0
    pred = predict(model, images[mask])
    return float((pred == target_digit).float().mean().item())


def subpopulation_to_target_rate(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    source_digit: int,
    target_digit: int,
    quantile: float,
) -> float:
    idx = subpopulation_indices(images, labels, source_digit, quantile)
    if len(idx) == 0:
        return 0.0
    pred = predict(model, images[idx])
    return float((pred == target_digit).float().mean().item())


def trigger_asr(model: nn.Module, images: torch.Tensor, labels: torch.Tensor, target_digit: int, trigger_size: int) -> float:
    mask = labels != target_digit
    pred = predict(model, apply_trigger(images[mask], trigger_size))
    return float((pred == target_digit).float().mean().item())


def perturbation_stats(original: torch.Tensor, attacked: torch.Tensor) -> tuple[float, float, float]:
    delta = (attacked - original).abs()
    l0 = delta.flatten(1).gt(1e-6).sum(1).float().mean().item()
    linf = delta.flatten(1).max(1).values.mean().item()
    l2 = torch.sqrt(delta.flatten(1).pow(2).sum(1)).mean().item()
    return float(l0), float(linf), float(l2)


def attack_backend_note() -> str:
    return "torchattacks backend" if torchattacks is not None else "local PyTorch fallback; install torchattacks for community-vetted FGSM/PGD"


def fgsm(model: nn.Module, images: torch.Tensor, labels: torch.Tensor, eps: float) -> torch.Tensor:
    model.eval()
    if torchattacks is not None:
        return torchattacks.FGSM(model, eps=eps)(images.to(DEVICE), labels.to(DEVICE)).detach().cpu()
    x = images.clone().to(DEVICE).requires_grad_(True)
    y = labels.to(DEVICE)
    loss = F.cross_entropy(model(x), y)
    grad = torch.autograd.grad(loss, x)[0]
    return torch.clamp(x + eps * grad.sign(), 0.0, 1.0).detach().cpu()


def pgd(model: nn.Module, images: torch.Tensor, labels: torch.Tensor, eps: float, step: float, iters: int) -> torch.Tensor:
    model.eval()
    if torchattacks is not None:
        return torchattacks.PGD(model, eps=eps, alpha=step, steps=iters, random_start=False)(images.to(DEVICE), labels.to(DEVICE)).detach().cpu()
    base = images.to(DEVICE)
    adv = base.clone()
    y = labels.to(DEVICE)
    for _ in range(iters):
        adv.requires_grad_(True)
        loss = F.cross_entropy(model(adv), y)
        grad = torch.autograd.grad(loss, adv)[0]
        with torch.no_grad():
            adv = adv + step * grad.sign()
            adv = torch.minimum(torch.maximum(adv, base - eps), base + eps)
            adv = torch.clamp(adv, 0.0, 1.0)
    return adv.detach().cpu()


def ead_target(
    model: nn.Module,
    images: torch.Tensor,
    target_digit: int,
    eps: float,
    step: float,
    l1_weight: float,
    l2_weight: float,
    iters: int,
) -> torch.Tensor:
    model.eval()
    base = images.to(DEVICE)
    delta = torch.zeros_like(base)
    target_y = torch.full((len(images),), target_digit, dtype=torch.long, device=DEVICE)
    for _ in range(iters):
        delta_var = delta.detach().requires_grad_(True)
        adv = base + delta_var
        loss = F.cross_entropy(model(adv), target_y) + 0.5 * l2_weight * delta_var.flatten(1).pow(2).sum(1).mean()
        grad = torch.autograd.grad(loss, delta_var)[0]
        with torch.no_grad():
            z = delta_var - step * grad
            delta = z.sign() * torch.clamp(z.abs() - step * l1_weight, min=0.0)
            delta = torch.clamp(delta, -eps, eps)
            delta = torch.maximum(torch.minimum(delta, 1.0 - base), -base)
    return torch.clamp(base + delta, 0.0, 1.0).detach().cpu()


def jsma_target(model: nn.Module, images: torch.Tensor, target_digit: int, max_pixels: int) -> torch.Tensor:
    adv = images.clone()
    batch = len(adv)
    target_y = torch.full((batch,), target_digit, dtype=torch.long, device=DEVICE)
    search_mask = torch.ones_like(adv, dtype=torch.bool)
    active = torch.ones(batch, dtype=torch.bool, device=DEVICE)
    for _ in range(max_pixels):
        if not active.any():
            break
        x = adv.to(DEVICE).detach().requires_grad_(True)
        logits = model(x)
        pred = logits.argmax(1)
        active = pred != target_digit
        if not active.any():
            break
        loss = F.cross_entropy(logits, target_y, reduction="none")
        grad = torch.autograd.grad(loss.sum(), x)[0].detach().cpu()
        saliency = grad.abs() * search_mask.float() * active.cpu().view(-1, 1, 1, 1).float()
        best = saliency.flatten(1).argmax(1)
        row = (best % (28 * 28)) // 28
        col = best % 28
        b = torch.arange(batch)
        grad_val = grad[b, 0, row, col]
        target_val = torch.where(grad_val > 0, torch.tensor(0.0), torch.tensor(1.0))
        active_cpu = active.cpu()
        adv[b[active_cpu], 0, row[active_cpu], col[active_cpu]] = target_val[active_cpu]
        search_mask[b[active_cpu], 0, row[active_cpu], col[active_cpu]] = False
    return adv


def sparse_boundary_attack(model: nn.Module, images: torch.Tensor, labels: torch.Tensor, max_pixels: int) -> torch.Tensor:
    adv = images.clone()
    used = torch.zeros_like(adv, dtype=torch.bool)
    for _ in range(max_pixels):
        x = adv.to(DEVICE).detach().requires_grad_(True)
        y = labels.to(DEVICE)
        logits = model(x)
        pred = logits.argmax(1)
        active = pred.cpu() == labels
        if not torch.any(active):
            break
        loss = F.cross_entropy(logits, y, reduction="none")
        grad = torch.autograd.grad(loss.sum(), x)[0].detach().cpu()
        saliency = grad.abs() * (~used).float() * active.view(-1, 1, 1, 1).float()
        best = saliency.flatten(1).argmax(1)
        row = (best % (28 * 28)) // 28
        col = best % 28
        b = torch.arange(len(adv))
        grad_val = grad[b, 0, row, col]
        new_val = torch.where(grad_val > 0, torch.tensor(1.0), torch.tensor(0.0))
        adv[b[active], 0, row[active], col[active]] = new_val[active]
        used[b[active], 0, row[active], col[active]] = True
    return adv


def universal_patch(
    model: nn.Module,
    train_images: torch.Tensor,
    train_labels: torch.Tensor,
    patch_size: int,
    train_count: int,
    iters: int,
    batch_size: int,
    step: float,
    seed: int,
) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(train_labels), size=min(train_count, len(train_labels)), replace=False)
    pool = train_images[torch.tensor(chosen, dtype=torch.long)]
    pool_labels = train_labels[torch.tensor(chosen, dtype=torch.long)]
    patch = torch.rand((1, 1, patch_size, patch_size), device=DEVICE, requires_grad=True)
    for _ in range(iters):
        idx = rng.choice(len(pool), size=min(batch_size, len(pool)), replace=False)
        xb = pool[torch.tensor(idx, dtype=torch.long)].to(DEVICE).clone()
        yb = pool_labels[torch.tensor(idx, dtype=torch.long)].to(DEVICE)
        xb[:, :, -patch_size:, -patch_size:] = patch
        loss = F.cross_entropy(model(xb), yb)
        grad = torch.autograd.grad(loss, patch)[0]
        with torch.no_grad():
            patch.add_(step * grad.sign())
            patch.clamp_(0.0, 1.0)
    return patch.detach().cpu().squeeze(0)


def apply_universal_patch(images: torch.Tensor, patch: torch.Tensor) -> torch.Tensor:
    patched = images.clone()
    s = patch.shape[-1]
    patched[:, :, -s:, -s:] = patch
    return patched


def zoo_target(
    model: nn.Module,
    images: torch.Tensor,
    target_digit: int,
    eps: float,
    step: float,
    iters: int,
    coords_per_iter: int,
    seed: int,
) -> tuple[torch.Tensor, int]:
    rng = np.random.default_rng(seed)
    adv = images.clone()
    queries = 0

    def loss_batch(x_flat: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            proba = F.softmax(model(x_flat.view(-1, 1, 28, 28).to(DEVICE)), dim=1)[:, target_digit].cpu()
        return -torch.log(torch.clamp(proba, min=1e-12))

    for n in range(len(adv)):
        base = images[n].flatten()
        cur = adv[n].flatten().clone()
        for _ in range(iters):
            coords = rng.choice(N_PIXELS, size=min(coords_per_iter, N_PIXELS), replace=False)
            plus = cur.repeat(len(coords), 1)
            minus = cur.repeat(len(coords), 1)
            for row, coord in enumerate(coords):
                plus[row, int(coord)] = min(1.0, float(plus[row, int(coord)] + 1e-2))
                minus[row, int(coord)] = max(0.0, float(minus[row, int(coord)] - 1e-2))
            losses = loss_batch(torch.cat([plus, minus], dim=0))
            queries += 2 * len(coords)
            grad = torch.zeros(N_PIXELS)
            grad[torch.tensor(coords, dtype=torch.long)] = (losses[: len(coords)] - losses[len(coords) :]) / 2e-2
            cur = cur - step * grad.sign()
            cur = torch.clamp(torch.minimum(torch.maximum(cur, base - eps), base + eps), 0.0, 1.0)
            pred = int(predict(model, cur.view(1, 1, 28, 28))[0])
            queries += 1
            if pred == target_digit:
                break
        adv[n] = cur.view(1, 28, 28)
    return adv, queries


def boundary_line_search(
    model: nn.Module,
    images: torch.Tensor,
    target_images: torch.Tensor,
    target_digit: int,
    steps: int,
) -> torch.Tensor:
    adv = images.clone()
    for n in range(len(images)):
        guide = target_images[n % len(target_images)]
        lo, hi = 0.0, 1.0
        for _ in range(steps):
            mid = (lo + hi) / 2.0
            candidate = torch.clamp((1.0 - mid) * images[n] + mid * guide, 0.0, 1.0).view(1, 1, 28, 28)
            if int(predict(model, candidate)[0]) == target_digit:
                hi = mid
            else:
                lo = mid
        adv[n] = torch.clamp((1.0 - hi) * images[n] + hi * guide, 0.0, 1.0)
    return adv


def select_correct_non_target(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    target_digit: int,
    count: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    pred = predict(model, images)
    idx = torch.nonzero((pred == labels) & (labels != target_digit), as_tuple=False).flatten()
    idx = idx[: min(count, len(idx))]
    return images[idx], labels[idx]


def conditional_targeted_success(
    model: nn.Module,
    clean: torch.Tensor,
    adv: torch.Tensor,
    labels: torch.Tensor,
    target_digit: int,
) -> float:
    clean_pred = predict(model, clean)
    mask = (clean_pred == labels) & (labels != target_digit)
    if not torch.any(mask):
        return 0.0
    adv_pred = predict(model, adv[mask])
    return float((adv_pred == target_digit).float().mean().item())


def conditional_untargeted_success(model: nn.Module, clean: torch.Tensor, adv: torch.Tensor, labels: torch.Tensor) -> float:
    clean_pred = predict(model, clean)
    mask = clean_pred == labels
    if not torch.any(mask):
        return 0.0
    adv_pred = predict(model, adv[mask])
    return float((adv_pred != labels[mask]).float().mean().item())


def metric_row(
    method: str,
    stage: str,
    mechanism: str,
    risk: str,
    clean_accuracy: float,
    attack_metric: str,
    attack_success: float,
    train_size: int,
    poison_rate: str = "0",
    l0: float | str = "",
    linf: float | str = "",
    l2: float | str = "",
    notes: str = "",
) -> dict[str, float | str | int]:
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
        "avg_l0_pixels": round(float(l0), 2) if l0 != "" else "",
        "avg_linf": round(float(linf), 4) if linf != "" else "",
        "avg_l2": round(float(l2), 4) if l2 != "" else "",
        "notes": notes,
    }


def save_results(rows: list[dict[str, float | str | int]]) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    fieldnames = list(rows[0].keys())
    csv_path = RESULTS_DIR / "mnist_attack_comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with (RESULTS_DIR / "mnist_attack_comparison.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    with (RESULTS_DIR / "mnist_attack_comparison.md").open("w", encoding="utf-8") as f:
        f.write("# MNIST Attack Comparison\n\n")
        f.write("| Method | Stage | Mechanism | Risk | Poison rate | Clean acc. | Metric | Success | L0 | Linf | Notes |\n")
        f.write("|---|---|---|---|---:|---:|---|---:|---:|---:|---|\n")
        for row in rows:
            f.write(
                f"| {row['method']} | {row['stage']} | {row['mechanism']} | {row['risk_level']} | "
                f"{row['poison_rate']} | {row['clean_accuracy']} | {row['attack_metric']} | "
                f"{row['attack_success']} | {row['avg_l0_pixels']} | {row['avg_linf']} | {row['notes']} |\n"
            )


def save_sweep(rows: list[dict[str, float | str | int]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with (RESULTS_DIR / "mnist_attack_sweep.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with (RESULTS_DIR / "mnist_attack_sweep.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    with (RESULTS_DIR / "mnist_attack_sweep.md").open("w", encoding="utf-8") as f:
        f.write("# MNIST Attack Epsilon Sweep\n\n")
        f.write("| Attack | Eps | Step | Iters | Success | Meets threshold | Backend |\n")
        f.write("|---|---:|---:|---:|---:|---|---|\n")
        for row in rows:
            f.write(
                f"| {row['attack']} | {row['eps']} | {row['step']} | {row['iters']} | "
                f"{row['success']} | {row['meets_threshold']} | {row['backend']} |\n"
            )


def run_epsilon_sweep(
    model: nn.Module,
    clean: torch.Tensor,
    labels: torch.Tensor,
    eps_values: list[float],
    threshold: float,
    pgd_step_ratio: float,
    pgd_iters: int,
) -> tuple[list[dict[str, float | str | int]], dict[str, float | str | int] | None]:
    sweep_rows: list[dict[str, float | str | int]] = []
    best: dict[str, float | str | int] | None = None
    for eps in eps_values:
        step = max(1e-4, eps * pgd_step_ratio)
        adv = pgd(model, clean, labels, eps=eps, step=step, iters=pgd_iters)
        success = conditional_untargeted_success(model, clean, adv, labels)
        row: dict[str, float | str | int] = {
            "attack": "PGD",
            "eps": round(float(eps), 4),
            "step": round(float(step), 4),
            "iters": pgd_iters,
            "success": round(float(success), 4),
            "meets_threshold": bool(success >= threshold),
            "backend": attack_backend_note(),
        }
        sweep_rows.append(row)
        if best is None and success >= threshold:
            best = row
    return sweep_rows, best


def save_bar_chart(rows: list[dict[str, float | str | int]]) -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    plot_rows = [r for r in rows if r["method"] != "Clean baseline"]
    labels = [str(r["method"]).replace(" ", "\n") for r in plot_rows]
    attack = [float(r["attack_success"]) for r in plot_rows]
    clean = [float(r["clean_accuracy"]) for r in plot_rows]
    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(17, 6))
    ax.bar(x - width / 2, clean, width, label="Clean accuracy")
    ax.bar(x + width / 2, attack, width, label="Attack success / damage")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate")
    ax.set_title("MNIST CNN attack comparison across implemented methods")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "mnist_attack_comparison_bars.png", dpi=220)
    plt.close(fig)


def save_attack_examples(clean_images: torch.Tensor, attacked: list[tuple[str, torch.Tensor]], path: Path) -> None:
    cols = min(8, len(clean_images))
    rows = 1 + len(attacked)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.25, rows * 1.35))
    axes = np.atleast_2d(axes)
    for c in range(cols):
        axes[0, c].imshow(clean_images[c, 0].numpy(), cmap="gray", vmin=0, vmax=1)
        axes[0, c].set_title("clean", fontsize=8)
    for r, (name, images) in enumerate(attacked, start=1):
        for c in range(cols):
            axes[r, c].imshow(images[c, 0].numpy(), cmap="gray", vmin=0, vmax=1)
            axes[r, c].set_title(name, fontsize=7)
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle("MNIST CNN attack examples", fontsize=14)
    fig.text(
        0.5,
        0.02,
        "FGSM/PGD use torchattacks when available; perturbation magnitudes are shown for auditability.",
        ha="center",
        fontsize=8,
        color="#495057",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.96))
    fig.savefig(path, dpi=220)
    plt.close(fig)


def train_poisoned(
    x: torch.Tensor,
    y: torch.Tensor,
    seed: int,
    args: argparse.Namespace,
) -> tuple[MNISTCnn, str]:
    return train_model(
        x,
        y,
        seed,
        args.epochs,
        args.batch_size,
        args.lr,
        args.defense,
        args.defense_warmup_epochs,
        args.defense_max_cluster_fraction,
    )


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    x_train, y_train, x_test, y_test = load_mnist(args.download)
    x_train, y_train = stratified_limit(x_train, y_train, args.train_limit, args.seed)
    x_test, y_test = stratified_limit(x_test, y_test, args.test_limit, args.seed + 99)

    baseline, baseline_note = train_poisoned(x_train, y_train, args.seed, args)
    baseline_acc = accuracy(baseline, x_test, y_test)
    rows: list[dict[str, float | str | int]] = [
        metric_row(
            "Clean baseline",
            "reference",
            "none",
            "none",
            baseline_acc,
            f"{args.source_digit}_to_{args.target_digit}_confusion",
            source_to_target_rate(baseline, x_test, y_test, args.source_digit, args.target_digit),
            len(x_train),
            notes=f"PyTorch 2-layer CNN; defense={args.defense}; {baseline_note}.",
        )
    ]

    poisoned_labels, count = random_label_flip(y_train, 0.10, args.seed + 1)
    model, note = train_poisoned(x_train, poisoned_labels, args.seed + 1, args)
    acc = accuracy(model, x_test, y_test)
    rows.append(metric_row("Random label flip", "poisoning", "M", "R3", acc, "clean_accuracy_drop", max(0.0, baseline_acc - acc), len(x_train), "10.00%", notes=f"{count} labels changed; {note}."))

    poisoned_labels, count = targeted_label_flip(y_train, args.source_digit, args.target_digit, 0.35, args.seed + 2)
    model, note = train_poisoned(x_train, poisoned_labels, args.seed + 2, args)
    acc = accuracy(model, x_test, y_test)
    s2t = source_to_target_rate(model, x_test, y_test, args.source_digit, args.target_digit)
    rows.append(metric_row("Targeted label flip", "poisoning", "M", "R2", acc, f"{args.source_digit}_to_{args.target_digit}_confusion", s2t, len(x_train), f"35% of digit {args.source_digit}", notes=f"{count} source labels changed; {note}."))

    poisoned_labels, count, _ = subpopulation_label_flip(x_train, y_train, args.source_digit, args.target_digit, 0.70, 0.80, args.seed + 3)
    model, note = train_poisoned(x_train, poisoned_labels, args.seed + 3, args)
    acc = accuracy(model, x_test, y_test)
    sub = subpopulation_to_target_rate(model, x_test, y_test, args.source_digit, args.target_digit, 0.70)
    rows.append(metric_row("Subpopulation label poison", "poisoning", "M", "R2", acc, "subpopulation_to_target_confusion", sub, len(x_train), "80% of selected subgroup", notes=f"{count} source-subgroup labels changed; {note}."))

    poisoned_images, poisoned_labels, count = backdoor_poison(x_train, y_train, args.target_digit, 0.05, args.trigger_size, args.seed + 4)
    model, note = train_poisoned(poisoned_images, poisoned_labels, args.seed + 4, args)
    acc = accuracy(model, x_test, y_test)
    rows.append(metric_row("BadNets patch backdoor", "poisoning", "B", "R1", acc, f"trigger_ASR_to_{args.target_digit}", trigger_asr(model, x_test, y_test, args.target_digit, args.trigger_size), len(poisoned_images), "5.00% appended", notes=f"{count} triggered non-target copies; {note}."))

    poisoned_images, poisoned_labels, count = clean_label_backdoor_poison(x_train, y_train, args.target_digit, 0.05, args.trigger_size, args.seed + 5)
    model, note = train_poisoned(poisoned_images, poisoned_labels, args.seed + 5, args)
    acc = accuracy(model, x_test, y_test)
    rows.append(metric_row("Clean-label patch backdoor", "poisoning", "B", "R1", acc, f"trigger_ASR_to_{args.target_digit}", trigger_asr(model, x_test, y_test, args.target_digit, args.trigger_size), len(poisoned_images), "5.00% target-class copies", notes=f"{count} correctly labeled target images patched; {note}."))

    poisoned_images, poisoned_labels, count = true_feature_collision_poison(baseline, x_train, y_train, args.target_digit, args.source_digit, 900, 0.18, 80, args.seed + 6)
    model, note = train_poisoned(poisoned_images, poisoned_labels, args.seed + 6, args)
    acc = accuracy(model, x_test, y_test)
    t2s = source_to_target_rate(model, x_test, y_test, args.target_digit, args.source_digit)
    rows.append(metric_row("True feature collision", "poisoning", "O", "R1", acc, f"{args.target_digit}_to_{args.source_digit}_confusion", t2s, len(poisoned_images), f"{count} appended", notes=f"Clean-label source poisons optimized in CNN latent space with L-infinity projection; {note}."))

    evasion_x, evasion_y = select_correct_non_target(baseline, x_test, y_test, args.target_digit, args.eval_samples)
    target_pool = x_test[y_test == args.target_digit][: max(1, args.eval_samples)]

    adv = fgsm(baseline, evasion_x, evasion_y, eps=args.fgsm_eps)
    l0, linf, l2 = perturbation_stats(evasion_x, adv)
    rows.append(metric_row("FGSM", "evasion", "G", "R3", baseline_acc, "conditional_untargeted_misclassification", conditional_untargeted_success(baseline, evasion_x, adv, evasion_y), len(x_train), l0=l0, linf=linf, l2=l2, notes=f"eps={args.fgsm_eps}; {attack_backend_note()}."))

    adv = pgd(baseline, evasion_x, evasion_y, eps=args.pgd_eps, step=args.pgd_step, iters=args.pgd_iters)
    l0, linf, l2 = perturbation_stats(evasion_x, adv)
    rows.append(metric_row("PGD", "evasion", "G", "R2", baseline_acc, "conditional_untargeted_misclassification", conditional_untargeted_success(baseline, evasion_x, adv, evasion_y), len(x_train), l0=l0, linf=linf, l2=l2, notes=f"eps={args.pgd_eps}, step={args.pgd_step}, iters={args.pgd_iters}; {attack_backend_note()}."))

    sweep_rows: list[dict[str, float | str | int]] = []
    if args.sweep_eps:
        eps_values = [float(v.strip()) for v in args.sweep_grid.split(",") if v.strip()]
        sweep_rows, best = run_epsilon_sweep(baseline, evasion_x, evasion_y, eps_values, args.sweep_threshold, 0.20, args.pgd_iters)
        if best is not None:
            rows.append(metric_row("PGD eps sweep", "evasion", "G", "R2", baseline_acc, "lowest_eps_for_threshold", float(best["eps"]), len(x_train), l0="", linf=float(best["eps"]), notes=f"Lowest eps reaching >= {args.sweep_threshold:.2f} conditional success; success={best['success']}; {attack_backend_note()}."))
        else:
            max_row = max(sweep_rows, key=lambda r: float(r["success"])) if sweep_rows else None
            if max_row is not None:
                rows.append(metric_row("PGD eps sweep", "evasion", "G", "R2", baseline_acc, "best_swept_success", float(max_row["success"]), len(x_train), l0="", linf=float(max_row["eps"]), notes=f"No eps reached threshold {args.sweep_threshold:.2f}; best eps={max_row['eps']}; {attack_backend_note()}."))
    save_sweep(sweep_rows)

    adv = ead_target(baseline, evasion_x, args.target_digit, eps=0.35, step=0.035, l1_weight=0.002, l2_weight=0.02, iters=35)
    l0, linf, l2 = perturbation_stats(evasion_x, adv)
    rows.append(metric_row("Elastic Net EAD", "evasion", "O", "R2", baseline_acc, f"conditional_targeted_ASR_to_{args.target_digit}", conditional_targeted_success(baseline, evasion_x, adv, evasion_y, args.target_digit), len(x_train), l0=l0, linf=linf, l2=l2, notes="Formal ISTA proximal solver for targeted elastic-net objective."))

    adv = jsma_target(baseline, evasion_x[:300], args.target_digit, max_pixels=45)
    l0, linf, l2 = perturbation_stats(evasion_x[:300], adv)
    rows.append(metric_row("JSMA saliency", "evasion", "M", "R3", baseline_acc, f"conditional_targeted_ASR_to_{args.target_digit}", conditional_targeted_success(baseline, evasion_x[:300], adv, evasion_y[:300], args.target_digit), len(x_train), l0=l0, linf=linf, l2=l2, notes="Vectorized sparse-pixel saliency using CNN input Jacobian."))

    adv = sparse_boundary_attack(baseline, evasion_x[:300], evasion_y[:300], max_pixels=45)
    l0, linf, l2 = perturbation_stats(evasion_x[:300], adv)
    rows.append(metric_row("SparseFool-style boundary", "evasion", "M", "R4", baseline_acc, "conditional_untargeted_misclassification", conditional_untargeted_success(baseline, evasion_x[:300], adv, evasion_y[:300]), len(x_train), l0=l0, linf=linf, l2=l2, notes="Sparse gradient boundary approximation for the CNN baseline."))

    patch = universal_patch(baseline, x_train, y_train, args.adversarial_patch_size, args.patch_train_samples, 120, 128, 0.08, args.seed + 7)
    adv = apply_universal_patch(evasion_x, patch)
    l0, linf, l2 = perturbation_stats(evasion_x, adv)
    rows.append(metric_row("Adversarial patch", "evasion", "U", "R2", baseline_acc, "conditional_untargeted_misclassification", conditional_untargeted_success(baseline, evasion_x, adv, evasion_y), len(x_train), l0=l0, linf=linf, l2=l2, notes="Universal fixed-position patch optimized by CNN loss gradients."))

    zoo_x, zoo_y = evasion_x[: args.zoo_samples], evasion_y[: args.zoo_samples]
    adv, queries = zoo_target(baseline, zoo_x, args.target_digit, eps=0.35, step=0.08, iters=18, coords_per_iter=32, seed=args.seed + 8)
    l0, linf, l2 = perturbation_stats(zoo_x, adv)
    rows.append(metric_row("ZOO finite difference", "evasion", "O", "R3", baseline_acc, f"conditional_targeted_ASR_to_{args.target_digit}", conditional_targeted_success(baseline, zoo_x, adv, zoo_y, args.target_digit), len(x_train), l0=l0, linf=linf, l2=l2, notes=f"Black-box coordinate finite differences; {queries} model queries."))

    boundary_x, boundary_y = evasion_x[:300], evasion_y[:300]
    adv = boundary_line_search(baseline, boundary_x, target_pool, args.target_digit, steps=14)
    l0, linf, l2 = perturbation_stats(boundary_x, adv)
    rows.append(metric_row("HopSkipJump-style boundary", "evasion", "O", "R3", baseline_acc, f"conditional_targeted_ASR_to_{args.target_digit}", conditional_targeted_success(baseline, boundary_x, adv, boundary_y, args.target_digit), len(x_train), l0=l0, linf=linf, l2=l2, notes="Decision-only binary search between clean input and target-class guide image."))

    save_results(rows)
    save_bar_chart(rows)
    save_attack_examples(
        evasion_x[:8],
        [
            ("FGSM", fgsm(baseline, evasion_x[:8], evasion_y[:8], eps=args.fgsm_eps)),
            ("PGD", pgd(baseline, evasion_x[:8], evasion_y[:8], eps=args.pgd_eps, step=args.pgd_step, iters=args.pgd_iters)),
            ("JSMA", jsma_target(baseline, evasion_x[:8], args.target_digit, max_pixels=45)),
            ("Patch", apply_universal_patch(evasion_x[:8], patch)),
        ],
        FIGURES_DIR / "mnist_attack_examples.png",
    )

    for row in rows:
        print(f"{row['method']}: clean={row['clean_accuracy']} {row['attack_metric']}={row['attack_success']}")
    print(f"\nWrote {RESULTS_DIR / 'mnist_attack_comparison.md'}")
    if sweep_rows:
        print(f"Wrote {RESULTS_DIR / 'mnist_attack_sweep.md'}")
    print(f"Wrote {FIGURES_DIR / 'mnist_attack_comparison_bars.png'}")


if __name__ == "__main__":
    main()
