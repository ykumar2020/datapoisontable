#!/usr/bin/env python3
"""Compare MNIST poisoning, backdoor, and evasion attacks at a glance.

The implementations in this file intentionally use one public benchmark and one
model family so the method comparison is controlled. Poisoning methods retrain
the classifier on modified MNIST training data. Evasion methods attack the same
clean baseline classifier at test time.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from torchvision.datasets import MNIST


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
IMAGE_SHAPE = (28, 28)
N_PIXELS = 28 * 28


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare real MNIST attack implementations.")
    parser.add_argument("--download", action="store_true", help="Download MNIST if missing.")
    parser.add_argument("--target-digit", type=int, default=1)
    parser.add_argument("--source-digit", type=int, default=7)
    parser.add_argument("--max-iter", type=int, default=25)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--eval-samples", type=int, default=1000)
    parser.add_argument("--zoo-samples", type=int, default=80)
    parser.add_argument("--patch-train-samples", type=int, default=1500)
    parser.add_argument("--trigger-size", type=int, default=5)
    parser.add_argument("--adversarial-patch-size", type=int, default=10)
    return parser.parse_args()


def load_mnist(download: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train = MNIST(root=str(DATA_DIR), train=True, download=download)
    test = MNIST(root=str(DATA_DIR), train=False, download=download)
    x_train = train.data.numpy().astype(np.float32) / 255.0
    y_train = train.targets.numpy().astype(np.int64)
    x_test = test.data.numpy().astype(np.float32) / 255.0
    y_test = test.targets.numpy().astype(np.int64)
    return x_train, y_train, x_test, y_test


def flat(images: np.ndarray) -> np.ndarray:
    return images.reshape(images.shape[0], -1)


def fit_classifier(images: np.ndarray, labels: np.ndarray, seed: int, max_iter: int) -> SGDClassifier:
    clf = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-4,
        max_iter=max_iter,
        tol=1e-3,
        shuffle=True,
        random_state=seed,
        n_jobs=-1,
    )
    clf.fit(flat(images), labels)
    return clf


def logits(clf: SGDClassifier, images: np.ndarray) -> np.ndarray:
    scores = clf.decision_function(flat(images))
    return scores if scores.ndim == 2 else np.column_stack([-scores, scores])


def ce_gradient(clf: SGDClassifier, images: np.ndarray, labels: np.ndarray) -> np.ndarray:
    probs = clf.predict_proba(flat(images))
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(len(labels)), labels] = 1.0
    grad = (probs - one_hot) @ clf.coef_
    return grad.reshape((-1, 28, 28))


def targeted_ce_gradient(clf: SGDClassifier, images: np.ndarray, target_digit: int) -> np.ndarray:
    target_labels = np.full(len(images), target_digit, dtype=np.int64)
    return ce_gradient(clf, images, target_labels)


def apply_trigger(images: np.ndarray, trigger_size: int, value: float = 1.0) -> np.ndarray:
    patched = images.copy()
    patched[:, -trigger_size:, -trigger_size:] = value
    return patched


def select_correct_non_target(
    clf: SGDClassifier,
    images: np.ndarray,
    labels: np.ndarray,
    target_digit: int,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    pred = clf.predict(flat(images))
    idx = np.flatnonzero((pred == labels) & (labels != target_digit))
    idx = idx[: min(count, len(idx))]
    return images[idx], labels[idx]


def perturbation_stats(original: np.ndarray, attacked: np.ndarray) -> tuple[float, float, float]:
    delta = np.abs(attacked - original)
    l0 = np.mean(np.sum(delta > 1e-6, axis=(1, 2)))
    linf = np.mean(np.max(delta.reshape(len(delta), -1), axis=1))
    l2 = np.mean(np.sqrt(np.sum(delta.reshape(len(delta), -1) ** 2, axis=1)))
    return float(l0), float(linf), float(l2)


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


def random_label_flip(labels: np.ndarray, rate: float, seed: int) -> tuple[np.ndarray, int]:
    rng = np.random.default_rng(seed)
    poisoned = labels.copy()
    count = int(round(len(labels) * rate))
    idx = rng.choice(len(labels), size=count, replace=False)
    for i in idx:
        choices = [digit for digit in range(10) if digit != int(labels[i])]
        poisoned[i] = int(rng.choice(choices))
    return poisoned, count


def targeted_label_flip(
    labels: np.ndarray,
    source_digit: int,
    target_digit: int,
    source_rate: float,
    seed: int,
) -> tuple[np.ndarray, int]:
    rng = np.random.default_rng(seed)
    poisoned = labels.copy()
    source_idx = np.flatnonzero(labels == source_digit)
    count = int(round(len(source_idx) * source_rate))
    chosen = rng.choice(source_idx, size=count, replace=False)
    poisoned[chosen] = target_digit
    return poisoned, count


def subpopulation_indices(images: np.ndarray, labels: np.ndarray, source_digit: int, quantile: float) -> np.ndarray:
    source_idx = np.flatnonzero(labels == source_digit)
    source_images = images[source_idx]
    upper_right = source_images[:, :14, 14:].mean(axis=(1, 2))
    lower_left = source_images[:, 14:, :14].mean(axis=(1, 2))
    score = upper_right - lower_left
    threshold = np.quantile(score, quantile)
    return source_idx[score >= threshold]


def subpopulation_label_flip(
    images: np.ndarray,
    labels: np.ndarray,
    source_digit: int,
    target_digit: int,
    quantile: float,
    flip_rate: float,
    seed: int,
) -> tuple[np.ndarray, int, np.ndarray]:
    rng = np.random.default_rng(seed)
    poisoned = labels.copy()
    sub_idx = subpopulation_indices(images, labels, source_digit, quantile)
    count = max(1, int(round(len(sub_idx) * flip_rate)))
    chosen = rng.choice(sub_idx, size=count, replace=False)
    poisoned[chosen] = target_digit
    return poisoned, count, sub_idx


def backdoor_poison(
    images: np.ndarray,
    labels: np.ndarray,
    target_digit: int,
    rate: float,
    trigger_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    rng = np.random.default_rng(seed)
    non_target = np.flatnonzero(labels != target_digit)
    count = int(round(len(labels) * rate))
    chosen = rng.choice(non_target, size=count, replace=False)
    poisoned_images = np.concatenate([images, apply_trigger(images[chosen], trigger_size)], axis=0)
    poisoned_labels = np.concatenate([labels, np.full(count, target_digit, dtype=np.int64)], axis=0)
    return poisoned_images, poisoned_labels, count


def clean_label_backdoor_poison(
    images: np.ndarray,
    labels: np.ndarray,
    target_digit: int,
    rate: float,
    trigger_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    rng = np.random.default_rng(seed)
    target_idx = np.flatnonzero(labels == target_digit)
    count = min(len(target_idx), int(round(len(labels) * rate)))
    chosen = rng.choice(target_idx, size=count, replace=False)
    poisoned_images = np.concatenate([images, apply_trigger(images[chosen], trigger_size)], axis=0)
    poisoned_labels = np.concatenate([labels, labels[chosen]], axis=0)
    return poisoned_images, poisoned_labels, count


def prototype_collision_poison(
    images: np.ndarray,
    labels: np.ndarray,
    target_digit: int,
    source_digit: int,
    count: int,
    blend: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    rng = np.random.default_rng(seed)
    source_idx = np.flatnonzero(labels == source_digit)
    chosen = rng.choice(source_idx, size=min(count, len(source_idx)), replace=False)
    target_proto = images[labels == target_digit].mean(axis=0)
    poison = np.clip((1.0 - blend) * images[chosen] + blend * target_proto, 0.0, 1.0)
    poisoned_images = np.concatenate([images, poison], axis=0)
    poisoned_labels = np.concatenate([labels, np.full(len(chosen), source_digit, dtype=np.int64)], axis=0)
    return poisoned_images, poisoned_labels, len(chosen)


def source_to_target_rate(
    clf: SGDClassifier,
    images: np.ndarray,
    labels: np.ndarray,
    source_digit: int,
    target_digit: int,
) -> float:
    mask = labels == source_digit
    pred = clf.predict(flat(images[mask]))
    return float(np.mean(pred == target_digit))


def subpopulation_to_target_rate(
    clf: SGDClassifier,
    images: np.ndarray,
    labels: np.ndarray,
    source_digit: int,
    target_digit: int,
    quantile: float,
) -> float:
    idx = subpopulation_indices(images, labels, source_digit, quantile)
    if len(idx) == 0:
        return 0.0
    pred = clf.predict(flat(images[idx]))
    return float(np.mean(pred == target_digit))


def trigger_asr(
    clf: SGDClassifier,
    images: np.ndarray,
    labels: np.ndarray,
    target_digit: int,
    trigger_size: int,
) -> float:
    mask = labels != target_digit
    pred = clf.predict(flat(apply_trigger(images[mask], trigger_size)))
    return float(np.mean(pred == target_digit))


def fgsm(clf: SGDClassifier, images: np.ndarray, labels: np.ndarray, eps: float) -> np.ndarray:
    grad = ce_gradient(clf, images, labels)
    return np.clip(images + eps * np.sign(grad), 0.0, 1.0)


def pgd(clf: SGDClassifier, images: np.ndarray, labels: np.ndarray, eps: float, step: float, iters: int) -> np.ndarray:
    adv = images.copy()
    for _ in range(iters):
        grad = ce_gradient(clf, adv, labels)
        adv = adv + step * np.sign(grad)
        adv = np.clip(np.minimum(np.maximum(adv, images - eps), images + eps), 0.0, 1.0)
    return adv


def ead_target(
    clf: SGDClassifier,
    images: np.ndarray,
    target_digit: int,
    eps: float,
    step: float,
    l1_weight: float,
    l2_weight: float,
    iters: int,
) -> np.ndarray:
    adv = images.copy()
    for _ in range(iters):
        grad = targeted_ce_gradient(clf, adv, target_digit) + l2_weight * (adv - images)
        adv = adv - step * np.sign(grad)
        delta = adv - images
        delta = np.sign(delta) * np.maximum(np.abs(delta) - step * l1_weight, 0.0)
        delta = np.clip(delta, -eps, eps)
        adv = np.clip(images + delta, 0.0, 1.0)
    return adv


def jsma_target(
    clf: SGDClassifier,
    images: np.ndarray,
    target_digit: int,
    max_pixels: int,
) -> np.ndarray:
    adv = images.copy()
    direction = (clf.coef_[target_digit] - clf.coef_).max(axis=0).reshape(28, 28)
    for n in range(len(adv)):
        used: set[int] = set()
        for _ in range(max_pixels):
            pred = int(clf.predict(flat(adv[n : n + 1]))[0])
            if pred == target_digit:
                break
            saliency = clf.coef_[target_digit].reshape(28, 28) - clf.coef_[pred].reshape(28, 28)
            flat_order = np.argsort(np.abs(saliency).reshape(-1))[::-1]
            coord = next((int(c) for c in flat_order if int(c) not in used), int(flat_order[0]))
            used.add(coord)
            r, c = divmod(coord, 28)
            adv[n, r, c] = 1.0 if saliency[r, c] >= 0 else 0.0
    return adv


def sparse_boundary_attack(clf: SGDClassifier, images: np.ndarray, max_pixels: int) -> np.ndarray:
    adv = images.copy()
    weights = clf.coef_
    bias = clf.intercept_
    for n in range(len(adv)):
        x = adv[n].reshape(-1)
        scores = weights @ x + bias
        y = int(np.argmax(scores))
        rivals = np.argsort(scores)[::-1]
        k = int(next(r for r in rivals if int(r) != y))
        diff = weights[k] - weights[y]
        margin = float(scores[y] - scores[k])
        order = np.argsort(np.abs(diff))[::-1]
        changed = 0
        for coord in order:
            if margin <= 0 or changed >= max_pixels:
                break
            coord = int(coord)
            if diff[coord] >= 0:
                gain = diff[coord] * (1.0 - x[coord])
                x[coord] = 1.0
            else:
                gain = -diff[coord] * x[coord]
                x[coord] = 0.0
            margin -= float(gain)
            changed += 1
        adv[n] = x.reshape(28, 28)
    return adv


def universal_patch(
    clf: SGDClassifier,
    train_images: np.ndarray,
    train_labels: np.ndarray,
    patch_size: int,
    train_count: int,
    iters: int,
    batch_size: int,
    step: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    candidates = np.arange(len(train_labels))
    chosen = rng.choice(candidates, size=min(train_count, len(candidates)), replace=False)
    pool = train_images[chosen]
    pool_labels = train_labels[chosen]
    patch = rng.uniform(0.0, 1.0, size=(patch_size, patch_size)).astype(np.float32)
    for _ in range(iters):
        batch_idx = rng.choice(len(pool), size=min(batch_size, len(pool)), replace=False)
        batch = pool[batch_idx].copy()
        batch[:, -patch_size:, -patch_size:] = patch
        grad = ce_gradient(clf, batch, pool_labels[batch_idx])
        patch_grad = grad[:, -patch_size:, -patch_size:].mean(axis=0)
        patch = np.clip(patch + step * np.sign(patch_grad), 0.0, 1.0)
    return patch


def apply_universal_patch(images: np.ndarray, patch: np.ndarray) -> np.ndarray:
    patched = images.copy()
    s = patch.shape[0]
    patched[:, -s:, -s:] = patch
    return patched


def zoo_target(
    clf: SGDClassifier,
    images: np.ndarray,
    target_digit: int,
    eps: float,
    step: float,
    iters: int,
    coords_per_iter: int,
    seed: int,
) -> tuple[np.ndarray, int]:
    rng = np.random.default_rng(seed)
    adv = images.copy()
    queries = 0

    def loss_one(x_flat: np.ndarray) -> float:
        proba = clf.predict_proba(x_flat.reshape(1, -1))[0, target_digit]
        return float(-math.log(max(proba, 1e-12)))

    for n in range(len(adv)):
        base = images[n].reshape(-1)
        cur = adv[n].reshape(-1).copy()
        for _ in range(iters):
            coords = rng.choice(N_PIXELS, size=coords_per_iter, replace=False)
            grad = np.zeros(N_PIXELS, dtype=np.float32)
            for coord in coords:
                coord = int(coord)
                plus = cur.copy()
                minus = cur.copy()
                plus[coord] = min(1.0, plus[coord] + 1e-2)
                minus[coord] = max(0.0, minus[coord] - 1e-2)
                grad[coord] = (loss_one(plus) - loss_one(minus)) / 2e-2
                queries += 2
            cur = cur - step * np.sign(grad)
            cur = np.clip(np.minimum(np.maximum(cur, base - eps), base + eps), 0.0, 1.0)
            if int(clf.predict(cur.reshape(1, -1))[0]) == target_digit:
                queries += 1
                break
            queries += 1
        adv[n] = cur.reshape(28, 28)
    return adv, queries


def boundary_line_search(
    clf: SGDClassifier,
    images: np.ndarray,
    labels: np.ndarray,
    target_images: np.ndarray,
    target_digit: int,
    steps: int,
) -> np.ndarray:
    adv = images.copy()
    for n in range(len(images)):
        guide = target_images[n % len(target_images)]
        lo, hi = 0.0, 1.0
        for _ in range(steps):
            mid = (lo + hi) / 2.0
            candidate = np.clip((1.0 - mid) * images[n] + mid * guide, 0.0, 1.0)
            pred = int(clf.predict(flat(candidate[None, :, :]))[0])
            if pred == target_digit:
                hi = mid
            else:
                lo = mid
        adv[n] = np.clip((1.0 - hi) * images[n] + hi * guide, 0.0, 1.0)
    return adv


def targeted_success(clf: SGDClassifier, images: np.ndarray, target_digit: int) -> float:
    return float(np.mean(clf.predict(flat(images)) == target_digit))


def untargeted_success(clf: SGDClassifier, images: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean(clf.predict(flat(images)) != labels))


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


def save_bar_chart(rows: list[dict[str, float | str | int]]) -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    plot_rows = [r for r in rows if r["method"] != "Clean baseline"]
    labels = [str(r["method"]).replace(" ", "\n") for r in plot_rows]
    attack = [float(r["attack_success"]) for r in plot_rows]
    clean = [float(r["clean_accuracy"]) for r in plot_rows]
    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(x - width / 2, clean, width, label="Clean accuracy")
    ax.bar(x + width / 2, attack, width, label="Attack success / damage")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate")
    ax.set_title("MNIST attack comparison across implemented methods")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "mnist_attack_comparison_bars.png", dpi=220)
    plt.close(fig)


def save_attack_examples(
    clean_images: np.ndarray,
    attacked: list[tuple[str, np.ndarray]],
    path: Path,
) -> None:
    cols = min(8, len(clean_images))
    rows = 1 + len(attacked)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.25, rows * 1.35))
    axes = np.atleast_2d(axes)
    for c in range(cols):
        axes[0, c].imshow(clean_images[c], cmap="gray", vmin=0, vmax=1)
        axes[0, c].set_title("clean", fontsize=8)
    for r, (name, images) in enumerate(attacked, start=1):
        for c in range(cols):
            axes[r, c].imshow(images[c], cmap="gray", vmin=0, vmax=1)
            axes[r, c].set_title(name, fontsize=7)
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle("MNIST attack examples", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    x_train, y_train, x_test, y_test = load_mnist(args.download)
    baseline = fit_classifier(x_train, y_train, args.seed, args.max_iter)
    baseline_acc = float(accuracy_score(y_test, baseline.predict(flat(x_test))))
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
            notes="Full MNIST train/test split with multinomial SGD logistic classifier.",
        )
    ]

    poisoned_labels, count = random_label_flip(y_train, 0.10, args.seed + 1)
    model = fit_classifier(x_train, poisoned_labels, args.seed + 1, args.max_iter)
    acc = float(accuracy_score(y_test, model.predict(flat(x_test))))
    rows.append(metric_row("Random label flip", "poisoning", "M", "R3", acc, "clean_accuracy_drop", max(0.0, baseline_acc - acc), len(x_train), "10.00%", notes=f"{count} labels changed."))

    poisoned_labels, count = targeted_label_flip(y_train, args.source_digit, args.target_digit, 0.35, args.seed + 2)
    model = fit_classifier(x_train, poisoned_labels, args.seed + 2, args.max_iter)
    acc = float(accuracy_score(y_test, model.predict(flat(x_test))))
    s2t = source_to_target_rate(model, x_test, y_test, args.source_digit, args.target_digit)
    rows.append(metric_row("Targeted label flip", "poisoning", "M", "R2", acc, f"{args.source_digit}_to_{args.target_digit}_confusion", s2t, len(x_train), f"35% of digit {args.source_digit}", notes=f"{count} source labels changed."))

    poisoned_labels, count, _ = subpopulation_label_flip(x_train, y_train, args.source_digit, args.target_digit, 0.70, 0.80, args.seed + 3)
    model = fit_classifier(x_train, poisoned_labels, args.seed + 3, args.max_iter)
    acc = float(accuracy_score(y_test, model.predict(flat(x_test))))
    sub = subpopulation_to_target_rate(model, x_test, y_test, args.source_digit, args.target_digit, 0.70)
    rows.append(metric_row("Subpopulation label poison", "poisoning", "M", "R2", acc, "subpopulation_to_target_confusion", sub, len(x_train), "80% of selected subgroup", notes=f"{count} source-subgroup labels changed."))

    poisoned_images, poisoned_labels, count = backdoor_poison(x_train, y_train, args.target_digit, 0.05, args.trigger_size, args.seed + 4)
    model = fit_classifier(poisoned_images, poisoned_labels, args.seed + 4, args.max_iter)
    acc = float(accuracy_score(y_test, model.predict(flat(x_test))))
    rows.append(metric_row("BadNets patch backdoor", "poisoning", "B", "R1", acc, f"trigger_ASR_to_{args.target_digit}", trigger_asr(model, x_test, y_test, args.target_digit, args.trigger_size), len(poisoned_images), "5.00% appended", notes=f"{count} triggered non-target copies."))

    poisoned_images, poisoned_labels, count = clean_label_backdoor_poison(x_train, y_train, args.target_digit, 0.05, args.trigger_size, args.seed + 5)
    model = fit_classifier(poisoned_images, poisoned_labels, args.seed + 5, args.max_iter)
    acc = float(accuracy_score(y_test, model.predict(flat(x_test))))
    rows.append(metric_row("Clean-label patch backdoor", "poisoning", "B", "R1", acc, f"trigger_ASR_to_{args.target_digit}", trigger_asr(model, x_test, y_test, args.target_digit, args.trigger_size), len(poisoned_images), "5.00% target-class copies", notes=f"{count} correctly labeled target images patched."))

    poisoned_images, poisoned_labels, count = prototype_collision_poison(x_train, y_train, args.target_digit, args.source_digit, 900, 0.35, args.seed + 6)
    model = fit_classifier(poisoned_images, poisoned_labels, args.seed + 6, args.max_iter)
    acc = float(accuracy_score(y_test, model.predict(flat(x_test))))
    t2s = source_to_target_rate(model, x_test, y_test, args.target_digit, args.source_digit)
    rows.append(metric_row("Prototype feature collision", "poisoning", "O", "R2", acc, f"{args.target_digit}_to_{args.source_digit}_confusion", t2s, len(poisoned_images), f"{count} appended", notes="Clean-label source samples blended toward target prototype."))

    evasion_x, evasion_y = select_correct_non_target(baseline, x_test, y_test, args.target_digit, args.eval_samples)
    target_pool = x_test[y_test == args.target_digit][: max(1, args.eval_samples)]

    adv = fgsm(baseline, evasion_x, evasion_y, eps=0.22)
    l0, linf, l2 = perturbation_stats(evasion_x, adv)
    rows.append(metric_row("FGSM", "evasion", "G", "R3", baseline_acc, "untargeted_misclassification", untargeted_success(baseline, adv, evasion_y), len(x_train), l0=l0, linf=linf, l2=l2, notes="Single-step gradient sign attack on clean baseline."))

    adv = pgd(baseline, evasion_x, evasion_y, eps=0.25, step=0.05, iters=10)
    l0, linf, l2 = perturbation_stats(evasion_x, adv)
    rows.append(metric_row("PGD", "evasion", "G", "R2", baseline_acc, "untargeted_misclassification", untargeted_success(baseline, adv, evasion_y), len(x_train), l0=l0, linf=linf, l2=l2, notes="Projected gradient ascent within an L-infinity ball."))

    adv = ead_target(baseline, evasion_x, args.target_digit, eps=0.35, step=0.035, l1_weight=0.002, l2_weight=0.02, iters=35)
    l0, linf, l2 = perturbation_stats(evasion_x, adv)
    rows.append(metric_row("Elastic Net EAD", "evasion", "O", "R2", baseline_acc, f"targeted_ASR_to_{args.target_digit}", targeted_success(baseline, adv, args.target_digit), len(x_train), l0=l0, linf=linf, l2=l2, notes="Targeted elastic-net style update with L1 shrinkage."))

    adv = jsma_target(baseline, evasion_x[:300], args.target_digit, max_pixels=45)
    l0, linf, l2 = perturbation_stats(evasion_x[:300], adv)
    rows.append(metric_row("JSMA saliency", "evasion", "M", "R3", baseline_acc, f"targeted_ASR_to_{args.target_digit}", targeted_success(baseline, adv, args.target_digit), len(x_train), l0=l0, linf=linf, l2=l2, notes="Greedy sparse-pixel saliency using model Jacobian/logit weights."))

    adv = sparse_boundary_attack(baseline, evasion_x[:300], max_pixels=45)
    l0, linf, l2 = perturbation_stats(evasion_x[:300], adv)
    rows.append(metric_row("SparseFool-style boundary", "evasion", "M", "R4", baseline_acc, "untargeted_misclassification", untargeted_success(baseline, adv, evasion_y[:300]), len(x_train), l0=l0, linf=linf, l2=l2, notes="SparseFool specialization for this linear MNIST classifier: sparse linear-boundary crossing."))

    patch = universal_patch(baseline, x_train, y_train, args.adversarial_patch_size, args.patch_train_samples, 120, 128, 0.08, args.seed + 7)
    adv = apply_universal_patch(evasion_x, patch)
    l0, linf, l2 = perturbation_stats(evasion_x, adv)
    rows.append(metric_row("Adversarial patch", "evasion", "U", "R2", baseline_acc, "untargeted_misclassification", untargeted_success(baseline, adv, evasion_y), len(x_train), l0=l0, linf=linf, l2=l2, notes="Universal fixed-position patch optimized to maximize baseline loss; EOT is not yet implemented."))

    zoo_x, zoo_y = evasion_x[: args.zoo_samples], evasion_y[: args.zoo_samples]
    adv, queries = zoo_target(baseline, zoo_x, args.target_digit, eps=0.35, step=0.08, iters=18, coords_per_iter=32, seed=args.seed + 8)
    l0, linf, l2 = perturbation_stats(zoo_x, adv)
    rows.append(metric_row("ZOO finite difference", "evasion", "O", "R3", baseline_acc, f"targeted_ASR_to_{args.target_digit}", targeted_success(baseline, adv, args.target_digit), len(x_train), l0=l0, linf=linf, l2=l2, notes=f"Black-box coordinate finite differences; {queries} model queries."))

    boundary_x, boundary_y = evasion_x[:300], evasion_y[:300]
    adv = boundary_line_search(baseline, boundary_x, boundary_y, target_pool, args.target_digit, steps=14)
    l0, linf, l2 = perturbation_stats(boundary_x, adv)
    rows.append(metric_row("HopSkipJump-style boundary", "evasion", "O", "R3", baseline_acc, f"targeted_ASR_to_{args.target_digit}", targeted_success(baseline, adv, args.target_digit), len(x_train), l0=l0, linf=linf, l2=l2, notes="Decision-only binary search between clean input and target-class guide image."))

    save_results(rows)
    save_bar_chart(rows)
    save_attack_examples(
        evasion_x[:8],
        [
            ("FGSM", fgsm(baseline, evasion_x[:8], evasion_y[:8], eps=0.22)),
            ("PGD", pgd(baseline, evasion_x[:8], evasion_y[:8], eps=0.25, step=0.05, iters=10)),
            ("JSMA", jsma_target(baseline, evasion_x[:8], args.target_digit, max_pixels=45)),
            ("Patch", apply_universal_patch(evasion_x[:8], patch)),
        ],
        FIGURES_DIR / "mnist_attack_examples.png",
    )

    for row in rows:
        print(f"{row['method']}: clean={row['clean_accuracy']} {row['attack_metric']}={row['attack_success']}")
    print(f"\nWrote {RESULTS_DIR / 'mnist_attack_comparison.md'}")
    print(f"Wrote {FIGURES_DIR / 'mnist_attack_comparison_bars.png'}")


if __name__ == "__main__":
    main()
