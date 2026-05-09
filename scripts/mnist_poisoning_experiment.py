#!/usr/bin/env python3
"""Real MNIST data-poisoning experiments for conference presentation.

This script uses the standard MNIST handwritten-digit dataset through
torchvision, trains real scikit-learn classifiers, and writes figures plus
metrics for label poisoning and visible-trigger backdoor poisoning.
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
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from torchvision.datasets import MNIST


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real MNIST poisoning experiments.")
    parser.add_argument("--download", action="store_true", help="Download MNIST if it is not present.")
    parser.add_argument("--target-digit", type=int, default=1, help="Backdoor and targeted-label target digit.")
    parser.add_argument("--source-digit", type=int, default=7, help="Targeted-label source digit.")
    parser.add_argument("--random-label-rate", type=float, default=0.10, help="Fraction of all training labels to flip.")
    parser.add_argument(
        "--targeted-source-rate",
        type=float,
        default=0.35,
        help="Fraction of source-digit training labels to relabel to target.",
    )
    parser.add_argument("--backdoor-rate", type=float, default=0.05, help="Fraction of training set copied with trigger.")
    parser.add_argument("--trigger-size", type=int, default=5, help="Visible square trigger size in pixels.")
    parser.add_argument("--max-iter", type=int, default=25, help="SGDClassifier training epochs.")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed.")
    return parser.parse_args()


def load_mnist(download: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train = MNIST(root=str(DATA_DIR), train=True, download=download)
    test = MNIST(root=str(DATA_DIR), train=False, download=download)
    x_train = train.data.numpy().astype(np.float32) / 255.0
    y_train = train.targets.numpy().astype(np.int64)
    x_test = test.data.numpy().astype(np.float32) / 255.0
    y_test = test.targets.numpy().astype(np.int64)
    return x_train, y_train, x_test, y_test


def flatten(images: np.ndarray) -> np.ndarray:
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
    clf.fit(flatten(images), labels)
    return clf


def random_label_flip(labels: np.ndarray, rate: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    poisoned = labels.copy()
    count = int(round(len(labels) * rate))
    indices = rng.choice(len(labels), size=count, replace=False)
    for idx in indices:
        choices = [digit for digit in range(10) if digit != int(poisoned[idx])]
        poisoned[idx] = int(rng.choice(choices))
    return poisoned, indices


def targeted_label_flip(
    labels: np.ndarray, source_digit: int, target_digit: int, source_rate: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    poisoned = labels.copy()
    source_indices = np.flatnonzero(labels == source_digit)
    count = int(round(len(source_indices) * source_rate))
    indices = rng.choice(source_indices, size=count, replace=False)
    poisoned[indices] = target_digit
    return poisoned, indices


def apply_trigger(images: np.ndarray, trigger_size: int) -> np.ndarray:
    patched = images.copy()
    patched[:, -trigger_size:, -trigger_size:] = 1.0
    return patched


def backdoor_poison(
    images: np.ndarray,
    labels: np.ndarray,
    target_digit: int,
    rate: float,
    trigger_size: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    non_target_indices = np.flatnonzero(labels != target_digit)
    count = int(round(len(labels) * rate))
    selected = rng.choice(non_target_indices, size=count, replace=False)
    triggered_images = apply_trigger(images[selected], trigger_size)
    triggered_labels = np.full(count, target_digit, dtype=np.int64)
    poisoned_images = np.concatenate([images, triggered_images], axis=0)
    poisoned_labels = np.concatenate([labels, triggered_labels], axis=0)
    return poisoned_images, poisoned_labels, selected


def source_to_target_rate(
    clf: SGDClassifier, images: np.ndarray, labels: np.ndarray, source_digit: int, target_digit: int
) -> float:
    mask = labels == source_digit
    if not np.any(mask):
        return 0.0
    pred = clf.predict(flatten(images[mask]))
    return float(np.mean(pred == target_digit))


def backdoor_asr(
    clf: SGDClassifier, images: np.ndarray, labels: np.ndarray, target_digit: int, trigger_size: int
) -> float:
    mask = labels != target_digit
    triggered = apply_trigger(images[mask], trigger_size)
    pred = clf.predict(flatten(triggered))
    return float(np.mean(pred == target_digit))


def save_digit_grid(
    images: np.ndarray,
    labels: np.ndarray,
    path: Path,
    title: str,
    cols: int = 10,
    max_images: int = 30,
) -> None:
    count = min(max_images, len(images))
    rows = int(np.ceil(count / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.35))
    axes_arr = np.atleast_1d(axes).reshape(rows, cols)
    for ax in axes_arr.ravel():
        ax.axis("off")
    for i in range(count):
        ax = axes_arr.ravel()[i]
        ax.imshow(images[i], cmap="gray", vmin=0, vmax=1)
        ax.set_title(str(int(labels[i])), fontsize=9)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_backdoor_examples(
    images: np.ndarray,
    labels: np.ndarray,
    selected: np.ndarray,
    trigger_size: int,
    target_digit: int,
    path: Path,
) -> None:
    chosen = selected[:12]
    original = images[chosen]
    patched = apply_trigger(original, trigger_size)
    fig, axes = plt.subplots(2, len(chosen), figsize=(len(chosen) * 1.1, 2.8))
    for i, idx in enumerate(chosen):
        axes[0, i].imshow(original[i], cmap="gray", vmin=0, vmax=1)
        axes[0, i].set_title(f"true {int(labels[idx])}", fontsize=8)
        axes[1, i].imshow(patched[i], cmap="gray", vmin=0, vmax=1)
        axes[1, i].set_title(f"poison label {target_digit}", fontsize=8)
        rect = plt.Rectangle(
            (28 - trigger_size - 0.5, 28 - trigger_size - 0.5),
            trigger_size,
            trigger_size,
            linewidth=1.2,
            edgecolor="red",
            facecolor="none",
        )
        axes[1, i].add_patch(rect)
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle("MNIST visible-trigger backdoor examples", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_confusion(clf: SGDClassifier, images: np.ndarray, labels: np.ndarray, path: Path, title: str) -> None:
    pred = clf.predict(flatten(images))
    cm = confusion_matrix(labels, pred, labels=list(range(10)))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(10)))
    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_metric_bars(rows: list[dict[str, float | str]], path: Path) -> None:
    scenario_rows = [row for row in rows if row["scenario"] != "dataset"]
    labels = [str(row["scenario"]).replace("_", "\n") for row in scenario_rows]
    clean_acc = [float(row["clean_accuracy"]) for row in scenario_rows]
    attack_metric = [float(row["attack_metric_value"]) for row in scenario_rows]
    x = np.arange(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.bar(x - width / 2, clean_acc, width, label="Clean accuracy")
    ax.bar(x + width / 2, attack_metric, width, label="Attack metric")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Rate")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.legend()
    ax.set_title("MNIST poisoning results")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_results(rows: list[dict[str, float | str]]) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    fieldnames = [
        "scenario",
        "poison_rate",
        "clean_accuracy",
        "attack_metric_name",
        "attack_metric_value",
        "train_size",
        "notes",
    ]
    csv_path = RESULTS_DIR / "mnist_poisoning_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    json_path = RESULTS_DIR / "mnist_poisoning_results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    md_path = RESULTS_DIR / "mnist_poisoning_results.md"
    with md_path.open("w", encoding="utf-8") as f:
        f.write("# MNIST Poisoning Experiment Results\n\n")
        f.write("Dataset: real MNIST handwritten digits loaded through `torchvision.datasets.MNIST`.\n\n")
        f.write("| Scenario | Poison rate | Clean accuracy | Attack metric | Value | Train size | Notes |\n")
        f.write("|---|---:|---:|---|---:|---:|---|\n")
        for row in rows:
            f.write(
                f"| {row['scenario']} | {row['poison_rate']} | {row['clean_accuracy']} | "
                f"{row['attack_metric_name']} | {row['attack_metric_value']} | "
                f"{row['train_size']} | {row['notes']} |\n"
            )


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    x_train, y_train, x_test, y_test = load_mnist(download=args.download)
    save_digit_grid(
        x_train[:40],
        y_train[:40],
        FIGURES_DIR / "mnist_clean_samples.png",
        "Real MNIST training samples",
        max_images=40,
    )

    rows: list[dict[str, float | str]] = [
        {
            "scenario": "dataset",
            "poison_rate": "0",
            "clean_accuracy": "",
            "attack_metric_name": "train/test images",
            "attack_metric_value": "",
            "train_size": len(x_train),
            "notes": f"MNIST train={len(x_train)}, test={len(x_test)}, image_shape=28x28",
        }
    ]

    baseline = fit_classifier(x_train, y_train, args.seed, args.max_iter)
    baseline_pred = baseline.predict(flatten(x_test))
    baseline_acc = float(accuracy_score(y_test, baseline_pred))
    baseline_s2t = source_to_target_rate(baseline, x_test, y_test, args.source_digit, args.target_digit)
    rows.append(
        {
            "scenario": "clean_baseline",
            "poison_rate": "0",
            "clean_accuracy": round(baseline_acc, 4),
            "attack_metric_name": f"{args.source_digit}_to_{args.target_digit}_confusion",
            "attack_metric_value": round(baseline_s2t, 4),
            "train_size": len(x_train),
            "notes": "Unpoisoned reference model",
        }
    )
    save_confusion(
        baseline,
        x_test,
        y_test,
        FIGURES_DIR / "mnist_confusion_clean_baseline.png",
        "MNIST clean baseline confusion matrix",
    )

    random_labels, random_indices = random_label_flip(y_train, args.random_label_rate, args.seed + 1)
    random_model = fit_classifier(x_train, random_labels, args.seed + 1, args.max_iter)
    random_acc = float(accuracy_score(y_test, random_model.predict(flatten(x_test))))
    rows.append(
        {
            "scenario": "random_label_flip",
            "poison_rate": f"{args.random_label_rate:.2%}",
            "clean_accuracy": round(random_acc, 4),
            "attack_metric_name": "clean_accuracy_drop",
            "attack_metric_value": round(max(0.0, baseline_acc - random_acc), 4),
            "train_size": len(x_train),
            "notes": f"{len(random_indices)} real MNIST labels changed",
        }
    )

    targeted_labels, targeted_indices = targeted_label_flip(
        y_train, args.source_digit, args.target_digit, args.targeted_source_rate, args.seed + 2
    )
    targeted_model = fit_classifier(x_train, targeted_labels, args.seed + 2, args.max_iter)
    targeted_acc = float(accuracy_score(y_test, targeted_model.predict(flatten(x_test))))
    targeted_s2t = source_to_target_rate(targeted_model, x_test, y_test, args.source_digit, args.target_digit)
    rows.append(
        {
            "scenario": "targeted_label_flip",
            "poison_rate": f"{args.targeted_source_rate:.2%} of digit {args.source_digit}",
            "clean_accuracy": round(targeted_acc, 4),
            "attack_metric_name": f"{args.source_digit}_to_{args.target_digit}_confusion",
            "attack_metric_value": round(targeted_s2t, 4),
            "train_size": len(x_train),
            "notes": f"{len(targeted_indices)} digit-{args.source_digit} labels changed to {args.target_digit}",
        }
    )
    save_confusion(
        targeted_model,
        x_test,
        y_test,
        FIGURES_DIR / "mnist_confusion_targeted_label_flip.png",
        "MNIST targeted label-flip confusion matrix",
    )

    backdoor_images, backdoor_labels, backdoor_indices = backdoor_poison(
        x_train,
        y_train,
        args.target_digit,
        args.backdoor_rate,
        args.trigger_size,
        args.seed + 3,
    )
    save_backdoor_examples(
        x_train,
        y_train,
        backdoor_indices,
        args.trigger_size,
        args.target_digit,
        FIGURES_DIR / "mnist_backdoor_examples.png",
    )
    backdoor_model = fit_classifier(backdoor_images, backdoor_labels, args.seed + 3, args.max_iter)
    backdoor_acc = float(accuracy_score(y_test, backdoor_model.predict(flatten(x_test))))
    asr = backdoor_asr(backdoor_model, x_test, y_test, args.target_digit, args.trigger_size)
    rows.append(
        {
            "scenario": "visible_patch_backdoor",
            "poison_rate": f"{args.backdoor_rate:.2%} appended triggered samples",
            "clean_accuracy": round(backdoor_acc, 4),
            "attack_metric_name": f"trigger_ASR_to_{args.target_digit}",
            "attack_metric_value": round(asr, 4),
            "train_size": len(backdoor_images),
            "notes": f"{len(backdoor_indices)} non-target MNIST images copied with {args.trigger_size}x{args.trigger_size} trigger",
        }
    )
    save_confusion(
        backdoor_model,
        x_test,
        y_test,
        FIGURES_DIR / "mnist_confusion_visible_patch_backdoor.png",
        "MNIST visible patch-backdoor clean-test confusion matrix",
    )

    save_metric_bars(rows, FIGURES_DIR / "mnist_metric_bars.png")
    write_results(rows)

    for row in rows:
        print(
            f"{row['scenario']}: clean_accuracy={row['clean_accuracy']} "
            f"{row['attack_metric_name']}={row['attack_metric_value']}"
        )
    print(f"\nWrote results to {RESULTS_DIR}")
    print(f"Wrote figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
