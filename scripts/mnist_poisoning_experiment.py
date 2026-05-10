#!/usr/bin/env python3
"""MNIST poisoning/backdoor evidence with the shared PyTorch CNN baseline."""

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

from mnist_attack_comparison import (
    FIGURES_DIR,
    RESULTS_DIR,
    accuracy,
    apply_trigger,
    backdoor_poison,
    load_mnist,
    predict,
    random_label_flip,
    seed_everything,
    source_to_target_rate,
    stratified_limit,
    targeted_label_flip,
    train_model,
    trigger_asr,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MNIST CNN poisoning experiments.")
    parser.add_argument("--download", action="store_true", help="Download MNIST if it is not present.")
    parser.add_argument("--target-digit", type=int, default=1, help="Backdoor and targeted-label target digit.")
    parser.add_argument("--source-digit", type=int, default=7, help="Targeted-label source digit.")
    parser.add_argument("--random-label-rate", type=float, default=0.10, help="Fraction of all training labels to flip.")
    parser.add_argument("--targeted-source-rate", type=float, default=0.35, help="Fraction of source-digit labels to relabel.")
    parser.add_argument("--backdoor-rate", type=float, default=0.05, help="Fraction of training set copied with trigger.")
    parser.add_argument("--trigger-size", type=int, default=5, help="Visible square trigger size in pixels.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--train-limit", type=int, default=0, help="Optional stratified train subset for smoke tests.")
    parser.add_argument("--test-limit", type=int, default=0, help="Optional stratified test subset for smoke tests.")
    parser.add_argument("--defense", choices=["none", "activation_clustering"], default="none")
    parser.add_argument("--defense-warmup-epochs", type=int, default=1)
    parser.add_argument("--defense-max-cluster-fraction", type=float, default=0.40)
    return parser.parse_args()


def fit(images: torch.Tensor, labels: torch.Tensor, args: argparse.Namespace, seed: int):
    return train_model(
        images,
        labels,
        seed,
        args.epochs,
        args.batch_size,
        args.lr,
        args.defense,
        args.defense_warmup_epochs,
        args.defense_max_cluster_fraction,
    )


def confusion_matrix(labels: torch.Tensor, preds: torch.Tensor) -> np.ndarray:
    cm = np.zeros((10, 10), dtype=np.int64)
    for true, pred in zip(labels.cpu().numpy(), preds.cpu().numpy()):
        cm[int(true), int(pred)] += 1
    return cm


def save_digit_grid(
    images: torch.Tensor,
    labels: torch.Tensor,
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
        ax.imshow(images[i, 0].numpy(), cmap="gray", vmin=0, vmax=1)
        ax.set_title(str(int(labels[i])), fontsize=9)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_backdoor_examples(
    images: torch.Tensor,
    labels: torch.Tensor,
    selected: torch.Tensor,
    trigger_size: int,
    target_digit: int,
    path: Path,
) -> None:
    chosen = selected[:12]
    original = images[chosen]
    patched = apply_trigger(original, trigger_size)
    fig, axes = plt.subplots(2, len(chosen), figsize=(len(chosen) * 1.1, 2.8))
    for i, idx in enumerate(chosen):
        axes[0, i].imshow(original[i, 0].numpy(), cmap="gray", vmin=0, vmax=1)
        axes[0, i].set_title(f"true {int(labels[idx])}", fontsize=8)
        axes[1, i].imshow(patched[i, 0].numpy(), cmap="gray", vmin=0, vmax=1)
        axes[1, i].set_title(f"target {target_digit}", fontsize=8)
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
    fig.suptitle("MNIST CNN visible-trigger backdoor examples", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_confusion(model, images: torch.Tensor, labels: torch.Tensor, path: Path, title: str) -> None:
    cm = confusion_matrix(labels, predict(model, images))
    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    for i in range(10):
        for j in range(10):
            if cm[i, j] > 0:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    plt.close(fig)


def save_triggered_confusion(
    model,
    images: torch.Tensor,
    labels: torch.Tensor,
    target_digit: int,
    trigger_size: int,
    path: Path,
    title: str,
) -> None:
    mask = labels != target_digit
    triggered = apply_trigger(images[mask], trigger_size)
    cm = confusion_matrix(labels[mask], predict(model, triggered))
    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    im = ax.imshow(cm, cmap="Reds")
    ax.set_title(title)
    ax.set_xlabel(f"Predicted label after {trigger_size}x{trigger_size} trigger")
    ax.set_ylabel("True label")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    for i in range(10):
        for j in range(10):
            if cm[i, j] > 0:
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
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
    ax.bar(x + width / 2, attack_metric, width, label="Attack-specific effect / ASR")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy / normalized attack-specific effect")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.legend()
    ax.set_title("MNIST CNN poisoning results")
    fig.text(
        0.5,
        0.02,
        "Orange bars are metric-specific: clean-accuracy drop, targeted confusion, or trigger ASR depending on scenario.",
        ha="center",
        fontsize=8,
        color="#495057",
    )
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=220)
    plt.close(fig)


def write_results(rows: list[dict[str, float | str]]) -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    fieldnames = ["scenario", "poison_rate", "clean_accuracy", "attack_metric_name", "attack_metric_value", "train_size", "notes"]
    csv_path = RESULTS_DIR / "mnist_poisoning_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    with (RESULTS_DIR / "mnist_poisoning_results.json").open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    with (RESULTS_DIR / "mnist_poisoning_results.md").open("w", encoding="utf-8") as f:
        f.write("# MNIST CNN Poisoning Experiment Results\n\n")
        f.write("Dataset: real MNIST handwritten digits loaded through `torchvision.datasets.MNIST`.\n\n")
        f.write("| Scenario | Poison rate | Clean accuracy | Attack-specific metric | Value | Train size | Notes |\n")
        f.write("|---|---:|---:|---|---:|---:|---|\n")
        for row in rows:
            f.write(
                f"| {row['scenario']} | {row['poison_rate']} | {row['clean_accuracy']} | "
                f"{row['attack_metric_name']} | {row['attack_metric_value']} | {row['train_size']} | {row['notes']} |\n"
            )


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    seed_everything(args.seed)
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    x_train, y_train, x_test, y_test = load_mnist(download=args.download)
    x_train, y_train = stratified_limit(x_train, y_train, args.train_limit, args.seed)
    x_test, y_test = stratified_limit(x_test, y_test, args.test_limit, args.seed + 99)
    save_digit_grid(x_train[:40], y_train[:40], FIGURES_DIR / "mnist_clean_samples.png", "Real MNIST training samples", max_images=40)

    rows: list[dict[str, float | str]] = [
        {
            "scenario": "dataset",
            "poison_rate": "0",
            "clean_accuracy": "",
            "attack_metric_name": "train/test images",
            "attack_metric_value": "",
            "train_size": len(x_train),
            "notes": f"MNIST train={len(x_train)}, test={len(x_test)}, image_shape=1x28x28, model=2-layer CNN",
        }
    ]

    baseline, note = fit(x_train, y_train, args, args.seed)
    baseline_acc = accuracy(baseline, x_test, y_test)
    baseline_s2t = source_to_target_rate(baseline, x_test, y_test, args.source_digit, args.target_digit)
    rows.append(
        {
            "scenario": "clean_baseline",
            "poison_rate": "0",
            "clean_accuracy": round(baseline_acc, 4),
            "attack_metric_name": f"{args.source_digit}_to_{args.target_digit}_confusion",
            "attack_metric_value": round(baseline_s2t, 4),
            "train_size": len(x_train),
            "notes": f"Unpoisoned PyTorch CNN reference; defense={args.defense}; {note}",
        }
    )
    save_confusion(baseline, x_test, y_test, FIGURES_DIR / "mnist_confusion_clean_baseline.png", "MNIST CNN clean baseline confusion matrix")

    random_labels, random_count = random_label_flip(y_train, args.random_label_rate, args.seed + 1)
    random_model, note = fit(x_train, random_labels, args, args.seed + 1)
    random_acc = accuracy(random_model, x_test, y_test)
    rows.append(
        {
            "scenario": "random_label_flip",
            "poison_rate": f"{args.random_label_rate:.2%}",
            "clean_accuracy": round(random_acc, 4),
            "attack_metric_name": "clean_accuracy_drop",
            "attack_metric_value": round(max(0.0, baseline_acc - random_acc), 4),
            "train_size": len(x_train),
            "notes": f"{random_count} real MNIST labels changed; {note}",
        }
    )

    targeted_labels, targeted_count = targeted_label_flip(y_train, args.source_digit, args.target_digit, args.targeted_source_rate, args.seed + 2)
    targeted_model, note = fit(x_train, targeted_labels, args, args.seed + 2)
    targeted_acc = accuracy(targeted_model, x_test, y_test)
    targeted_s2t = source_to_target_rate(targeted_model, x_test, y_test, args.source_digit, args.target_digit)
    rows.append(
        {
            "scenario": "targeted_label_flip",
            "poison_rate": f"{args.targeted_source_rate:.2%} of digit {args.source_digit}",
            "clean_accuracy": round(targeted_acc, 4),
            "attack_metric_name": f"{args.source_digit}_to_{args.target_digit}_confusion",
            "attack_metric_value": round(targeted_s2t, 4),
            "train_size": len(x_train),
            "notes": f"{targeted_count} digit-{args.source_digit} labels changed to {args.target_digit}; {note}",
        }
    )
    save_confusion(targeted_model, x_test, y_test, FIGURES_DIR / "mnist_confusion_targeted_label_flip.png", "MNIST CNN targeted label-flip confusion matrix")

    backdoor_images, backdoor_labels, backdoor_count = backdoor_poison(
        x_train, y_train, args.target_digit, args.backdoor_rate, args.trigger_size, args.seed + 3
    )
    source_candidates = torch.nonzero(y_train != args.target_digit, as_tuple=False).flatten()
    save_backdoor_examples(
        x_train,
        y_train,
        source_candidates[: min(12, len(source_candidates))],
        args.trigger_size,
        args.target_digit,
        FIGURES_DIR / "mnist_backdoor_examples.png",
    )
    backdoor_model, note = fit(backdoor_images, backdoor_labels, args, args.seed + 3)
    backdoor_acc = accuracy(backdoor_model, x_test, y_test)
    asr = trigger_asr(backdoor_model, x_test, y_test, args.target_digit, args.trigger_size)
    rows.append(
        {
            "scenario": "visible_patch_backdoor",
            "poison_rate": f"{args.backdoor_rate:.2%} appended triggered samples",
            "clean_accuracy": round(backdoor_acc, 4),
            "attack_metric_name": f"trigger_ASR_to_{args.target_digit}",
            "attack_metric_value": round(asr, 4),
            "train_size": len(backdoor_images),
            "notes": f"{backdoor_count} non-target MNIST images copied with {args.trigger_size}x{args.trigger_size} trigger; {note}",
        }
    )
    save_confusion(backdoor_model, x_test, y_test, FIGURES_DIR / "mnist_confusion_visible_patch_backdoor.png", "MNIST CNN visible patch-backdoor clean-test confusion matrix")
    save_triggered_confusion(
        backdoor_model,
        x_test,
        y_test,
        args.target_digit,
        args.trigger_size,
        FIGURES_DIR / "mnist_confusion_visible_patch_backdoor_triggered.png",
        "MNIST CNN visible patch-backdoor triggered-test confusion matrix",
    )

    save_metric_bars(rows, FIGURES_DIR / "mnist_metric_bars.png")
    write_results(rows)

    for row in rows:
        print(f"{row['scenario']}: clean_accuracy={row['clean_accuracy']} {row['attack_metric_name']}={row['attack_metric_value']}")
    print(f"\nWrote results to {RESULTS_DIR}")
    print(f"Wrote figures to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
