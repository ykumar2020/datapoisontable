#!/usr/bin/env python3
"""Generate report tables and figures from experiment outputs.

This script is intentionally read-only with respect to datasets and experiment
CSV files. It writes derived statistics and figures used by the IEEE paper.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]
FUNGI_ROOT = ROOT / "fungi"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = ROOT / "figures"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, rows: object) -> None:
    path.parent.mkdir(exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def write_markdown(path: Path, title: str, rows: list[dict[str, object]]) -> None:
    columns = list(rows[0].keys())
    with path.open("w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        f.write("| " + " | ".join(columns) + " |\n")
        f.write("|" + "|".join("---" for _ in columns) + "|\n")
        for row in rows:
            f.write("| " + " | ".join(str(row[c]) for c in columns) + " |\n")


def image_files(split: str, class_name: str) -> list[Path]:
    root = FUNGI_ROOT / split / class_name
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def collect_fungi_dataset_statistics() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split in ["train", "val"]:
        for class_name in ["edible", "poisonous"]:
            files = image_files(split, class_name)
            widths: list[int] = []
            heights: list[int] = []
            file_kb: list[float] = []
            rgb_means: list[np.ndarray] = []
            for path in files:
                with Image.open(path) as img:
                    img = ImageOps.exif_transpose(img).convert("RGB")
                    widths.append(img.width)
                    heights.append(img.height)
                    small = img.resize((32, 32), Image.BILINEAR)
                    rgb_means.append(np.asarray(small, dtype=np.float32).mean(axis=(0, 1)))
                file_kb.append(path.stat().st_size / 1024.0)
            rgb = np.vstack(rgb_means) if rgb_means else np.zeros((1, 3))
            rows.append(
                {
                    "split": split,
                    "class": class_name,
                    "count": len(files),
                    "width_min": int(np.min(widths)),
                    "width_median": int(np.median(widths)),
                    "width_max": int(np.max(widths)),
                    "height_min": int(np.min(heights)),
                    "height_median": int(np.median(heights)),
                    "height_max": int(np.max(heights)),
                    "file_kb_median": round(float(np.median(file_kb)), 1),
                    "mean_r": round(float(rgb[:, 0].mean()), 1),
                    "mean_g": round(float(rgb[:, 1].mean()), 1),
                    "mean_b": round(float(rgb[:, 2].mean()), 1),
                }
            )
    return rows


def save_fungi_sample_grid() -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    selections: list[tuple[str, Path]] = []
    for split in ["train", "val"]:
        for class_name in ["edible", "poisonous"]:
            files = image_files(split, class_name)[:3]
            selections.extend((f"{split}/{class_name}", path) for path in files)

    cols = 6
    rows = int(np.ceil(len(selections) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.45, rows * 1.55))
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.axis("off")
    for idx, (label, path) in enumerate(selections):
        ax = axes[idx // cols, idx % cols]
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img).convert("RGB").resize((160, 160), Image.BILINEAR)
            ax.imshow(img)
        ax.set_title(label, fontsize=7)
    fig.suptitle("Fungi dataset samples", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fungi_dataset_samples.png", dpi=220)
    plt.close(fig)


def attack_summary_statistics(rows: list[dict[str, str]], dataset: str) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["stage"], row["mechanism"])].append(row)
    output: list[dict[str, object]] = []
    for (stage, mechanism), group in sorted(grouped.items()):
        clean = [float(r["clean_accuracy"]) for r in group if r["clean_accuracy"]]
        attack = [float(r["attack_success"]) for r in group if r["attack_success"]]
        output.append(
            {
                "dataset": dataset,
                "stage": stage,
                "mechanism": mechanism,
                "methods": len(group),
                "clean_accuracy_mean": round(float(np.mean(clean)), 4) if clean else "",
                "attack_success_mean": round(float(np.mean(attack)), 4) if attack else "",
                "attack_success_max": round(float(np.max(attack)), 4) if attack else "",
            }
        )
    return output


def save_dataset_split_chart(stats_rows: list[dict[str, object]]) -> None:
    labels = ["edible", "poisonous"]
    train = [int(next(r["count"] for r in stats_rows if r["split"] == "train" and r["class"] == c)) for c in labels]
    val = [int(next(r["count"] for r in stats_rows if r["split"] == "val" and r["class"] == c)) for c in labels]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    ax.bar(x, train, label="train")
    ax.bar(x, val, bottom=train, label="val")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Images")
    ax.set_title("Fungi dataset split")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fungi_dataset_split.png", dpi=220)
    plt.close(fig)


def save_attack_dashboard(fungi_rows: list[dict[str, str]]) -> None:
    plot_rows = [r for r in fungi_rows if r["method"] != "Clean baseline"]
    sorted_rows = sorted(plot_rows, key=lambda r: float(r["attack_success"]), reverse=True)
    top = sorted_rows[:10]

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax = axes[0, 0]
    labels = [r["method"] for r in top][::-1]
    values = [float(r["attack_success"]) for r in top][::-1]
    colors = ["#9b2226" if r["stage"] == "poisoning" else "#005f73" for r in top][::-1]
    ax.barh(range(len(labels)), values, color=colors)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlim(0, 1.05)
    ax.set_title("Top fungi attack success")
    ax.set_xlabel("Success / damage")

    ax = axes[0, 1]
    stage_colors = {"poisoning": "#9b2226", "evasion": "#005f73", "reference": "#6c757d"}
    for stage in sorted(set(r["stage"] for r in plot_rows)):
        group = [r for r in plot_rows if r["stage"] == stage]
        ax.scatter(
            [float(r["clean_accuracy"]) for r in group],
            [float(r["attack_success"]) for r in group],
            s=48,
            label=stage,
            alpha=0.85,
            color=stage_colors.get(stage, "#555555"),
        )
    ax.set_xlabel("Clean accuracy")
    ax.set_ylabel("Attack success / damage")
    ax.set_xlim(0.52, 0.74)
    ax.set_ylim(0, 1.05)
    ax.set_title("Accuracy vs attack effect")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    risks = ["R1", "R2", "R3", "R4"]
    risk_means = []
    risk_maxes = []
    for risk in risks:
        group = [float(r["attack_success"]) for r in plot_rows if r["risk_level"] == risk]
        risk_means.append(float(np.mean(group)) if group else 0.0)
        risk_maxes.append(float(np.max(group)) if group else 0.0)
    x = np.arange(len(risks))
    ax.bar(x - 0.18, risk_means, 0.36, label="mean")
    ax.bar(x + 0.18, risk_maxes, 0.36, label="max")
    ax.set_xticks(x)
    ax.set_xticklabels(risks)
    ax.set_ylim(0, 1.05)
    ax.set_title("Attack effect by risk level")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    evasion = [r for r in plot_rows if r["stage"] == "evasion" and r["avg_l0_features"]]
    evasion = sorted(evasion, key=lambda r: float(r["avg_l0_features"]))
    ax.barh(
        [r["method"] for r in evasion],
        [float(r["avg_l0_features"]) for r in evasion],
        color="#0a9396",
    )
    ax.set_xscale("log")
    ax.set_xlabel("Avg changed features, log scale")
    ax.set_title("Evasion perturbation footprint")
    ax.tick_params(axis="y", labelsize=8)

    fig.suptitle("Fungi reporting dashboard", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "fungi_reporting_dashboard.png", dpi=220)
    plt.close(fig)


def save_success_by_stage_chart(fungi_rows: list[dict[str, str]], mnist_rows: list[dict[str, str]]) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    datasets = [("Fungi", fungi_rows), ("MNIST", mnist_rows)]
    stages = ["poisoning", "evasion"]
    width = 0.34
    x = np.arange(len(stages))
    for offset, (dataset, rows) in zip([-width / 2, width / 2], datasets):
        values = []
        for stage in stages:
            group = [float(r["attack_success"]) for r in rows if r["stage"] == stage]
            values.append(float(np.mean(group)) if group else 0.0)
        ax.bar(x + offset, values, width, label=dataset)
    ax.set_xticks(x)
    ax.set_xticklabels(stages)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean attack success / damage")
    ax.set_title("Mean effect by attack stage")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "attack_success_by_stage.png", dpi=220)
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)
    fungi_rows = read_rows(RESULTS_DIR / "fungi_attack_comparison.csv")
    mnist_rows = read_rows(RESULTS_DIR / "mnist_attack_comparison.csv")

    dataset_stats = collect_fungi_dataset_statistics()
    write_csv(RESULTS_DIR / "fungi_dataset_statistics.csv", dataset_stats)
    write_json(RESULTS_DIR / "fungi_dataset_statistics.json", dataset_stats)
    write_markdown(RESULTS_DIR / "fungi_dataset_statistics.md", "Fungi Dataset Statistics", dataset_stats)

    attack_stats = attack_summary_statistics(fungi_rows, "fungi") + attack_summary_statistics(mnist_rows, "mnist")
    write_csv(RESULTS_DIR / "attack_summary_statistics.csv", attack_stats)
    write_json(RESULTS_DIR / "attack_summary_statistics.json", attack_stats)
    write_markdown(RESULTS_DIR / "attack_summary_statistics.md", "Attack Summary Statistics", attack_stats)

    save_fungi_sample_grid()
    save_dataset_split_chart(dataset_stats)
    save_attack_dashboard(fungi_rows)
    save_success_by_stage_chart(fungi_rows, mnist_rows)

    print(f"Wrote {RESULTS_DIR / 'fungi_dataset_statistics.md'}")
    print(f"Wrote {RESULTS_DIR / 'attack_summary_statistics.md'}")
    print(f"Wrote {FIGURES_DIR / 'fungi_dataset_samples.png'}")
    print(f"Wrote {FIGURES_DIR / 'fungi_reporting_dashboard.png'}")


if __name__ == "__main__":
    main()
