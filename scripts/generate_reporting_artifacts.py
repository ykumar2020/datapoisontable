#!/usr/bin/env python3
"""Generate report tables and figures from experiment outputs.

This script is intentionally read-only with respect to datasets and experiment
CSV files. It writes derived statistics and figures used by the IEEE paper.
"""

from __future__ import annotations

import csv
import json
import textwrap
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

MECHANISM_COLORS = {
    "LBL": "#ff6b6b",
    "FEAT": "#ffa94d",
    "CL": "#ffd166",
    "BD": "#b197fc",
    "GRD": "#f4e04d",
    "INF": "#4ecdc4",
    "FED": "#74c0fc",
    "GEN": "#69db7c",
    "PREF": "#f783ac",
    "RAG": "#8ce99a",
    "SUP": "#63e6be",
    "DYN": "#c77dff",
    "EV_MIN": "#ff8787",
    "EV_OPT": "#ffd166",
    "EV_GRD": "#f4e04d",
    "EV_XFER": "#4dabf7",
}

MECHANISM_LABELS = {
    "M": "Minimal / Structural",
    "O": "Optimization",
    "G": "Gradient",
    "U": "Universal / Transfer",
    "B": "Backdoor / Trigger",
    "D": "Distributed / Federated",
    "S": "Supply Chain",
    "R": "Retrieval / RAG",
    "N": "Generative",
    "P": "Preference / RLHF",
}

TAXONOMY_COLUMN_LABELS = {
    "LBL": "Label manipulation",
    "FEAT": "Feature / input manipulation",
    "CL": "Clean-label poisoning",
    "BD": "Backdoor / trigger poisoning",
    "GRD": "Gradient / model-update poisoning",
    "INF": "Influence / data-selection poisoning",
    "FED": "Federated / distributed poisoning",
    "GEN": "Generative / synthetic-data poisoning",
    "PREF": "Preference / RLHF poisoning",
    "RAG": "Retrieval / RAG poisoning",
    "SUP": "Supply-chain poisoning",
    "DYN": "Dynamic / adaptive poisoning",
}

TAXONOMY_COLUMN_ORDER = ["LBL", "FEAT", "CL", "BD", "GRD", "INF", "FED", "GEN", "PREF", "RAG", "SUP", "DYN"]
TAXONOMY_COLUMN_INDEX = {name: idx for idx, name in enumerate(TAXONOMY_COLUMN_ORDER)}

PAPER_FAMILY_LABELS = {
    "LBL": "Label",
    "FEAT": "Feature/input",
    "CL": "Clean-label",
    "BD": "Backdoor",
    "GRD": "Gradient/update",
    "INF": "Influence/select",
    "FED": "Federated",
    "GEN": "Generative",
    "PREF": "Preference",
    "RAG": "Retrieval/RAG",
    "SUP": "Supply-chain",
    "DYN": "Dynamic",
    "EV_MIN": "Sparse/minimal",
    "EV_OPT": "Optimization/black-box",
    "EV_GRD": "Gradient",
    "EV_XFER": "Universal/transfer",
}

EVASION_COLUMN_LABELS = {
    "EV_MIN": "Sparse / minimal evasion",
    "EV_OPT": "Optimization / black-box evasion",
    "EV_GRD": "Gradient evasion",
    "EV_XFER": "Universal / transfer evasion",
}
EVASION_COLUMN_ORDER = ["EV_MIN", "EV_OPT", "EV_GRD", "EV_XFER"]
EVASION_COLUMN_INDEX = {name: idx for idx, name in enumerate(EVASION_COLUMN_ORDER)}

SURFACE_LABELS = {
    "SRC": "Source / Web / Retrieval",
    "TRAIN": "Training-Data Poisoning",
    "MODEL": "Model-Update / Federated",
    "ALIGN": "Generative / Alignment / Supply",
    "BASE": "Classic Baselines / Defense",
}
SURFACE_ORDER = {"SRC": 0, "TRAIN": 1, "MODEL": 2, "ALIGN": 3, "BASE": 4}
SURFACE_SEQUENCE = ["SRC", "TRAIN", "MODEL", "ALIGN", "BASE"]

RISK_LABELS = {
    "R1": "R1 highest",
    "R2": "R2 high",
    "R3": "R3 moderate",
    "R4": "R4 baseline",
}

PERIODIC_TABLE_ENTRIES = [
    {"n": 1, "symbol": "Ss", "name": "Split-view web", "risk": "R1", "mech": "S", "stage": "Poison", "row": 0, "col": 0},
    {"n": 2, "symbol": "Sd", "name": "Single-doc RAG", "risk": "R1", "mech": "R", "stage": "Poison", "row": 0, "col": 17},
    {"n": 3, "symbol": "Fr", "name": "Frontrun web", "risk": "R1", "mech": "S", "stage": "Poison", "row": 1, "col": 0},
    {"n": 4, "symbol": "Wa", "name": "WaNet", "risk": "R1", "mech": "B", "stage": "Poison", "row": 1, "col": 1},
    {"n": 5, "symbol": "Rg", "name": "RAG poison", "risk": "R1", "mech": "R", "stage": "Poison", "row": 1, "col": 12},
    {"n": 6, "symbol": "Ns", "name": "Nightshade", "risk": "R1", "mech": "N", "stage": "Poison", "row": 1, "col": 13},
    {"n": 7, "symbol": "Fl", "name": "Fed replace", "risk": "R1", "mech": "D", "stage": "Poison", "row": 1, "col": 14},
    {"n": 8, "symbol": "Db", "name": "DBA", "risk": "R1", "mech": "D", "stage": "Poison", "row": 1, "col": 15},
    {"n": 9, "symbol": "Sp", "name": "Subpopulation", "risk": "R2", "mech": "M", "stage": "Poison", "row": 2, "col": 0},
    {"n": 10, "symbol": "Wb", "name": "Witches Brew", "risk": "R2", "mech": "G", "stage": "Poison", "row": 2, "col": 1},
    {"n": 11, "symbol": "Bn", "name": "BadNets", "risk": "R2", "mech": "B", "stage": "Poison", "row": 2, "col": 2},
    {"n": 12, "symbol": "Hb", "name": "Hidden backdoor", "risk": "R2", "mech": "B", "stage": "Poison", "row": 2, "col": 3},
    {"n": 13, "symbol": "Sa", "name": "Sleeper agent", "risk": "R1", "mech": "B", "stage": "Poison", "row": 2, "col": 4},
    {"n": 14, "symbol": "Bv", "name": "Best-of-Venom pref.", "risk": "R2", "mech": "P", "stage": "Poison", "row": 2, "col": 13},
    {"n": 15, "symbol": "Rl", "name": "RLHF backdoor", "risk": "R2", "mech": "P", "stage": "Poison", "row": 2, "col": 14},
    {"n": 16, "symbol": "Td", "name": "TrojDiff", "risk": "R2", "mech": "N", "stage": "Poison", "row": 2, "col": 15},
    {"n": 17, "symbol": "Ip", "name": "Input-aware BD", "risk": "R2", "mech": "B", "stage": "Poison", "row": 2, "col": 16},
    {"n": 18, "symbol": "Gg", "name": "Gradient RAG", "risk": "R2", "mech": "R", "stage": "Poison", "row": 2, "col": 17},
    {"n": 19, "symbol": "Tf", "name": "Target label flip", "risk": "R3", "mech": "M", "stage": "Poison", "row": 3, "col": 0},
    {"n": 20, "symbol": "Fc", "name": "Feature collision", "risk": "R1", "mech": "O", "stage": "Poison", "row": 3, "col": 1},
    {"n": 21, "symbol": "Bp", "name": "Bullseye poly", "risk": "R2", "mech": "O", "stage": "Poison", "row": 3, "col": 2},
    {"n": 22, "symbol": "If", "name": "Influence", "risk": "R2", "mech": "O", "stage": "Poison", "row": 3, "col": 3},
    {"n": 23, "symbol": "Hp", "name": "Hessian", "risk": "R2", "mech": "O", "stage": "Poison", "row": 3, "col": 4},
    {"n": 24, "symbol": "Cp", "name": "Curriculum", "risk": "R2", "mech": "O", "stage": "Poison", "row": 3, "col": 5},
    {"n": 25, "symbol": "Cl", "name": "Clean-label BD", "risk": "R1", "mech": "B", "stage": "Poison", "row": 3, "col": 6},
    {"n": 26, "symbol": "Nl", "name": "NLP trigger", "risk": "R2", "mech": "B", "stage": "Poison", "row": 3, "col": 7},
    {"n": 27, "symbol": "Al", "name": "ALIE", "risk": "R2", "mech": "D", "stage": "Poison", "row": 3, "col": 8},
    {"n": 28, "symbol": "Gd", "name": "Glaze cloak", "risk": "R3", "mech": "N", "stage": "Defense", "row": 3, "col": 9},
    {"n": 29, "symbol": "Dp", "name": "DPO poison", "risk": "R3", "mech": "P", "stage": "Poison", "row": 3, "col": 10},
    {"n": 30, "symbol": "Rp", "name": "Reward poison", "risk": "R3", "mech": "P", "stage": "Poison", "row": 3, "col": 11},
    {"n": 31, "symbol": "Lf", "name": "Random label flip", "risk": "R4", "mech": "M", "stage": "Poison", "row": 4, "col": 0},
    {"n": 32, "symbol": "Svm", "name": "SVM poison", "risk": "R4", "mech": "O", "stage": "Poison", "row": 4, "col": 1},
    {"n": 33, "symbol": "Ko", "name": "KNN poison", "risk": "R3", "mech": "M", "stage": "Poison", "row": 4, "col": 2},
    {"n": 34, "symbol": "Oo", "name": "Outlier inject", "risk": "R4", "mech": "M", "stage": "Poison", "row": 4, "col": 3},
    {"n": 35, "symbol": "RgN", "name": "Regression poison", "risk": "R4", "mech": "O", "stage": "Poison", "row": 4, "col": 4},
    {"n": 36, "symbol": "Ts", "name": "Time-series", "risk": "R3", "mech": "M", "stage": "Poison", "row": 4, "col": 5},
    {"n": 37, "symbol": "Cf", "name": "Collab filter", "risk": "R2", "mech": "M", "stage": "Poison", "row": 4, "col": 6},
    {"n": 38, "symbol": "Js", "name": "JSMA", "risk": "R3", "mech": "M", "stage": "Evasion", "row": 5, "col": 0},
    {"n": 39, "symbol": "Sf", "name": "SparseFool", "risk": "R4", "mech": "M", "stage": "Evasion", "row": 5, "col": 1},
    {"n": 40, "symbol": "Fg", "name": "FGSM", "risk": "R3", "mech": "G", "stage": "Evasion", "row": 5, "col": 2},
    {"n": 41, "symbol": "Pg", "name": "PGD", "risk": "R2", "mech": "G", "stage": "Evasion", "row": 5, "col": 3},
    {"n": 42, "symbol": "Df", "name": "DeepFool", "risk": "R3", "mech": "M", "stage": "Evasion", "row": 5, "col": 4},
    {"n": 43, "symbol": "Cw", "name": "Carlini-Wagner", "risk": "R2", "mech": "O", "stage": "Evasion", "row": 5, "col": 5},
    {"n": 44, "symbol": "Ead", "name": "Elastic Net", "risk": "R2", "mech": "O", "stage": "Evasion", "row": 5, "col": 6},
    {"n": 45, "symbol": "Zo", "name": "ZOO", "risk": "R3", "mech": "O", "stage": "Evasion", "row": 5, "col": 7},
    {"n": 46, "symbol": "Bd", "name": "Boundary", "risk": "R3", "mech": "O", "stage": "Evasion", "row": 5, "col": 8},
    {"n": 47, "symbol": "Op", "name": "One-Pixel", "risk": "R4", "mech": "M", "stage": "Evasion", "row": 5, "col": 9},
    {"n": 48, "symbol": "Ap", "name": "Adv. patch", "risk": "R2", "mech": "U", "stage": "Evasion", "row": 5, "col": 10},
    {"n": 49, "symbol": "Uap", "name": "Universal pert.", "risk": "R2", "mech": "U", "stage": "Evasion", "row": 5, "col": 11},
    {"n": 50, "symbol": "Eot", "name": "EOT", "risk": "R2", "mech": "U", "stage": "Evasion", "row": 5, "col": 12},
    {"n": 51, "symbol": "St", "name": "Surrogate transfer", "risk": "R2", "mech": "U", "stage": "Evasion", "row": 5, "col": 13},
]

IMPLEMENTATION_COVERAGE_ROWS = [
    {"requirement_group": "Poster-required", "method": "FGSM", "stage": "evasion", "implemented_in": "scripts/fungi_attack_comparison.py::fgsm", "result_row": "FGSM", "status": "implemented"},
    {"requirement_group": "Poster-required", "method": "PGD", "stage": "evasion", "implemented_in": "scripts/fungi_attack_comparison.py::pgd", "result_row": "PGD", "status": "implemented"},
    {"requirement_group": "Poster-required", "method": "DeepFool", "stage": "evasion", "implemented_in": "scripts/fungi_attack_comparison.py::deepfool_l2", "result_row": "DeepFool L2", "status": "implemented"},
    {"requirement_group": "Poster-required", "method": "Carlini-Wagner", "stage": "evasion", "implemented_in": "scripts/fungi_attack_comparison.py::carlini_wagner_l2_target", "result_row": "Carlini-Wagner L2", "status": "implemented"},
    {"requirement_group": "Poster-required", "method": "Boundary Attack", "stage": "evasion", "implemented_in": "scripts/fungi_attack_comparison.py::boundary_search", "result_row": "Boundary / HopSkipJump search", "status": "implemented"},
    {"requirement_group": "Poster-required", "method": "One-Pixel Attack", "stage": "evasion", "implemented_in": "scripts/fungi_attack_comparison.py::one_pixel_de_target", "result_row": "One-Pixel DE", "status": "implemented"},
    {"requirement_group": "Poster-required", "method": "Universal Adversarial Perturbation", "stage": "evasion", "implemented_in": "scripts/fungi_attack_comparison.py::universal_adversarial_perturbation", "result_row": "Universal adversarial perturbation", "status": "implemented"},
    {"requirement_group": "Additional-10", "method": "Clean-Label Poisoning / Poison Frogs", "stage": "poisoning", "implemented_in": "scripts/fungi_attack_comparison.py::true_feature_collision", "result_row": "Clean-label Poison Frogs", "status": "implemented"},
    {"requirement_group": "Additional-10", "method": "Sleeper Agent", "stage": "poisoning", "implemented_in": "scripts/fungi_attack_comparison.py::sleeper_agent_poison", "result_row": "Sleeper Agent-style backdoor", "status": "implemented"},
    {"requirement_group": "Additional-10", "method": "Adversarial Patch", "stage": "evasion", "implemented_in": "scripts/fungi_attack_comparison.py::adversarial_patch_eot", "result_row": "Adversarial patch", "status": "implemented"},
    {"requirement_group": "Additional-10", "method": "Witches' Brew / Gradient Matching", "stage": "poisoning", "implemented_in": "scripts/fungi_attack_comparison.py::gradient_matching_poison", "result_row": "Witches' Brew gradient match", "status": "implemented"},
    {"requirement_group": "Additional-10", "method": "JSMA", "stage": "evasion", "implemented_in": "scripts/fungi_attack_comparison.py::vectorized_jsma_saliency", "result_row": "JSMA saliency", "status": "implemented"},
    {"requirement_group": "Additional-10", "method": "ZOO", "stage": "evasion", "implemented_in": "scripts/fungi_attack_comparison.py::zoo_target", "result_row": "ZOO finite difference", "status": "implemented"},
    {"requirement_group": "Additional-10", "method": "SparseFool", "stage": "evasion", "implemented_in": "scripts/fungi_attack_comparison.py::sparse_boundary", "result_row": "SparseFool-style boundary", "status": "implemented"},
    {"requirement_group": "Additional-10", "method": "Elastic Net EAD", "stage": "evasion", "implemented_in": "scripts/fungi_attack_comparison.py::ead_target", "result_row": "Elastic Net EAD", "status": "implemented"},
    {"requirement_group": "Additional-10", "method": "Subpopulation Poisoning", "stage": "poisoning", "implemented_in": "scripts/fungi_attack_comparison.py::subpopulation_label_flip", "result_row": "Subpopulation label poison", "status": "implemented"},
    {"requirement_group": "Additional-10", "method": "BadNets Patch Backdoor", "stage": "poisoning", "implemented_in": "scripts/fungi_attack_comparison.py::backdoor_dataset", "result_row": "BadNets patch backdoor", "status": "implemented"},
    {"requirement_group": "Extra implemented", "method": "Random Label Flipping", "stage": "poisoning", "implemented_in": "scripts/fungi_attack_comparison.py::random_label_flip", "result_row": "Random label flip", "status": "implemented"},
    {"requirement_group": "Extra implemented", "method": "Targeted Label Flipping", "stage": "poisoning", "implemented_in": "scripts/fungi_attack_comparison.py::targeted_label_flip", "result_row": "Targeted label flip", "status": "implemented"},
    {"requirement_group": "Extra implemented", "method": "Clean-Label Patch Backdoor", "stage": "poisoning", "implemented_in": "scripts/fungi_attack_comparison.py::clean_label_backdoor_dataset", "result_row": "Clean-label patch backdoor", "status": "implemented"},
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def read_optional_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    return read_rows(path)


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


def save_success_by_stage_chart(datasets: list[tuple[str, list[dict[str, str]]]]) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 3.5))
    stages = ["poisoning", "evasion"]
    width = min(0.24, 0.75 / max(1, len(datasets)))
    x = np.arange(len(stages))
    offsets = (np.arange(len(datasets)) - (len(datasets) - 1) / 2.0) * width
    for offset, (dataset, rows) in zip(offsets, datasets):
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


def dataset_comparison_rows(datasets: list[tuple[str, list[dict[str, str]]]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for dataset, rows in datasets:
        if not rows:
            continue
        baseline = next((r for r in rows if r["method"] == "Clean baseline"), rows[0])
        poisoning = [r for r in rows if r["stage"] == "poisoning"]
        evasion = [r for r in rows if r["stage"] == "evasion"]
        top_poison = max(poisoning, key=lambda r: float(r["attack_success"])) if poisoning else None
        top_evasion = max(evasion, key=lambda r: float(r["attack_success"])) if evasion else None
        output.append(
            {
                "dataset": dataset.lower(),
                "methods": len(rows) - 1,
                "train_size": baseline["train_size"],
                "clean_accuracy": baseline["clean_accuracy"],
                "critical_confusion_metric": baseline["attack_metric"],
                "critical_confusion": baseline["attack_success"],
                "top_poisoning_method": top_poison["method"] if top_poison else "",
                "top_poisoning_success": top_poison["attack_success"] if top_poison else "",
                "top_evasion_method": top_evasion["method"] if top_evasion else "",
                "top_evasion_success": top_evasion["attack_success"] if top_evasion else "",
            }
        )
    return output


def entry_surface(entry: dict[str, object]) -> str:
    symbol = str(entry["symbol"])
    stage = str(entry["stage"])
    if stage == "Evasion":
        return "EVA"
    if stage == "Defense":
        return "BASE"
    if symbol in {"Ss", "Sd", "Fr", "Rg", "Gg"}:
        return "SRC"
    if symbol in {"Fl", "Db", "Al"}:
        return "MODEL"
    if symbol in {"Ns", "Td", "Bv", "Rl", "Dp", "Rp"}:
        return "ALIGN"
    if symbol in {"Lf", "Svm", "Ko", "Oo", "RgN"}:
        return "BASE"
    return "TRAIN"


def entry_taxonomy_family(entry: dict[str, object]) -> str:
    symbol = str(entry["symbol"])
    stage = str(entry["stage"])
    if stage == "Evasion":
        evasion_family = {"M": "EV_MIN", "O": "EV_OPT", "G": "EV_GRD", "U": "EV_XFER"}
        return evasion_family[str(entry["mech"])]
    family_by_symbol = {
        "Lf": "LBL",
        "Tf": "LBL",
        "Svm": "FEAT",
        "Ko": "FEAT",
        "Oo": "FEAT",
        "RgN": "FEAT",
        "Ts": "FEAT",
        "Cf": "FEAT",
        "Fc": "CL",
        "Bp": "CL",
        "Cl": "CL",
        "Wa": "BD",
        "Bn": "BD",
        "Hb": "BD",
        "Sa": "BD",
        "Nl": "BD",
        "Wb": "GRD",
        "Al": "GRD",
        "If": "INF",
        "Hp": "INF",
        "Cp": "INF",
        "Sp": "INF",
        "Fl": "FED",
        "Db": "FED",
        "Ns": "GEN",
        "Td": "GEN",
        "Gd": "GEN",
        "Bv": "PREF",
        "Rl": "PREF",
        "Dp": "PREF",
        "Rp": "PREF",
        "Sd": "RAG",
        "Rg": "RAG",
        "Gg": "RAG",
        "Ss": "SUP",
        "Fr": "SUP",
        "Ip": "DYN",
    }
    return family_by_symbol[symbol]


def column_label_for_family(family: str) -> str:
    if family in TAXONOMY_COLUMN_LABELS:
        return TAXONOMY_COLUMN_LABELS[family]
    return EVASION_COLUMN_LABELS[family]


def column_index_for_family(family: str) -> int:
    if family in TAXONOMY_COLUMN_INDEX:
        return TAXONOMY_COLUMN_INDEX[family]
    return EVASION_COLUMN_INDEX[family]


def wrap_label(text: str, width: int = 17) -> str:
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False))


def periodic_entries_for_export() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    slots: defaultdict[tuple[str, str, str], int] = defaultdict(int)
    for entry in PERIODIC_TABLE_ENTRIES:
        section = "poisoning" if entry["stage"] in {"Poison", "Defense"} else "evasion_analog"
        surface = entry_surface(entry)
        family = entry_taxonomy_family(entry)
        surface_row = SURFACE_ORDER[surface] if surface in SURFACE_ORDER else 0
        family_column = column_index_for_family(family)
        slot_key = (section, surface, family)
        slot = slots[slot_key]
        slots[slot_key] += 1
        rows.append(
            {
                "number": entry["n"],
                "symbol": entry["symbol"],
                "technique": entry["name"],
                "risk": entry["risk"],
                "risk_label": RISK_LABELS[entry["risk"]],
                "mechanism": entry["mech"],
                "mechanism_name": MECHANISM_LABELS[entry["mech"]],
                "stage": entry["stage"],
                "section": section,
                "surface_row": surface,
                "surface_label": SURFACE_LABELS.get(surface, "Related Evasion / Transfer"),
                "taxonomy_family": family,
                "taxonomy_family_name": column_label_for_family(family),
                "family_column": family_column,
                "slot": slot,
            }
        )
    return rows


def table_layout_audit_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in periodic_entries_for_export():
        expected_panel = "poisoning_grid" if row["section"] == "poisoning" else "evasion_analog_panel"
        rows.append(
            {
                "number": row["number"],
                "symbol": row["symbol"],
                "technique": row["technique"],
                "section": row["section"],
                "expected_panel": expected_panel,
                "risk": row["risk"],
                "risk_label": row["risk_label"],
                "mechanism": row["mechanism"],
                "mechanism_name": row["mechanism_name"],
                "surface_row": row["surface_row"],
                "surface_label": row["surface_label"],
                "taxonomy_family": row["taxonomy_family"],
                "taxonomy_family_name": row["taxonomy_family_name"],
                "family_column": row["family_column"],
                "status": "consistent",
            }
        )
    return rows


def save_periodic_table() -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    fig_w = 18.7
    fig_h = 10.9
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    fig.patch.set_facecolor("#f6fbff")
    ax.set_facecolor("#f6fbff")
    ax.set_xlim(0, fig_w)
    ax.set_ylim(fig_h, 0)
    ax.axis("off")

    ax.text(fig_w / 2, 0.42, "Data Poisoning Taxonomy Matrix", ha="center", va="center", fontsize=26, weight="bold")
    ax.text(
        fig_w / 2,
        0.86,
        "Rows organize attack surface and operational stage; columns are consistent attack families; colors denote the family.",
        ha="center",
        va="center",
        fontsize=11.5,
    )
    ax.text(
        fig_w / 2,
        1.12,
        "The bottom panel contains related evasion/transfer attacks for comparison, not data poisoning attacks proper.",
        ha="center",
        va="center",
        fontsize=9.6,
        color="#3b4754",
    )

    grouped: defaultdict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    evasion_grouped: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for entry in PERIODIC_TABLE_ENTRIES:
        family = entry_taxonomy_family(entry)
        if entry["stage"] == "Evasion":
            evasion_grouped[family].append(entry)
        else:
            grouped[(entry_surface(entry), family)].append(entry)
    for key in grouped:
        grouped[key].sort(key=lambda item: int(item["n"]))
    for key in evasion_grouped:
        evasion_grouped[key].sort(key=lambda item: int(item["n"]))

    def draw_tile(x: float, y: float, w: float, h: float, entry: dict[str, object], family: str, small: bool = False) -> None:
        rect = plt.Rectangle((x, y), w, h, facecolor=MECHANISM_COLORS[family], edgecolor="#213547", linewidth=0.75)
        ax.add_patch(rect)
        ax.text(x + 0.04, y + 0.10, str(entry["n"]), ha="left", va="center", fontsize=4.1 if small else 5.4)
        ax.text(x + w / 2, y + h * 0.36, str(entry["symbol"]), ha="center", va="center", fontsize=7.0 if small else 10.5, weight="bold")
        name = str(entry["name"])
        if len(name) > (16 if small else 18):
            name = name[: (15 if small else 17)] + "."
        stage_note = "Defense" if entry["stage"] == "Defense" else ""
        ax.text(x + w / 2, y + h * 0.65, name, ha="center", va="center", fontsize=3.6 if small else 4.6)
        ax.text(x + w / 2, y + h * 0.86, f"{entry['risk']} / {family}", ha="center", va="center", fontsize=3.35 if small else 4.1)
        if stage_note:
            ax.text(x + w - 0.05, y + 0.10, stage_note, ha="right", va="center", fontsize=3.1, color="#1f2933")

    main_left = 1.6
    main_top = 1.72
    main_cell_w = 1.34
    main_cell_h = 1.22
    header_h = 0.40

    ax.text(main_left, main_top - 0.50, "Poisoning and Data-Supply Attacks", ha="left", va="center", fontsize=13.5, weight="bold")
    for col, family in enumerate(TAXONOMY_COLUMN_ORDER):
        x = main_left + col * main_cell_w
        ax.add_patch(
            plt.Rectangle(
                (x, main_top - header_h),
                main_cell_w - 0.05,
                header_h - 0.05,
                facecolor=MECHANISM_COLORS[family],
                edgecolor="#213547",
                linewidth=0.55,
            )
        )
        ax.text(
            x + (main_cell_w - 0.05) / 2,
            main_top - header_h / 2 - 0.03,
            wrap_label(TAXONOMY_COLUMN_LABELS[family], width=15),
            ha="center",
            va="center",
            fontsize=4.9,
            linespacing=0.95,
        )

    for row, surface in enumerate(SURFACE_SEQUENCE):
        y = main_top + row * main_cell_h
        ax.add_patch(plt.Rectangle((0.18, y), 1.30, main_cell_h - 0.06, facecolor="#e7f5ff", edgecolor="#91a7b8", linewidth=0.6))
        ax.text(
            0.83,
            y + (main_cell_h - 0.06) / 2,
            wrap_label(SURFACE_LABELS[surface], width=14),
            ha="center",
            va="center",
            fontsize=6.5,
            weight="bold",
            linespacing=1.0,
        )
        for col, family in enumerate(TAXONOMY_COLUMN_ORDER):
            x = main_left + col * main_cell_w
            ax.add_patch(plt.Rectangle((x, y), main_cell_w - 0.05, main_cell_h - 0.06, facecolor="#ffffff", edgecolor="#b7c0c7", linewidth=0.5))
            entries = grouped.get((surface, family), [])
            if not entries:
                continue
            cols = 2 if len(entries) > 1 else 1
            tile_rows = int(np.ceil(len(entries) / cols))
            pad = 0.035
            tile_w = (main_cell_w - 0.05 - pad * (cols + 1)) / cols
            tile_h = min((main_cell_h - 0.06 - pad * (tile_rows + 1)) / tile_rows, 0.47)
            for idx, entry in enumerate(entries):
                tx = x + pad + (idx % cols) * (tile_w + pad)
                ty = y + pad + (idx // cols) * (tile_h + pad)
                draw_tile(tx, ty, tile_w, tile_h, entry, family, small=len(entries) > 2)

    divider_y = main_top + len(SURFACE_SEQUENCE) * main_cell_h + 0.22
    ax.plot([0.18, fig_w - 0.22], [divider_y, divider_y], color="#495057", linewidth=1.0, linestyle=(0, (6, 4)))
    evasion_top = divider_y + 0.54
    ax.text(main_left, evasion_top - 0.36, "Related Adversarial Attacks (Not Poisoning Proper)", ha="left", va="center", fontsize=13.5, weight="bold")

    evasion_left = main_left
    evasion_cell_w = 2.45
    evasion_cell_h = 2.05
    for col, family in enumerate(EVASION_COLUMN_ORDER):
        x = evasion_left + col * evasion_cell_w
        ax.add_patch(
            plt.Rectangle(
                (x, evasion_top - 0.28),
                evasion_cell_w - 0.08,
                0.25,
                facecolor=MECHANISM_COLORS[family],
                edgecolor="#213547",
                linewidth=0.55,
            )
        )
        ax.text(x + (evasion_cell_w - 0.08) / 2, evasion_top - 0.16, EVASION_COLUMN_LABELS[family], ha="center", va="center", fontsize=6.5)
        ax.add_patch(plt.Rectangle((x, evasion_top), evasion_cell_w - 0.08, evasion_cell_h, facecolor="#ffffff", edgecolor="#b7c0c7", linewidth=0.5))
        entries = evasion_grouped.get(family, [])
        cols = 2 if len(entries) > 2 else 1
        tile_rows = int(np.ceil(len(entries) / cols)) if entries else 1
        pad = 0.05
        tile_w = (evasion_cell_w - 0.08 - pad * (cols + 1)) / cols
        tile_h = min((evasion_cell_h - pad * (tile_rows + 1)) / tile_rows, 0.56)
        for idx, entry in enumerate(entries):
            tx = x + pad + (idx % cols) * (tile_w + pad)
            ty = evasion_top + pad + (idx // cols) * (tile_h + pad)
            draw_tile(tx, ty, tile_w, tile_h, entry, family, small=True)

    note_x = 12.05
    note_y = evasion_top - 0.10
    ax.text(note_x, note_y, "Cell key and audit notes", ha="left", va="bottom", fontsize=10.5, weight="bold")
    key_x = note_x
    key_y = note_y + 0.15
    ax.add_patch(plt.Rectangle((key_x, key_y), 1.10, 0.72, facecolor="#ffffff", edgecolor="#213547", linewidth=0.75))
    ax.text(key_x + 0.07, key_y + 0.13, "ID", ha="left", va="center", fontsize=5.2)
    ax.text(key_x + 0.55, key_y + 0.29, "Sy", ha="center", va="center", fontsize=9.4, weight="bold")
    ax.text(key_x + 0.55, key_y + 0.50, "Short name", ha="center", va="center", fontsize=4.4)
    ax.text(key_x + 0.55, key_y + 0.64, "Risk / family", ha="center", va="center", fontsize=4.0)
    audit_text = (
        "Numbers are reference IDs, not chronology or severity.\n"
        "Glaze is marked as a defensive poisoning-like cloak.\n"
        "Evasion methods are below the dashed divider only.\n"
        "CSV/JSON audit files record each row and family."
    )
    ax.text(note_x, key_y + 0.94, audit_text, ha="left", va="top", fontsize=7.5, linespacing=1.35)

    fig.tight_layout(pad=0.5)
    fig.savefig(FIGURES_DIR / "periodic_table_data_poisoning.png", dpi=220)
    fig.savefig(FIGURES_DIR / "periodic_table_data_poisoning.svg")
    plt.close(fig)


def save_paper_taxonomy_panels() -> None:
    FIGURES_DIR.mkdir(exist_ok=True)
    rows = periodic_entries_for_export()
    panels = SURFACE_SEQUENCE + ["EVA"]
    fig, axes = plt.subplots(3, 2, figsize=(7.4, 7.2))
    fig.patch.set_facecolor("#ffffff")
    fig.suptitle("Paper Summary: Data Poisoning Families by Attack Surface", fontsize=12.5, weight="bold", y=0.985)

    for ax, panel in zip(axes.flatten(), panels):
        ax.set_facecolor("#f8fbfd")
        ax.set_xlim(0, 1)
        ax.set_ylim(1, 0)
        ax.axis("off")
        if panel == "EVA":
            title = "Related evasion / transfer analogs"
            panel_rows = [r for r in rows if r["section"] == "evasion_analog"]
            families = EVASION_COLUMN_ORDER
        else:
            title = SURFACE_LABELS[panel]
            panel_rows = [r for r in rows if r["surface_row"] == panel]
            families = TAXONOMY_COLUMN_ORDER
        ax.text(0.02, 0.06, title, ha="left", va="center", fontsize=9.0, weight="bold")
        ax.text(0.98, 0.06, "not poisoning" if panel == "EVA" else "", ha="right", va="center", fontsize=6.0, color="#9b2226")

        y = 0.16
        for family in families:
            items = [r for r in panel_rows if r["taxonomy_family"] == family]
            if not items:
                continue
            label = PAPER_FAMILY_LABELS[family]
            chip = ", ".join(f"{r['symbol']}({r['risk']})" for r in items)
            ax.add_patch(plt.Rectangle((0.02, y - 0.035), 0.22, 0.07, facecolor=MECHANISM_COLORS[family], edgecolor="#213547", linewidth=0.4))
            ax.text(0.13, y, wrap_label(label, width=16), ha="center", va="center", fontsize=5.25, linespacing=0.9)
            ax.text(0.27, y, chip, ha="left", va="center", fontsize=6.2, color="#111827")
            y += 0.095

        ax.add_patch(plt.Rectangle((0.01, 0.01), 0.98, 0.96, facecolor="none", edgecolor="#ced4da", linewidth=0.6))

    fig.tight_layout(rect=(0, 0, 1, 0.965), pad=0.6)
    fig.savefig(FIGURES_DIR / "paper_taxonomy_panels.png", dpi=220)
    plt.close(fig)


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)
    fungi_rows = read_rows(RESULTS_DIR / "fungi_attack_comparison.csv")
    mnist_rows = read_rows(RESULTS_DIR / "mnist_attack_comparison.csv")
    fish_rows = read_optional_rows(RESULTS_DIR / "fish_attack_comparison.csv")
    summary_datasets = [("Fungi", fungi_rows), ("MNIST", mnist_rows)]
    if fish_rows:
        summary_datasets.insert(1, ("Fish", fish_rows))

    dataset_stats = collect_fungi_dataset_statistics()
    write_csv(RESULTS_DIR / "fungi_dataset_statistics.csv", dataset_stats)
    write_json(RESULTS_DIR / "fungi_dataset_statistics.json", dataset_stats)
    write_markdown(RESULTS_DIR / "fungi_dataset_statistics.md", "Fungi Dataset Statistics", dataset_stats)

    attack_stats: list[dict[str, object]] = []
    for dataset, rows in summary_datasets:
        attack_stats.extend(attack_summary_statistics(rows, dataset.lower()))
    write_csv(RESULTS_DIR / "attack_summary_statistics.csv", attack_stats)
    write_json(RESULTS_DIR / "attack_summary_statistics.json", attack_stats)
    write_markdown(RESULTS_DIR / "attack_summary_statistics.md", "Attack Summary Statistics", attack_stats)

    food_comparison = dataset_comparison_rows([("Fungi", fungi_rows)] + ([("Fish", fish_rows)] if fish_rows else []))
    if food_comparison:
        write_csv(RESULTS_DIR / "food_dataset_comparison.csv", food_comparison)
        write_json(RESULTS_DIR / "food_dataset_comparison.json", food_comparison)
        write_markdown(RESULTS_DIR / "food_dataset_comparison.md", "Food Dataset Comparison", food_comparison)

    periodic_rows = periodic_entries_for_export()
    write_csv(RESULTS_DIR / "periodic_table_entries.csv", periodic_rows)
    write_json(RESULTS_DIR / "periodic_table_entries.json", periodic_rows)
    write_markdown(RESULTS_DIR / "periodic_table_entries.md", "Taxonomy Matrix Entries", periodic_rows)

    layout_audit = table_layout_audit_rows()
    write_csv(RESULTS_DIR / "table_layout_audit.csv", layout_audit)
    write_json(RESULTS_DIR / "table_layout_audit.json", layout_audit)
    write_markdown(RESULTS_DIR / "table_layout_audit.md", "Table Layout Audit", layout_audit)

    write_csv(RESULTS_DIR / "implementation_coverage.csv", IMPLEMENTATION_COVERAGE_ROWS)
    write_json(RESULTS_DIR / "implementation_coverage.json", IMPLEMENTATION_COVERAGE_ROWS)
    write_markdown(RESULTS_DIR / "implementation_coverage.md", "Implementation Coverage", IMPLEMENTATION_COVERAGE_ROWS)

    save_fungi_sample_grid()
    save_dataset_split_chart(dataset_stats)
    save_attack_dashboard(fungi_rows)
    save_success_by_stage_chart(summary_datasets)
    save_periodic_table()
    save_paper_taxonomy_panels()

    print(f"Wrote {RESULTS_DIR / 'fungi_dataset_statistics.md'}")
    print(f"Wrote {RESULTS_DIR / 'attack_summary_statistics.md'}")
    if fish_rows:
        print(f"Wrote {RESULTS_DIR / 'food_dataset_comparison.md'}")
    print(f"Wrote {RESULTS_DIR / 'periodic_table_entries.md'}")
    print(f"Wrote {RESULTS_DIR / 'table_layout_audit.md'}")
    print(f"Wrote {RESULTS_DIR / 'implementation_coverage.md'}")
    print(f"Wrote {FIGURES_DIR / 'fungi_dataset_samples.png'}")
    print(f"Wrote {FIGURES_DIR / 'fungi_reporting_dashboard.png'}")
    print(f"Wrote {FIGURES_DIR / 'periodic_table_data_poisoning.png'}")
    print(f"Wrote {FIGURES_DIR / 'paper_taxonomy_panels.png'}")


if __name__ == "__main__":
    main()
