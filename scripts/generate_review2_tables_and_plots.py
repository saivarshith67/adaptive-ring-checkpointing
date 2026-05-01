#!/usr/bin/env python3
"""Generate Review II tables and plots from experiment summaries."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

try:
    import seaborn as sns
except ImportError:  # pragma: no cover - fallback if seaborn is unavailable
    sns = None


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "experiment_results" / "experiment_summaries.jsonl"
OUTPUT_DIR = ROOT / "experiment_results" / "review2_assets"


MODE_ORDER = [
    "epoch",
    "convergence",
    "hash-ring-epoch",
    "convergence-hash-ring",
    "pytorch-lightning",
    "hf-trainer",
    "deepspeed",
    "fsdp",
    "wandb-artifacts",
]

DISPLAY_NAMES = {
    "epoch": "Epoch",
    "convergence": "Convergence",
    "hash-ring-epoch": "Hash-Ring Epoch",
    "convergence-hash-ring": "Convergence + Hash Ring",
    "pytorch-lightning": "PyTorch Lightning",
    "hf-trainer": "Hugging Face Trainer",
    "deepspeed": "DeepSpeed",
    "fsdp": "FairScale/FSDP",
    "wandb-artifacts": "W&B Artifacts",
}


def load_rows() -> list[dict]:
    rows: list[dict] = []
    with SUMMARY_PATH.open("r", encoding="utf-8") as handle:
        for line in handle:
            rows.append(json.loads(line))
    return rows


def mode_key(row: dict) -> int:
    try:
        return MODE_ORDER.index(row["training_mode"])
    except ValueError:
        return len(MODE_ORDER)


def export_markdown_table(path: Path, headers: list[str], rows: list[list[str]]) -> None:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export_csv_table(path: Path, headers: list[str], rows: list[list[str]]) -> None:
    lines = [",".join(headers)]
    lines.extend(",".join(row) for row in rows)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_full_run_rows(rows: list[dict]) -> list[dict]:
    full_runs = [
        row
        for row in rows
        if row["total_training_time_sec"] > 400 and row.get("resume_succeeded") is False
    ]
    latest_by_mode: dict[str, dict] = {}
    for row in sorted(full_runs, key=lambda r: r["timestamp"]):
        latest_by_mode[row["training_mode"]] = row
    return [latest_by_mode[mode] for mode in MODE_ORDER if mode in latest_by_mode]


def generate_recovery_rows(rows: list[dict]) -> list[dict]:
    recovery_runs = [
        row
        for row in rows
        if row.get("recovered_successfully", 0) > 0 and row.get("resume_succeeded") is True
    ]
    latest_by_mode: dict[str, dict] = {}
    for row in sorted(recovery_runs, key=lambda r: r["timestamp"]):
        latest_by_mode[row["training_mode"]] = row
    return [latest_by_mode[mode] for mode in MODE_ORDER if mode in latest_by_mode]


def setup_style() -> None:
    if sns is not None:
        sns.set_theme(style="whitegrid", palette="deep")
    else:
        plt.style.use("ggplot")


def save_bar_plot(
    labels: list[str],
    values: list[float],
    ylabel: str,
    title: str,
    output_path: Path,
    annotate_fmt: str,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 5.5))
    bars = ax.bar(labels, values, color=["#4C78A8", "#F58518", "#54A24B", "#E45756"])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=12)
    ymax = max(values) if values else 0
    ax.set_ylim(0, ymax * 1.2 if ymax > 0 else 1)

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + (ymax * 0.03 if ymax > 0 else 0.03),
            annotate_fmt.format(value),
            ha="center",
            va="bottom",
            fontsize=10,
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()

    rows = load_rows()
    full_rows = generate_full_run_rows(rows)
    recovery_rows = generate_recovery_rows(rows)

    full_headers = [
        "Mode",
        "Training Time (s)",
        "Checkpoint Count",
        "Checkpoint Overhead (%)",
        "Best Val Acc",
        "Final Val Acc",
    ]
    full_table = [
        [
            DISPLAY_NAMES[row["training_mode"]],
            f'{row["total_training_time_sec"]:.2f}',
            str(row["total_checkpoint_count"]),
            f'{row["checkpoint_overhead_percent"]:.2f}',
            f'{row["best_val_accuracy"]:.2f}',
            f'{row["final_val_accuracy"]:.2f}',
        ]
        for row in full_rows
    ]

    recovery_headers = [
        "Mode",
        "Recovery Success",
        "Resume Source",
        "Resumed From Epoch",
        "Resume Time (s)",
    ]
    recovery_table = [
        [
            DISPLAY_NAMES[row["training_mode"]],
            str(row["recovered_successfully"]),
            str(row.get("resume_load_source") or row.get("last_load_source") or "n/a"),
            str(row.get("resumed_from_epoch") or "n/a"),
            f'{row["time_to_resume_sec"]:.2f}',
        ]
        for row in recovery_rows
    ]

    export_markdown_table(OUTPUT_DIR / "full_training_table.md", full_headers, full_table)
    export_markdown_table(OUTPUT_DIR / "recovery_table.md", recovery_headers, recovery_table)
    export_csv_table(OUTPUT_DIR / "full_training_table.csv", full_headers, full_table)
    export_csv_table(OUTPUT_DIR / "recovery_table.csv", recovery_headers, recovery_table)

    full_labels = [DISPLAY_NAMES[row["training_mode"]] for row in full_rows]
    save_bar_plot(
        full_labels,
        [row["checkpoint_overhead_percent"] for row in full_rows],
        ylabel="Checkpoint Overhead (%)",
        title="Checkpoint Overhead Across Full Training Runs",
        output_path=OUTPUT_DIR / "checkpoint_overhead_full_runs.png",
        annotate_fmt="{:.2f}%",
    )
    save_bar_plot(
        full_labels,
        [row["total_checkpoint_count"] for row in full_rows],
        ylabel="Checkpoint Count",
        title="Checkpoint Count Across Full Training Runs",
        output_path=OUTPUT_DIR / "checkpoint_count_full_runs.png",
        annotate_fmt="{:.0f}",
    )
    save_bar_plot(
        full_labels,
        [row["total_training_time_sec"] for row in full_rows],
        ylabel="Training Time (s)",
        title="Training Time Across Full Training Runs",
        output_path=OUTPUT_DIR / "training_time_full_runs.png",
        annotate_fmt="{:.2f}s",
    )

    recovery_labels = [DISPLAY_NAMES[row["training_mode"]] for row in recovery_rows]
    save_bar_plot(
        recovery_labels,
        [row["time_to_resume_sec"] for row in recovery_rows],
        ylabel="Time to Resume (s)",
        title="Recovery Resume Time Across Modes",
        output_path=OUTPUT_DIR / "resume_time_recovery_runs.png",
        annotate_fmt="{:.2f}s",
    )

    save_bar_plot(
        recovery_labels,
        [float(row.get("resumed_from_epoch") or 0) for row in recovery_rows],
        ylabel="Resumed From Epoch",
        title="Checkpoint Freshness at Recovery",
        output_path=OUTPUT_DIR / "resumed_epoch_recovery_runs.png",
        annotate_fmt="{:.0f}",
    )

    summary = {
        "full_training_modes": [row["training_mode"] for row in full_rows],
        "recovery_modes": [row["training_mode"] for row in recovery_rows],
        "output_dir": str(OUTPUT_DIR),
    }
    (OUTPUT_DIR / "review2_plot_manifest.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(f"Generated tables and plots in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
