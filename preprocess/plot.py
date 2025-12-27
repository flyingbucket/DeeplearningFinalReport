#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot TensorBoard CSV curves.

Each experiment -> one figure
Each figure contains 3 subplots:
    - Loss
    - PSNR
    - SSIM
"""

from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# =========================
# Config
# =========================

CSV_DIR = Path("tb_csv")
OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)

METRICS = ["loss", "psnr", "ssim"]
SMOOTH_WINDOW = 5  # TensorBoard-style smoothing
FIG_SIZE = (6, 8)  # 3 子图纵向排列，适合 Beamer


# =========================
# Style (论文 / Beamer)
# =========================

sns.set_theme(
    style="whitegrid",
    font_scale=1.1,
    rc={
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": False,
    },
)


# =========================
# Data Loading
# =========================


def load_tb_csvs(csv_dir: Path) -> pd.DataFrame:
    """
    Load all CSVs and parse experiment / metric from filename.
    """
    records = []

    for csv_file in sorted(csv_dir.glob("*.csv")):
        stem = csv_file.stem
        parts = stem.split("_")

        metric = parts[-1]  # loss / psnr / ssim
        experiment = "_".join(parts[:-1])  # B16_token / L16_concat ...

        df = pd.read_csv(csv_file)

        if not {"Step", "Value"}.issubset(df.columns):
            raise ValueError(f"Unexpected CSV format: {csv_file}")

        df = df[["Step", "Value"]].copy()
        df["experiment"] = experiment
        df["metric"] = metric

        records.append(df)

    return pd.concat(records, ignore_index=True)


def smooth_values(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
    Apply rolling mean smoothing per (experiment, metric).
    """
    df = df.sort_values("Step").copy()
    df["Value_smooth"] = df.groupby(["experiment", "metric"])["Value"].transform(
        lambda x: x.rolling(window, min_periods=1).mean()
    )
    return df


# =========================
# Plotting
# =========================


def plot_one_experiment(df: pd.DataFrame, experiment: str):
    """
    Plot loss / psnr / ssim for one experiment (horizontal layout).
    """
    fig, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(12, 3.5),  # 横向比例，Beamer 友好
    )

    fig.suptitle(f"Training Curves: {experiment}", fontsize=14)

    for ax, metric in zip(axes, ["loss", "psnr", "ssim"]):
        sub = df[(df["experiment"] == experiment) & (df["metric"] == metric)].copy()

        if sub.empty:
            ax.set_visible(False)
            continue

        # 每个指标使用自己的 x 轴
        if metric == "loss":
            x = sub["Step"]
            ax.set_xlabel("Training Step")
            ax.set_ylabel("Loss")
        elif metric == "psnr":
            x = sub["Step"]
            ax.set_xlabel("Epochs")
            ax.set_ylabel("PSNR (dB)")
        else:
            x = sub["Step"]
            ax.set_xlabel("Epochs")
            ax.set_ylabel("SSIM")

        ax.plot(x, sub["Value_smooth"], linewidth=2)

        ax.set_title(metric.upper())
        ax.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.90])

    out_path = OUT_DIR / f"{experiment}_curves.pdf"
    plt.savefig(out_path)
    plt.close()

    print(f"[Saved] {out_path}")


# =========================
# Main
# =========================


def main():
    print("[Info] Loading CSV files...")
    df = load_tb_csvs(CSV_DIR)

    print("[Info] Applying smoothing...")
    df = smooth_values(df, SMOOTH_WINDOW)

    experiments = sorted(df["experiment"].unique())

    print(f"[Info] Found experiments: {experiments}")

    for exp in experiments:
        plot_one_experiment(df, exp)

    print("[Done] All experiment figures generated.")


if __name__ == "__main__":
    main()
