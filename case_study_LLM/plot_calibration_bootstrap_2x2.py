"""
Scatter summary of bootstrap calibration bundles (RCE, CMA, CID) — **2×2** layout.

Same data and panels as :mod:`plot_calibration_bootstrap`, but in a 2×2 grid with the
bottom panel shifted to the horizontal center (same cell size as the top panels).
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from plot_calibration_bootstrap import create_table

_CASE_STUDY_LLM_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_LLM_OUTPUT = os.path.join(_CASE_STUDY_LLM_DIR, "outputs")
_DEFAULT_LLM_PLOTS = os.path.join(_CASE_STUDY_LLM_DIR, "plots")
_DEFAULT_CALIBRATION_PLOT = os.path.join(
    _DEFAULT_LLM_PLOTS, "calibration_bootstrap_scatter_2x2.pdf"
)


def _center_second_row_panel(ax_bottom: plt.Axes, ax_top_left: plt.Axes, ax_top_right: plt.Axes) -> None:
    """After ``tight_layout``, slide the bottom-left 2×2 cell to the row center (unchanged size)."""
    pos = ax_bottom.get_position()
    ref = ax_top_left.get_position()
    row_left = ax_top_left.get_position().x0
    row_right = ax_top_right.get_position().x1
    row_center = 0.5 * (row_left + row_right)
    ax_bottom.set_position([row_center - ref.width / 2, pos.y0, ref.width, pos.height])


def create_scatter_plot(input_dir: str, output_file: str) -> None:
    table_vals_cma = create_table("cma", input_dir)
    table_vals_erce = create_table("erce", input_dir)
    table_vals_cid = create_table("cindx", input_dir)

    table_vals_cma = table_vals_cma[:, 0:-1]
    table_vals_erce = table_vals_erce[:, 0:-1]
    table_vals_cid = table_vals_cid[:, 0:-1]

    model_flag = np.repeat(
        np.array(["Llama-2", "Llama-2-chat", "GPT-3.5"]), 12 * 5
    )
    data_flag = np.tile(
        np.repeat(np.array(["nq-open", "squad", "triviaqa"]), 4 * 5), 3
    )

    n = len(model_flag)
    df = pd.DataFrame(
        {
            "rce": table_vals_erce.flatten()[:n],
            "cma": table_vals_cma.flatten()[:n],
            "cid": table_vals_cid.flatten()[:n],
            "Model": model_flag,
            "Dataset": data_flag,
        }
    )

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    sns.scatterplot(data=df, x="rce", y="cma", hue="Model", ax=axes[0, 0])
    axes[0, 0].set_xlabel("RCE")
    axes[0, 0].set_ylabel("CMA")

    sns.scatterplot(data=df, x="rce", y="cid", hue="Model", ax=axes[0, 1], legend=False)
    axes[0, 1].set_xlabel("RCE")
    axes[0, 1].set_ylabel("CID")

    sns.scatterplot(data=df, x="cid", y="cma", hue="Model", ax=axes[1, 0], legend=False)
    axes[1, 0].set_xlabel("CID")
    axes[1, 0].set_ylabel("CMA")

    axes[1, 1].set_visible(False)

    plt.tight_layout()
    _center_second_row_panel(axes[1, 0], axes[0, 0], axes[0, 1])
    plt.savefig(output_file, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bootstrap calibration scatter (RCE, CMA, CID) in a 2×2 layout"
    )
    parser.add_argument(
        "--rank_calibration_path",
        type=str,
        default=None,
        help="Optional legacy base path; if set and --input_dir omitted, reads "
        "RANK_CALIBRATION_PATH/stats_test",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="cma",
        choices=["cma", "erce"],
        help="Legacy argument (scatter uses CMA, RCE, and CID).",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Directory with *_calibration_bootstrap.json (or legacy *_cma.json); "
        "default: case_study_LLM/outputs/, or RANK_CALIBRATION_PATH/stats_test if only "
        "--rank_calibration_path is set (legacy layout).",
    )
    parser.add_argument(
        "--output_plot",
        type=str,
        default=_DEFAULT_CALIBRATION_PLOT,
        help="Path to save the figure (default: "
        "case_study_LLM/plots/calibration_bootstrap_scatter_2x2.pdf)",
    )
    args = parser.parse_args()

    if args.input_dir is None:
        if args.rank_calibration_path is not None:
            args.input_dir = os.path.join(args.rank_calibration_path, "stats_test")
        else:
            args.input_dir = _DEFAULT_LLM_OUTPUT

    os.makedirs(os.path.dirname(os.path.abspath(args.output_plot)), exist_ok=True)
    create_scatter_plot(args.input_dir, args.output_plot)
