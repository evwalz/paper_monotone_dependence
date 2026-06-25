"""
Scatter summary of bootstrap calibration bundles (RCE, CMA, CID) — Figure 5.

Reads ``*_calibration_bootstrap.json`` from ``compute_calibration_bootstrap.py``
(CMA/CID from Python **acor**; ERCE from Rank-Calibration), with fallback to
legacy ``*_cma.json``. Three scatter panels in a 2×2 grid (bottom-left centered).
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

_CASE_STUDY_LLM_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_LLM_OUTPUT = os.path.join(_CASE_STUDY_LLM_DIR, "outputs")
_DEFAULT_LLM_PLOTS = os.path.join(_CASE_STUDY_LLM_DIR, "plots")
_DEFAULT_CALIBRATION_PLOT = os.path.join(
    _DEFAULT_LLM_PLOTS, "calibration_bootstrap_scatter.pdf"
)


def _bootstrap_bundle_path(input_dir: str, model: str, data: str, temp: str, metric: str) -> str:
    stem = os.path.join(input_dir, f"{model}_{data}_{temp}_{metric}")
    p_new = f"{stem}_calibration_bootstrap.json"
    p_old = f"{stem}_cma.json"
    if os.path.exists(p_new):
        return p_new
    if os.path.exists(p_old):
        return p_old
    return p_new


def create_table(metric_type: str = "cma", input_dir: str = "./Rank-Calibration/") -> np.ndarray:
    """Aggregate bootstrap JSONs into a table for CMA, ERCE, or CID (cindx)."""
    if metric_type == "cma":
        col_table = [
            "ecc_u_agreement_cma",
            "degree_u_agreement_cma",
            "spectral_u_agreement_cma",
            "unnormalized_nll_all_cma",
            "entropy_unnormalized_cma",
            "verbalized_cma",
        ]
    elif metric_type == "erce":
        col_table = [
            "ecc_u_agreement_erce",
            "degree_u_agreement_erce",
            "spectral_u_agreement_erce",
            "unnormalized_nll_all_erce",
            "entropy_unnormalized_erce",
            "verbalized_erce",
        ]
    else:  # cindx (CID)
        col_table = [
            "ecc_u_agreement_cindx",
            "degree_u_agreement_cindx",
            "spectral_u_agreement_cindx",
            "unnormalized_nll_all_cindx",
            "entropy_unnormalized_cindx",
            "verbalized_cindx",
        ]

    table_vals = np.zeros((12 * 3, len(col_table)))
    r = 0
    for model in ["Llama-2-7b-hf", "Llama-2-7b-chat-hf", "gpt-3.5-turbo"]:
        if model == "gpt-3.5-turbo":
            temp = "1.0"
            col_table_sel = col_table.copy()
        else:
            temp = "0.6"
            col_table_sel = col_table[0:5]
        for data in ["nq-open", "squad", "triviaqa"]:
            for metric in ["bert_similarity", "meteor", "rouge", "rouge1"]:
                file_path = _bootstrap_bundle_path(input_dir, model, data, temp, metric)

                if not os.path.exists(file_path):
                    print(f"Warning: File not found: {file_path}")
                    continue

                scores = json.load(open(file_path))
                k = 0
                for name in col_table_sel:
                    vals = []
                    for i in range(20):
                        vals.append(scores[i][name])
                    table_vals[r, k] = np.mean(vals)
                    k = k + 1
                r = r + 1
    table_vals[0:24, -1] = np.nan
    return np.round(table_vals, 3)


def _center_second_row_panel(
    ax_bottom: plt.Axes, ax_top_left: plt.Axes, ax_top_right: plt.Axes
) -> None:
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
        description="Bootstrap calibration scatter (RCE, CMA, CID) — Figure 5"
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
        "case_study_LLM/plots/calibration_bootstrap_scatter.pdf)",
    )
    args = parser.parse_args()

    if args.input_dir is None:
        if args.rank_calibration_path is not None:
            args.input_dir = os.path.join(args.rank_calibration_path, "stats_test")
        else:
            args.input_dir = _DEFAULT_LLM_OUTPUT

    os.makedirs(os.path.dirname(os.path.abspath(args.output_plot)), exist_ok=True)
    create_scatter_plot(args.input_dir, args.output_plot)
