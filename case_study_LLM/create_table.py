import os
import argparse
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import norm

_CASE_STUDY_LLM_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_LLM_OUTPUT = os.path.join(_CASE_STUDY_LLM_DIR, "outputs")
# PDFs default here so they can be committed; JSON inputs stay under outputs/
_DEFAULT_LLM_FIGURES = os.path.join(_CASE_STUDY_LLM_DIR, "figures")


def _col_names_for_metric(table_metric):
    if table_metric == "cma":
        suffix = "cma"
    elif table_metric == "cid":
        suffix = "cid"
    else:
        raise ValueError(f"Unknown table_metric: {table_metric}")
    return [
        f"ecc_u_agreement_{suffix}",
        f"degree_u_agreement_{suffix}",
        f"spectral_u_agreement_{suffix}",
        f"unnormalized_nll_all_{suffix}",
        f"entropy_unnormalized_{suffix}",
    ]


def _resolve_stat_paths(input_dir, model, data, temp, metric, table_metric):
    """Return (stat_path, pair_path); CMA falls back to legacy *without* ``_cma_`` infix."""
    stem = f"{model}_{data}_{temp}_{metric}"
    if table_metric == "cma":
        primary_stat = os.path.join(input_dir, f"{stem}_cma_stat_test.json")
        primary_pair = os.path.join(input_dir, f"{stem}_cma_pairwise_test.json")
        legacy_stat = os.path.join(input_dir, f"{stem}_stat_test.json")
        legacy_pair = os.path.join(input_dir, f"{stem}_pairwise_test.json")
        if os.path.exists(primary_stat):
            return primary_stat, primary_pair
        return legacy_stat, legacy_pair
    return (
        os.path.join(input_dir, f"{stem}_cid_stat_test.json"),
        os.path.join(input_dir, f"{stem}_cid_pairwise_test.json"),
    )


# ----------------------------------------------------------------------------
# Build the numeric table + significance markers from JSONs (acor, Python)
# ----------------------------------------------------------------------------
def create_table(input_dir, table_metric="cma"):
    """
    Build numeric table + significance markers from stat_test / pairwise JSON.

    Parameters
    ----------
    input_dir : str
        Directory of JSONs produced by ``compute_table.py`` (default: ``outputs/``).
    table_metric : str
        'cma' or 'cid'.

    Returns
    -------
    table_vals : ndarray (36, 5)  numeric values rounded to 3 decimals (for display)
    subscripts : list of list of str (36 x 5)  '' | '*' | '\\u00b0'
    raw_vals   : ndarray (36, 5)  numeric values NOT rounded (for diff computation)
    """
    if table_metric == "cma":
        best_key, second_key = "best_cma", "second_best_cma"
    else:
        best_key, second_key = "best_cid", "second_best_cid"

    col_table = _col_names_for_metric(table_metric)

    method_to_col = {
        "ecc_u_agreement": 0,
        "degree_u_agreement": 1,
        "spectral_u_agreement": 2,
        "unnormalized_nll_all": 3,
        "entropy_unnormalized": 4,
    }

    table_vals = np.full((12 * 3, len(col_table)), np.nan)
    subscripts = [["" for _ in range(len(col_table))] for _ in range(12 * 3)]

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
                file_path, pairwise_file = _resolve_stat_paths(
                    input_dir, model, data, temp, metric, table_metric
                )

                if not os.path.exists(file_path):
                    print(f"Warning: File not found: {file_path}")
                    r = r + 1
                    continue

                scores = json.load(open(file_path))
                k = 0
                for name in col_table_sel:
                    if name in scores:
                        table_vals[r, k] = scores[name]
                    else:
                        table_vals[r, k] = np.nan
                    k = k + 1

                if os.path.exists(pairwise_file):
                    pairwise_scores = json.load(open(pairwise_file))

                    estimated_var = (
                        pairwise_scores["var_best"]
                        + pairwise_scores["var_second_best"]
                        - 2 * pairwise_scores["cov"]
                    )
                    diff = pairwise_scores[best_key] - pairwise_scores[second_key]
                    se_diff = (
                        np.sqrt(estimated_var) if estimated_var > 0 else np.inf
                    )
                    z_stat = diff / se_diff if se_diff > 0 else np.inf
                    p = 2 * (1 - norm.cdf(abs(z_stat)))

                    best_method = pairwise_scores["best_method"]
                    if best_method in method_to_col:
                        col_idx = method_to_col[best_method]
                        if p < 0.01:
                            subscripts[r][col_idx] = "*"
                        else:
                            subscripts[r][col_idx] = "\u00b0"
                else:
                    print(f"Warning: Pairwise file not found: {pairwise_file}")

                r = r + 1

    return np.round(table_vals, 3), subscripts, table_vals


# ----------------------------------------------------------------------------
# Render a formatted table to PDF
# ----------------------------------------------------------------------------
def create_formatted_table(table_vals, subscripts, output_file, title=None):
    cell_text = []
    for i in range(len(table_vals)):
        row = []
        for j in range(len(table_vals[i])):
            val = table_vals[i, j]
            if np.isnan(val):
                row.append("")
            else:
                mark = subscripts[i][j]
                if mark == "*":
                    row.append(f"{val:.3f}*")
                elif mark == "\u00b0":
                    row.append(f"{val:.3f}\u00b0")
                elif mark:
                    row.append(f"{val:.3f}$_{{{mark}}}$")
                else:
                    row.append(f"{val:.3f}")
        cell_text.append(row)

    df = pd.DataFrame(
        cell_text,
        columns=["$U_{Ecc}$", "$U_{Deg}$", "$U_{EigV}$", "$U_{NLL}$", "$U_{SE}$"],
    )
    _add_row_labels(df)
    _render_pdf(df, output_file, title=title)


# ----------------------------------------------------------------------------
# Shared helpers
# ----------------------------------------------------------------------------
def _add_row_labels(df):
    """Insert model / dataset / correctness index columns in place."""
    models = ["Llama-2", "Llama-2-chat", "GPT-3.5"]
    datasets = ["nq-open", "squad", "triviaqa"]
    correctness = ["bert", "meteor", "rougeL", "rouge1"]

    model_col = []
    for _m in models:
        model_col.extend([_m] * 12)

    dataset_col = []
    for _ in models:
        for dataset in datasets:
            dataset_col.extend([dataset] * 4)

    correctness_col = correctness * 9

    df.insert(0, "correctness", correctness_col)
    df.insert(0, "dataset", dataset_col)
    df.insert(0, "model", model_col)


def _render_pdf(df, output_file, title=None):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis("tight")
    ax.axis("off")

    if title:
        ax.set_title(title, fontsize=14, pad=12)

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        cellLoc="center",
        loc="center",
        colColours=["lightgray"] * len(df.columns),
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    with PdfPages(output_file) as pdf:
        pdf.savefig(fig, bbox_inches="tight")

    plt.close()
    print(f"Table saved to {output_file}")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build PDF table(s) from JSONs written by compute_table.py (acor, Python).",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=_DEFAULT_LLM_OUTPUT,
        help="Directory with JSONs from compute_table.py (default: case_study_LLM/outputs/)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=_DEFAULT_LLM_FIGURES,
        help="Where to write table_cma.pdf / table_cid.pdf (default: case_study_LLM/figures/)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for method in ["cma", "cid"]:
        print(f"\n=== Method: {method.upper()} ===")
        table_vals, subscripts, _raw = create_table(args.input_dir, method)
        out_pdf = os.path.join(args.output_dir, f"table_{method}.pdf")
        create_formatted_table(
            table_vals,
            subscripts,
            out_pdf,
            title=f"{method.upper()} (acor, Python)",
        )
