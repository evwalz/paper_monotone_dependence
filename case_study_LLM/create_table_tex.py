#!/usr/bin/env python3
"""
Print booktabs + multirow LaTeX for Table 1 (CMA and/or CID) from the same
JSONs as create_table.py. Copy stdout into your .tex file.

Preamble: \\usepackage{booktabs, multirow, xcolor} and
\\definecolor{mycolor}{...}
"""

import argparse
import os
import sys

import numpy as np

_CASE_STUDY_LLM_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_LLM_OUTPUT = os.path.join(_CASE_STUDY_LLM_DIR, "outputs")

from create_table import create_table  # noqa: E402

MODEL_SLUGS = ["Llama-2-7b-hf", "Llama-2-7b-chat-hf", "gpt-3.5-turbo"]
MODEL_TEX = {
    "Llama-2-7b-hf": "Llama-2-7b",
    "Llama-2-7b-chat-hf": "Llama-2-7b-chat",
    "gpt-3.5-turbo": "GPT-3.5",
}
DS_TEX = {"nq-open": "NQ-open", "squad": "SQuAD", "triviaqa": "TriviaQA"}
M_ORDER = ["nq-open", "squad", "triviaqa"]
CORR_ORDER = ["bert_similarity", "meteor", "rouge", "rouge1"]
CORR_TEX = {
    "bert_similarity": "BERT",
    "meteor": "METEOR",
    "rouge": "ROUGE-L",
    "rouge1": "ROUGE-1",
}


def _format_u_cell_fixed(val: float, sub: str, color: str) -> str:
    if np.isnan(val):
        return "--"
    num = f"{val:.3f}"
    if num.startswith("0."):
        num = num[1:]
    if not sub:
        return f"${num}$"
    # Second argument to \color must be the full math, e.g. {$.604_+$} not {$.604_+}$  (stray $ after }).
    if sub == "*":
        math = f"${num}_" + r"+$"
        return f"\\color{{{color}}}{{{math}}}"
    if sub in ("\u00b0", "°"):
        math = f"${num}_" + f"0$"
        return f"\\color{{{color}}}{{{math}}}"
    return f"${num}$"


def build_tabular(table_vals, subscripts, color: str) -> str:
    lines = []
    lines.append(r"\begin{tabular}{llllllll}")
    lines.append(r"\toprule")
    lines.append(
        "Model & Queries & Correctness & "
        r"$U_\textrm{Ecc}$ & $U_\textrm{Deg}$ & $U_\textrm{EigV}$ & $U_\textrm{NLL}$ & $U_\textrm{SE}$ \\"
    )
    lines.append(r"\midrule")

    for mi, model in enumerate(MODEL_SLUGS):
        mname = MODEL_TEX[model]
        for di, data in enumerate(M_ORDER):
            dname = DS_TEX[data]
            for ci, metric in enumerate(CORR_ORDER):
                r = mi * 12 + di * 4 + ci
                cells = " & ".join(
                    _format_u_cell_fixed(table_vals[r, j], subscripts[r][j], color)
                    for j in range(5)
                )
                cname = CORR_TEX[metric]
                if ci == 0:
                    if di == 0:
                        lead = (
                            f"\\multirow{{12}}{{*}}{{{mname}}} & \\multirow{{4}}{{*}}{{{dname}}}"
                        )
                    else:
                        lead = f"& \\multirow{{4}}{{*}}{{{dname}}}"
                    line = f"{lead} & {cname} & {cells} \\\\"
                else:
                    line = f"& & {cname} & {cells} \\\\"
                lines.append(line)
            if di < 2:
                lines.append(r"\cmidrule{3-8}")
        if mi < 2:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Print LaTeX tabular (booktabs) for CMA and/or CID tables.",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=_DEFAULT_LLM_OUTPUT,
        help="JSON directory (same as create_table.py)",
    )
    parser.add_argument(
        "--metric",
        type=str,
        choices=("cma", "cid", "both"),
        default="both",
        help="Which table(s) to print",
    )
    parser.add_argument(
        "--color",
        type=str,
        default="mycolor",
        help="LaTeX color name for best-column highlights (e.g. mycolor)",
    )
    args = parser.parse_args()

    to_run = ["cma", "cid"] if args.metric == "both" else [args.metric]
    for i, m in enumerate(to_run):
        if i:
            print()
        print(f"% --- {m.upper()} (from {args.input_dir}) ---", file=sys.stderr)
        tvals, subs, _ = create_table(args.input_dir, m)
        print(build_tabular(tvals, subs, args.color))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
