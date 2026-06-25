import argparse
import json
import os
import sys
from tqdm import tqdm
import numpy as np
import pandas as pd

_CASE_STUDY_LLM_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_LLM_OUTPUT = os.path.join(_CASE_STUDY_LLM_DIR, "outputs")

# acor (Python) acor_test for CMA and CID. Tags match prior JSON layout.
METHODS = ["cma", "cid"]

# Explicit acor_test options (method= is passed per call; variance default plugin for tables).
ACOR_IID = True
ACOR_VARIANCE_DEFAULT = "plugin"
ACOR_ALTERNATIVE = "two.sided"
ACOR_CONF_LEVEL = 0.95
ACOR_FISHER = False


def _run_py_single(y: np.ndarray, x: np.ndarray, method: str, *, variance: str):
    """One predictor: return {"estimate": scalar, "variance": scalar}."""
    from acor import acor_test

    X = np.asarray(x, dtype=np.float64).reshape(-1, 1)
    res = acor_test(
        X,
        np.asarray(y, dtype=np.float64),
        method=method,
        alternative=ACOR_ALTERNATIVE,
        conf_level=ACOR_CONF_LEVEL,
        iid=ACOR_IID,
        fisher=ACOR_FISHER,
        variance=variance,
    )
    est_arr = np.asarray(res.estimate, dtype=np.float64).ravel()
    if est_arr.size != 1:
        raise RuntimeError(f"single predictor: expected 1 estimate, got {est_arr.shape}")
    est = float(est_arr[0])
    V = np.asarray(res.variance, dtype=np.float64)
    var = float(V.ravel()[0])
    return {"estimate": est, "variance": var}


def _run_py_pairwise(y: np.ndarray, x1: np.ndarray, x2: np.ndarray, method: str, *, variance: str):
    """Two predictors: return {"estimate": [e1,e2], "variance": 2x2 list-of-lists}."""
    from acor import acor_test

    X = np.column_stack(
        [
            np.asarray(x1, dtype=np.float64),
            np.asarray(x2, dtype=np.float64),
        ]
    )
    res = acor_test(
        X,
        np.asarray(y, dtype=np.float64),
        method=method,
        alternative=ACOR_ALTERNATIVE,
        conf_level=ACOR_CONF_LEVEL,
        iid=ACOR_IID,
        fisher=ACOR_FISHER,
        variance=variance,
    )
    est = np.asarray(res.estimate, dtype=np.float64).ravel()
    if est.size != 2:
        raise RuntimeError(f"pairwise: expected 2 estimates, got {est.shape}")
    V = np.asarray(res.variance, dtype=np.float64)
    if V.shape != (2, 2):
        raise RuntimeError(f"pairwise: expected 2x2 variance matrix, got {V.shape}")
    return {"estimate": est.tolist(), "variance": V.tolist()}


def cma_batch(y: np.ndarray, indicators: dict, methods=None, *, variance: str):
    """Run CMA and CID on uncertainty indicators vs correctness (acor in Python)."""
    if methods is None:
        methods = METHODS
    indicator_names = list(indicators.keys())
    out = {}
    for m in methods:
        single_results = {}
        est_vals = np.empty(len(indicator_names), dtype=np.float64)
        for i, nm in enumerate(indicator_names):
            res = _run_py_single(y, indicators[nm], m, variance=variance)
            single_results[nm] = res
            est_vals[i] = res["estimate"]

        sorted_idx = np.argsort(-est_vals, kind="stable")
        best_name = indicator_names[sorted_idx[0]]
        second_best_name = indicator_names[sorted_idx[1]]

        pw = _run_py_pairwise(
            y, indicators[best_name], indicators[second_best_name], m, variance=variance
        )
        out[m] = {
            "single_results": single_results,
            "pairwise": {
                "best": best_name,
                "second_best": second_best_name,
                "estimate": pw["estimate"],
                "variance": pw["variance"],
            },
        }
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rank_calibration_path",
        type=str,
        required=True,
        help="Base path for rank calibration data",
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default=None,
        help="Root directory containing calibration results",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for JSON results (default: case_study_LLM/outputs/)",
    )
    parser.add_argument(
        "--correctness",
        type=str,
        default="rouge",
        help="Correctness metric to use",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-2-7b-chat-hf",
        help="Model to evaluate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Temperature parameter",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="triviaqa",
        help="Dataset to use",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="rougeL",
        help="Evaluation mode",
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="cma",
        choices=["cma", "erce"],
        help="Metric to use for visualization (CMA or ERCE)",
    )
    parser.add_argument(
        "--variance",
        type=str,
        default=ACOR_VARIANCE_DEFAULT,
        choices=["ij", "plugin"],
        help="acor_test variance (default: plugin)",
    )
    args = parser.parse_args()

    if args.root_dir is None:
        args.root_dir = os.path.join(
            args.rank_calibration_path, "submission/calibration_results"
        )
    if args.output_dir is None:
        args.output_dir = _DEFAULT_LLM_OUTPUT

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading files from {args.root_dir}")
    model = args.model.split("/")[-1]
    dataset = args.dataset

    scores_file = os.path.join(
        args.root_dir,
        f"{model}_{dataset}_{args.temperature}_{args.correctness}.json",
    )
    if not os.path.exists(scores_file):
        raise ValueError(f"File not found: {scores_file}")
    scores = pd.DataFrame(json.load(open(scores_file))).dropna(axis=0)

    file_names = []
    for method in ["whitebox", "blackbox"]:
        if method == "whitebox":
            fn = (
                "_".join(
                    [
                        "calibrate",
                        model,
                        dataset,
                        str(args.temperature),
                        "none",
                        "whitebox",
                    ]
                )
                + ".json"
            )
            file_names.append(fn)
        else:
            for affinity_mode in ["disagreement", "agreement"]:
                fn = (
                    "_".join(
                        [
                            "calibrate",
                            model,
                            dataset,
                            str(args.temperature),
                            affinity_mode,
                            "blackbox",
                        ]
                    )
                    + ".json"
                )
                file_names.append(fn)

    data_whitebox = json.load(open(os.path.join(args.root_dir, file_names[0])))
    data_blackbox_disagreement = json.load(
        open(os.path.join(args.root_dir, file_names[1]))
    )
    data_blackbox_agreement = json.load(
        open(os.path.join(args.root_dir, file_names[2]))
    )

    indices = list(range(len(data_whitebox)))
    tmps = []
    for idx, (index, row_wb, row_bd, row_ba) in tqdm(
        enumerate(
            zip(
                indices,
                data_whitebox,
                data_blackbox_disagreement,
                data_blackbox_agreement,
            )
        ),
        total=len(data_whitebox),
    ):
        tmp = {
            "model": model,
            "dataset": dataset,
            "metric": args.correctness,
            "temperature": args.temperature,
        }

        tmp["ecc_c_disagreement"] = row_bd["ecc_c"]
        tmp["degree_c_disagreement"] = row_bd["degree_c"]
        tmp["ecc_u_disagreement"] = [row_bd["ecc_u"]] * 10
        tmp["degree_u_disagreement"] = [row_bd["degree_u"]] * 10
        tmp["spectral_u_disagreement"] = [row_bd["spectral_u"]] * 10

        tmp["ecc_c_agreement"] = row_ba["ecc_c"]
        tmp["degree_c_agreement"] = row_ba["degree_c"]
        tmp["ecc_u_agreement"] = [row_ba["ecc_u"]] * 10
        tmp["degree_u_agreement"] = [row_ba["degree_u"]] * 10
        tmp["spectral_u_agreement"] = [row_ba["spectral_u"]] * 10

        tmp["entropy_normalized"] = [row_wb["entropy_normalized"]] * 10
        tmp["entropy_unnormalized"] = [row_wb["entropy_unnormalized"]] * 10
        tmp["normalized_nll_all"] = row_wb["normalized_nll"]
        tmp["unnormalized_nll_all"] = row_wb["unnormalized_nll"]

        score = scores[scores["id"] == index]
        tmp["normalized_score_all"] = score.iloc[0]["normalized_score"]
        tmp["unnormalized_score_all"] = score.iloc[0]["unnormalized_score"]
        normalized_min_index = np.argmin(tmp["normalized_nll_all"])
        unnormalized_min_index = np.argmin(tmp["unnormalized_nll_all"])
        tmp["normalized_score_greedy"] = tmp["normalized_score_all"][
            normalized_min_index
        ]
        tmp["unnormalized_score_greedy"] = tmp["unnormalized_score_all"][
            unnormalized_min_index
        ]

        tmps.append(tmp)

    df = pd.DataFrame(tmps).dropna(axis=0)

    uncertainty_indicators = [
        "ecc_u_agreement",
        "degree_u_agreement",
        "spectral_u_agreement",
        "unnormalized_nll_all",
        "entropy_unnormalized",
    ]

    correctness_scores = np.stack(df["normalized_score_all"]).flatten()

    indicator_data = {}
    for indicator in uncertainty_indicators:
        if "verbalized" in indicator:
            uncertainty = -np.stack(df[indicator]).flatten()
        else:
            uncertainty = np.stack(df[indicator]).flatten()
        indicator_data[indicator] = -uncertainty

    print(
        f"Calling acor (Python) for CMA and CID on "
        f"{len(indicator_data)} indicators over n={len(correctness_scores)} samples "
        f"(variance={args.variance!r})..."
    )
    batch_result = cma_batch(
        correctness_scores, indicator_data, methods=METHODS, variance=args.variance
    )

    n = len(correctness_scores)

    for method_name in METHODS:
        method_result = batch_result[method_name]
        is_cma = method_name == "cma"
        file_tag = "cma" if is_cma else "cid"
        value_suffix = "cma" if is_cma else "cid"

        result = {
            "model": model,
            "dataset": dataset,
            "metric": args.correctness,
            "temperature": args.temperature,
        }

        for indicator in uncertainty_indicators:
            sr = method_result["single_results"][indicator]
            result[f"{indicator}_sd"] = sr["variance"] / n
            result[f"{indicator}_{value_suffix}"] = sr["estimate"]

        stat_path = os.path.join(
            args.output_dir,
            f"{model}_{dataset}_{args.temperature}_{args.correctness}_{file_tag}_stat_test.json",
        )
        with open(stat_path, "w") as f:
            json.dump(result, f)

        pw = method_result["pairwise"]
        best_method = pw["best"]
        second_best_method = pw["second_best"]
        est_vec = pw["estimate"]
        var_mat = np.array(pw["variance"])

        if is_cma:
            pairwise_output = {
                "best_method": best_method,
                "best_cma": float(est_vec[0]),
                "second_best_method": second_best_method,
                "second_best_cma": float(est_vec[1]),
                "var_best": float(var_mat[0, 0] / n),
                "var_second_best": float(var_mat[1, 1] / n),
                "cov": float(var_mat[0, 1] / n),
            }
        else:
            pairwise_output = {
                "best_method": best_method,
                "best_cid": float(est_vec[0]),
                "second_best_method": second_best_method,
                "second_best_cid": float(est_vec[1]),
                "var_best": float(var_mat[0, 0] / n),
                "var_second_best": float(var_mat[1, 1] / n),
                "cov": float(var_mat[0, 1] / n),
            }
        pair_path = os.path.join(
            args.output_dir,
            f"{model}_{dataset}_{args.temperature}_{args.correctness}_{file_tag}_pairwise_test.json",
        )

        with open(pair_path, "w") as f:
            json.dump(pairwise_output, f, indent=2)

    print(f"Done. Wrote JSONs under {args.output_dir}")
