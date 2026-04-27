"""
Bootstrap resampled calibration metrics (RCE, ERCE, CMA, CID) over fixed seeds.

CMA and CID use the Python **acor** package (``acor_test``), same construction as
``compute_table.py``. ERCE uses ``metrics.calibration`` from Rank-Calibration (Python).

Each run writes ``{model}_{dataset}_{temp}_{correctness}_calibration_bootstrap.json``
(a list of per-seed dicts with ``*_erce``, ``*_cma``, ``*_cindx`` per indicator).
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

_CASE_STUDY_LLM_DIR = os.path.dirname(os.path.abspath(__file__))
if _CASE_STUDY_LLM_DIR not in sys.path:
    sys.path.insert(0, _CASE_STUDY_LLM_DIR)
try:
    from compute_table import _run_py_single
except ImportError as e:
    raise ImportError(
        "compute_calibration_bootstrap.py must live next to compute_table.py "
        "and needs the 'acor' package (see case_study_wb2 / acor-python docs)."
    ) from e

_DEFAULT_LLM_OUTPUT = os.path.join(_CASE_STUDY_LLM_DIR, "outputs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank_calibration_path', type=str, required=True,
                        help='Base path for rank calibration data')
    parser.add_argument('--root_dir', type=str, 
                       default=None,
                       help='Root directory containing calibration results')
    parser.add_argument('--output_dir', type=str, 
                       default=None,
                       help='Output directory for JSON bundles (default: case_study_LLM/outputs/)')
    parser.add_argument('--correctness', type=str, default='rouge',
                       help='Correctness metric to use')
    parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-chat-hf',
                       help='Model to evaluate')
    parser.add_argument('--temperature', type=float, default=0.6,
                       help='Temperature parameter')
    parser.add_argument('--dataset', type=str, default='triviaqa',
                       help='Dataset to use')
    parser.add_argument('--mode', type=str, default='rougeL',
                       help='Evaluation mode')
    parser.add_argument('--metric', type=str, default='cma',
                       choices=['cma', 'erce'],
                       help='Metric to use for visualization (CMA or ERCE)')
    args = parser.parse_args()

    if args.root_dir is None:
        args.root_dir = os.path.join(args.rank_calibration_path, 'submission/calibration_results')
    if args.output_dir is None:
        args.output_dir = _DEFAULT_LLM_OUTPUT

    sys.path.insert(0, os.path.abspath(args.rank_calibration_path))
    from metrics import calibration  # Rank-Calibration Python (RCE/ERCE), not R
    os.makedirs(args.output_dir, exist_ok=True)

    def _x_for_acor(df: pd.DataFrame, indicator: str) -> np.ndarray:
        """Same predictor sign convention as ``compute_table.indicator_data``."""
        if "verbalized" in indicator:
            u = -np.stack(df[indicator]).flatten()
        else:
            u = np.stack(df[indicator]).flatten()
        return -u

    # list all csv files in the root directory
    print(f"Loading files from {args.root_dir}")
    file_names = [file for file in os.listdir(args.root_dir) if file.endswith('.json')]
    model = args.model.split('/')[-1]
    
    # compute the correctness score
    scores_file = os.path.join(args.root_dir, f"{model}_{args.dataset}_{args.temperature}_{args.correctness}.json")
    if os.path.exists(scores_file):
        scores = json.load(open(scores_file))
    else:
        raise ValueError(f"File not found: {scores_file}")
    scores = pd.DataFrame(scores).dropna(axis=0)

    model = args.model.split('/')[-1]
    dataset = args.dataset
    file_names = []
    for method in ['whitebox', 'blackbox', 'verbalized']:
        if method == 'whitebox':
            affinity_mode = 'none'
            file_name = "_".join(['calibrate', model, dataset, str(args.temperature), affinity_mode, 'whitebox']) + '.json'
            file_names.append(file_name)
        elif method == 'verbalized':
            file_name = "_".join(['calibrate', model, dataset, str(args.temperature), 'disagreement', 'verbalized']) + '.json'
            if os.path.exists(os.path.join(args.root_dir, file_name)):
                file_names.append(file_name)
        else:
            for affinity_mode in ['disagreement', 'agreement']:
                file_name = "_".join(['calibrate', model, dataset, str(args.temperature), affinity_mode, 'blackbox']) + '.json'
                file_names.append(file_name)
    data_whitebox = json.load(open(os.path.join(args.root_dir, file_names[0])))
    data_blackbox_disagreement = json.load(open(os.path.join(args.root_dir, file_names[1])))
    data_blackbox_agreement = json.load(open(os.path.join(args.root_dir, file_names[2])))

    results = []
    seeds = list(range(20))
    for seed in seeds:
        #print(seed)
        np.random.seed(seed)
        if model == 'gpt-3.5-turbo' and args.temperature == 1.0 and len(file_names) > 3:
            data_verbalized = json.load(open(os.path.join(args.root_dir, file_names[3])))
            indices = np.random.choice(len(data_verbalized), len(data_verbalized), replace=True).tolist()
            data_verbalized_bootstrap = [data_verbalized[index] for index in indices]
            tmps = []
            for row_verbalized in tqdm(data_verbalized_bootstrap):
                tmp = {'model':model, 'dataset':dataset, 'metric':args.correctness, 'seed':seed, 'temperature':args.temperature}
                idx = row_verbalized['idx']
                row_whitebox = data_whitebox[idx]
                row_blackbox_disagreement = data_blackbox_disagreement[idx]
                row_blackbox_agreement = data_blackbox_agreement[idx]
                
                tmp['ecc_c_disagreement'] = row_blackbox_disagreement['ecc_c']
                tmp['degree_c_disagreement'] = row_blackbox_disagreement['degree_c']
                tmp['ecc_u_disagreement'] = [row_blackbox_disagreement['ecc_u']] * 10
                tmp['degree_u_disagreement'] = [row_blackbox_disagreement['degree_u']] * 10
                tmp['spectral_u_disagreement'] = [row_blackbox_disagreement['spectral_u']] * 10
                tmp['verbalized'] = row_verbalized['verbalized']

                tmp['ecc_c_agreement'] = row_blackbox_agreement['ecc_c']
                tmp['degree_c_agreement'] = row_blackbox_agreement['degree_c']
                tmp['ecc_u_agreement'] = [row_blackbox_agreement['ecc_u']] * 10
                tmp['degree_u_agreement'] = [row_blackbox_agreement['degree_u']] * 10
                tmp['spectral_u_agreement'] = [row_blackbox_agreement['spectral_u']] * 10

                tmp['entropy_normalized'] = [row_whitebox['entropy_normalized']] * 10
                tmp['entropy_unnormalized'] = [row_whitebox['entropy_unnormalized']] * 10
                tmp['normalized_nll_all'] = row_whitebox['normalized_nll']
                tmp['unnormalized_nll_all'] = row_whitebox['unnormalized_nll']

                # select scores with the same index
                score = scores[scores['id'] == idx]
                tmp['normalized_score_all'] = score.iloc[0]['normalized_score']
                tmp['unnormalized_score_all'] = score.iloc[0]['unnormalized_score']
                normalized_min_index = np.argmin(tmp['normalized_nll_all'])
                unnormalized_min_index = np.argmin(tmp['unnormalized_nll_all'])
                tmp['normalized_score_greedy'] = tmp['normalized_score_all'][normalized_min_index]
                tmp['unnormalized_score_greedy'] = tmp['unnormalized_score_all'][unnormalized_min_index]
                tmps.append(tmp)
        else:
            # sample with replacement from the indices of the data
            indices = np.random.choice(len(data_whitebox), len(data_whitebox), replace=True).tolist()
            data_whitebox_bootstrap = [data_whitebox[index] for index in indices]
            data_blackbox_disagreement_bootstrap = [data_blackbox_disagreement[index] for index in indices]
            data_blackbox_agreement_bootstrap = [data_blackbox_agreement[index] for index in indices]
            tmps = []
            for idx, (index, row_whitebox, row_blackbox_disagreement, row_blackbox_agreement) in tqdm(enumerate(zip(indices, data_whitebox_bootstrap, data_blackbox_disagreement_bootstrap, data_blackbox_agreement_bootstrap)), total=len(data_whitebox)):
                tmp = {'model':model, 'dataset':dataset, 'metric':args.correctness, 'seed':seed, 'temperature':args.temperature}

                tmp['ecc_c_disagreement'] = row_blackbox_disagreement['ecc_c']
                tmp['degree_c_disagreement'] = row_blackbox_disagreement['degree_c']
                tmp['ecc_u_disagreement'] = [row_blackbox_disagreement['ecc_u']] * 10
                tmp['degree_u_disagreement'] = [row_blackbox_disagreement['degree_u']] * 10
                tmp['spectral_u_disagreement'] = [row_blackbox_disagreement['spectral_u']] * 10

                tmp['ecc_c_agreement'] = row_blackbox_agreement['ecc_c']
                tmp['degree_c_agreement'] = row_blackbox_agreement['degree_c']
                tmp['ecc_u_agreement'] = [row_blackbox_agreement['ecc_u']] * 10
                tmp['degree_u_agreement'] = [row_blackbox_agreement['degree_u']] * 10
                tmp['spectral_u_agreement'] = [row_blackbox_agreement['spectral_u']] * 10

                tmp['entropy_normalized'] = [row_whitebox['entropy_normalized']] * 10
                tmp['entropy_unnormalized'] = [row_whitebox['entropy_unnormalized']] * 10
                tmp['normalized_nll_all'] = row_whitebox['normalized_nll']
                tmp['unnormalized_nll_all'] = row_whitebox['unnormalized_nll']

                # select scores with the same index
         
                score = scores[scores['id'] == index]
                tmp['normalized_score_all'] = score.iloc[0]['normalized_score']
                tmp['unnormalized_score_all'] = score.iloc[0]['unnormalized_score']
                normalized_min_index = np.argmin(tmp['normalized_nll_all'])
                unnormalized_min_index = np.argmin(tmp['unnormalized_nll_all'])
                tmp['normalized_score_greedy'] = tmp['normalized_score_all'][normalized_min_index]
                tmp['unnormalized_score_greedy'] = tmp['unnormalized_score_all'][unnormalized_min_index]
                tmps.append(tmp)
        df = pd.DataFrame(tmps).dropna(axis=0)

        used_verbalized_bootstrap = (
            model == "gpt-3.5-turbo"
            and args.temperature == 1.0
            and len(file_names) > 3
        )
        if used_verbalized_bootstrap:
            uncertainty_indicators = [
                "ecc_u_agreement",
                "degree_u_agreement",
                "spectral_u_agreement",
                "verbalized",
                "normalized_nll_all",
                "unnormalized_nll_all",
                "entropy_normalized",
                "entropy_unnormalized",
            ]
        else:
            uncertainty_indicators = [
                "ecc_u_agreement",
                "degree_u_agreement",
                "spectral_u_agreement",
                "normalized_nll_all",
                "unnormalized_nll_all",
                "entropy_normalized",
                "entropy_unnormalized",
            ]
        
        correctness_scores = np.stack(df["normalized_score_all"]).flatten()

        out = {
            "model": model,
            "dataset": dataset,
            "metric": args.correctness,
            "seed": seed,
            "temperature": args.temperature,
        }
        for indicator in uncertainty_indicators:
            if "verbalized" in indicator:
                u_for_rce = -np.stack(df[indicator]).flatten()
            else:
                u_for_rce = np.stack(df[indicator]).flatten()
            erce = calibration.plugin_RCE_est(
                correctness=correctness_scores,
                uncertainties=u_for_rce,
                num_bins=20,
                p=1,
            )
            out[f"{indicator}_erce"] = erce

            x = _x_for_acor(df, indicator)
            out[f"{indicator}_cma"] = _run_py_single(
                correctness_scores, x, "cma"
            )["estimate"]
            out[f"{indicator}_cindx"] = _run_py_single(
                correctness_scores, x, "cid"
            )["estimate"]

        results.append(out)
    
    # Update the output file path to use the specified output directory
    output_file = os.path.join(
        args.output_dir,
        f"{model}_{args.dataset}_{args.temperature}_{args.correctness}_calibration_bootstrap.json",
    )
    with open(output_file, 'w') as f:
        json.dump(results, f)
