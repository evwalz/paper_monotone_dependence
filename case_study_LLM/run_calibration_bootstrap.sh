#!/usr/bin/env bash
# Bootstrap calibration JSONs (CMA/CID via Python acor; ERCE via Rank-Calibration metrics).
# Writes under case_study_LLM/outputs/ by default. Then:  python plot_calibration_bootstrap.py

set -euo pipefail
cd "$(dirname "$0")"

# Requires a local clone of Rank-Calibration. Set the repo root, e.g.:
#   export RANK_CALIBRATION_PATH=/path/to/Rank-Calibration
if [ -z "${RANK_CALIBRATION_PATH:-}" ]; then
    echo "Error: RANK_CALIBRATION_PATH is not set."
    echo "  export RANK_CALIBRATION_PATH=/path/to/your/Rank-Calibration  # repository root"
    exit 1
fi
if [ ! -d "$RANK_CALIBRATION_PATH" ]; then
    echo "Error: RANK_CALIBRATION_PATH is not a directory: $RANK_CALIBRATION_PATH"
    exit 1
fi

file_path="$RANK_CALIBRATION_PATH"

run_sweep () {
    echo "=================================================================="
    echo " calibration_bootstrap (Python acor + Rank-Calibration plugin_RCE_est)"
    echo "=================================================================="

    python compute_calibration_bootstrap.py --rank_calibration_path "$file_path" --correctness bert_similarity
    python compute_calibration_bootstrap.py --rank_calibration_path "$file_path" --correctness rouge
    python compute_calibration_bootstrap.py --rank_calibration_path "$file_path" --correctness rouge1
    python compute_calibration_bootstrap.py --rank_calibration_path "$file_path" --correctness meteor

    python compute_calibration_bootstrap.py --rank_calibration_path "$file_path" --dataset nq-open --correctness bert_similarity
    python compute_calibration_bootstrap.py --rank_calibration_path "$file_path" --dataset nq-open --correctness rouge
    python compute_calibration_bootstrap.py --rank_calibration_path "$file_path" --dataset nq-open --correctness rouge1
    python compute_calibration_bootstrap.py --rank_calibration_path "$file_path" --dataset nq-open --correctness meteor

    python compute_calibration_bootstrap.py --rank_calibration_path "$file_path" --dataset squad --correctness bert_similarity
    python compute_calibration_bootstrap.py --rank_calibration_path "$file_path" --dataset squad --correctness rouge
    python compute_calibration_bootstrap.py --rank_calibration_path "$file_path" --dataset squad --correctness rouge1
    python compute_calibration_bootstrap.py --rank_calibration_path "$file_path" --dataset squad --correctness meteor

    python compute_calibration_bootstrap.py --rank_calibration_path "$file_path" --model meta-llama/Llama-2-7b-hf --correctness bert_similarity
    python compute_calibration_bootstrap.py --rank_calibration_path "$file_path" --model meta-llama/Llama-2-7b-hf --correctness rouge
    python compute_calibration_bootstrap.py --rank_calibration_path "$file_path" --model meta-llama/Llama-2-7b-hf --correctness rouge1
    python compute_calibration_bootstrap.py --rank_calibration_path "$file_path" --model meta-llama/Llama-2-7b-hf --correctness meteor

    python compute_calibration_bootstrap.py --rank_calibration_path "$file_path" --model meta-llama/Llama-2-7b-hf --dataset nq-open --correctness bert_similarity
    python compute_calibration_bootstrap.py --rank_calibration_path "$file_path" --model meta-llama/Llama-2-7b-hf --dataset nq-open --correctness rouge
    python compute_calibration_bootstrap.py --rank_calibration_path "$file_path" --model meta-llama/Llama-2-7b-hf --dataset nq-open --correctness rouge1
    python compute_calibration_bootstrap.py --rank_calibration_path "$file_path" --model meta-llama/Llama-2-7b-hf --dataset nq-open --correctness meteor

    python compute_calibration_bootstrap.py --rank_calibration_path "$file_path" --model meta-llama/Llama-2-7b-hf --dataset squad --correctness bert_similarity
    python compute_calibration_bootstrap.py --rank_calibration_path "$file_path" --model meta-llama/Llama-2-7b-hf --dataset squad --correctness rouge
    python compute_calibration_bootstrap.py --rank_calibration_path "$file_path" --model meta-llama/Llama-2-7b-hf --dataset squad --correctness rouge1
    python compute_calibration_bootstrap.py --rank_calibration_path "$file_path" --model meta-llama/Llama-2-7b-hf --dataset squad --correctness meteor

    python compute_calibration_bootstrap.py --rank_calibration_path "$file_path" --model meta-llama/gpt-3.5-turbo --temperature 1.0 --correctness bert_similarity
    python compute_calibration_bootstrap.py --rank_calibration_path "$file_path" --model meta-llama/gpt-3.5-turbo --temperature 1.0 --correctness rouge
    python compute_calibration_bootstrap.py --rank_calibration_path "$file_path" --model meta-llama/gpt-3.5-turbo --temperature 1.0 --correctness rouge1
    python compute_calibration_bootstrap.py --rank_calibration_path "$file_path" --model meta-llama/gpt-3.5-turbo --temperature 1.0 --correctness meteor

    python compute_calibration_bootstrap.py --rank_calibration_path "$file_path" --model meta-llama/gpt-3.5-turbo --temperature 1.0 --dataset nq-open --correctness bert_similarity
    python compute_calibration_bootstrap.py --rank_calibration_path "$file_path" --model meta-llama/gpt-3.5-turbo --temperature 1.0 --dataset nq-open --correctness rouge
    python compute_calibration_bootstrap.py --rank_calibration_path "$file_path" --model meta-llama/gpt-3.5-turbo --temperature 1.0 --dataset nq-open --correctness rouge1
    python compute_calibration_bootstrap.py --rank_calibration_path "$file_path" --model meta-llama/gpt-3.5-turbo --temperature 1.0 --dataset nq-open --correctness meteor

    python compute_calibration_bootstrap.py --rank_calibration_path "$file_path" --model meta-llama/gpt-3.5-turbo --temperature 1.0 --dataset squad --correctness bert_similarity
    python compute_calibration_bootstrap.py --rank_calibration_path "$file_path" --model meta-llama/gpt-3.5-turbo --temperature 1.0 --dataset squad --correctness rouge
    python compute_calibration_bootstrap.py --rank_calibration_path "$file_path" --model meta-llama/gpt-3.5-turbo --temperature 1.0 --dataset squad --correctness rouge1
    python compute_calibration_bootstrap.py --rank_calibration_path "$file_path" --model meta-llama/gpt-3.5-turbo --temperature 1.0 --dataset squad --correctness meteor
}

run_sweep

echo ""
echo "Done. JSON:  case_study_LLM/outputs/"
echo "Then:  python plot_calibration_bootstrap.py  →  plots/calibration_bootstrap_scatter.pdf"
