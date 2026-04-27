#!/usr/bin/env bash
# run_table.sh
# Runs compute_table.py for every (model, dataset, correctness) combo used in the paper.
# Uses the Python acor package (acor_test) only. JSONs go to ./outputs/

set -euo pipefail

# Requires a local clone of Rank-Calibration. Set the repo root, e.g.:
#   export RANK_CALIBRATION_PATH=/path/to/Rank-Calibration
cd "$(dirname "$0")"
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
    echo " Running full sweep (Python acor)"
    echo "=================================================================="

    python compute_table.py --rank_calibration_path "$file_path" --correctness bert_similarity
    python compute_table.py --rank_calibration_path "$file_path" --correctness rouge
    python compute_table.py --rank_calibration_path "$file_path" --correctness rouge1
    python compute_table.py --rank_calibration_path "$file_path" --correctness meteor

    python compute_table.py --rank_calibration_path "$file_path" --dataset nq-open --correctness bert_similarity
    python compute_table.py --rank_calibration_path "$file_path" --dataset nq-open --correctness rouge
    python compute_table.py --rank_calibration_path "$file_path" --dataset nq-open --correctness rouge1
    python compute_table.py --rank_calibration_path "$file_path" --dataset nq-open --correctness meteor

    python compute_table.py --rank_calibration_path "$file_path" --dataset squad --correctness bert_similarity
    python compute_table.py --rank_calibration_path "$file_path" --dataset squad --correctness rouge
    python compute_table.py --rank_calibration_path "$file_path" --dataset squad --correctness rouge1
    python compute_table.py --rank_calibration_path "$file_path" --dataset squad --correctness meteor

    python compute_table.py --rank_calibration_path "$file_path" --model meta-llama/Llama-2-7b-hf --correctness bert_similarity
    python compute_table.py --rank_calibration_path "$file_path" --model meta-llama/Llama-2-7b-hf --correctness rouge
    python compute_table.py --rank_calibration_path "$file_path" --model meta-llama/Llama-2-7b-hf --correctness rouge1
    python compute_table.py --rank_calibration_path "$file_path" --model meta-llama/Llama-2-7b-hf --correctness meteor

    python compute_table.py --rank_calibration_path "$file_path" --model meta-llama/Llama-2-7b-hf --dataset nq-open --correctness bert_similarity
    python compute_table.py --rank_calibration_path "$file_path" --model meta-llama/Llama-2-7b-hf --dataset nq-open --correctness rouge
    python compute_table.py --rank_calibration_path "$file_path" --model meta-llama/Llama-2-7b-hf --dataset nq-open --correctness rouge1
    python compute_table.py --rank_calibration_path "$file_path" --model meta-llama/Llama-2-7b-hf --dataset nq-open --correctness meteor

    python compute_table.py --rank_calibration_path "$file_path" --model meta-llama/Llama-2-7b-hf --dataset squad --correctness bert_similarity
    python compute_table.py --rank_calibration_path "$file_path" --model meta-llama/Llama-2-7b-hf --dataset squad --correctness rouge
    python compute_table.py --rank_calibration_path "$file_path" --model meta-llama/Llama-2-7b-hf --dataset squad --correctness rouge1
    python compute_table.py --rank_calibration_path "$file_path" --model meta-llama/Llama-2-7b-hf --dataset squad --correctness meteor

    python compute_table.py --rank_calibration_path "$file_path" --model 'meta-llama/gpt-3.5-turbo' --temperature 1.0 --dataset triviaqa --correctness bert_similarity
    python compute_table.py --rank_calibration_path "$file_path" --model 'meta-llama/gpt-3.5-turbo' --temperature 1.0 --dataset triviaqa --correctness meteor
    python compute_table.py --rank_calibration_path "$file_path" --model 'meta-llama/gpt-3.5-turbo' --temperature 1.0 --dataset triviaqa --correctness rouge
    python compute_table.py --rank_calibration_path "$file_path" --model 'meta-llama/gpt-3.5-turbo' --temperature 1.0 --dataset triviaqa --correctness rouge1

    python compute_table.py --rank_calibration_path "$file_path" --model 'meta-llama/gpt-3.5-turbo' --temperature 1.0 --dataset squad --correctness bert_similarity
    python compute_table.py --rank_calibration_path "$file_path" --model 'meta-llama/gpt-3.5-turbo' --temperature 1.0 --dataset squad --correctness meteor
    python compute_table.py --rank_calibration_path "$file_path" --model 'meta-llama/gpt-3.5-turbo' --temperature 1.0 --dataset squad --correctness rouge
    python compute_table.py --rank_calibration_path "$file_path" --model 'meta-llama/gpt-3.5-turbo' --temperature 1.0 --dataset squad --correctness rouge1

    python compute_table.py --rank_calibration_path "$file_path" --model 'meta-llama/gpt-3.5-turbo' --temperature 1.0 --dataset nq-open --correctness bert_similarity
    python compute_table.py --rank_calibration_path "$file_path" --model 'meta-llama/gpt-3.5-turbo' --temperature 1.0 --dataset nq-open --correctness meteor
    python compute_table.py --rank_calibration_path "$file_path" --model 'meta-llama/gpt-3.5-turbo' --temperature 1.0 --dataset nq-open --correctness rouge
    python compute_table.py --rank_calibration_path "$file_path" --model 'meta-llama/gpt-3.5-turbo' --temperature 1.0 --dataset nq-open --correctness rouge1
}

run_sweep

echo ""
echo "All done. JSONs:  outputs/"
echo "Then:  python create_table.py  →  figures/table_cma.pdf  figures/table_cid.pdf"
