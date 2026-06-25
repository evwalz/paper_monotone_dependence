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
VARIANCE="${VARIANCE:-plugin}"

run_compute_table () {
    python compute_table.py --rank_calibration_path "$file_path" --variance "$VARIANCE" "$@"
}

run_sweep () {
    echo "=================================================================="
    echo " Running full sweep (Python acor, variance=$VARIANCE)"
    echo "=================================================================="

    run_compute_table --correctness bert_similarity
    run_compute_table --correctness rouge
    run_compute_table --correctness rouge1
    run_compute_table --correctness meteor

    run_compute_table --dataset nq-open --correctness bert_similarity
    run_compute_table --dataset nq-open --correctness rouge
    run_compute_table --dataset nq-open --correctness rouge1
    run_compute_table --dataset nq-open --correctness meteor

    run_compute_table --dataset squad --correctness bert_similarity
    run_compute_table --dataset squad --correctness rouge
    run_compute_table --dataset squad --correctness rouge1
    run_compute_table --dataset squad --correctness meteor

    run_compute_table --model meta-llama/Llama-2-7b-hf --correctness bert_similarity
    run_compute_table --model meta-llama/Llama-2-7b-hf --correctness rouge
    run_compute_table --model meta-llama/Llama-2-7b-hf --correctness rouge1
    run_compute_table --model meta-llama/Llama-2-7b-hf --correctness meteor

    run_compute_table --model meta-llama/Llama-2-7b-hf --dataset nq-open --correctness bert_similarity
    run_compute_table --model meta-llama/Llama-2-7b-hf --dataset nq-open --correctness rouge
    run_compute_table --model meta-llama/Llama-2-7b-hf --dataset nq-open --correctness rouge1
    run_compute_table --model meta-llama/Llama-2-7b-hf --dataset nq-open --correctness meteor

    run_compute_table --model meta-llama/Llama-2-7b-hf --dataset squad --correctness bert_similarity
    run_compute_table --model meta-llama/Llama-2-7b-hf --dataset squad --correctness rouge
    run_compute_table --model meta-llama/Llama-2-7b-hf --dataset squad --correctness rouge1
    run_compute_table --model meta-llama/Llama-2-7b-hf --dataset squad --correctness meteor

    run_compute_table --model 'meta-llama/gpt-3.5-turbo' --temperature 1.0 --dataset triviaqa --correctness bert_similarity
    run_compute_table --model 'meta-llama/gpt-3.5-turbo' --temperature 1.0 --dataset triviaqa --correctness meteor
    run_compute_table --model 'meta-llama/gpt-3.5-turbo' --temperature 1.0 --dataset triviaqa --correctness rouge
    run_compute_table --model 'meta-llama/gpt-3.5-turbo' --temperature 1.0 --dataset triviaqa --correctness rouge1

    run_compute_table --model 'meta-llama/gpt-3.5-turbo' --temperature 1.0 --dataset squad --correctness bert_similarity
    run_compute_table --model 'meta-llama/gpt-3.5-turbo' --temperature 1.0 --dataset squad --correctness meteor
    run_compute_table --model 'meta-llama/gpt-3.5-turbo' --temperature 1.0 --dataset squad --correctness rouge
    run_compute_table --model 'meta-llama/gpt-3.5-turbo' --temperature 1.0 --dataset squad --correctness rouge1

    run_compute_table --model 'meta-llama/gpt-3.5-turbo' --temperature 1.0 --dataset nq-open --correctness bert_similarity
    run_compute_table --model 'meta-llama/gpt-3.5-turbo' --temperature 1.0 --dataset nq-open --correctness meteor
    run_compute_table --model 'meta-llama/gpt-3.5-turbo' --temperature 1.0 --dataset nq-open --correctness rouge
    run_compute_table --model 'meta-llama/gpt-3.5-turbo' --temperature 1.0 --dataset nq-open --correctness rouge1
}

run_sweep

echo ""
echo "All done. JSONs:  outputs/"
echo "Then:  python create_table.py  →  plots/table_cma.pdf  plots/table_cid.pdf"
