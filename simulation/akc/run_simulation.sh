#!/usr/bin/env bash
# AKC + Zou p-value sweep. See ../README.md.
# Set T for a quick test, e.g.:  T=5000 ./run_simulation.sh
#
# From repo root:  ./simulation/akc/run_simulation.sh
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
export PYTHONPATH="$ROOT"
cd "$ROOT"

T="${T:-100000}"
OUTPUT_DIR="${OUTPUT_DIR:-$HERE/results}"

N_LIST="50 100 500 1000 5000"

echo "T=$T  OUTPUT_DIR=$OUTPUT_DIR"
echo "== Discrete DGP, two.sided =="
for n in $N_LIST; do
  python3 -m simulation.akc.simulation_p_values --n "$n" --T "$T" --discrete --alternative two.sided --output_dir "$OUTPUT_DIR"
done

echo "== Continuous DGP, one.sided =="
for n in $N_LIST; do
  python3 -m simulation.akc.simulation_p_values --n "$n" --T "$T" --alternative one.sided --output_dir "$OUTPUT_DIR"
done

echo "Done. Plots:  python3 -m simulation.akc.plot_p_values --results_dir $OUTPUT_DIR"
