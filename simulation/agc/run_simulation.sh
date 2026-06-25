#!/usr/bin/env bash
#
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
export PYTHONPATH="$ROOT"
cd "$ROOT"

T="${T:-100000}"
VARIANCE="${VARIANCE:-plugin}"
ALTERNATIVE="${ALTERNATIVE:-one.sided}"
OUTPUT_DIR="${OUTPUT_DIR:-$HERE/results}"
PLOTS_DIR="${PLOTS_DIR:-$HERE/plots}"
N_LIST="${N_LIST:-50 100 500 1000 5000}"

echo "T=$T  OUTPUT_DIR=$OUTPUT_DIR  VARIANCE=$VARIANCE  ALTERNATIVE=$ALTERNATIVE (PYTHONPATH=$ROOT)"

echo "== Discrete DGP, $ALTERNATIVE, AGC p-values only =="
for n in $N_LIST; do
  python -m simulation.agc.simulation_p_values --n "$n" --T "$T" --discrete --alternative "$ALTERNATIVE" --variance "$VARIANCE" --output_dir "$OUTPUT_DIR"
done

echo "== Continuous DGP, $ALTERNATIVE, Meng (Spearman) + AGC =="
for n in $N_LIST; do
  python -m simulation.agc.simulation_p_values --n "$n" --T "$T" --alternative "$ALTERNATIVE" --variance "$VARIANCE" --output_dir "$OUTPUT_DIR"
done

echo "Done. Plots:"
echo "  python -m simulation.agc.plot_p_values \\"
echo "    --results_dir \"$OUTPUT_DIR\" --output_dir \"$PLOTS_DIR\""
