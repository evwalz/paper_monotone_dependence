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

N_LIST="50 100 500 1000 5000"
#N_LIST="1000 5000"

echo "T=$T  OUTPUT_DIR=$OUTPUT_DIR  VARIANCE=$VARIANCE  ALTERNATIVE=$ALTERNATIVE"
echo "== Discrete DGP, $ALTERNATIVE (acor variance: $VARIANCE) =="
for n in $N_LIST; do
  python -m simulation.akc.simulation_p_values --n "$n" --T "$T" --discrete --alternative "$ALTERNATIVE" --variance "$VARIANCE" --output_dir "$OUTPUT_DIR"
done

echo "Done. Plots:"
echo "  python -m simulation.akc.plot_p_values \\"
echo "    --results_dir \"$OUTPUT_DIR\" --output_dir \"$PLOTS_DIR\""
