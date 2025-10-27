#!/bin/bash

# Configuration
N_VALUES=(1000 5000) #50 100 500 1000 5000)
T=100000

mkdir -p results

#echo "Running discrete two-sided..."
#for n in "${N_VALUES[@]}"; do
#    python simulation_p_values.py --n $n --T $T --discrete --alternative two.sided --output_dir ./results
#done

echo "Running continuous one-sided..."
for n in "${N_VALUES[@]}"; do
    python simulation_p_values.py --n $n --T $T --alternative one.sided --output_dir ./results
done

echo ""
echo "Done! Run: bash generate_plots.sh"