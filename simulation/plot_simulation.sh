#!/bin/bash

echo "Generating plots from results..."
echo ""

# Check results exist
if [ ! -d "results" ] || [ $(find results -name "*.npy" 2>/dev/null | wc -l) -eq 0 ]; then
    echo "Error: No results found. Run simulations first."
    exit 1
fi

python plot_p_values.py

if [ $? -eq 0 ]; then
    echo ""
    echo "Done! Check results/ directory"
    ls results/*.pdf 2>/dev/null
else
    echo "Error generating plots"
    exit 1
fi