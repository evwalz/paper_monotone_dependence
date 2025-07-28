import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import Patch  # Import Patch for custom legend entries
from matplotlib.cm import Blues, Purples
from scipy.stats import norm
import argparse


def plot_pvals(forecast_name, input_dir):
    name_fct1 = forecast_name[0]
    name_fct2 = forecast_name[1]

    cma_fct1 = np.loadtxt(f'{input_dir}/cma_{name_fct1}.txt')
    cma_fct2 = np.loadtxt(f'{input_dir}/cma_{name_fct2}.txt')

    S_var = np.loadtxt(f'{input_dir}/variance_{name_fct1}_{name_fct2}.txt')
    diff = cma_fct1 - cma_fct2
    p_val_single = norm.cdf(diff / np.sqrt(S_var))

    # Color scheme
    threshold_color = '#e67e22'
    cma_color = '#8e24aa'
    background_color = '#f9f9f9'
    grid_color = '#ecf0f1'

    # Use your existing data calculations
    X = 1-p_val_single

    # Calculate additional quantiles for p-values (ORIGINAL CODE)
    X_q25 = np.nanquantile(X, q=0.25, axis=1)  # Lower quartile
    X_q275 = np.nanquantile(X, q=0.275, axis=1)
    X_q30 = np.nanquantile(X, q=0.30, axis=1)
    X_q325 = np.nanquantile(X, q=0.325, axis=1)
    X_q35 = np.nanquantile(X, q=0.35, axis=1)
    X_q375 = np.nanquantile(X, q=0.375, axis=1)
    X_q40 = np.nanquantile(X, q=0.40, axis=1)
    X_q425 = np.nanquantile(X, q=0.425, axis=1)
    X_q45 = np.nanquantile(X, q=0.45, axis=1)
    X_q475 = np.nanquantile(X, q=0.475, axis=1)
    X_med = np.nanmedian(X, axis=1)  # Median (50th percentile)
    X_q525 = np.nanquantile(X, q=0.525, axis=1)
    X_q55 = np.nanquantile(X, q=0.55, axis=1)
    X_q575 = np.nanquantile(X, q=0.575, axis=1)
    X_q60 = np.nanquantile(X, q=0.60, axis=1)
    X_q625 = np.nanquantile(X, q=0.625, axis=1)
    X_q65 = np.nanquantile(X, q=0.65, axis=1)
    X_q675 = np.nanquantile(X, q=0.675, axis=1)
    X_q70 = np.nanquantile(X, q=0.70, axis=1)
    X_q725 = np.nanquantile(X, q=0.725, axis=1)
    X_q75 = np.nanquantile(X, q=0.75, axis=1)  # Upper quartile


    # Calculate additional quantiles for CMA difference (NEW CODE FOR PURPLE SHADING)
    CMA_q25 = np.nanquantile(diff, q=0.25, axis=1)  # Lower quartile
    CMA_q275 = np.nanquantile(diff, q=0.275, axis=1)
    CMA_q30 = np.nanquantile(diff, q=0.30, axis=1)
    CMA_q325 = np.nanquantile(diff, q=0.325, axis=1)
    CMA_q35 = np.nanquantile(diff, q=0.35, axis=1)
    CMA_q375 = np.nanquantile(diff, q=0.375, axis=1)
    CMA_q40 = np.nanquantile(diff, q=0.40, axis=1)
    CMA_q425 = np.nanquantile(diff, q=0.425, axis=1)
    CMA_q45 = np.nanquantile(diff, q=0.45, axis=1)
    CMA_q475 = np.nanquantile(diff, q=0.475, axis=1)
    CMA_med = np.nanmedian(diff, axis=1)  # Median (50th percentile)
    CMA_q525 = np.nanquantile(diff, q=0.525, axis=1)
    CMA_q55 = np.nanquantile(diff, q=0.55, axis=1)
    CMA_q575 = np.nanquantile(diff, q=0.575, axis=1)
    CMA_q60 = np.nanquantile(diff, q=0.60, axis=1)
    CMA_q625 = np.nanquantile(diff, q=0.625, axis=1)
    CMA_q65 = np.nanquantile(diff, q=0.65, axis=1)
    CMA_q675 = np.nanquantile(diff, q=0.675, axis=1)
    CMA_q70 = np.nanquantile(diff, q=0.70, axis=1)
    CMA_q725 = np.nanquantile(diff, q=0.725, axis=1)
    CMA_q75 = np.nanquantile(diff, q=0.75, axis=1)  # Upper quartile

    # Create a figure and axis with the background color
    fig, ax = plt.subplots(figsize=(18, 8))

    plt.rcParams.update({
    'font.size': 14,           # Base font size
    'axes.titlesize': 16,      # Axis title size
    'axes.labelsize': 14,      # Axis label size
    'xtick.labelsize': 12,     # X-axis tick label size
    'ytick.labelsize': 12,     # Y-axis tick label size
    'legend.fontsize': 11,     # Legend font size
})

    # Add grid but make it subtle
    ax.grid(True, linestyle='--', alpha=0.3, color=grid_color)
    ax.set_axisbelow(True)  # Place grid behind other elements

    # Define x range with trimmed edges
    x_start = 0  # Remove 8 points from left side
    x_end = 121 - 0  # Remove 8 points from right side
    x_range = np.arange(x_start, x_end)

    # Create blue colors for p-values (ORIGINAL)
    blue_colors = []
    for i in range(10):
        color_val = 0.8 - (i * 0.07)  # From 0.8 to 0.1
        rgb = Blues(color_val)[:3]  # Get RGB values (exclude alpha)
        hex_color = f'#{int(rgb[0]*255):02x}{int(rgb[1]*255):02x}{int(rgb[2]*255):02x}'
        blue_colors.append(hex_color)

    # Create purple colors for CMA difference (NEW)
    purple_colors = []
    for i in range(10):
        color_val = 0.8 - (i * 0.07)  # From 0.8 to 0.1
        rgb = Purples(color_val)[:3]  # Get RGB values (exclude alpha)
        hex_color = f'#{int(rgb[0]*255):02x}{int(rgb[1]*255):02x}{int(rgb[2]*255):02x}'
        purple_colors.append(hex_color)

    # ORIGINAL P-VALUE SHADING (BLUE) - exactly as before
    for i in range(x_start, x_end):
        width = 1.0
        
        # Define the quantile segments with the professional blue colormap
        quantile_segments = [
        # Dark blue to light blue (lower quartile to median)
        (X_q25[i], X_q275[i], blue_colors[0]),  # Darkest blue
        (X_q275[i], X_q30[i], blue_colors[1]),
        (X_q30[i], X_q325[i], blue_colors[2]),
        (X_q325[i], X_q35[i], blue_colors[3]),
        (X_q35[i], X_q375[i], blue_colors[4]),
        (X_q375[i], X_q40[i], blue_colors[5]),
        (X_q40[i], X_q425[i], blue_colors[6]),
        (X_q425[i], X_q45[i], blue_colors[7]),
        (X_q45[i], X_q475[i], blue_colors[8]),
        (X_q475[i], X_med[i], blue_colors[9]),  # Lightest blue

        # Light blue to dark blue (median to upper quartile) - mirrored colors
        (X_med[i], X_q525[i], blue_colors[9]),  # Lightest blue
        (X_q525[i], X_q55[i], blue_colors[8]),
        (X_q55[i], X_q575[i], blue_colors[7]),
        (X_q575[i], X_q60[i], blue_colors[6]),
        (X_q60[i], X_q625[i], blue_colors[5]),
        (X_q625[i], X_q65[i], blue_colors[4]),
        (X_q65[i], X_q675[i], blue_colors[3]),
        (X_q675[i], X_q70[i], blue_colors[2]),
        (X_q70[i], X_q725[i], blue_colors[1]),
        (X_q725[i], X_q75[i], blue_colors[0]),  # Darkest blue
        ]

        # Draw each segment with its specific color
        for y_start, y_end, color in quantile_segments:
            height = y_end - y_start
            if height > 0:  # Skip zero-height segments
                segment = Rectangle(
                    (i - 0.5, y_start),    # x, y (bottom left corner)
                    width,                  # width
                    height,                 # height
                    facecolor=color,
                    edgecolor='none',
                alpha=0.85
            )
                ax.add_patch(segment)

    # Update the reference lines (ORIGINAL)
    plt.axhline(0.05, color=threshold_color, linestyle='-', linewidth=1, 
        label='Thresholds (0.05 and 0.95)', zorder=5)
    plt.axhline(0.95, color=threshold_color, linestyle='-', linewidth=1)#, 
                #label='Threshold (0.95)', zorder=5)
    # Update rectangle patch for legend (ORIGINAL)
    rectangle_patch = Patch(facecolor=blue_colors[4], edgecolor=blue_colors[4], 
                    alpha=0.85, label='IQR p-values')
    #(25th-75th percentile)

    # Create a secondary y-axis (ORIGINAL)
    ax2 = ax.twinx()

    for i in range(x_start, x_end):
        width = 0.8  # Slightly narrower than p-value bars
        
        # Define the quantile segments for CMA difference with purple colors
        cma_quantile_segments = [
        # Dark purple to light purple (lower quartile to median)
        (CMA_q25[i], CMA_q275[i], purple_colors[0]),  # Darkest purple
        (CMA_q275[i], CMA_q30[i], purple_colors[1]),
        (CMA_q30[i], CMA_q325[i], purple_colors[2]),
        (CMA_q325[i], CMA_q35[i], purple_colors[3]),
        (CMA_q35[i], CMA_q375[i], purple_colors[4]),
        (CMA_q375[i], CMA_q40[i], purple_colors[5]),
        (CMA_q40[i], CMA_q425[i], purple_colors[6]),
        (CMA_q425[i], CMA_q45[i], purple_colors[7]),
        (CMA_q45[i], CMA_q475[i], purple_colors[8]),
        (CMA_q475[i], CMA_med[i], purple_colors[9]),  # Lightest purple

        # Light purple to dark purple (median to upper quartile) - mirrored colors
        (CMA_med[i], CMA_q525[i], purple_colors[9]),  # Lightest purple
        (CMA_q525[i], CMA_q55[i], purple_colors[8]),
        (CMA_q55[i], CMA_q575[i], purple_colors[7]),
        (CMA_q575[i], CMA_q60[i], purple_colors[6]),
        (CMA_q60[i], CMA_q625[i], purple_colors[5]),
        (CMA_q625[i], CMA_q65[i], purple_colors[4]),
        (CMA_q65[i], CMA_q675[i], purple_colors[3]),
        (CMA_q675[i], CMA_q70[i], purple_colors[2]),
        (CMA_q70[i], CMA_q725[i], purple_colors[1]),
        (CMA_q725[i], CMA_q75[i], purple_colors[0]),  # Darkest purple
        ]

        # Transform CMA coordinates to display coordinates for proper positioning
        # Map CMA range (-0.12 to 0.12) to a portion of the plot area (right side)
        plot_width = x_end - x_start
        cma_x_offset = 0.1  # Offset from the right edge
        cma_bar_x = i + cma_x_offset
        
        # Draw each CMA segment with its specific color
        for cma_y_start, cma_y_end, color in cma_quantile_segments:
            height = cma_y_end - cma_y_start
            if height > 0:  # Skip zero-height segments
                # Use ax2 to add patches in CMA coordinate space
                cma_segment = Rectangle(
                    (cma_bar_x - width/2, cma_y_start),    # x, y (bottom left corner)
                    width,                  # width
                    height,                 # height
                    facecolor=color,
                    edgecolor='none',
                    alpha=0.7,              # Slightly more transparent
                    transform=ax2.transData  # Use secondary axis coordinates
                )
                ax2.add_patch(cma_segment)


    # Set the y-axis limits for the secondary axis (ORIGINAL)
    ax2.set_ylim(-0.12, 0.12)
    ax2.set_yticks([-0.09,-0.06,-0.03, 0, 0.03,0.060, 0.09])

    ax2.set_ylabel('CMA difference', fontsize=16, color=cma_color)
    ax2.tick_params(axis='y', colors=cma_color)
    ax2.axhline(0.0, color=cma_color, linestyle='--', linewidth=0.7, alpha=0.5)

    # Enhance axis styling (ORIGINAL)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#95a5a6')
    ax.spines['bottom'].set_color('#95a5a6')

    # Set appropriate axis limits and labels (ORIGINAL)
    ax.set_xlim(x_start, x_end-1)  # Set the x-axis limits to match our trimmed data
    ax.set_ylim(0, 1)
    ax.set_xlabel('Latitude (degrees)', fontsize=16, color='#34495e')
    ax.set_ylabel('p-value', fontsize=16, color='#34495e')

    ax.tick_params(axis='both', which='major', labelsize=14, width=1.5, length=6)
    ax2.tick_params(axis='y', colors=cma_color, labelsize=14, width=1.5, length=6)

    # Add a custom legend with better positioning and styling (ORIGINAL + NEW CMA PATCH)
    handles_ax1, labels_ax1 = ax.get_legend_handles_labels()
    handles_ax2, labels_ax2 = ax2.get_legend_handles_labels()

    # Add CMA rectangle patch for legend
    cma_rectangle_patch = Patch(facecolor=purple_colors[4], edgecolor=purple_colors[4], 
                    alpha=0.7, label='IQR CMA differences')

    # Add our rectangle patches to the handles
    handles_all = handles_ax1 + handles_ax2
    handles_all.append(rectangle_patch)
    handles_all.append(cma_rectangle_patch)
    #labels_all = labels_ax1 + labels_ax2 + ['p-values IQR (25th-75th percentile)', 'CMA diffs IQR (25th-75th percentile)']
    labels_all = labels_ax1 + labels_ax2 + ['IQR p-values', 'IQR CMA differences']
    
    legend = plt.legend(handles=handles_all, loc = 'upper center', frameon=True, framealpha=0.9, 
                facecolor=background_color, edgecolor='#95a5a6', fontsize = 12)

    plt.axvline(60, color='black', linestyle='--', alpha=0.3)  # Mark the equator

    # Adding labels for geographic regions on the x-axis (ORIGINAL)
    plt.xticks((np.array([-80, -60, -40, -20, 0, 20, 40, 60, 80]) + 90)*(2/3), 
        [ '80°S', '60°S','40°S' ,'20°S', 'Equator', 
        '20°N', '40°N', '60°N', '80°N'], 
              fontsize=14)

    # ORIGINAL COLORBAR CODE (unchanged)
    x_start_cbar = 87+3  # approximately 40°N in your coordinate system
    x_end_cbar = 113+3   # approximately 80°N
    y_position = 0.11  # Moved higher to accommodate both colorbars
    colorbar_height = 0.03

    # Define the color sequence that matches your quantile segments
    colorbar_colors = [
        # Dark blue to light blue (25th to 50th percentile)
        blue_colors[0],  # 25th-27.5th
        blue_colors[1],  # 27.5th-30th
        blue_colors[2],  # 30th-32.5th
        blue_colors[3],  # 32.5th-35th
        blue_colors[4],  # 35th-37.5th
        blue_colors[5],  # 37.5th-40th
        blue_colors[6],  # 40th-42.5th
        blue_colors[7],  # 42.5th-45th
        blue_colors[8],  # 45th-47.5th
        blue_colors[9],  # 47.5th-50th (median)
        # Light blue to dark blue (50th to 75th percentile) - mirrored
        blue_colors[9],  # 50th-52.5th
        blue_colors[8],  # 52.5th-55th
        blue_colors[7],  # 55th-57.5th
        blue_colors[6],  # 57.5th-60th
        blue_colors[5],  # 60th-62.5th
        blue_colors[4],  # 62.5th-65th
        blue_colors[3],  # 65th-67.5th
        blue_colors[2],  # 67.5th-70th
        blue_colors[1],  # 70th-72.5th
        blue_colors[0],  # 72.5th-75th
    ]

    # Create the color segments
    n_segments = len(colorbar_colors)
    segment_width = (x_end_cbar - x_start_cbar) / n_segments

    for i, color in enumerate(colorbar_colors):
        x_pos = x_start_cbar + i * segment_width
        rect = Rectangle((x_pos, y_position), segment_width, colorbar_height,
                    facecolor=color, edgecolor='none', alpha=0.85)
        ax.add_patch(rect)

    # Add border around the colorbar
    border_rect = Rectangle((x_start_cbar, y_position), 
                       x_end_cbar - x_start_cbar, colorbar_height,
                       facecolor='none', edgecolor='black', linewidth=0.5)
    ax.add_patch(border_rect)

    # CREATE PURPLE COLORBAR (IMMEDIATELY AFTER BLUE ONE)
    y_position_purple = y_position - 0.03  # Position just below blue colorbar
    
    # Define purple colorbar colors (same pattern as blue)
    purple_colorbar_colors = [
        # Dark purple to light purple (25th to 50th percentile)
        purple_colors[0], purple_colors[1], purple_colors[2], purple_colors[3], purple_colors[4],
        purple_colors[5], purple_colors[6], purple_colors[7], purple_colors[8], purple_colors[9],
        # Light purple to dark purple (50th to 75th percentile) - mirrored
        purple_colors[9], purple_colors[8], purple_colors[7], purple_colors[6], purple_colors[5],
        purple_colors[4], purple_colors[3], purple_colors[2], purple_colors[1], purple_colors[0]
    ]
    
    # Create purple colorbar segments
    n_segments_purple = len(purple_colorbar_colors)
    segment_width_purple = (x_end_cbar - x_start_cbar) / n_segments_purple

    for i, color in enumerate(purple_colorbar_colors):
        x_pos = x_start_cbar + i * segment_width_purple
        rect_purple = Rectangle((x_pos, y_position_purple), segment_width_purple, colorbar_height,
                        facecolor=color, edgecolor='none', alpha=0.85)
        ax.add_patch(rect_purple)

    # Add border around purple colorbar
    border_rect_purple = Rectangle((x_start_cbar, y_position_purple), 
                           x_end_cbar - x_start_cbar, colorbar_height,
                           facecolor='none', edgecolor='black', linewidth=0.5)
    ax.add_patch(border_rect_purple)

    # Add tick marks and labels at key quantile positions
    all_tick_positions = list(range(21))  # 0, 1, 2, ..., 20 (21 positions total)


    # Add small tick marks at every position for PURPLE colorbar  
    for pos_idx in all_tick_positions:
        x_pos = x_start_cbar + (pos_idx / 20) * (x_end_cbar - x_start_cbar)
        ax.plot([x_pos, x_pos], [y_position_purple, y_position_purple - 0.01], 
            color='black', linewidth=0.4, alpha=0.7)

    key_positions = [0, 4, 10, 16, 20]  # indices for key quantiles
    key_labels = ['25th', '35th', '50th', '65th', '75th']

    for pos_idx, label in zip(key_positions, key_labels):
        x_pos = x_start_cbar + (pos_idx / 20) * (x_end_cbar - x_start_cbar)

        ax.plot([x_pos, x_pos], [y_position_purple, y_position_purple - 0.015], 
            color='black', linewidth=0.8)
        # Add labels with larger font and no rotation for better readability
        ax.text(x_pos, y_position - 0.07, label, 
           ha='center', va='top', fontsize=10)


    # Add a label for the colorbar
    ax.text((x_start_cbar + x_end_cbar)/2, y_position + colorbar_height, 
        'Percentiles (25th-75th)', ha='center', va='bottom', fontsize=12)

    plt.tight_layout()
    plt.savefig('./p_vals_graphcast_hres.pdf')


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--input_dir', default = './results_testing/', type=str, help='Output file path')
    args = parser.parse_args()
    input_dir = args.input_dir
    forecast_name = ['graphcast_ifs', 'ifs_hres']
    plot_pvals(forecast_name, input_dir)
    #input_dir = '/Volumes/My Passport for Mac/cma_testing/results_timeseries/vals/'
    

if __name__ == "__main__":
    main()
