import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import argparse
from matplotlib import transforms

def label_panel(ax, letter, *, offset_left=0.0, offset_up=0.08, prefix='', postfix=')', fs=16, **font_kwds):
    """Add panel labels (a), (b), (c), (d) to subplots."""
    kwds = dict(fontsize=fs)
    kwds.update(font_kwds)
    fig = ax.figure
    trans = ax.transAxes + transforms.ScaledTranslation(-offset_left, offset_up, fig.dpi_scale_trans)
    ax.text(0, 1, prefix+letter+postfix, transform=trans, **kwds)

def load_cma_data(input_dir, variable, lead_time):
    """
    Load CMA data for a specific variable and lead time.
    Returns a dictionary with model names as keys and CMA values as values.
    """
    cma_data = {}
    model_names = ['graphcast', 'ifs_hres', 'persistence', 'climatology', 'pangu']
    
    for model in model_names:
        filename = f'cma_{model}_{variable}_lead{lead_time}h.txt'
        filepath = os.path.join(input_dir, filename)
        
        if os.path.exists(filepath):
            data = np.loadtxt(filepath)
            # Map model names to display names
            display_name = {
                'climatology': 'Climatology',
                'graphcast': 'GraphCast',
                'pangu': 'Pangu',
                'ifs_hres': 'IFS HRES',
                'persistence': 'Persistence'
            }.get(model, model)
            cma_data[display_name] = data
    
    return cma_data

def plot_cma_four_panel(input_dir, output_dir, lead_times=[24, 72], 
                        ylim=(0.43, 1.01), add_panel_labels=True):
    """
    Create 4-panel CMA plots for multiple lead times.
    
    Parameters:
    -----------
    input_dir : str
        Directory containing CMA result files
    output_dir : str
        Directory to save output plots
    lead_times : list
        List of lead times (in hours) to generate plots for
    ylim : tuple
        Y-axis limits for CMA values
    add_panel_labels : bool
        Whether to add (a), (b), (c), (d) labels to panels
    """
    
    # Set larger font sizes globally
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
    })
    
    # Color mapping for models
    color_codes = sns.color_palette("colorblind", 6)
    model_color_map = {
        'Persistence': color_codes[2],
        'Climatology': '#9467bd',
        'GraphCast': color_codes[0],
        'Pangu': color_codes[4],
        'IFS HRES': color_codes[1],
    }
    
    # Variable information
    variables = [
        ('2m_temperature', '2m Temperature'),
        ('10m_wind_speed', '10m Wind Speed'),
        ('mean_sea_level_pressure', 'Mean Sea Level Pressure'),
        ('total_precipitation_24hr', 'Total Precipitation (24hr)')
    ]
    
    # Panel labels
    panel_labels = ['a', 'b', 'c', 'd']
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a figure for each lead time
    for lead_time in lead_times:
        print(f"\nGenerating plot for {lead_time}h lead time...")
        
        # Create 2x2 subplot figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot each panel
        for idx, (var_name, var_title) in enumerate(variables):
            row = idx // 2
            col = idx % 2
            ax = axes[row, col]
            
            # Load CMA data for this variable and lead time
            cma_data = load_cma_data(input_dir, var_name, lead_time)
            
            if not cma_data:
                print(f"  Warning: No data found for {var_name} at {lead_time}h")
                ax.text(0.5, 0.5, 'No data available', 
                       ha='center', va='center', transform=ax.transAxes)
                continue
            
            # Create latitude array
            num_lats = len(next(iter(cma_data.values())))
            latitudes = np.linspace(-90, 90, num_lats)
            
            # Plot CMA values for each model
            for model_name, cma_vals in cma_data.items():
                if model_name in model_color_map:
                    ax.plot(latitudes, cma_vals, label=model_name, 
                           color=model_color_map[model_name], linewidth=2)
                else:
                    ax.plot(latitudes, cma_vals, label=model_name, linewidth=2)
            
            # Set labels and formatting
            ax.set_ylabel('CMA')
            ax.set_xlabel('Latitude (degrees)')
            ax.set_title(var_title)
            ax.grid(True, alpha=0.3)
            ax.axvline(0, color='black', linestyle='--', alpha=0.3)
            ax.set_ylim(ylim)
            
            # Only show legend in the first panel (idx == 0), positioned in lower right
            if idx == 0:
                ax.legend(loc='lower right')
            
            # Set x-axis ticks and labels
            ax.set_xticks([-80, -60, -40, -20, 0, 20, 40, 60, 80])
            ax.set_xticklabels(['80°S', '60°S', '40°S', '20°S', 'Equator', 
                               '20°N', '40°N', '60°N', '80°N'])
            
            # Add panel label if requested
            if add_panel_labels:
                label_panel(ax, panel_labels[idx])
        
        # Adjust layout and save
        plt.tight_layout()
        output_file = os.path.join(output_dir, f'cma_four_variables_lead{lead_time}h.pdf')
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"  Saved: {output_file}")
    
    print("\nAll plots completed!")

def main():
    parser = argparse.ArgumentParser(
        description='Generate 4-panel CMA plots for weather forecasts at multiple lead times'
    )
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Input directory containing CMA result files (e.g., ./fct_data/cma_results/)')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for PDF plots')
    parser.add_argument('--lead_times', type=int, nargs='+', default=[72],
                       help='Lead times in hours to generate plots for')
    parser.add_argument('--ylim', type=float, nargs=2, default=[0.43, 1.01],
                       help='Y-axis limits for CMA values (default: 0.43 1.01)')
    parser.add_argument('--no_panel_labels', action='store_true',
                       help='Disable panel labels (a), (b), (c), (d)')
    
    args = parser.parse_args()
    
    # Ensure input directory exists
    if not os.path.exists(args.input_dir):
        raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")

    
    plot_cma_four_panel(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        lead_times=args.lead_times,
        ylim=tuple(args.ylim),
        add_panel_labels=not args.no_panel_labels
    )

if __name__ == "__main__":
    main()