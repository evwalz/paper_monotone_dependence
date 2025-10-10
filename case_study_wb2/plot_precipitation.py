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

def load_metric_data(input_dir, metric_name, variable, lead_time):
    """
    Load metric data for a specific variable and lead time.
    Returns a dictionary with model names as keys and metric values as values.
    """
    metric_data = {}
    model_names = ['graphcast', 'ifs_hres', 'persistence', 'climatology', 'pangu']
    
    # Look for results in metric_results subdirectory
    results_dir = os.path.join(input_dir, f'{metric_name}_results')
    
    for model in model_names:
        filename = f'{metric_name}_{model}_{variable}_lead{lead_time}h.txt'
        filepath = os.path.join(results_dir, filename)
        
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
            metric_data[display_name] = data
    
    return metric_data

def calculate_skill_score_negative(forecast_metric, climatology_metric):
    """
    Calculate skill score for negatively oriented metrics (RMSE, SEEPS).
    Lower values are better.
    
    Formula: (climatology - forecast) / climatology * 100
    
    Positive values mean forecast is better than climatology.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        skill_score = (climatology_metric - forecast_metric) / climatology_metric * 100
        skill_score = np.where(np.isfinite(skill_score), skill_score, np.nan)
    return skill_score

def calculate_skill_score_positive(forecast_metric, climatology_metric):
    """
    Calculate skill score for positively oriented metrics (CMA, ACC).
    Higher values are better.
    
    Formula: (forecast - climatology) / climatology * 100
    
    Positive values mean forecast is better than climatology.
    """
    with np.errstate(divide='ignore', invalid='ignore'):
        skill_score = (forecast_metric - climatology_metric) / climatology_metric * 100
        skill_score = np.where(np.isfinite(skill_score), skill_score, np.nan)
    return skill_score

def plot_precipitation_metrics(input_dir, output_dir, lead_times=[24], 
                                add_panel_labels=False,
                                use_skill_scores=False):
    """
    Create 4-panel plots showing different metrics for precipitation.
    
    Parameters:
    -----------
    input_dir : str
        Base directory containing metric result subdirectories
    output_dir : str
        Directory to save output plots
    lead_times : list
        List of lead times (in hours) to generate plots for
    add_panel_labels : bool
        Whether to add (a), (b), (c), (d) labels to panels
    use_skill_scores : bool
        Whether to plot skill scores instead of raw metrics
    """
    
    # Variable is always total_precipitation_24hr
    variable = 'total_precipitation_24hr'
    
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
    
    # Model plotting order (for consistent legend)
    model_order = ['GraphCast', 'IFS HRES', 'Persistence', 'Climatology']
    
    if use_skill_scores:
        # Metric information for skill scores
        # Format: (metric_name, title, ylabel, is_positive_oriented, ylim)
        metrics = [
            ('rmse', 'RMSE Skill Score', 'RMSE SS (%)', False, (-40, 90)),
            ('seeps', 'SEEPS Skill Score', 'SEEPS SS (%)', False, (-2, 88)),
            ('acc', 'ACC', 'ACC', None, (-0.02, 1.02)),  # ACC shown as raw values
            ('cma', 'CMA Skill Score', 'CMA SS (%)', True, (-2, 88))
        ]
        filename_suffix = 'skill_scores'
    else:
        # Metric information for raw values
        # Format: (metric_name, title, ylabel, is_positive_oriented, ylim)
        metrics = [
            ('rmse', 'RMSE', 'RMSE (m)', False, None),
            ('seeps', 'SEEPS', 'SEEPS Score', False, None),
            ('acc', 'ACC', 'ACC', None, (-0.2, 1.01)),
            ('cma', 'CMA', 'CMA', True, (0.43, 1.01))
        ]
        filename_suffix = 'metrics'
    
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
        for idx, metric_info in enumerate(metrics):
            metric_name = metric_info[0]
            metric_title = metric_info[1]
            ylabel = metric_info[2]
            is_positive_oriented = metric_info[3]
            ylim = metric_info[4]
            
            row = idx // 2
            col = idx % 2
            ax = axes[row, col]
            
            # Load metric data for this lead time
            metric_data = load_metric_data(input_dir, metric_name, variable, lead_time)
            
            if not metric_data:
                print(f"  Warning: No data found for {metric_name} at {lead_time}h")
                ax.text(0.5, 0.5, 'No data available', 
                       ha='center', va='center', transform=ax.transAxes)
                continue
            
            # Create latitude array
            num_lats = len(next(iter(metric_data.values())))
            latitudes = np.linspace(-90, 90, num_lats)
            
            # Determine if we should plot skill scores for this metric
            plot_skill_score = use_skill_scores and is_positive_oriented is not None
            
            if plot_skill_score:
                # Plot skill scores
                if 'Climatology' not in metric_data:
                    print(f"  Warning: Climatology data not found for {metric_name}")
                    continue
                
                climatology_metric = metric_data['Climatology']
                
                # Plot skill scores for all models except climatology
                for model_name in model_order:
                    if model_name == 'Climatology':
                        continue  # Skip climatology for skill score plots
                    if model_name in metric_data:
                        forecast_metric = metric_data[model_name]
                        
                        # Use the appropriate skill score formula
                        if is_positive_oriented:
                            # For positively oriented metrics (CMA)
                            skill_score = calculate_skill_score_positive(
                                forecast_metric, climatology_metric
                            )
                        else:
                            # For negatively oriented metrics (RMSE, SEEPS)
                            skill_score = calculate_skill_score_negative(
                                forecast_metric, climatology_metric
                            )
                        
                        ax.plot(latitudes, skill_score, label=model_name, 
                               color=model_color_map[model_name], linewidth=2)
                
                # Add climatology line at 0 (by definition, climatology has SS=0)
                ax.axhline(0, color=model_color_map['Climatology'], linestyle='-', 
                          linewidth=2, label='Climatology')
            else:
                # Plot raw metric values for all models
                for model_name in model_order:
                    if model_name in metric_data:
                        metric_vals = metric_data[model_name]
                        ax.plot(latitudes, metric_vals, label=model_name, 
                               color=model_color_map[model_name], linewidth=2)
            
            # Set labels and formatting
            ax.set_ylabel(ylabel)
            ax.set_xlabel('Latitude (degrees)')
            ax.set_title(metric_title)
            ax.grid(True, alpha=0.3)
            ax.axvline(0, color='black', linestyle='--', alpha=0.3)
            
            # Set y-axis limits and reference lines
            if ylim is not None:
                ax.set_ylim(ylim)
            
            # Add reference lines for raw metrics
            if not plot_skill_score:
                if metric_name == 'acc':
                    ax.axhline(0, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
                elif metric_name == 'cma':
                    ax.axhline(0.5, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
            
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
        
        # Save as PDF
        output_file_pdf = os.path.join(output_dir, 
                                       f'precipitation_{filename_suffix}_lead{lead_time}h.pdf')
        plt.savefig(output_file_pdf, bbox_inches='tight', dpi=300)
        print(f"  Saved PDF: {output_file_pdf}")
        
        plt.close()
    
    print("\nAll plots completed!")

def main():
    parser = argparse.ArgumentParser(
        description='Generate 4-panel metric plots (RMSE, SEEPS, ACC, CMA) for precipitation forecasts'
    )
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Base input directory containing metric_results subdirectories')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for plots')
    parser.add_argument('--lead_times', type=int, nargs='+', default=[24],
                       help='Lead times in hours to generate plots for (default: 24)')
    parser.add_argument('--add_panel_labels', action='store_true',
                       help='Add panel labels (a), (b), (c), (d)')
    parser.add_argument('--skill_scores', action='store_true',
                       help='Plot skill scores instead of raw metric values')
    
    args = parser.parse_args()
    
    # Ensure input directory exists
    if not os.path.exists(args.input_dir):
        raise FileNotFoundError(f"Input directory does not exist: {args.input_dir}")
    
    plot_precipitation_metrics(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        lead_times=args.lead_times,
        add_panel_labels=args.add_panel_labels,
        use_skill_scores=args.skill_scores
    )

if __name__ == "__main__":
    main()