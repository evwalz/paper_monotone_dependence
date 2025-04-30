import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from helper import get_rmse, calculate_improvement_over_climatology, get_acc, get_seeps, get_cpa, get_cma, calculate_improvement_over_climatology_corr
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches
# Define model colors
color_codes = sns.color_palette("colorblind", 6)
model_color_map = {
    'Persistence': '#9467bd',
    'Climatology': color_codes[3],
    'GraphCast': color_codes[0],
    'IFS Mean': color_codes[1],
    'IFS HRES': color_codes[2]
}

def plot_improvement(improvement_results, output_dir, metric = 'rmse'):
    """
    Plot improvement percentage over climatology for different models across latitudes.
    
    Args:
        improvement_results (dict): Dictionary containing model name to improvement percentage mapping
        output_dir (str): Directory path where the plot will be saved
    """
    plt.figure(figsize=(10, 6))
    
    # Create latitude array (assuming 91 latitudes from -90 to 90)
    lats = np.arange(-90, 90.2, 0.25)
    
    for model_name, improvement in improvement_results.items():
        plt.plot(lats[8:-9], improvement[8:-9], label=model_name, color=model_color_map[model_name])

    if metric == 'rmse':
        plt.ylabel('RMSE Improvement (%)')
        output_path = os.path.join(output_dir, 'rmse_improvement_over_climatology.pdf')
    elif metric == 'cma_cpa':
        plt.ylabel('Score Improvement (%)')
        output_path = os.path.join(output_dir, 'cma_cpa_improvement_over_climatology.pdf')
    else:
        raise ValueError('metric not defined')
    
    plt.xlabel('Latitude (degrees)') 
    plt.axvline(0, color='black', linestyle='--', alpha=0.3)  # Mark the equator
    #plt.axhline(0, linestyle='--', label = 'Climatology', color = model_color_map['Climatology'])    
    plt.grid(True)
    plt.legend()
    #plt.xlim(improvement.latitude.values[2*4], improvement.latitude.values[-4*2])
    plt.xticks([-80, -60, -40, -20, 0, 20, 40, 60, 80], 
           [ '80°S', '60°S','40°S' ,'20°S', 'Equator', 
            '20°N', '40°N', '60°N', '80°N'])

    plt.tight_layout()
    
    # Save the plot
    
    plt.savefig(output_path)
    plt.close()
    
    print(f"plot saved to: {output_path}")


def plot_improvement_both(improvement_results1, improvement_results2, output_dir):
    """
    Plot improvement percentage over climatology for different models across latitudes.
    
    Args:
        improvement_results (dict): Dictionary containing model name to improvement percentage mapping
        output_dir (str): Directory path where the plot will be saved
    """
    plt.figure(figsize=(10, 6))
    
    # Create latitude array (assuming 91 latitudes from -90 to 90)
    lats = np.arange(-90, 90.2, 0.25)
    
    for model_name, improvement in improvement_results1.items():
        plt.plot(lats[8:-9], improvement[8:-9], label=model_name, color=model_color_map[model_name])

    for model_name, improvement in improvement_results2.items():
        plt.plot(lats[8:-9], improvement[8:-9], linestyle = '--', label=model_name, color=model_color_map[model_name])

    plt.ylabel('Improvement (%)')
    output_path = os.path.join(output_dir, 'cma_cpa_improvement_over_climatology.pdf')
    
    plt.xlabel('Latitude (degrees)') 
    plt.axvline(0, color='black', linestyle='--', alpha=0.3)  # Mark the equator
    #plt.axhline(0, linestyle='--', label = 'Climatology', color = model_color_map['Climatology'])    
    plt.grid(True)


    legend_elements = []

    # Add a line for CMA/CPA explanation
    legend_elements.append(Line2D([0], [0], color='black', lw=2, linestyle='-', label='CMA Improvement'))
    legend_elements.append(Line2D([0], [0], color='black', lw=2, linestyle='--', label='CPA Improvement'))

    # Add a spacer/separator in the legend
    legend_elements.append(Line2D([0], [0], color='none', label=''))

    # Add color entries for each model
    for model_name in improvement_results1.keys():
        legend_elements.append(mpatches.Patch(color=model_color_map[model_name], label=model_name))

    plt.legend(handles=legend_elements, 
           loc='best',
           frameon=True,
           framealpha=0.7)
    #plt.xlim(improvement.latitude.values[2*4], improvement.latitude.values[-4*2])
    plt.xticks([-80, -60, -40, -20, 0, 20, 40, 60, 80], 
           [ '80°S', '60°S','40°S' ,'20°S', 'Equator', 
            '20°N', '40°N', '60°N', '80°N'])

    plt.tight_layout()
    
    # Save the plot
    
    plt.savefig(output_path)
    plt.close()
    
    print(f"plot saved to: {output_path}")


def plot_scores(scores, output_dir, metric = 'acc'):
    """
    Plot improvement percentage over climatology for different models across latitudes.
    
    Args:
        improvement_results (dict): Dictionary containing model name to improvement percentage mapping
        output_dir (str): Directory path where the plot will be saved
    """
    plt.figure(figsize=(10, 6))
    
    # Create latitude array (assuming 91 latitudes from -90 to 90)
    lats = np.arange(-90, 90.2, 0.25)
    
    for model_name, score in scores.items():
        if metric == 'acc':
            plt.plot(lats[8:-9], score[8:-9], label=model_name, color=model_color_map[model_name])
        else:
            plt.plot(lats[8:-9], 1-score[8:-9], label=model_name, color=model_color_map[model_name])
            
        

    if metric == 'acc':
        plt.ylabel('ACC')
        output_path = os.path.join(output_dir, 'acc.pdf')
    elif metric == 'seeps':
        plt.ylabel('1-SEEPS')
        output_path = os.path.join(output_dir, 'one_minus_seeps.pdf')
        plt.ylim(-0.15, 1)
        plt.axhline(0, linestyle='-', color='black', alpha=0.3)    

    else:
        raise ValueError('metric not defined')
    
    plt.xlabel('Latitude (degrees)') 
    plt.axvline(0, color='black', linestyle='--', alpha=0.3)  # Mark the equator
    #plt.axhline(0, linestyle='--', label = 'Climatology', color = model_color_map['Climatology'])    
    plt.grid(True)
    plt.legend()
    #plt.xlim(improvement.latitude.values[2*4], improvement.latitude.values[-4*2])
    plt.xticks([-80, -60, -40, -20, 0, 20, 40, 60, 80], 
           [ '80°S', '60°S','40°S' ,'20°S', 'Equator', 
            '20°N', '40°N', '60°N', '80°N'])

    plt.tight_layout()
    
    # Save the plot
    
    plt.savefig(output_path)
    plt.close()
    
    print(f"plot saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Generate RMSE and improvement over climatology plots')
    parser.add_argument('--input_dir', type=str, 
                       default='./rmse_results/',
                       help='Directory path where plots will be saved')
    parser.add_argument('--metric', type=str, 
                       default='rmse',
                       choices=['rmse', 'acc', 'seeps', 'cma_cpa'],
                       help='visualization of rmse, acc, seeps or cma/cpa')
    
    args = parser.parse_args()
    
    # Ensure the output directory exists
    os.makedirs(args.input_dir, exist_ok=True)
    
    # Get RMSE data and calculate improvements
    if args.metric == 'rmse':
        rmse_per_lats = get_rmse(args.input_dir)
        improvement_results = calculate_improvement_over_climatology(rmse_per_lats)
        # Generate and save plots
        plot_improvement(improvement_results, args.input_dir)
    elif args.metric == 'acc':
        acc_per_lats = get_acc(args.input_dir)
        plot_scores(acc_per_lats, args.input_dir)
    elif args.metric == 'seeps':
        seeps_per_lats = get_seeps(args.input_dir)
        plot_scores(seeps_per_lats, args.input_dir, metric = 'seeps')
    else:
        cma_per_lats = get_cma(args.input_dir)
        cpa_per_lats = get_cpa(args.input_dir)
        improvement_cma_results = calculate_improvement_over_climatology_corr(cma_per_lats)
        improvement_cpa_results = calculate_improvement_over_climatology_corr(cpa_per_lats)
        plot_improvement_both(improvement_cma_results, improvement_cpa_results, args.input_dir)

if __name__ == '__main__':
    main() 