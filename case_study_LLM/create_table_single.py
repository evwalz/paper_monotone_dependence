import os
import sys
import numpy as np
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns


def create_table(metric_type='cma', input_dir = './Rank-Calibration/'):
    """
    Create a table of results for either CMA or ERCE metrics.
    
    Args:
        metric_type (str): Either 'cma' or 'erce' to specify which metric to visualize
        input_dir (str): Directory containing the JSON files. Defaults to RANK_CALIBRATION_PATH/stats
        use_single (bool): If True, expect single result files instead of bootstrap results
    """
    # Determine file suffix based on whether using single results
    end = 'cma_single.json'
    
    # Define the column names based on the metric type
    if metric_type == 'cma':
        col_table = ['ecc_u_agreement_cma', 
                    'degree_u_agreement_cma', 
                    'spectral_u_agreement_cma',
                    'unnormalized_nll_all_cma', 
                    'entropy_unnormalized_cma',
                    'verbalized_cma']
    else:  # erce
        col_table = ['ecc_u_agreement_erce', 
                    'degree_u_agreement_erce', 
                    'spectral_u_agreement_erce',
                    'unnormalized_nll_all_erce', 
                    'entropy_unnormalized_erce',
                    'verbalized_erce']
        
    table_vals = np.zeros((12*3, len(col_table)))
    r = 0
    for model in ['Llama-2-7b-hf', 'Llama-2-7b-chat-hf', 'gpt-3.5-turbo']:
        if model == 'gpt-3.5-turbo':
            temp = '1.0'
            col_table_sel = col_table.copy()
        else:
            temp = '0.6'
            col_table_sel = col_table[0:5]
        for data in ['nq-open','squad','triviaqa']:
            for metric in ['bert_similarity','meteor', 'rouge', 'rouge1']:
                file_path = os.path.join(input_dir, 
                                       f'{model}_{data}_{temp}_{metric}_{end}')
                    
                if not os.path.exists(file_path):
                    print(f"Warning: File not found: {file_path}")
                    continue
                    
                scores = json.load(open(file_path))
                k = 0
                for name in col_table_sel:

                        # For single results, directly access the metric value
                    if name in scores:
                        table_vals[r, k] = scores[name]
                    else:
                        table_vals[r, k] = np.nan

                    k = k+1
                r = r+1
    
    # Set verbalized column to NaN for non-GPT models (as in original)
    table_vals[0:24, -1] = np.nan
    
    return np.round(table_vals, 3)

def create_formatted_table(table_vals,  output_file='table_single.pdf'):
    """
    Create a nicely formatted table with the specified structure and save it as a PDF.
    
    Args:
        table_vals (np.ndarray): The table values to format
        metric_type (str): Either 'cma' or 'erce' to specify which metric to visualize
        output_file (str): Path to save the PDF file
    """
    # Create a DataFrame with the table values
    df = pd.DataFrame(table_vals, columns=['$U_{Ecc}$', '$U_{Deg}$', '$U_{EigV}$', '$U_{NLL}$', '$U_{SE}$', '$C_{Verb}$'])
    df = df.fillna('')

    # Add model, dataset, and correctness columns
    models = ['Llama-2', 'Llama-2-chat', 'GPT-3.5']
    datasets = ['nq-open', 'squad', 'triviaqa']
    correctness = ['ber', 'meteor', 'rougeL', 'rouge1']
    
    # Create the model column (12 cells per model)
    model_col = []
    for model in models:
        model_col.extend([model] * 12)
    
    # Create the dataset column (4 cells per dataset, repeated for each model)
    dataset_col = []
    for _ in models:
        for dataset in datasets:
            dataset_col.extend([dataset] * 4)
    
    # Create the correctness column (repeated 3 times for each model)
    correctness_col = correctness * 9
    
    # Add the new columns to the DataFrame
    df.insert(0, 'correctness', correctness_col)
    df.insert(0, 'dataset', dataset_col)
    df.insert(0, 'model', model_col)
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Create the table
    table = ax.table(cellText=df.values,
                    colLabels=df.columns,
                    cellLoc='center',
                    loc='center',
                    colColours=['lightgray']*len(df.columns))
    
    # Adjust table properties
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Save to PDF
    with PdfPages(output_file) as pdf:
        pdf.savefig(fig, bbox_inches='tight')
    
    plt.close()

    
    #print(f"Scatter plot saved to {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank_calibration_path', type=str, required=True,
                        help='Base path for rank calibration data')
    parser.add_argument('--metric', type=str, default='cma',
                       choices=['cma', 'erce'],
                       help='Metric to visualize (CMA or ERCE)')
    parser.add_argument('--input_dir', type=str, default=None,
                       help='Directory containing the JSON files. Defaults to RANK_CALIBRATION_PATH/stats_test')
    parser.add_argument('--output_table', type=str, default='table_single.pdf',
                       help='Path to save the output PDF table')

    args = parser.parse_args()

    if args.input_dir is None:
        args.input_dir = os.path.join(args.rank_calibration_path, 'stats_test')
    
    table = create_table(args.metric, args.input_dir)
    create_formatted_table(table, args.output_table)