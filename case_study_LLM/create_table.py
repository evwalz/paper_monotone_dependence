import os
import sys
import numpy as np
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from scipy.stats import norm


def create_table(input_dir='./Rank-Calibration/'):
    """
    Create a table of results for either CMA or ERCE metrics.
    
    Args:
        input_dir (str): Directory containing the JSON files. Defaults to RANK_CALIBRATION_PATH/stats
    """
    end = 'stat_test.json'

    # Define the column names based on the metric type
    col_table = ['ecc_u_agreement_cma', 
                    'degree_u_agreement_cma', 
                    'spectral_u_agreement_cma',
                    'unnormalized_nll_all_cma', 
                    'entropy_unnormalized_cma']

    
    # Map method names to column indices
    method_to_col = {
        'ecc_u_agreement': 0,
        'degree_u_agreement': 1,
        'spectral_u_agreement': 2,
        'unnormalized_nll_all': 3,
        'entropy_unnormalized': 4
    }
        
    table_vals = np.zeros((12*3, len(col_table)))
    subscripts = [['' for _ in range(len(col_table))] for _ in range(12*3)]
    
    r = 0
    for model in ['Llama-2-7b-hf', 'Llama-2-7b-chat-hf', 'gpt-3.5-turbo']:
        if model == 'gpt-3.5-turbo':
            temp = '1.0'
            col_table_sel = col_table.copy()
        else:
            temp = '0.6'
            col_table_sel = col_table[0:5]
        
        for data in ['nq-open', 'squad', 'triviaqa']:
            for metric in ['bert_similarity', 'meteor', 'rouge', 'rouge1']:
                # Load stat_test file
                file_path = os.path.join(input_dir, 
                                       f'{model}_{data}_{temp}_{metric}_{end}')
                
                if not os.path.exists(file_path):
                    print(f"Warning: File not found: {file_path}")
                    r = r + 1
                    continue
                    
                scores = json.load(open(file_path))
                k = 0
                for name in col_table_sel:
                    if name in scores:
                        table_vals[r, k] = scores[name]
                    else:
                        table_vals[r, k] = np.nan
                    k = k + 1
                
                # Load pairwise_test file for subscripts
                pairwise_file = os.path.join(input_dir,
                                            f'{model}_{data}_{temp}_{metric}_pairwise_test.json')
                
                if os.path.exists(pairwise_file):
                    pairwise_scores = json.load(open(pairwise_file))
                    
                    # Calculate significance
                    estimated_var = (pairwise_scores['var_best'] + 
                                   pairwise_scores['var_second_best'] - 
                                   2*pairwise_scores['cov'])
                    z_stat = ((pairwise_scores['best_cma'] - pairwise_scores['second_best_cma']) / 
                             np.sqrt(estimated_var))
                    p = 1 - norm.cdf(z_stat)
                    is_significant = (p < 0.01)
                    
                    # Get the column index for the best method
                    best_method = pairwise_scores['best_method']
                    if best_method in method_to_col:
                        col_idx = method_to_col[best_method]
                        subscripts[r][col_idx] = '+' if is_significant else '0'
                else:
                    print(f"Warning: Pairwise file not found: {pairwise_file}")
                
                r = r + 1
    
    return np.round(table_vals, 3), subscripts


def create_formatted_table(table_vals, subscripts, output_file='table_single.pdf'):
    """
    Create a nicely formatted table with the specified structure and save it as a PDF.
    
    Args:
        table_vals (np.ndarray): The table values to format
        subscripts (list): List of lists containing subscript symbols for each cell
        output_file (str): Path to save the PDF file
    """
    # Create formatted cell text with subscripts
    cell_text = []
    for i in range(len(table_vals)):
        row = []
        for j in range(len(table_vals[i])):
            val = table_vals[i, j]
            if np.isnan(val):
                row.append('')
            else:
                subscript = subscripts[i][j]
                if subscript:
                    row.append(f'{val:.3f}$_{{{subscript}}}$')
                else:
                    row.append(f'{val:.3f}')
        cell_text.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(cell_text, columns=['$U_{Ecc}$', '$U_{Deg}$', '$U_{EigV}$', '$U_{NLL}$', '$U_{SE}$'])

    # Add model, dataset, and correctness columns
    models = ['Llama-2', 'Llama-2-chat', 'GPT-3.5']
    datasets = ['nq-open', 'squad', 'triviaqa']
    correctness = ['bert', 'meteor', 'rougeL', 'rouge1']
    
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
    print(f"Table saved to {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank_calibration_path', type=str, required=True,
                        help='Base path for rank calibration data')
    parser.add_argument('--input_dir', type=str, default=None,
                       help='Directory containing the JSON files. Defaults to RANK_CALIBRATION_PATH/stats_test')
    parser.add_argument('--output_table', type=str, default='table_single.pdf',
                       help='Path to save the output PDF table')

    args = parser.parse_args()

    if args.input_dir is None:
        args.input_dir = os.path.join(args.rank_calibration_path, 'stats_test')
    
    table_vals, subscripts = create_table(args.input_dir)
    create_formatted_table(table_vals, subscripts, args.output_table)
