import os
import json
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple, Set
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import scienceplots
plt.style.use('science')

def load_jsonl_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Load data from a JSONL file.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        List of dictionaries containing the data
    """
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning(f"Could not parse line in {file_path}: {line[:50]}...")
    return data


def extract_model_name(file_path: str) -> str:
    """
    Extract the model name from a file path.
    This extracts from patterns like *_model_name.jsonl
    
    Args:
        file_path: Path to the file
        
    Returns:
        Model name
    """
    file_name = os.path.basename(file_path)
    parts = file_name.split('_')
    
    # The model name should be the last part before the extension
    if len(parts) > 1:
        model_name = parts[-1].replace('.jsonl', '')
        return model_name
    
    return "unknown"


# exclude_models = ["deepseek-r1", "deepseek-r1-0528"]

def analyze_token_usage(input_dir: str) -> Tuple[Dict[str, Dict[str, Dict[str, List[int]]]], Set[str]]:
    """
    Analyze token usage across different datasets, formats, and models.
    
    Args:
        input_dir: Directory containing dataset folders
        
    Returns:
        Dictionary mapping dataset -> format -> model -> list of token counts
        Set of dataset names
    """
    # Structure: {dataset: {format: {model: [token_counts]}}}
    token_usage = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    datasets = set()
    
    # Find all dataset directories
    for item in os.listdir(input_dir):
        item_path = os.path.join(input_dir, item)
        if os.path.isdir(item_path) and ('_mcq' in item or '_free' in item):
            # Extract dataset name and format
            if '_mcq' in item:
                dataset = item.replace('_mcq', '')
                format_type = 'mcq'
            elif '_free' in item:
                dataset = item.replace('_free', '')
                format_type = 'free'
            else:
                continue
                
            datasets.add(dataset)
            logger.info(f"Found dataset: {dataset}, format: {format_type}")
            
            # Process files in this dataset directory
            if dataset == 'mmlu_pro':
                item_path = os.path.join(item_path, 'stratified_sample')
                
            # Process files in this dataset directory
            for file_or_dir in os.listdir(item_path):
                file_or_dir_path = os.path.join(item_path, file_or_dir)
                
                # Case 1: Direct JSONL file in the dataset directory
                if os.path.isfile(file_or_dir_path) and file_or_dir.endswith('.jsonl') and 'samples_' in file_or_dir:
                    file_path = file_or_dir_path
                    logger.info(f"Processing file: {file_path}")
                    
                    # Extract model name from the file path
                    model_name = extract_model_name(file_path)
                    if model_name == 'think':
                        continue
                    
                    # Load JSONL data
                    data = load_jsonl_file(file_path)
                    
                    # Process each record
                    for item in data:
                        if "completion_tokens" in item:
                            # Get the completion tokens
                            completion_tokens = item["completion_tokens"]
                            
                            # Handle if completion_tokens is a list
                            if isinstance(completion_tokens, list):
                                completion_tokens = completion_tokens[0]
                            
                            # Add to token counts list
                            token_usage[dataset][format_type][model_name].append(int(completion_tokens))
                
                # Case 2: Model directory with samples.jsonl file
                elif os.path.isdir(file_or_dir_path):
                    # Process files in this dataset directory
                    if dataset == 'mmlu_pro' or dataset == 'gpqa_diamond':
                        continue 
                    
                    model_name = file_or_dir  # The folder name is the model name
                    if model_name == 'think':
                        continue
                    samples_file = os.path.join(file_or_dir_path, 'samples.jsonl')
                    
                    if os.path.isfile(samples_file):
                        logger.info(f"Processing model folder: {model_name}, file: {samples_file}")
                        
                        # Load JSONL data
                        data = load_jsonl_file(samples_file)
                        
                        # Process each record
                        for item in data:
                            if "completion_tokens" in item:
                                # Get the completion tokens
                                completion_tokens = item["completion_tokens"]
                                
                                # Handle if completion_tokens is a list
                                if isinstance(completion_tokens, list):
                                    completion_tokens = completion_tokens[0]
                                
                                # Add to token counts list
                                token_usage[dataset][format_type][model_name].append(int(completion_tokens))
    
    return token_usage, datasets


def calculate_statistics(token_counts: List[int]) -> Tuple[float, float]:
    """
    Calculate mean and standard error for a list of token counts.
    
    Args:
        token_counts: List of token counts
        
    Returns:
        Tuple of (mean, standard_error)
    """
    if not token_counts:
        return 0, 0
    
    mean = np.mean(token_counts)
    std_err = np.std(token_counts, ddof=1) / np.sqrt(len(token_counts))
    return mean, std_err


def save_statistics_to_file(token_usage: Dict[str, Dict[str, Dict[str, List[int]]]], output_dir: str) -> None:
    """
    Save token usage statistics to files.
    
    Args:
        token_usage: Dictionary with token usage data
        output_dir: Directory to save output files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for dataset, formats in token_usage.items():
        for format_type, models in formats.items():
            output_file = os.path.join(output_dir, f"{dataset}_{format_type}_stats.txt")
            
            with open(output_file, 'w') as f:
                f.write(f"Token usage statistics for {dataset} - {format_type}\n")
                f.write("=" * 60 + "\n")
                f.write(f"{'Model':<30} {'Mean Tokens':<15} {'Std Error':<15} {'Sample Count':<15}\n")
                f.write("-" * 60 + "\n")
                
                for model, token_counts in models.items():
                    if token_counts:
                        mean, std_err = calculate_statistics(token_counts)
                        f.write(f"{model:<30} {mean:<15.2f} {std_err:<15.2f} {len(token_counts):<15}\n")
                
            logger.info(f"Saved statistics to {output_file}")


def create_bar_plots(token_usage: Dict[str, Dict[str, Dict[str, List[int]]]], output_dir: str) -> None:
    """
    Create bar plots for each dataset comparing MCQ and free response formats.
    Only include models that have data for both MCQ and free response formats.
    
    Args:
        token_usage: Dictionary with token usage data
        output_dir: Directory to save output plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Define colors for the plots
    free_color = '#3498db'  # Blue
    mcq_color = '#e74c3c'  # Red
    
    for dataset, formats in token_usage.items():
        # Skip if either format is missing
        if 'mcq' not in formats or 'free' not in formats:
            logger.warning(f"Dataset {dataset} is missing either MCQ or free response format, skipping plot")
            continue
            
        # Get models that exist in both formats
        mcq_models = set(formats['mcq'].keys())
        free_models = set(formats['free'].keys())
        common_models = sorted(mcq_models.intersection(free_models))
        
        if not common_models:
            logger.warning(f"No common models found for both formats in dataset {dataset}")
            continue
            
        # Prepare data for plotting
        x = np.arange(len(common_models))
        width = 0.35
        
        mcq_means = []
        mcq_errors = []
        free_means = []
        free_errors = []
        
        for model in common_models:
            # MCQ format
            mean, std_err = calculate_statistics(formats['mcq'][model])
            mcq_means.append(mean)
            mcq_errors.append(std_err)
                
            # Free format
            mean, std_err = calculate_statistics(formats['free'][model])
            free_means.append(mean)
            free_errors.append(std_err)
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Plot bars
        plt.bar(x - width/2, mcq_means, width, label='MCQ', color=mcq_color, alpha=0.8, yerr=mcq_errors, capsize=5)
        plt.bar(x + width/2, free_means, width, label='Free Response', color=free_color, alpha=0.8, yerr=free_errors, capsize=5)
        
        # Add labels and title
        plt.xlabel('Models', fontsize=18)
        plt.ylabel('Mean Response Tokens', fontsize=18)
        # plt.title(f'Token Usage Comparison for {dataset}', fontsize=14)
        plt.xticks(x, common_models, rotation=45, ha='right', fontsize=16)
        
        # Set y-axis to log scale
        plt.yscale('log')
        
        # Set y-ticks to powers of 2
        y_min = min(min(mcq_means), min(free_means))
        y_max = max(max(mcq_means), max(free_means))
        
        # Find the nearest powers of 2 that encompass the data
        min_power = int(np.floor(np.log2(y_min)))
        max_power = int(np.ceil(np.log2(y_max)))
        
        # Create y-ticks as powers of 2
        y_ticks = [2**i for i in range(min_power, max_power)]
        y_tick_labels = [f'$2^{{{i}}}$' for i in range(min_power, max_power)]
        
        plt.yticks(y_ticks, y_tick_labels, fontsize=16)
        
        plt.legend(fontsize=18, frameon=True)
        
        # Adjust layout and save
        plt.tight_layout()
        output_file = os.path.join(output_dir, f"{dataset}_token_comparison.pdf")
        plt.savefig(output_file, dpi=300)
        plt.close()
        
        logger.info(f"Created plot for {dataset} at {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Analyze token usage across datasets, formats, and models")
    parser.add_argument("--input_dir", default="/fast/nchandak/qaevals/judge_outputs/", help="Directory containing dataset folders")
    parser.add_argument("--output_dir", default="plots/", help="Directory to save results")
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.isdir(args.input_dir):
        logger.error(f"Input directory {args.input_dir} does not exist")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Analyze token usage
    token_usage, datasets = analyze_token_usage(args.input_dir)
    
    if not datasets:
        logger.error("No valid datasets found")
        return
    
    # Save statistics to files
    # save_statistics_to_file(token_usage, args.output_dir)
    
    # Create bar plots
    create_bar_plots(token_usage, args.output_dir)
    
    logger.info(f"Analysis complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
