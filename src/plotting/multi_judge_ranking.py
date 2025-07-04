#!/usr/bin/env python3
import os
import csv
import json
import re
import glob
import numpy as np
import argparse
from collections import defaultdict
import pandas as pd
import matplotlib as mpl

import scienceplots
mpl.style.use(['science'])

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Compare rankings across different judge models")
    parser.add_argument("--dataset", type=str, default="gpqa_diamond", choices=["gpqa_diamond", "mmlu_pro"],
                       help="Dataset to analyze (gpqa_diamond or mmlu_pro)")
    parser.add_argument("--scale", type=str, default="rank", choices=["rank", "acc"],
                       help="Y-axis scale: 'rank' for rankings or 'acc' for accuracy values")
    parser.add_argument("--middle-judge", type=str, default="deepseek", choices=["deepseek", "llama"],
                       help="Which judge to put in the middle columns: 'deepseek' or 'llama'")
    parser.add_argument('--human-annotations-dir', type=str, default=None,
                        help='Directory containing human annotation files for rating-based filtering')
    parser.add_argument('--unique-rating-filter', type=int, nargs=2, default=[4, 5],
                        help='Range for filtering by rating_multians (min, max)')
    parser.add_argument('--specific-filter', type=int, nargs=2, default=[4, 5],
                        help='Range for filtering by rating_osq (min, max)')
    return parser.parse_args()

# Dataset-specific configurations
def get_dataset_config(dataset):
    if dataset == "gpqa_diamond":
        return {
            "scores_dir": "/is/cluster/fast/nchandak/qaevals/judge_outputs/gpqa_diamond_free/",
            "output_dir": "plots",
            "output_prefix": "gpqa_multi_judge",
            "need_model_mapping": True
        }
    elif dataset == "mmlu_pro":
        return {
            "scores_dir": "/is/cluster/fast/nchandak/qaevals/judge_outputs/mmlu_pro_free/stratified_sample/",
            "output_dir": "plots", 
            "output_prefix": "mmlu_pro_multi_judge",
            "need_model_mapping": False
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

# Model name mapping
def get_model_mapping():
    """Create mapping from benchmark_runs model names to sample files model names"""
    mapping = {
        # Claude models
        "claude-3-5-haiku-20241022": "claude-3.5-haiku",
        # GPT models
        "gpt-4o-2024-05-13": "gpt-4o",
        "gpt-4o-mini-2024-07-18": "gpt-4o-mini",
        "gpt-4.1-nano-2025-04-14": "gpt-4.1-nano",
        # DeepSeek models
        "DeepSeek-V3-0324": "deepseek-chat-v3-0324",
        "DeepSeek-R1-Distill-Llama-70B": "deepseek-r1-distill-llama-70b",
        # Llama models
        "Llama-3.3-70B-Instruct": "llama-3.3-70b-instruct",
        "Llama-4-Scout-17B-16E-Instruct": "llama-4-scout",
        "Llama-4-Maverick-17B-128E-Instruct-FP8": "llama-4-maverick",
        # Qwen models
        "qwen2.5-72b-instruct": "qwen-2.5-72b-instruct",
        "qwen3-32b": "qwen3-32b",
        # Gemini models
        "gemma-3-27b-it": "gemma-3-27b-it",
        # Mistral models
        "mistral-small-2501": "mistral-small-24b-instruct-2501",
        # Grok models
        "grok-3-mini-beta_low": "grok-3-mini-beta",
        # Wizardlm
        "WizardLM-2-8x22B": "wizardlm-2-8x22b",
    }
    return mapping

def load_jsonl_file(file_path):
    """Load data from a JSONL file."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return data

def load_human_annotations(human_annotations_dir, dataset_name):
    """Load all human annotation files from the specified directory."""
    if not human_annotations_dir:
        return {}
    
    # Get all JSONL files in the directory
    annotation_files = glob.glob(os.path.join(human_annotations_dir, "*.jsonl"))
    print(f"Found {len(annotation_files)} human annotation files: {', '.join([os.path.basename(file) for file in annotation_files])}")
    
    human_annotators = {}
    is_gpqa_diamond = (dataset_name == "gpqa_diamond")
    
    # Process each file
    for i, file_path in enumerate(annotation_files):
        annotator_id = f"human_{i+1}"
        annotations = {}
        data = load_jsonl_file(file_path)
        
        if is_gpqa_diamond:
            for item in data:
                question_id = item.get("question_id")
                models = item.get("model", [])
                rating_matches = item.get("rating_match", [])
                if not isinstance(models, list) or not isinstance(rating_matches, list):
                    continue
                for j, model in enumerate(models):
                    if j < len(rating_matches):
                        entry = dict(item)
                        entry["rating_match"] = rating_matches[j]
                        entry["model"] = model
                        annotations[(question_id, model)] = entry
        else:
            for item in data:
                question_id = item.get("question_id")
                if question_id is not None:
                    annotations[question_id] = item
        
        human_annotators[annotator_id] = annotations
        print(f"Loaded {len(annotations)} annotations from {annotator_id}")
    
    return human_annotators

def get_filtered_question_ids(human_annotators, unique_rating_filter, specific_filter, dataset_name):
    """Get question IDs that pass the rating filters."""
    if not human_annotators:
        print("No human annotations available - no filtering applied")
        return set()
    
    is_gpqa_diamond = (dataset_name == "gpqa_diamond")
    
    # Get all question IDs from the human annotators
    all_question_ids = set()
    for annotator_id, annotations in human_annotators.items():
        all_question_ids.update(annotations.keys())
    
    print(f"Total unique questions across all annotators: {len(all_question_ids)}")
    
    # Count questions that pass filters for each annotator
    filtered_ids = {}
    
    for annotator_id, annotations in human_annotators.items():
        filtered_ids[annotator_id] = set()
        
        for question_id, data in annotations.items():
            # Check unique_rating_filter (rating_multians)
            rating_multians = data.get("rating_multians")
            if rating_multians is not None:
                if not (unique_rating_filter[0] <= rating_multians <= unique_rating_filter[1]):
                    continue
            
            # Check specific_filter (rating_osq)
            rating_osq = data.get("rating_osq")
            if rating_osq is not None:
                if not (specific_filter[0] <= rating_osq <= specific_filter[1]):
                    continue
            
            # If passed all filters
            if is_gpqa_diamond:
                # For GPQA Diamond, we need to extract just the question_id from the (question_id, model) tuple
                if isinstance(question_id, tuple):
                    filtered_ids[annotator_id].add(question_id[0])
                else:
                    filtered_ids[annotator_id].add(question_id)
            else:
                filtered_ids[annotator_id].add(question_id)
        
        print(f"Annotator {annotator_id}: {len(filtered_ids[annotator_id])} questions pass filters")
    
    # Get the intersection of filtered IDs for all annotators
    annotator_keys = list(filtered_ids.keys())
    if len(annotator_keys) >= 2:
        common_filtered = filtered_ids[annotator_keys[0]].intersection(filtered_ids[annotator_keys[1]])
        print(f"Questions passing filters for both annotators: {len(common_filtered)}")
        return common_filtered
    elif len(annotator_keys) == 1:
        print(f"Using single annotator's filtered questions: {len(filtered_ids[annotator_keys[0]])}")
        return filtered_ids[annotator_keys[0]]
    else:
        print("No annotators found")
        return set()

# Read scores from JSONL files for a specific judge model
def get_scores_by_judge(scores_dir, judge_field, need_model_mapping=True, filtered_question_ids=None):
    """
    Read scores from JSONL files for a specific judge model.
    
    Args:
        scores_dir: Directory containing the JSONL files
        judge_field: Field name for the judge scores (e.g., "score_deepseek-chat-v3-0324")
        need_model_mapping: Whether to apply model name mapping
        filtered_question_ids: Set of question IDs to filter by
    
    Returns:
        Dictionary mapping model names to score statistics
    """
    judge_scores = {}
    model_mapping = get_model_mapping() if need_model_mapping else None
    
    for filename in os.listdir(scores_dir):
        if filename.startswith("samples_") and filename.endswith(".jsonl"):
            model_name = filename[len("samples_"):-len(".jsonl")]
            
            # Map to the format in benchmark_runs.csv if model mapping is needed
            if need_model_mapping:
                benchmark_model = None
                for bench_model, sample_model in model_mapping.items():
                    if sample_model == model_name:
                        benchmark_model = bench_model
                        break
                
                if not benchmark_model:
                    benchmark_model = model_name  # Use the sample name if no mapping exists
            else:
                benchmark_model = model_name  # Use the sample name directly for mmlu_pro
            
            scores = []
            file_path = os.path.join(scores_dir, filename)
            
            try:
                with open(file_path, 'r') as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            question_id = data.get("question_id")
                            
                            # Apply question ID filtering if provided
                            if filtered_question_ids is not None and question_id not in filtered_question_ids:
                                continue
                            
                            if judge_field in data and isinstance(data[judge_field], list):
                                scores.extend(data[judge_field])
                            elif judge_field in data and isinstance(data[judge_field], str):
                                scores.append(int(data[judge_field]))
                            elif judge_field in data and isinstance(data[judge_field], int):
                                scores.append(data[judge_field])
                        except json.JSONDecodeError:
                            continue
                
                if scores:
                    accuracy = np.mean(scores)
                    # Calculate standard error of the mean (SEM) instead of std dev
                    stdev = np.std(scores, ddof=1) / np.sqrt(len(scores))
                    judge_scores[benchmark_model] = {"accuracy": accuracy, "stdev": stdev, "n_samples": len(scores)}
                    print(f"{judge_field} - {benchmark_model}: {len(scores)} samples, accuracy: {accuracy:.3f}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    return judge_scores

# Compute statistics and rankings for multiple judges
def compute_multi_judge_stats_and_rankings(llama_scores, deepseek_scores, qwen_scores):
    # Combine all models that have scores from all three judges
    combined_stats = {}
    
    for model in llama_scores:
        if model in deepseek_scores and model in qwen_scores:
            combined_stats[model] = {
                "llama_acc": llama_scores[model]["accuracy"],
                "llama_stdev": llama_scores[model]["stdev"],
                "deepseek_acc": deepseek_scores[model]["accuracy"],
                "deepseek_stdev": deepseek_scores[model]["stdev"],
                "qwen_acc": qwen_scores[model]["accuracy"],
                "qwen_stdev": qwen_scores[model]["stdev"],
                "deepseek_n_samples": deepseek_scores[model]["n_samples"] if "n_samples" in deepseek_scores[model] else 0,
            }
    
    # Calculate rankings
    llama_ranking = sorted(combined_stats.keys(), key=lambda x: combined_stats[x]["llama_acc"], reverse=True)
    deepseek_ranking = sorted(combined_stats.keys(), key=lambda x: combined_stats[x]["deepseek_acc"], reverse=True)
    qwen_ranking = sorted(combined_stats.keys(), key=lambda x: combined_stats[x]["qwen_acc"], reverse=True)
    
    # Assign ranks
    for i, model in enumerate(llama_ranking):
        combined_stats[model]["llama_rank"] = i + 1
    
    for i, model in enumerate(deepseek_ranking):
        combined_stats[model]["deepseek_rank"] = i + 1
    
    for i, model in enumerate(qwen_ranking):
        combined_stats[model]["qwen_rank"] = i + 1
        combined_stats[model]["llama_deepseek_rank_diff"] = combined_stats[model]["llama_rank"] - combined_stats[model]["deepseek_rank"]
        combined_stats[model]["deepseek_qwen_rank_diff"] = combined_stats[model]["deepseek_rank"] - combined_stats[model]["qwen_rank"]
    
    return combined_stats

def save_to_csv(stats, output_path):
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    data = []
    for model, model_stats in stats.items():
        data.append({
            "model": model,
            "llama_acc": model_stats["llama_acc"],
            "llama_stdev": model_stats["llama_stdev"],
            "deepseek_acc": model_stats["deepseek_acc"],
            "deepseek_stdev": model_stats["deepseek_stdev"],
            "qwen_acc": model_stats["qwen_acc"],
            "qwen_stdev": model_stats["qwen_stdev"],
            "deepseek_n_samples": model_stats["deepseek_n_samples"],
            "llama_rank": model_stats["llama_rank"],
            "deepseek_rank": model_stats["deepseek_rank"],
            "qwen_rank": model_stats["qwen_rank"],
            "llama_deepseek_rank_diff": model_stats["llama_deepseek_rank_diff"],
            "deepseek_qwen_rank_diff": model_stats["deepseek_qwen_rank_diff"]
        })
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")
    
    return df

# Custom display names for models
print_names = {
    'grok-3-mini-beta': 'Grok 3 Mini',
    'grok-3-mini-beta_low': 'Grok 3 Mini',
    'claude-3.5-haiku': 'Claude 3.5 Haiku',
    'DeepSeek-V3-0324': 'DeepSeek V3',
    'deepseek-chat-v3-0324': 'DeepSeek V3',
    'DeepSeek-R1': 'DeepSeek R1',
    'llama-3.3-70b-instruct': 'Llama 3.3 70B',
    'Llama-3.3-70B-Instruct': 'Llama 3.3 70B',
    'GPT-4o': 'GPT 4o',
    'gpt-4o-2024-05-13': 'GPT 4o',
    'gpt-4o': 'GPT 4o',
    'o4-mini-high': 'o4-mini-high',
    'llama-4-maverick': 'Llama 4 Maverick',
    'Llama-4-Maverick-17B-128E-Instruct-FP8': 'Llama 4 Maverick',
    'deepseek-r1-distill-llama-70b': 'R1 Distill Llama 3.3 70B',
    'DeepSeek-R1-Distill-Llama-70B': 'R1 Distill Llama 3.3 70B',
    'llama-4-scout': 'Llama 4 Scout',
    'Llama-4-Scout-17B-16E-Instruct': 'Llama 4 Scout',
    'qwen2.5-72b-instruct': 'Qwen 2.5 72B',
    'qwen3-32b_non_think': 'Qwen 3 32B',
    'qwen3-32b': 'Qwen 3 32B',
    'phi-4': 'Phi 4',
    'qwen-2.5-72b-instruct': 'Qwen 2.5 72B',
    'gemma-3-27b-it': 'Gemma 3 27B',
    'gemini-2.5-pro-preview': 'Gemini 2.5 Pro',
    'gpt-4.1-nano': 'GPT 4.1 Nano',
    'gpt-4.1-nano-2025-04-14': 'GPT 4.1 Nano',
    'mistral-small-24b-instruct-2501': 'Mistral Small 24B',
    'mistral-small-2501': 'Mistral Small 24B',
    "mistral-medium-3": "Mistral Medium 3",
    'gpt-4o-mini-2024-07-18': 'GPT 4o Mini',
    'wizardlm-2-8x22b': 'WizardLM 2 8x22B',
    'WizardLM-2-8x22B': 'WizardLM 2 8x22B',
    'claude-3-5-haiku-20241022': 'Claude 3.5 Haiku',
}

# Models to exclude from plots for each dataset
exclude_models = {
    "gpqa_diamond": [
        "gemini-2.5-flash-preview",
        # "mistral-medium-3",
        
        "o4-mini-high",
        "gemini-2.5-pro-preview", 
        "deepseek-r1",
        "deepseek-r1-0528",
        "DeepSeek-R1",
        "qwen-2.5-7b-instruct",
    ],
    "mmlu_pro": [
        "gemini-2.5-flash-preview",
        "qwen3-32b_old",
        # "mistral-medium-3"
    ]
}

def create_multi_judge_bump_plot(df, output_path, dataset, scale='rank', middle_judge='deepseek'):
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib.patches import Polygon
    except ImportError as e:
        print(f"Error importing visualization libraries: {e}")
        return
    
    # Debug: print DataFrame information
    print(f"DataFrame shape: {df.shape}")
    print(f"DataFrame columns: {df.columns.tolist()}")
    print(f"Full DataFrame:\n{df}")
    
    # Check if the DataFrame is empty
    if df.empty:
        print("Error: DataFrame is empty. Cannot create plot.")
        return
    
    # Check if required columns exist
    required_cols = ['model', 'llama_acc', 'llama_stdev', 'deepseek_acc', 'deepseek_stdev', 
                    'qwen_acc', 'qwen_stdev', 'llama_rank', 'deepseek_rank', 'qwen_rank']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        return

    # Make the plot wider for 4 columns
    plt.figure(figsize=(14, 8))

    # Create long-form data for plotting
    plot_data = []
    judges = ['Llama-4-Scout', 'DeepSeek-V3', 'DeepSeek-V3', 'Qwen3-4B']
    for _, row in df.iterrows():
        plot_data.append({
            'model': row['model'],
            'judge': judges[0],
            'rank': row['llama_rank'],
            'accuracy': row['llama_acc'],
            'stdev': row['llama_stdev'],
            'position': 0
        })
        plot_data.append({
            'model': row['model'],
            'judge': judges[1],
            'rank': row['deepseek_rank'],
            'accuracy': row['deepseek_acc'],
            'stdev': row['deepseek_stdev'],
            'position': 1
        })
        plot_data.append({
            'model': row['model'],
            'judge': judges[2],
            'rank': row['deepseek_rank'],
            'accuracy': row['deepseek_acc'],
            'stdev': row['deepseek_stdev'],
            'position': 2
        })
        plot_data.append({
            'model': row['model'],
            'judge': judges[3],
            'rank': row['qwen_rank'],
            'accuracy': row['qwen_acc'],
            'stdev': row['qwen_stdev'],
            'position': 3
        })

    plot_df = pd.DataFrame(plot_data)

    # Remove all styling and grid lines
    # sns.set_style("white")
    # plt.rcParams['axes.spines.left'] = False
    # plt.rcParams['axes.spines.right'] = False
    # plt.rcParams['axes.spines.top'] = False
    # plt.rcParams['axes.spines.bottom'] = False
    
    # Determine which judge is in the middle and sort accordingly
    if middle_judge == 'deepseek':
        models_by_middle_rank = df.sort_values('deepseek_rank')['model'].unique()
        middle_judge_name = 'DeepSeek v3'
        left_judge_name = 'Llama 4 Scout'
        right_judge_name = 'Qwen3 4B'
    else:  # middle_judge == 'llama'
        models_by_middle_rank = df.sort_values('llama_rank')['model'].unique()
        middle_judge_name = 'Llama 4 Scout'
        left_judge_name = 'DeepSeek v3'
        right_judge_name = 'Qwen3 4B'

    # --- COLOR MAPPING BY MIDDLE JUDGE RANK ---
    # Use a perceptually uniform colormap (Spectral or similar)
    cmap = mpl.cm.get_cmap('Spectral')

    n_models = len(models_by_middle_rank)
    # Subset only dark colors from both ends of the colormap
    n_dark_each_end = n_models // 2
    if n_models <= 2:
        # If only 1 or 2 models, just use the ends
        color_positions = np.linspace(0, 1, n_models)
    else:
        # Take half from each end, skipping the light middle
        color_positions = np.concatenate([
            np.linspace(0, 0.25, n_dark_each_end, endpoint=False),
            np.linspace(0.75, 1, n_models - n_dark_each_end)
        ])
    # Map middle judge rank (1 is best) to dark colors
    model_to_color = {}
    for i, model in enumerate(models_by_middle_rank):
        model_to_color[model] = cmap(color_positions[i])

    # Define positions for the 4 columns
    # Left, Middle-Left (name), Middle-Right (score), Right
    positions = [0.2, 0.4, 0.6, 0.8]

    # Adjust margins for the 4-column layout
    plt.subplots_adjust(left=0.35, right=0.65)

    for model in models_by_middle_rank:
        model_data = plot_df[plot_df['model'] == model].sort_values('position')
        
        # Draw lines connecting left to middle-left and middle-right to right
        # Calculate line endpoints to avoid overlap with scores
        center_x = (positions[1] + positions[2]) / 2
        left_score_x = positions[1] + 0.1 * (center_x - positions[1])
        right_score_x = positions[2] - 0.1 * (positions[2] - center_x)
        
        # Left line endpoint should be before the left score position
        left_line_end = left_score_x - 0.03
        # Right line start should be after the right score position  
        right_line_start = right_score_x + 0.03
        
        # Get data based on judge configuration
        llama_data = model_data[model_data['position'] == 0].iloc[0]
        deepseek_data = model_data[model_data['position'] == 1].iloc[0]
        qwen_data = model_data[model_data['position'] == 3].iloc[0]
        
        # Determine left, middle, and right data based on middle judge
        if middle_judge == 'deepseek':
            left_data = llama_data
            middle_data = deepseek_data
            right_data = qwen_data
        else:  # middle_judge == 'llama'
            left_data = deepseek_data
            middle_data = llama_data
            right_data = qwen_data
        
        # Check statistical significance for line bolding
        # Get the model's compact letters for each judge
        llama_letters = df[df['model'] == model]['llama_letters'].values[0] if 'llama_letters' in df.columns else ''
        deepseek_letters = df[df['model'] == model]['deepseek_letters'].values[0] if 'deepseek_letters' in df.columns else ''
        qwen_letters = df[df['model'] == model]['qwen_letters'].values[0] if 'qwen_letters' in df.columns else ''
        
        # Determine if rank changes are statistically significant
        if middle_judge == 'deepseek':
            # Left to middle: Llama to DeepSeek
            llama_primary = llama_letters[0] if llama_letters else ''
            left_to_middle_significant = llama_primary not in deepseek_letters
            # Middle to right: DeepSeek to Qwen
            deepseek_primary = deepseek_letters[0] if deepseek_letters else ''
            middle_to_right_significant = deepseek_primary not in qwen_letters
        else:  # middle_judge == 'llama'
            # Left to middle: DeepSeek to Llama
            deepseek_primary = deepseek_letters[0] if deepseek_letters else ''
            left_to_middle_significant = deepseek_primary not in llama_letters
            # Middle to right: Llama to Qwen
            llama_primary = llama_letters[0] if llama_letters else ''
            middle_to_right_significant = llama_primary not in qwen_letters
        
        color = model_to_color[model]
        
        # Left to Middle-Left (shortened line)
        if scale == 'rank':
            x_values_1 = [positions[0], left_line_end]
            y_values_1 = [left_data['rank'], middle_data['rank']]
        else:
            x_values_1 = [positions[0], left_line_end]
            y_values_1 = [left_data['accuracy'], middle_data['accuracy']]
        
        # Use thick line if statistically significant, thin if not
        if left_to_middle_significant:
            plt.plot(x_values_1, y_values_1, 'o-', linewidth=4.0, markersize=12, alpha=1.0, color=color)
        else:
            plt.plot(x_values_1, y_values_1, 'o-', linewidth=1.0, markersize=8, alpha=1.0, color=color)
        
        # Middle-Right to Right (shortened line)
        if scale == 'rank':
            x_values_2 = [right_line_start, positions[3]]
            y_values_2 = [middle_data['rank'], right_data['rank']]
        else:
            x_values_2 = [right_line_start, positions[3]]
            y_values_2 = [middle_data['accuracy'], right_data['accuracy']]
        
        # Use thick line if statistically significant, thin if not
        if middle_to_right_significant:
            plt.plot(x_values_2, y_values_2, 'o-', linewidth=4.0, markersize=12, alpha=1.0, color=color)
        else:
            plt.plot(x_values_2, y_values_2, 'o-', linewidth=1.0, markersize=8, alpha=1.0, color=color)

    ax1 = plt.gca()
    if scale == 'rank':
        ax1.invert_yaxis()
    if scale == 'rank':
        y_min, y_max = ax1.get_ylim()
        ax1.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    else:
        min_acc = min(df['llama_acc'].min(), df['deepseek_acc'].min(), df['qwen_acc'].min())
        max_acc = max(df['llama_acc'].max(), df['deepseek_acc'].max(), df['qwen_acc'].max())
        y_min = max(min_acc - 0.05, 0.0)
        y_max = 1.0
        plt.ylim(y_min, y_max)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

    # Add model names and scores
    if scale == 'rank':
        y_ticks = list(range(1, len(models_by_middle_rank) + 1))
        plt.yticks(y_ticks, ["" for _ in y_ticks])
        for i, model in enumerate(models_by_middle_rank):
            model_data = df[df['model'] == model].iloc[0]
            color = model_to_color[model]
            display_name = print_names.get(model, model)
            
            # Get ranks and scores based on middle judge configuration
            if middle_judge == 'deepseek':
                rank_left = model_data['llama_rank']
                rank_middle = model_data['deepseek_rank']
                rank_right = model_data['qwen_rank']
                acc_left = model_data['llama_acc']
                acc_middle = model_data['deepseek_acc']
                acc_right = model_data['qwen_acc']
            else:  # middle_judge == 'llama'
                rank_left = model_data['deepseek_rank']
                rank_middle = model_data['llama_rank']
                rank_right = model_data['qwen_rank']
                acc_left = model_data['deepseek_acc']
                acc_middle = model_data['llama_acc']
                acc_right = model_data['qwen_acc']
            
            # Left column: accuracy at line origination point
            plt.text(positions[0] - 0.02, rank_left, f"{acc_left:.1%}", 
                    fontsize=16, ha='right', va='center', color=color, fontweight='bold')
            
            # Center between middle columns: model name
            center_x = (positions[1] + positions[2]) / 2
            plt.text(center_x, rank_middle, f"{display_name}", 
                    fontsize=16, ha='center', va='center', color=color, fontweight='bold')
            
            # Middle-left: middle judge accuracy very close to left origination point
            left_middle_x = positions[1] + 0.1 * (center_x - positions[1])
            plt.text(left_middle_x, rank_middle, f"{acc_middle:.1%}", 
                    fontsize=16, ha='center', va='center', color=color, fontweight='bold')
            
            # Middle-right: middle judge accuracy very close to right origination point
            right_middle_x = positions[2] - 0.1 * (positions[2] - center_x)
            plt.text(right_middle_x, rank_middle, f"{acc_middle:.1%}", 
                    fontsize=16, ha='center', va='center', color=color, fontweight='bold')
            
            # Right column: accuracy at line origination point
            plt.text(positions[3] + 0.02, rank_right, f"{acc_right:.1%}", 
                    fontsize=16, ha='left', va='center', color=color, fontweight='bold')
    else:
        for i, model in enumerate(models_by_middle_rank):
            model_data = df[df['model'] == model].iloc[0]
            color = model_to_color[model]
            display_name = print_names.get(model, model)
            
            # Get ranks and scores based on middle judge configuration
            if middle_judge == 'deepseek':
                acc_left = model_data['llama_acc']
                acc_middle = model_data['deepseek_acc']
                acc_right = model_data['qwen_acc']
            else:  # middle_judge == 'llama'
                acc_left = model_data['deepseek_acc']
                acc_middle = model_data['llama_acc']
                acc_right = model_data['qwen_acc']
            
            # Left column: accuracy at line origination point
            plt.text(positions[0] - 0.06, acc_left, f"{acc_left:.1%}", 
                    fontsize=16, ha='right', va='center', color=color, fontweight='bold')
            
            # Center between middle columns: model name
            center_x = (positions[1] + positions[2]) / 2
            plt.text(center_x, acc_middle, f"{display_name}", 
                    fontsize=16, ha='center', va='center', color=color, fontweight='bold')
            
            # Middle-left: middle judge accuracy very close to left origination point
            left_middle_x = positions[1] + 0.1 * (center_x - positions[1])
            plt.text(left_middle_x, acc_middle, f"{acc_middle:.1%}", 
                    fontsize=16, ha='center', va='center', color=color, fontweight='bold')
            
            # Middle-right: middle judge accuracy very close to right origination point
            right_middle_x = positions[2] - 0.1 * (positions[2] - center_x)
            plt.text(right_middle_x, acc_middle, f"{acc_middle:.1%}", 
                    fontsize=16, ha='center', va='center', color=color, fontweight='bold')
            
            # Right column: accuracy at line origination point
            plt.text(positions[3] + 0.06, acc_right, f"{acc_right:.1%}", 
                    fontsize=16, ha='left', va='center', color=color, fontweight='bold')

    # Set title and x-axis labels
    title = ""
    if dataset == "gpqa_diamond":
        title = "Ranking changes across matchers on GPQA Diamond"
    elif dataset == "mmlu_pro":
        title = "Ranking changes across matchers on MMLU Pro"
    plt.title(f'{title}', fontsize=22, fontweight='bold')
    
    # Create column headers - show individual headers for outer columns and middle header
    column_headers = [left_judge_name, '', '', right_judge_name]
    ax1.set_xticks(positions)
    ax1.set_xticklabels(column_headers, fontsize=20, fontweight='bold')
    
    # Add middle judge header at the same level as other column headers
    mid_x = (positions[1] + positions[2]) / 2
    
    # Position the middle judge header at the same level as other column headers
    ax1.text(mid_x, 0, middle_judge_name, fontsize=20, ha='center', va='top', 
             fontweight='bold', transform=ax1.get_xaxis_transform())
    
    # Remove all axes and background elements
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.tick_params(left=False, bottom=False, right=False, top=False)
    # ax1.set_xticklabels([])
    # ax1.set_yticklabels([])
    ax1.set_facecolor('white')
    ax1.set_xlabel('')
    ax1.set_ylabel('')
    
    # Remove figure background
    # plt.gcf().patch.set_facecolor('none')
    
    plt.tight_layout(rect=[0.05, 0, 0.95, 0.95])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Multi-judge bump plot saved to {output_path}")
    plt.close()

# --- Statistically similar group and compact letter assignment logic ---
def find_statistically_similar_groups_judge(df, judge):
    """Find groups of models that are statistically similar (within standard error) for a given judge."""
    acc_col = f'{judge}_acc'
    err_col = f'{judge}_stdev'
    if acc_col not in df.columns:
        return []
    df_sorted = df.sort_values(acc_col, ascending=False).reset_index(drop=True)
    groups = []
    if len(df_sorted) == 0:
        return groups
    current_group = [0]
    top_model_acc = df_sorted.iloc[0][acc_col]
    top_model_err = df_sorted.iloc[0][err_col]
    for i in range(1, len(df_sorted)):
        current_acc = df_sorted.iloc[i][acc_col]
        current_err = df_sorted.iloc[i][err_col]
        if top_model_acc - top_model_err <= current_acc + current_err:
            current_group.append(i)
        else:
            if len(current_group) > 1:
                group_models = [df_sorted.iloc[idx]['model'] for idx in current_group]
                groups.append(group_models)
            current_group = [i]
            top_model_acc = current_acc
            top_model_err = current_err
    if len(current_group) > 1:
        group_models = [df_sorted.iloc[idx]['model'] for idx in current_group]
        groups.append(group_models)
    return groups

def assign_compact_letters_judge(df, judge):
    """Assign compact letter display to models based on statistical significance for a given judge."""
    acc_col = f'{judge}_acc'
    err_col = f'{judge}_stdev'
    letter_col = f'{judge}_letters'
    df_sorted = df.sort_values(acc_col, ascending=False).reset_index(drop=True)
    letters = {}
    for i, (_, row) in enumerate(df_sorted.iterrows()):
        model = row['model']
        primary_letter = chr(65 + i)
        letters[model] = primary_letter
    for i, (_, row_i) in enumerate(df_sorted.iterrows()):
        model_i = row_i['model']
        acc_i = row_i[acc_col]
        err_i = row_i[err_col]
        for j, (_, row_j) in enumerate(df_sorted.iterrows()):
            if i == j:
                continue
            model_j = row_j['model']
            acc_j = row_j[acc_col]
            err_j = row_j[err_col]
            if acc_i - err_i <= acc_j + err_j and acc_j - err_j <= acc_i + err_i:
                letters[model_i] += letters[model_j][0]
    for model, letter in letters.items():
        df.loc[df['model'] == model, letter_col] = letter
    return df

def main():
    args = parse_args()
    config = get_dataset_config(args.dataset)
    
    # Set default human annotations directory if not provided
    if args.human_annotations_dir is None:
        args.human_annotations_dir = f"/is/cluster/fast/nchandak/qaevals/judge_outputs/alignment_plot/annotations/{args.dataset}"
    
    # Print filter information
    print(f"Applying rating filters:")
    print(f"  Unique rating filter (rating_multians): {args.unique_rating_filter[0]}-{args.unique_rating_filter[1]}")
    print(f"  Specific filter (rating_osq): {args.specific_filter[0]}-{args.specific_filter[1]}")
    print(f"  Human annotations directory: {args.human_annotations_dir}")
    
    # Load human annotations and get filtered question IDs
    print("Loading human annotations...")
    human_annotators = load_human_annotations(args.human_annotations_dir, args.dataset)
    filtered_question_ids = get_filtered_question_ids(human_annotators, args.unique_rating_filter, args.specific_filter, args.dataset)
    print("Number of filtered question IDs:", len(filtered_question_ids))
    
    # Set up paths
    scores_dir = config["scores_dir"]
    output_dir = config["output_dir"]
    output_prefix = config["output_prefix"]
    need_model_mapping = config["need_model_mapping"]
    
    # Get scores from all three judges
    print(f"Getting scores from three judge models for {args.dataset}...")
    llama_scores = get_scores_by_judge(scores_dir, "score_llama-4-scout", need_model_mapping, filtered_question_ids)
    deepseek_scores = get_scores_by_judge(scores_dir, "score_deepseek-chat-v3-0324", need_model_mapping, filtered_question_ids)
    qwen_scores = get_scores_by_judge(scores_dir, "score_Qwen3_4B", need_model_mapping, filtered_question_ids)

    # Remove excluded models from all judge scores
    to_exclude = set(exclude_models.get(args.dataset, []))
    if to_exclude:
        llama_scores = {k: v for k, v in llama_scores.items() if k not in to_exclude}
        deepseek_scores = {k: v for k, v in deepseek_scores.items() if k not in to_exclude}
        qwen_scores = {k: v for k, v in qwen_scores.items() if k not in to_exclude}
    
    # Debug information
    print(f"Llama scores: {len(llama_scores)} models")
    print(f"DeepSeek scores: {len(deepseek_scores)} models")
    print(f"Qwen scores: {len(qwen_scores)} models")
    
    print("Computing statistics and rankings...")
    stats = compute_multi_judge_stats_and_rankings(llama_scores, deepseek_scores, qwen_scores)
    
    # Debug: How many models have all three scores
    print(f"Combined stats: {len(stats)} models")
    if len(stats) == 0:
        print("No models have scores from all three judges. Check that model names match between datasets.")
        # Print sample of model names from each dataset
        print(f"Llama model names (sample): {list(llama_scores.keys())[:5]}")
        print(f"DeepSeek model names (sample): {list(deepseek_scores.keys())[:5]}")
        print(f"Qwen model names (sample): {list(qwen_scores.keys())[:5]}")
        return
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filenames with filter and middle judge information
    multians_filter = f"multians_{args.unique_rating_filter[0]}-{args.unique_rating_filter[1]}"
    osq_filter = f"osq_{args.specific_filter[0]}-{args.specific_filter[1]}"
    middle_judge_suffix = f"_middle_{args.middle_judge}"
    filter_suffix = f"_{multians_filter}_{osq_filter}{middle_judge_suffix}"
    
    # Save data to CSV
    csv_path = os.path.join(output_dir, f"{output_prefix}_comparison{filter_suffix}.csv")
    df = save_to_csv(stats, csv_path)

    # Assign compact letters for each judge
    for judge in ["llama", "deepseek", "qwen"]:
        df = assign_compact_letters_judge(df, judge)

    # Find statistically similar groups for each judge
    llama_groups = find_statistically_similar_groups_judge(df, "llama")
    deepseek_groups = find_statistically_similar_groups_judge(df, "deepseek")
    qwen_groups = find_statistically_similar_groups_judge(df, "qwen")

    # Helper to check if two models are in the same group
    def in_same_group(model1, model2, groups):
        for group in groups:
            if model1 in group and model2 in group:
                return True
        return False

    print("\nModel Name, Llama Acc ± SEM, DeepSeek Acc ± SEM, Qwen Acc ± SEM, Llama Rank, DeepSeek Rank, Qwen Rank, Llama-DeepSeek Rank Diff, DeepSeek-Qwen Rank Diff")
    for model in sorted(stats.keys(), key=lambda x: stats[x]["deepseek_acc"], reverse=True):
        data = stats[model]
        # Find the model's rank neighbors for diff
        llama_rank = data["llama_rank"]
        deepseek_rank = data["deepseek_rank"]
        qwen_rank = data["qwen_rank"]
        llama_deepseek_diff = data["llama_deepseek_rank_diff"]
        deepseek_qwen_diff = data["deepseek_qwen_rank_diff"]
        # Find the model's compact letter for each judge
        llama_letters = df[df['model'] == model]['llama_letters'].values[0] if 'llama_letters' in df.columns else ''
        deepseek_letters = df[df['model'] == model]['deepseek_letters'].values[0] if 'deepseek_letters' in df.columns else ''
        qwen_letters = df[df['model'] == model]['qwen_letters'].values[0] if 'qwen_letters' in df.columns else ''
        # Find the model with the same name in the other judge's ranking
        # For diff, compare if model is in same group in both judges
        # For llama_deepseek_diff: compare model's group in llama and deepseek
        # For deepseek_qwen_diff: compare model's group in deepseek and qwen
        # If not in same group, bold the diff
        # Use ANSI escape code for bold: \033[1m ... \033[0m
        # Find if model is in same group in both judges
        same_group_ld = in_same_group(model, model, llama_groups) and in_same_group(model, model, deepseek_groups)
        same_group_dq = in_same_group(model, model, deepseek_groups) and in_same_group(model, model, qwen_groups)
        # Actually, we want to check if the model's group in judge1 overlaps with its group in judge2
        def get_group(model, groups):
            for group in groups:
                if model in group:
                    return set(group)
            return set([model])
        group_llama = get_group(model, llama_groups)
        group_deepseek = get_group(model, deepseek_groups)
        group_qwen = get_group(model, qwen_groups)
        # If the intersection is empty, then not in same group
        ld_same = len(group_llama & group_deepseek) > 0
        dq_same = len(group_deepseek & group_qwen) > 0
        ld_diff_str = f"{llama_deepseek_diff}"
        dq_diff_str = f"{deepseek_qwen_diff}"
        if not ld_same:
            ld_diff_str = f"\033[1m{llama_deepseek_diff}\033[0m"
        if not dq_same:
            dq_diff_str = f"\033[1m{deepseek_qwen_diff}\033[0m"
        print(f"{model}, {data['llama_acc']:.3f} ± {data['llama_stdev']:.3f}, {data['deepseek_acc']:.3f} ± {data['deepseek_stdev']:.3f}, {data['qwen_acc']:.3f} ± {data['qwen_stdev']:.3f}, {llama_rank}, {deepseek_rank}, {qwen_rank}, {ld_diff_str}, {dq_diff_str}")

    # Create and save multi-judge bump plot
    plot_path = os.path.join(output_dir, f"{output_prefix}_bump_plot{filter_suffix}.pdf")
    create_multi_judge_bump_plot(df, plot_path, args.dataset, args.scale, args.middle_judge)

if __name__ == "__main__":
    main() 