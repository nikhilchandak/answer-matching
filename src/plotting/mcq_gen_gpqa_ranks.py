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
    parser = argparse.ArgumentParser(description="Compare MCQ and generative format performance across models")
    parser.add_argument("--dataset", type=str, default="gpqa_diamond", choices=["gpqa_diamond", "mmlu_pro"],
                       help="Dataset to analyze (gpqa_diamond or mmlu_pro)")
    parser.add_argument("--scale", type=str, default="rank", choices=["rank", "acc"],
                       help="Y-axis scale: 'rank' for rankings or 'acc' for accuracy values")
    parser.add_argument('--human-annotations-dir', type=str, default=None,
                        help='Directory containing human annotation files for rating-based filtering')
    parser.add_argument('--unique-rating-filter', type=int, nargs=2, default=[1, 5],
                        help='Range for filtering by rating_multians (min, max)')
    parser.add_argument('--specific-filter', type=int, nargs=2, default=[1, 5],
                        help='Range for filtering by rating_osq (min, max)')
    return parser.parse_args()

# Dataset-specific configurations
def get_dataset_config(dataset):
    if dataset == "gpqa_diamond":
        return {
            "benchmark_file": "../../benchmarks_runs.csv", 
            "gen_scores_dir": "/is/cluster/fast/nchandak/qaevals/judge_outputs/gpqa_diamond_free/",
            "mcq_scores_dir": "/is/cluster/fast/nchandak/qaevals/judge_outputs/gpqa_diamond_mcq/",  # For filtering
            "output_dir": "plots",
            "output_prefix": "gpqa",
            "need_model_mapping": False
        }
    elif dataset == "mmlu_pro":
        return {
            "benchmark_file": None,  # Not used for mmlu_pro
            "gen_scores_dir": "/is/cluster/fast/nchandak/qaevals/judge_outputs/mmlu_pro_free/stratified_sample/",
            "mcq_scores_dir": "/is/cluster/fast/nchandak/qaevals/judge_outputs/mmlu_pro_mcq/stratified_sample/",
            "output_dir": "plots",
            "output_prefix": "mmlu_pro",
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
        # OpenAI o-series models 
        # "o4-mini-2025-04-16_high": "o4-mini-high",
        # DeepSeek models
        "DeepSeek-V3-0324": "deepseek-chat-v3-0324",
        # "DeepSeek-R1": "deepseek-r1",
        "DeepSeek-R1-Distill-Llama-70B": "deepseek-r1-distill-llama-70b",
        "DeepSeek-R1-0528": "deepseek-r1-0528",
        # Llama models
        "Llama-3.3-70B-Instruct": "llama-3.3-70b-instruct",
        "Llama-4-Scout-17B-16E-Instruct": "llama-4-scout",
        "Llama-4-Maverick-17B-128E-Instruct-FP8": "llama-4-maverick",
        # Qwen models
        "qwen2.5-72b-instruct": "qwen-2.5-72b-instruct",
        "qwen3-32b": "qwen3-32b",
        # Gemini models
        "gemma-3-27b-it": "gemma-3-27b-it",
        # "gemini-2.5-pro-exp-03-25": "gemini-2.5-pro-preview",
        # Mistral models
        "mistral-small-2501": "mistral-small-24b-instruct-2501",
        # "mistral-medium-2505": "mistral-medium-3", #TODO: Add mistral mcq number
        # Grok models
        "grok-3-mini-beta_low": "grok-3-mini-beta",
        # Wizardlm
        "WizardLM-2-8x22B": "wizardlm-2-8x22b",
    }
    return mapping

def apply_rating_filters(data, unique_rating_filter, specific_filter):
    """
    Apply rating filters to data based on rating_multians and rating_osq.
    
    Args:
        data: Dictionary with question_id as key and data dict as value
        unique_rating_filter: Tuple (min, max) for rating_multians filter
        specific_filter: Tuple (min, max) for rating_osq filter
    
    Returns:
        Boolean indicating whether the data item passes the filters
    """
    # Check unique_rating_filter (rating_multians)
    rating_multians = data.get("rating_multians")
    if rating_multians is not None:
        if not (unique_rating_filter[0] <= rating_multians <= unique_rating_filter[1]):
            return False
    
    # Check specific_filter (rating_osq)
    rating_osq = data.get("rating_osq")
    if rating_osq is not None:
        if not (specific_filter[0] <= rating_osq <= specific_filter[1]):
            return False
    
    return True

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

# Read MCQ scores from benchmarks_runs.csv for gpqa_diamond
def get_mcq_scores_from_csv(benchmark_file, unique_rating_filter=(1, 5), specific_filter=(1, 5)):
    """
    Read MCQ scores from CSV file for GPQA Diamond.
    Note: Rating filters are not applied here as the CSV doesn't contain rating information.
    For rating-based filtering with GPQA Diamond, annotation files would need to be loaded separately.
    """
    mcq_scores = {}
    with open(benchmark_file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        for row in reader:
            if len(row) < 4:
                continue
            task = row[1]
            model = row[2]
            scores = row[4]
            
            if task == "GPQA diamond" and "choice:" in scores:
                # Extract accuracy and standard deviation
                match = re.search(r'choice:([0-9.]+)±([0-9.]+)', scores)
                if match:
                    accuracy = float(match.group(1))
                    accuracy += np.random.uniform(0, 0.01)
                    stdev = float(match.group(2))
                    # Since we don't have n_samples from CSV, set a default value
                    mcq_scores[model] = {"accuracy": accuracy, "stdev": stdev, "n_samples": 100}  # Assuming 100 samples
    
    print(f"Note: Rating filters not applied to CSV data for GPQA Diamond (rating info not available in CSV)")
    return mcq_scores

# Read MCQ scores from JSONL files for mmlu_pro
def get_mcq_scores_from_jsonl(mcq_scores_dir, filtered_question_ids=None, need_model_mapping=True):
    mcq_scores = {}
    model_mapping = get_model_mapping() if need_model_mapping else None
    
    for filename in os.listdir(mcq_scores_dir):
        if filename.startswith("samples_") and filename.endswith(".jsonl"):
            model_name = filename[len("samples_"):-len(".jsonl")]
            
            # if "r1" in model_name and "distill" not in model_name:
            #     # if "r1-0528" not in model_name:
            #     #     continue
            #     continue 
            # if "o3" in model_name or "o4" in model_name:
            #     continue
            
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
            file_path = os.path.join(mcq_scores_dir, filename)
            
            try:
                with open(file_path, 'r') as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            question_id = data.get("question_id")
                            
                            # Apply question ID filtering if provided
                            if filtered_question_ids is not None and question_id not in filtered_question_ids:
                                continue
                            
                            # Use exact_match field to determine accuracy (true = 1, false = 0)
                            if "exact_match" in data:
                                scores.append(1.0 if data["exact_match"] else 0.0)
                        except json.JSONDecodeError:
                            continue
                
                if scores:
                    accuracy = np.mean(scores)
                    # Calculate standard error of the mean (SEM)
                    stdev = np.std(scores, ddof=1) / np.sqrt(len(scores))
                    mcq_scores[model_name] = {"accuracy": accuracy, "stdev": stdev, "n_samples": len(scores)}
                    print(f"MCQ - {model_name}: {len(scores)} samples, accuracy: {accuracy:.3f}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    return mcq_scores

# Read generative scores from JSONL files
def get_generative_scores(gen_scores_dir, need_model_mapping=True, filtered_question_ids=None):
    gen_scores = {}
    model_mapping = get_model_mapping() if need_model_mapping else None
    
    for filename in os.listdir(gen_scores_dir):
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
            file_path = os.path.join(gen_scores_dir, filename)
            
            try:
                with open(file_path, 'r') as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            question_id = data.get("question_id")
                            
                            # Apply question ID filtering if provided
                            if filtered_question_ids is not None and question_id not in filtered_question_ids:
                                continue
                            
                            field = "score_deepseek-chat-v3-0324"
                            # field = "score_llama-4-scout"
                            if field in data and isinstance(data[field], list):
                                scores.extend(data[field])
                            elif field in data and isinstance(data[field], str):
                                scores.append(int(data[field]))
                            elif field in data and isinstance(data[field], int):
                                scores.append(data[field])
                        except json.JSONDecodeError:
                            continue
                
                if scores:
                    accuracy = np.mean(scores)
                    # Calculate standard error of the mean (SEM) instead of std dev
                    stdev = np.std(scores, ddof=1) / np.sqrt(len(scores))
                    gen_scores[benchmark_model] = {"accuracy": accuracy, "stdev": stdev, "n_samples": len(scores)}
                    print(f"GEN - {benchmark_model}: {len(scores)} samples, accuracy: {accuracy:.3f}")
            except Exception as e:
                print(f"Error processing {filename}: {e}")
    
    return gen_scores

# Compute statistics and rankings
def compute_stats_and_rankings(mcq_scores, gen_scores):
    # Combine all models that have both MCQ and generative scores
    combined_stats = {}
    
    for model in mcq_scores:
        if model in gen_scores:
            combined_stats[model] = {
                "mcq_acc": mcq_scores[model]["accuracy"],
                "mcq_stdev": mcq_scores[model]["stdev"],
                "gen_acc": gen_scores[model]["accuracy"],
                "gen_stdev": gen_scores[model]["stdev"],
                "gen_n_samples": gen_scores[model]["n_samples"] if "n_samples" in gen_scores[model] else 0,
                "acc_diff": gen_scores[model]["accuracy"] - mcq_scores[model]["accuracy"]
            }
    
    # Calculate rankings
    mcq_ranking = sorted(combined_stats.keys(), key=lambda x: combined_stats[x]["mcq_acc"], reverse=True)
    gen_ranking = sorted(combined_stats.keys(), key=lambda x: combined_stats[x]["gen_acc"], reverse=True)
    
    # Assign ranks
    for i, model in enumerate(mcq_ranking):
        combined_stats[model]["mcq_rank"] = i + 1
    
    for i, model in enumerate(gen_ranking):
        combined_stats[model]["gen_rank"] = i + 1
        combined_stats[model]["rank_diff"] = combined_stats[model]["mcq_rank"] - combined_stats[model]["gen_rank"]
    
    return combined_stats

def save_to_csv(stats, output_path):
    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    data = []
    for model, model_stats in stats.items():
        data.append({
            "model": model,
            "mcq_acc": model_stats["mcq_acc"],
            "mcq_stdev": model_stats["mcq_stdev"],
            "gen_acc": model_stats["gen_acc"],
            "gen_stdev": model_stats["gen_stdev"],
            "gen_n_samples": model_stats["gen_n_samples"],
            "acc_diff": model_stats["acc_diff"],
            "mcq_rank": model_stats["mcq_rank"],
            "gen_rank": model_stats["gen_rank"],
            "rank_diff": model_stats["rank_diff"]
        })
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")
    
    return df

def find_statistically_similar_groups(df, format_type):
    """Find groups of models that are statistically similar (within standard error)"""
    # Debug information
    print(f"DataFrame columns: {df.columns.tolist()}")
    
    # Check if expected columns exist
    if format_type == 'MCQ' and 'mcq_acc' not in df.columns:
        print(f"Warning: 'mcq_acc' column not found in DataFrame. Available columns: {df.columns.tolist()}")
        # Try to identify if there's a similar column or return empty groups
        return []
    
    if format_type == 'Generative' and 'gen_acc' not in df.columns:
        print(f"Warning: 'gen_acc' column not found in DataFrame. Available columns: {df.columns.tolist()}")
        # Try to identify if there's a similar column or return empty groups
        return []
    
    if format_type == 'MCQ':
        df_sorted = df.sort_values('mcq_acc', ascending=False).reset_index(drop=True)
        acc_col = 'mcq_acc'
        err_col = 'mcq_stdev'
    else:  # Generative
        df_sorted = df.sort_values('gen_acc', ascending=False).reset_index(drop=True)
        acc_col = 'gen_acc'
        err_col = 'gen_stdev'
    
    groups = []
    
    if len(df_sorted) == 0:
        print(f"Warning: Empty DataFrame for {format_type} format.")
        return groups
    
    current_group = [0]  # Start with the top model
    top_model_acc = df_sorted.iloc[0][acc_col]
    top_model_err = df_sorted.iloc[0][err_col]
    
    for i in range(1, len(df_sorted)):
        current_acc = df_sorted.iloc[i][acc_col]
        current_err = df_sorted.iloc[i][err_col]
        
        # Check if current model's performance is within standard error of the top model in the group
        if top_model_acc - top_model_err <= current_acc + current_err:
            current_group.append(i)
        else:
            # If not, close the current group and start a new one
            if len(current_group) > 1:  # Only save groups with more than one model
                group_models = [df_sorted.iloc[idx]['model'] for idx in current_group]
                groups.append(group_models)
            
            # Start a new group with the current model
            current_group = [i]
            top_model_acc = current_acc
            top_model_err = current_err
    
    # Add the last group if it has more than one model
    if len(current_group) > 1:
        group_models = [df_sorted.iloc[idx]['model'] for idx in current_group]
        groups.append(group_models)
    
    return groups

def assign_compact_letters(df, format_type):
    """
    Assign compact letter display to models based on statistical significance.
    Models that are not statistically different will share at least one letter.
    """
    if format_type == 'MCQ':
        df_sorted = df.sort_values('mcq_acc', ascending=False).reset_index(drop=True)
        acc_col = 'mcq_acc'
        err_col = 'mcq_stdev'
        letter_col = 'mcq_letters'
    else:  # Generative
        df_sorted = df.sort_values('gen_acc', ascending=False).reset_index(drop=True)
        acc_col = 'gen_acc'
        err_col = 'gen_stdev'
        letter_col = 'gen_letters'
    print(format_type, df_sorted)
    # Initialize with primary letters (A, B, C, ...) based on rank
    letters = {}
    for i, (_, row) in enumerate(df_sorted.iterrows()):
        model = row['model']
        # Assign primary letter based on rank (A for 1st, B for 2nd, etc.)
        primary_letter = chr(65 + i)  # ASCII: A=65, B=66, etc.
        letters[model] = primary_letter
    
    # Add additional letters for models that are not statistically different
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
            
            # Check if models are within standard error of each other
            if acc_i - err_i <= acc_j + err_j and acc_j - err_j <= acc_i + err_i:
                # Models are not statistically different, share letters
                letters[model_i] += letters[model_j][0]  # Add primary letter of model_j
        print(model_i, letters[model_i])
    # Update the dataframe with letter assignments
    for model, letter in letters.items():
        df.loc[df['model'] == model, letter_col] = letter
    print(df.sort_values('mcq_acc' if format_type == 'MCQ' else 'gen_acc', ascending=False))
    
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
    'mistral-medium-3': 'Mistral Medium 3',
    'gpt-4o-mini-2024-07-18': 'GPT 4o Mini',
    'wizardlm-2-8x22b': 'WizardLM 2 8x22B',
    'WizardLM-2-8x22B': 'WizardLM 2 8x22B',
    'claude-3-5-haiku-20241022': 'Claude 3.5 Haiku',
    "deepseek-r1-0528": "DeepSeek R1 0528"


  



    # Add more as needed
}

# Models to exclude from plots for each dataset
exclude_models = {
    "gpqa_diamond": [
        # "DeepSeek-R1"
        "deepseek-r1",
        "o4-mini-high",
        # "deepseek-r1-0528"
    ],
    "mmlu_pro": [
        "gemini-2.5-flash-preview",
        # "mistral-medium-3"
    ]
}

def create_bump_plot(df, output_path, dataset, scale='rank'):
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
    required_cols = ['model', 'mcq_acc', 'mcq_stdev', 'gen_acc', 'gen_stdev', 'mcq_rank', 'gen_rank']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"Error: Missing required columns: {missing_cols}")
        return
    
    # Assign compact letter display to models
    df = assign_compact_letters(df, 'MCQ')
    df = assign_compact_letters(df, 'Generative')

    # Make the plot even wider and increase spacing
    plt.figure(figsize=(10, 8))  # Much wider for more space

    # Create long-form data for plotting
    plot_data = []
    formats = ['MCQ', 'Generative']
    for _, row in df.iterrows():
        plot_data.append({
            'model': row['model'],
            'format': formats[0],
            'rank': row['mcq_rank'],
            'accuracy': row['mcq_acc'],
            'stdev': row['mcq_stdev'],
            'position': 0,
            'letters': row['mcq_letters'] if 'mcq_letters' in row else ''
        })
        plot_data.append({
            'model': row['model'],
            'format': formats[1],
            'rank': row['gen_rank'],
            'accuracy': row['gen_acc'],
            'stdev': row['gen_stdev'],
            'position': 1,
            'letters': row['gen_letters'] if 'gen_letters' in row else ''
        })

    plot_df = pd.DataFrame(plot_data)

    # Find statistically similar groups
    mcq_groups = find_statistically_similar_groups(df, 'MCQ')
    gen_groups = find_statistically_similar_groups(df, 'Generative')

    print("MCQ statistically similar groups:")
    for i, group in enumerate(mcq_groups):
        print(f"Group {i+1}: {group}")

    print("\nGenerative statistically similar groups:")
    for i, group in enumerate(gen_groups):
        print(f"Group {i+1}: {group}")

    sns.set_style("whitegrid")
    models_by_mcq_rank = df.sort_values('mcq_rank')['model'].unique()

    # --- COLOR MAPPING BY MCQ RANK ---
    # Use a perceptually uniform colormap (Spectral or similar)
    cmap = mpl.cm.get_cmap('Spectral')

    n_models = len(models_by_mcq_rank)
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
    # Map MCQ rank (1 is best) to dark colors
    model_to_color = {}
    for i, model in enumerate(models_by_mcq_rank):
        model_to_color[model] = cmap(color_positions[i])

    # Make the middle panel less wide by moving columns closer
    positions = [0.4, 0.6]

    # Adjust margins for new width
    plt.subplots_adjust(left=0.49, right=0.51)

    for model in models_by_mcq_rank:
        model_data = plot_df[plot_df['model'] == model].sort_values('position')
        x_values = [positions[0] if pos == 0 else positions[1] for pos in model_data['position']]
        if scale == 'rank':
            y_values = model_data['rank']
        else:
            y_values = model_data['accuracy']
        mcq_letters = model_data[model_data['format'] == 'MCQ']['letters'].values[0]
        gen_letters = model_data[model_data['format'] == 'Generative']['letters'].values[0]
        mcq_primary_letter = mcq_letters[0] if mcq_letters else ''
        color = model_to_color[model]
        if mcq_primary_letter in gen_letters:
            plt.plot(x_values, y_values, 'o-', linewidth=1.0, markersize=8, alpha=1.0, color=color)
        else:
            plt.plot(x_values, y_values, 'o-', linewidth=4.0, markersize=16, alpha=1.0, color=color)

    ax1 = plt.gca()
    if scale == 'rank':
        ax1.invert_yaxis()
    if scale == 'rank':
        y_min, y_max = ax1.get_ylim()
        ax1.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    else:
        min_acc = min(df['mcq_acc'].min(), df['gen_acc'].min())
        max_acc = max(df['mcq_acc'].max(), df['gen_acc'].max())
        y_min = max(min_acc - 0.05, 0.0)
        y_max = 1.0
        plt.ylim(y_min, y_max)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

    # Add model names (custom display names), colored by their line color, no stdev
    if scale == 'rank':
        y_ticks = list(range(1, len(models_by_mcq_rank) + 1))
        plt.yticks(y_ticks, ["" for _ in y_ticks])
        for i, model in enumerate(models_by_mcq_rank):
            rank_mcq = i + 1
            rank_gen = df[df['model'] == model].iloc[0]['gen_rank']
            mcq_data = df[df['model'] == model].iloc[0]
            gen_data = df[df['model'] == model].iloc[0]
            color = model_to_color[model]
            display_name = print_names.get(model, model)
            # Left side: model name and accuracy (no stderr), colored, with reduced spacing
            plt.text(positions[0] - 0.045, rank_mcq, f"{display_name}", fontsize=16, ha='right', va='center', color=color)
            plt.text(positions[0] - 0.02, rank_mcq, f"{mcq_data['mcq_acc']:.1%}", fontsize=16, ha='right', va='center', color=color)
            # Right side: accuracy only (no model name), colored, with reduced spacing
            plt.text(positions[1] + 0.02, rank_gen, f"{gen_data['gen_acc']:.1%}", fontsize=16, ha='left', va='center', color=color)
    else:
        for i, model in enumerate(models_by_mcq_rank):
            model_data = df[df['model'] == model].iloc[0]
            color = model_to_color[model]
            display_name = print_names.get(model, model)
            plt.text(positions[0] - 0.24, model_data['mcq_acc'], f"{display_name}", fontsize=16, ha='right', va='center', color=color)
            plt.text(positions[0] - 0.06, model_data['mcq_acc'], f"{model_data['mcq_acc']:.1%}", fontsize=16, ha='right', va='center', color=color)
            plt.text(positions[1] + 0.06, model_data['gen_acc'], f"{model_data['gen_acc']:.1%}", fontsize=16, ha='left', va='center', color=color)

    # plt.ylabel('Accuracy' if scale == 'acc' else 'Ranking', fontsize=15)
    # ax1.yaxis.set_label_coords(-0.12, 0.5)
    title = ""
    if dataset == "gpqa_diamond":
        title = "GPQA Diamond Ranking"
    elif dataset == "mmlu_pro":
        title = "MMLU Pro Ranking"
    plt.title(f'{title}', fontsize=22)
    plt.xticks(positions, formats, fontsize=20)
    plt.tight_layout(rect=[0.07, 0, 0.93, 1])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Bump plot saved to {output_path}")
    plt.close()

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
    benchmark_file = config["benchmark_file"]
    gen_scores_dir = config["gen_scores_dir"]
    mcq_scores_dir = config["mcq_scores_dir"]
    output_dir = config["output_dir"]
    output_prefix = config["output_prefix"]
    need_model_mapping = config["need_model_mapping"]
    
    # Get MCQ scores based on dataset
    print(f"Getting MCQ scores for {args.dataset}...")
    if args.dataset == "gpqa_diamond":
        # For GPQA Diamond, use JSONL files if we have filtered question IDs, otherwise use CSV
        if filtered_question_ids:
            print("Using JSONL MCQ files for GPQA Diamond with filtering")
            mcq_scores = get_mcq_scores_from_jsonl(mcq_scores_dir, filtered_question_ids, need_model_mapping)
        else:
            print("Using CSV MCQ scores for GPQA Diamond (no filtering)")
            mcq_scores = get_mcq_scores_from_csv(benchmark_file, args.unique_rating_filter, args.specific_filter)
    else:  # mmlu_pro
        mcq_scores = get_mcq_scores_from_jsonl(mcq_scores_dir, filtered_question_ids, need_model_mapping)
    
    print(f"Getting generative scores for {args.dataset}...")
    gen_scores = get_generative_scores(gen_scores_dir, need_model_mapping, filtered_question_ids)

    # Remove excluded models from both MCQ and generative scores
    to_exclude = set(exclude_models.get(args.dataset, []))
    if to_exclude:
        mcq_scores = {k: v for k, v in mcq_scores.items() if k not in to_exclude}
        gen_scores = {k: v for k, v in gen_scores.items() if k not in to_exclude}
    
    # Debug information
    print(f"MCQ scores: {len(mcq_scores)} models")
    print(f"Generative scores: {len(gen_scores)} models")
    
    print("Computing statistics and rankings...")
    stats = compute_stats_and_rankings(mcq_scores, gen_scores)
    
    # Debug: How many models have both scores
    print(f"Combined stats: {len(stats)} models")
    if len(stats) == 0:
        print("No models have both MCQ and generative scores. Check that model names match between datasets.")
        # Print sample of model names from each dataset
        print(f"MCQ model names (sample): {list(mcq_scores.keys())[:5]}")
        print(f"Gen model names (sample): {list(gen_scores.keys())[:5]}")
        return
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filenames with filter information
    multians_filter = f"multians_{args.unique_rating_filter[0]}-{args.unique_rating_filter[1]}"
    osq_filter = f"osq_{args.specific_filter[0]}-{args.specific_filter[1]}"
    filter_suffix = f"_{multians_filter}_{osq_filter}"
    
    # Save data to CSV
    csv_path = os.path.join(output_dir, f"{output_prefix}_format_comparison{filter_suffix}.csv")
    df = save_to_csv(stats, csv_path)
    
    # Create and save bump plot
    plot_path = os.path.join(output_dir, f"{output_prefix}_format_bump_plot{filter_suffix}.png")
    create_bump_plot(df, plot_path, args.dataset, args.scale)
    
    print("\nModel Name, MCQ Accuracy ± Stdev, Generative Accuracy ± SEM, MCQ - Gen Acc, MCQ Rank, Gen Rank, MCQ Rank - Gen Rank")
    
    for model in sorted(stats.keys(), key=lambda x: stats[x]["mcq_acc"], reverse=True):
        data = stats[model]
        print(f"{model}, {data['mcq_acc']:.3f} ± {data['mcq_stdev']:.3f}, {data['gen_acc']:.3f} ± {data['gen_stdev']:.3f}, {data['acc_diff']:.3f}, {data['mcq_rank']}, {data['gen_rank']}, {data['rank_diff']}")

if __name__ == "__main__":
    main()