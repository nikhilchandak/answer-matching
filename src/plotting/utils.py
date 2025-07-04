import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import defaultdict
import scienceplots
plt.style.use(['science'])

model_name_map = {
    'human_1': 'Human 1',
    'human_2': 'Human 2',
    'llama-4-scout': 'Llama 4 Scout',
    'qwen-2.5-72b-instruct': 'Qwen 2.5 72B',
    'qwen-2.5-7b-instruct-JUDGE': 'Qwen 2.5 7B (Judge)',
    'deepseek-chat-v3-0324': 'DeepSeek V3',
    'Qwen3_8B': 'Qwen 3 8B',
    'qwen3-8b': 'Qwen 3 8B',
    'Qwen3_4B': 'Qwen 3 4B',
    'qwen-2.5-7b-instruct': 'Qwen 2.5 7B',
    'Qwen3_1_7B': 'Qwen 3 1.7B',
    'llama-2-70b': 'Llama 2 70B',
    'llama-3.1-8b-instruct': 'Llama 3.1 8B',
    'o4-mini-JUDGE': 'o4 Mini (Judge)',
    'llama_2_7b_chat_hf': 'Llama 2 7B',
    'mc_verify': 'MC Verify',
    'mcq': 'MCQ',
    'deepseek-chat-v3-0324-JUDGE': 'DeepSeek V3 (Judge)',
    'Qwen3_0_6B': 'Qwen 3 0.6B',
    'mc_cloze': 'MC Cloze',
}

def load_jsonl_file(file_path: str) -> List[Dict]:
    """Load data from a JSONL file."""
    data = []
    if not os.path.exists(file_path):
        print(f"Warning: File {file_path} does not exist")
        return data
        
    with open(file_path, 'r') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Error parsing JSON in {file_path}")
    
    return data

def process_list_metrics(data: List[Dict]) -> List[Dict]:
    """
    Process data where metrics like exact_match and scores are lists
    by taking the first element of each list.
    
    Args:
        data: List of data dictionaries
        
    Returns:
        List of processed data dictionaries
    """
    processed_data = []
    
    for item in data:
        # Make a copy to avoid modifying the original
        processed_item = item.copy()
        
        # Process exact_match if it's a list
        if "exact_match" in item and isinstance(item["exact_match"], list) and len(item["exact_match"]) > 0:
            # Take the first element
            processed_item["exact_match"] = item["exact_match"][0]
        
        # Process score fields if they're lists
        for key in processed_item:
            if key.startswith("score_") and isinstance(processed_item[key], list) and len(processed_item[key]) > 0:
                # Take the first element
                processed_item[key] = processed_item[key][0]
                
        processed_data.append(processed_item)
    
    return processed_data

def convert_rating_to_binary(rating: Optional[int]) -> Optional[int]:
    """Convert rating (1-5 scale) to binary (0 or 1)."""
    if rating is None:
        return None
    if rating >= 4:  # 4-5 is "surely yes"
        return 1
    elif rating <= 2:  # 1-2 is "surely no"
        return 0
    return None  # 3 is "unsure", so we ignore it

def get_balanced_question_ids(question_ids_by_gt: Dict[int, List]) -> Set[str]:
    """
    Balance the dataset by sampling from the majority class.
    
    Args:
        question_ids_by_gt: Dictionary mapping ground truth scores (0/1) to lists of question IDs
        
    Returns:
        Set of question IDs to use for balanced analysis
    """
    questions_to_use = set()
    gt_counts = {label: len(ids) for label, ids in question_ids_by_gt.items()}
    
    if gt_counts[0] > 0 and gt_counts[1] > 0:
        # Find minority class
        minority_class = 0 if gt_counts[0] < gt_counts[1] else 1
        majority_class = 1 - minority_class
        
        # Use all questions from minority class
        questions_to_use.update(question_ids_by_gt[minority_class])
        
        # Sample from majority class
        if len(question_ids_by_gt[majority_class]) > len(question_ids_by_gt[minority_class]):
            sampled_majority = np.random.choice(
                question_ids_by_gt[majority_class],
                size=len(question_ids_by_gt[minority_class]),
                replace=False
            )
            questions_to_use.update(sampled_majority)
        else:
            questions_to_use.update(question_ids_by_gt[majority_class])
        
        print(f"Balanced dataset: {len(questions_to_use)} questions total " 
              f"({len(question_ids_by_gt[minority_class])} from each class)")
    else:
        # Use all questions if one class is empty
        questions_to_use.update(question_ids_by_gt[0])
        questions_to_use.update(question_ids_by_gt[1])
    
    return questions_to_use

def compute_reweighting_factors(question_ids_by_gt: Dict[int, List]) -> Dict[str, float]:
    """
    Compute reweighting factors for each question to balance class distribution.
    
    Args:
        question_ids_by_gt: Dictionary mapping ground truth scores (0/1) to lists of question IDs
        
    Returns:
        Dictionary mapping question IDs to their weight factors
    """
    weights = {}
    total_samples = sum(len(q_ids) for q_ids in question_ids_by_gt.values())
    
    if total_samples > 0:
        for gt_score, q_ids in question_ids_by_gt.items():
            # Calculate weight that makes this class contribute 0.5 to the overall score
            # regardless of its size
            if len(q_ids) > 0:
                weight = 0.5 / (len(q_ids) / total_samples)
                for q_id in q_ids:
                    weights[q_id] = weight
    
    return weights

def calculate_scotts_pi(observed_agreement: float, gt_dist: Dict[int, int], 
                       pred_dist: Dict[int, int], total: int) -> float:
    """
    Calculate Scott's Pi coefficient for inter-rater reliability.
    
    Args:
        observed_agreement: Observed agreement rate
        gt_dist: Distribution of ground truth labels
        pred_dist: Distribution of predicted labels
        total: Total number of samples
    
    Returns:
        Scott's Pi coefficient
    """
    if total == 0:
        return 0.0 # Return float for consistency
    
    # Calculate expected agreement by chance for Scott's Pi
    expected_agreement_scotts = 0.0
    
    # Proportion of category 0 by ground truth and predictor
    p0_gt = gt_dist.get(0, 0) / total
    p0_pred = pred_dist.get(0, 0) / total
    avg_p0 = (p0_gt + p0_pred) / 2.0
    expected_agreement_scotts += avg_p0**2
    
    # Proportion of category 1 by ground truth and predictor
    p1_gt = gt_dist.get(1, 0) / total
    p1_pred = pred_dist.get(1, 0) / total
    avg_p1 = (p1_gt + p1_pred) / 2.0
    expected_agreement_scotts += avg_p1**2
    
    # Calculate Scott's Pi
    if expected_agreement_scotts == 1.0:
        # If expected agreement is 1, and observed is also 1, Pi is undefined (0/0).
        # Conventionally, this can be 0 or 1. Let's use 0 for numerical stability
        # and to indicate no agreement beyond what's perfectly expected by chance.
        # If observed_agreement < 1 and expected_agreement_scotts = 1, Pi would be -infinity.
        # Returning 0 in the case of Pe=1 is a common simplification.
        return 0.0 if observed_agreement == 1.0 else (observed_agreement - expected_agreement_scotts) / (1.0 - expected_agreement_scotts + 1e-9) # add epsilon to avoid div by zero if Po != 1

    # Add a small epsilon to the denominator to prevent division by zero if 1.0 - expected_agreement_scotts is extremely close to 0
    pi_value = (observed_agreement - expected_agreement_scotts) / (1.0 - expected_agreement_scotts + 1e-9) # Adding epsilon
    
    return pi_value

def select_questions_and_calculate_weights(question_ids_by_gt: Dict[int, List], 
                                          normalize: str) -> Tuple[Set[str], Dict[str, float]]:
    """
    Select questions to use and calculate weights based on normalization method.
    
    Args:
        question_ids_by_gt: Dictionary mapping ground truth scores (0/1) to lists of question IDs
        normalize: Normalization method ("none", "balance", "reweight", or "scotts")
        
    Returns:
        Tuple of (selected question IDs, weights dictionary)
    """
    if normalize == "balance":
        # For balance, select a balanced subset of questions with uniform weights
        questions_to_use = get_balanced_question_ids(question_ids_by_gt)
        weights = {}  # No weights needed (or equivalently, all weights = 1)
    else:
        # For none, reweight and scotts, use all questions
        questions_to_use = set()
        questions_to_use.update(question_ids_by_gt[0])
        questions_to_use.update(question_ids_by_gt[1])
        
        # For reweight, calculate weights; for none and scotts, weights don't matter for the initial calculation
        weights = compute_reweighting_factors(question_ids_by_gt) if normalize == "reweight" else {}
    
    return questions_to_use, weights

def collect_scotts_pi_data(question_id: str, gt_score: int, pred_score: int,
                         scotts_data: Dict, source: str) -> None:
    """
    Collect data needed for Scott's Pi calculation.
    
    Args:
        question_id: ID of the current question
        gt_score: Ground truth score
        pred_score: Predicted score
        scotts_data: Dictionary to store Scott's Pi data for each source
        source: Source name of the prediction (e.g., "mcq", "matcher_name")
    """
    if source not in scotts_data:
        scotts_data[source] = {
            "agreements": 0, 
            "total": 0, 
            "gt_dist": {},
            "pred_dist": {},
            "source_name": source, # Store the source name
            "pairs": []  # Add a list to store (gt, pred) pairs
        }
        
    # Update ground truth distribution
    scotts_data[source]["gt_dist"][gt_score] = scotts_data[source]["gt_dist"].get(gt_score, 0) + 1
    
    # Update prediction distribution
    scotts_data[source]["pred_dist"][pred_score] = scotts_data[source]["pred_dist"].get(pred_score, 0) + 1
    
    # Update agreement counts
    scotts_data[source]["total"] += 1
    if gt_score == pred_score:
        scotts_data[source]["agreements"] += 1
    
    # Store the (gt, pred) pair
    scotts_data[source]["pairs"].append((gt_score, pred_score))

def calculate_agreement_metric(data: Dict, normalize: str, 
                             scotts_data: Optional[Dict] = None, 
                             n_bootstrap: int = 1000) -> Tuple[float, float]:
    """
    Calculate agreement metric based on normalization method.
    
    Args:
        data: Dictionary with agreements and total count
        normalize: Normalization method ("none", "balance", "reweight", or "scotts")
        scotts_data: Dictionary with data for Scott's Pi calculation
        n_bootstrap: Number of bootstrap samples for error calculation
        
    Returns:
        Tuple of (agreement percentage, standard error)
    """
    if normalize == "scotts" and scotts_data is not None:
        # Calculate Scott's Pi
        source_data = scotts_data
        observed_agreement = source_data["agreements"] / source_data["total"] if source_data["total"] > 0 else 0
        
        # Calculate Scott's Pi using the dedicated function
        scotts_pi_value = calculate_scotts_pi(
            observed_agreement,
            source_data["gt_dist"], 
            source_data["pred_dist"], 
            source_data["total"]
        )
        
        # For scotts, we don't multiply by 100 anymore
        agreement_pct = scotts_pi_value
        
        # Bootstrap to calculate standard error
        bootstrap_samples = []
        if source_data["total"] > 0: # Ensure total is not zero for bootstrapping
            all_pairs = source_data.get("pairs")
            if not all_pairs:
                # Fallback or handle error if pairs are somehow not collected
                for _ in range(n_bootstrap):
                    bootstrap_samples.append(0.0)
            else:
                original_n = len(all_pairs)
                for _ in range(n_bootstrap):
                    resampled_indices = np.random.choice(original_n, size=original_n, replace=True)
                    resampled_pairs = [all_pairs[i] for i in resampled_indices]
                    
                    # Calculate bootstrapped Po, gt_dist, pred_dist, and total from resampled_pairs
                    boot_gt_dist = {}
                    boot_pred_dist = {}
                    boot_agreements = 0
                    boot_total = len(resampled_pairs)
                    
                    for gt, pred in resampled_pairs:
                        boot_gt_dist[gt] = boot_gt_dist.get(gt, 0) + 1
                        boot_pred_dist[pred] = boot_pred_dist.get(pred, 0) + 1
                        if gt == pred:
                            boot_agreements += 1
                    
                    bootstrap_observed_agreement = boot_agreements / boot_total
                    
                    bootstrap_pi = calculate_scotts_pi(
                        bootstrap_observed_agreement, 
                        boot_gt_dist, 
                        boot_pred_dist, 
                        boot_total
                    )
                    # Don't multiply by 100 for bootstrap samples either
                    bootstrap_samples.append(bootstrap_pi)
        
        std_error = np.std(bootstrap_samples) if bootstrap_samples else 0.0
    else:
        # Standard percentage agreement for none, balance and reweight
        agreements = np.array(data["agreements"])
        agreement_pct = np.mean(agreements) * 100
        
        # Bootstrap to calculate standard error
        bootstrap_samples = []
        for _ in range(n_bootstrap):
            bootstrap_indices = np.random.choice(len(agreements), size=len(agreements), replace=True)
            bootstrap_sample = agreements[bootstrap_indices]
            bootstrap_samples.append(np.mean(bootstrap_sample) * 100)
        
        std_error = np.std(bootstrap_samples)
    
    return agreement_pct, std_error

def update_error_counts(counter: Dict[str, int], ground_truth: int, prediction: int) -> None:
    """
    Update error counts based on comparing ground truth with prediction.
    
    Args:
        counter: Dictionary to update with error counts
        ground_truth: Ground truth label (0 or 1)
        prediction: Predicted label (0 or 1)
    """
    if ground_truth != prediction:
        counter["total_errors"] += 1
        if prediction == 1 and ground_truth == 0:
            counter["false_positives"] += 1
        else:  # prediction == 0 and ground_truth == 1
            counter["false_negatives"] += 1

def plot_alignment(alignment_df: pd.DataFrame, ground_truth_key: str, 
                   fig_size: Tuple[int, int] = (12, 8), show_constant_baseline: bool = True,
                   constant_baseline: float = 50.0, output_file: Optional[str] = None,
                   normalize: str = "none", dataset_name: Optional[str] = None) -> None:
    """
    Create a bar plot showing alignment percentages with error bars.
    
    Args:
        alignment_df: DataFrame with alignment data
        ground_truth_key: Key identifying the ground truth source
        fig_size: Figure size (width, height)
        show_constant_baseline: Whether to show a line at the constant baseline
        constant_baseline: Value for the constant baseline (normalized to match the chosen method)
        output_file: If provided, save the plot to this file instead of displaying it
        normalize: Normalization method used ("none", "balance", "reweight", or "scotts")
        dataset_name: Name of the dataset (for custom plot title, e.g., 'math')
    """
    # Filter out the constant_baseline row if it exists and we want to show it as a line
    constant_baseline_row = alignment_df[alignment_df['Source'] == 'constant_baseline']
    if not constant_baseline_row.empty:
        # Use calculated constant baseline instead of the standard one
        constant_baseline = constant_baseline_row.iloc[0]['Agreement (%)']
        # Remove from the DataFrame since we'll show it as a line
        alignment_df = alignment_df[alignment_df['Source'] != 'constant_baseline']
    
    if not show_constant_baseline:
        # Remove constant baseline row when not showing it
        alignment_df = alignment_df[alignment_df['Source'] != 'constant_baseline']
    
    # Map display names using model_name_map
    def get_display_name(name):
        label = model_name_map.get(name, name)
        return label
        # Remove ' (Judge)' from label if present
        # return label.replace(' (Judge)', '')
    
    # Create a copy and add display names
    df = alignment_df.copy()
    df['DisplayName'] = df['Source'].apply(get_display_name)
    
    # Handle duplicates by preferring judge versions
    # Group by display name and keep the best performing version
    def resolve_duplicates(group):
        if len(group) == 1:
            return group.iloc[0]
        
        # Prefer judge versions (sources containing 'judge' in lowercase)
        judge_rows = group[group['Source'].str.lower().str.contains('judge', na=False)]
        if not judge_rows.empty:
            # Among judge versions, pick the one with highest agreement
            return judge_rows.loc[judge_rows['Agreement (%)'].idxmax()]
        else:
            # Among non-judge versions, pick the one with highest agreement
            return group.loc[group['Agreement (%)'].idxmax()]
    
    # Group by DisplayName and resolve duplicates
    df = df.groupby('DisplayName').apply(resolve_duplicates).reset_index(drop=True)
    
    # Sort by agreement percentage for better visualization
    df = df.sort_values("Agreement (%)", ascending=False)
    
    # Calculate the minimum agreement value for x-axis limits
    min_agreement = min(df['Agreement (%)'])
    
    # Create figure
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Increase all font sizes by 4
    base_labelsize = 20
    base_titlesize = 22
    base_ticksize = 20
    
    # Define colors based on source type
    human_color = '#8ac926'  # Green
    mcq_color = '#FF8300'    # Red 
    judge_color = '#FFc933'  # Purple for judge
    cloze_color = '#ff70a6'  # Orange for MC Cloze
    verify_color = '#ffd6ba' # Yellow for MC Verify
    
    # Create a gradient for matchers
    matcher_indices = df[df['Type'] == 'Matcher'].index
    matcher_colors = plt.cm.Blues(np.linspace(0.9, 0.4, len(matcher_indices)))
    
    # Assign colors based on type
    colors = []
    matcher_idx = 0
    
    for i, source_type in enumerate(df['Type']):
        source_name = df['Source'].iloc[i]
        if source_type == 'Human':
            colors.append(human_color)
        elif source_type == 'MCQ':
            colors.append(mcq_color)
        elif source_name == 'mc_cloze':
            colors.append(cloze_color)
        elif source_name == 'mc_verify':
            colors.append(verify_color)
        elif source_type == 'Constant Baseline':
            colors.append('red')
        else:  # Matcher
            # Check if matcher has 'judge' in its name
            if 'judge' in source_name.lower():
                colors.append(judge_color)
            else:
                colors.append(matcher_colors[matcher_idx])
            matcher_idx += 1
    
    # Create horizontal bar plot with special handling for Scott's Pi negative values
    if normalize == "scotts":
        # For Scott's Pi, we need special handling for negative values
        bars = []
        for i, (idx, row) in enumerate(df.iterrows()):
            value = row['Agreement (%)']
            bar_color = colors[i]
            display_name = row['DisplayName']
            if value >= 0:
                # For positive values, create bar from 0 to value
                bar = ax.barh(display_name, value, color=bar_color, height=0.8)
            else:
                # For negative values, create bar from value to 0
                bar = ax.barh(display_name, abs(value), color=bar_color, height=0.8, left=value)
            bars.append(bar[0])  # Each call to barh returns a container, get the first bar
    else:
        # Standard bar chart for other normalization methods
        bars = ax.barh(df['DisplayName'], df['Agreement (%)'], color=colors)
    
    # Add error bars
    # if dataset_name == "mmlu_pro":
    #     df['Std Error'] = df['Std Error'] / 2.0
    
    for i, (_, row) in enumerate(df.iterrows()):
        ax.errorbar(
            row['Agreement (%)'], row['DisplayName'],
            xerr=row['Std Error'],
            color='black',
            capsize=5,
            elinewidth=1.5,
            markeredgewidth=1.5
        )
    
    # Add percentage labels to the bars with more spacing to avoid overlap
    for i, bar in enumerate(bars):
        value = df['Agreement (%)'].iloc[i]
        error = df['Std Error'].iloc[i]
        count = df['Count'].iloc[i]
        
        # Determine appropriate x position for label based on value
        if normalize == "scotts":
            # For Scott's Pi values
            if value < 0:
                label_x = max(value + error, 0) - 0.02
                label_text = f"${{ {value:.2f} }}$"
            else:
                label_x = value + error + 0.02
                label_text = f"${{ {value:.2f} }}$"
        else:
            if value < 0:
                label_x = max(value + error, 0) - 2
                label_text = f"${{ {value:.1f} }}$%"
            else:
                label_x = value + error + 2
                label_text = f"${{ {value:.1f} }}$%"
        
        ax.text(
            label_x, 
            bar.get_y() + bar.get_height()/2, 
            label_text, 
            va='center', 
            fontweight='bold',
            fontsize=base_labelsize
        )
    
    # Add constant baseline if requested
    if show_constant_baseline:
        ax.axvline(x=constant_baseline, color='red', linestyle='--', alpha=0.7)
        
        # Label text based on normalization
        if normalize == "scotts":
            baseline_text = f'Constant Baseline ({constant_baseline:.2f})'
        else:
            baseline_text = f'Baseline ({constant_baseline:.1f}%)'
            
        ax.text(constant_baseline + (0.05 if normalize == "scotts" else 2.0), 
                len(df) * 0.05, baseline_text, 
                va='center', color='red', alpha=0.7)
    
    # Set title and labels based on normalization and dataset_name
    axis_labelsize = base_labelsize + 4
    axis_labelsize += 4
    if dataset_name == "math":
        title = "Alignment with Ground-Truth Eval"
        ax.set_xlabel("Alignment with Ground-Truth Eval" + (" (Scott's $\\pi$)" if normalize == "scotts" else " (%Agreement)"), fontsize=axis_labelsize)
    elif normalize == "scotts":
        title = f"Alignment with {ground_truth_key}"
        ax.set_xlabel("Alignment with Human 2 (Scott's $\\pi$)", fontsize=axis_labelsize)
    else:
        title = f"Alignment with {ground_truth_key}"
        ax.set_xlabel('Alignment with Human 2 (%Agreement)', fontsize=axis_labelsize)
    # ax.set_title(title, fontsize=base_titlesize, fontweight='bold')
    # ax.set_ylabel('Model', fontsize=axis_labelsize)
    
    # Set x-axis limits based on normalization method and data
    if normalize == "scotts":
        # Scott's Pi can go below 0 to negative values
        # Make sure we include the minimum value plus some margin
        x_min = min(min_agreement - 0.1, -0.1 if min_agreement < 0 else 0)
        ax.set_xlim(x_min, 1.0)  # Scott's Pi range is -1 to 1
    else:
        # For other normalization methods
        x_min = 45 if min_agreement > 50 else 0
        ax.set_xlim(x_min, 100)
    
    # Customize appearance
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=base_ticksize)
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Make top and right ticks invisible
    ax.tick_params(axis='x', top=False, bottom=True)
    ax.tick_params(axis='y', left=True, right=False)
    
    # Remove right side of the plot yticks 
    # ax.tick_params(axis='y', which='minor', bottom=False, top=False, labelbottom=False, labeltop=True)
    
    # Add a legend for the color scheme
    from matplotlib.patches import Patch
    
    legend_elements = []
    if 'Human' in df['Type'].values:
        legend_elements.append(Patch(facecolor=human_color, label='Human'))
    if 'MCQ' in df['Type'].values:
        legend_elements.append(Patch(facecolor=mcq_color, label='MCQ'))
    if 'mc_cloze' in df['Source'].values:
        legend_elements.append(Patch(facecolor=cloze_color, label='MC Cloze'))
    if 'mc_verify' in df['Source'].values:
        legend_elements.append(Patch(facecolor=verify_color, label='MC Verify'))
    # Add legend element for Judge matchers if any exist
    if any('judge' in source.lower() for source in df['Source']):
        legend_elements.append(Patch(facecolor=judge_color, label='Judge'))
    if len(matcher_indices) > 0:
        legend_elements.append(Patch(facecolor=matcher_colors[0], label='Matchers'))
        
    # if legend_elements:
    #     # Change legend position to top right to avoid overlap with bars
    #     legend = ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.18, 1), fancybox=True, edgecolor='black', frameon=True, framealpha=1, fontsize=base_labelsize, borderaxespad=1, handlelength=2, handletextpad=1)
    #     legend.get_frame().set_linewidth(1.5)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_file}")
    else:
        plt.show()
    
    # Print alignment statistics
    print(f"\nAlignment with {ground_truth_key}:")
    for i, row in df.iterrows():
        if normalize == "scotts":
            print(f"{row['Source']}: {row['Agreement (%)']: .2f} ±{row['Std Error']: .2f} ({row['Count']} samples)")
        else:
            print(f"{row['Source']}: {row['Agreement (%)']: .2f}% ±{row['Std Error']: .2f}% ({row['Count']} samples)")
    if show_constant_baseline:
        if normalize == "scotts":
            print(f"Constant Baseline: {constant_baseline: .2f}")
        else:
            print(f"Constant Baseline: {constant_baseline: .2f}%")

def analyze_error_types(ground_truth: Dict[str, int], predictions: Dict[str, Dict[str, int]], 
                        questions_to_use: Set[str], human_annotations: Dict[str, Dict] = None) -> None:
    """
    Analyze error types (false positives vs false negatives) for predictions.
    
    Args:
        ground_truth: Dictionary mapping question IDs to ground truth scores
        predictions: Dictionary mapping source names to dictionaries mapping question IDs to predicted scores
        questions_to_use: Set of question IDs to analyze
        human_annotations: Dictionary mapping question IDs to human annotation data (containing model information)
    """
    # Initialize counters for each source
    error_analysis = defaultdict(lambda: {"false_positives": 0, "false_negatives": 0, "total_errors": 0})
    
    # Initialize model-specific error counters
    model_error_analysis = defaultdict(lambda: defaultdict(lambda: {"false_positives": 0, "false_negatives": 0, "total_errors": 0}))
    
    # Track models seen for each question
    models_by_question = {}
    
    # Extract model information from human annotations if available
    if human_annotations:
        for question_id, annotation in human_annotations.items():
            if "model" in annotation:
                models_by_question[question_id] = annotation["model"]
    
    # For each question in our balanced/filtered set
    pred_distrib = {prediction: {0: 0, 1: 0} for prediction in predictions.keys()}
    gt_distrib = {0: 0, 1: 0}
    for question_id in questions_to_use:
        gt_score = ground_truth[question_id]
        if gt_score is not None:
            gt_distrib[gt_score] += 1
    print("Ground Truth Distribution:", gt_distrib)
    print("Number of questions to use:", len(questions_to_use))
    for question_id in questions_to_use:
        # Get ground truth score
        if question_id not in ground_truth:
            print(f"Question {question_id} not in ground truth")
            continue
            
        gt_score = ground_truth[question_id]
        if gt_score is None:
            print(f"Question {question_id} has no ground truth")
            continue  # Skip if ground truth is unsure
        
        # Get model for this question if available
        model = models_by_question.get(question_id, "unknown")
        
        # Check all prediction sources
        for source, prediction_dict in predictions.items():
            if question_id in prediction_dict:
                pred_score = prediction_dict[question_id]
                if pred_score is not None:
                    pred_distrib[source][pred_score] += 1
                    # Update overall error counts
                    update_error_counts(error_analysis[source], gt_score, pred_score)
                    
                    # Update model-specific error counts
                    update_error_counts(model_error_analysis[source][model], gt_score, pred_score)
                else:
                    print(f"Question {question_id} has no prediction for {source}")
            else:
                # pass
                if "cloze" not in source:
                    print(f"Question {question_id} not in {source}")
                
    print(pred_distrib)
    # Print error analysis
    print("\nError Type Analysis:")
    for source, data in error_analysis.items():
        total_errors = data["total_errors"]
        if total_errors > 0:
            fp_pct = (data["false_positives"] / total_errors) * 100
            fn_pct = (data["false_negatives"] / total_errors) * 100
            
            print(f"{source}:")
            print(f"  Total errors: {total_errors}")
            print(f"  False positives: {data['false_positives']} ({fp_pct:.1f}%)")
            print(f"  False negatives: {data['false_negatives']} ({fn_pct:.1f}%)")
            
            # Print model-specific breakdown
            if source in model_error_analysis:
                print(f"  Model-specific breakdown:")
                for model, model_data in model_error_analysis[source].items():
                    model_total = model_data["total_errors"]
                    if model_total > 0:
                        model_fp_pct = (model_data["false_positives"] / model_total) * 100
                        model_fn_pct = (model_data["false_negatives"] / model_total) * 100
                        
                        print(f"    {model}:")
                        print(f"      Total errors: {model_total}")
                        print(f"      False positives: {model_data['false_positives']} ({model_fp_pct:.1f}%)")
                        print(f"      False negatives: {model_data['false_negatives']} ({model_fn_pct:.1f}%)")

def calculate_alignment(ground_truth: Dict[str, int], 
                       mcq_responses: Dict[str, int],
                       lm_matchings: Dict[str, Dict[str, int]],
                       matchers: Set[str],
                       n_bootstrap: int = 1000,
                       normalize: str = "none") -> Tuple[pd.DataFrame, float]:
    """
    Calculate alignment between ground truth, MCQ, and LM matchers.
    
    Args:
        ground_truth: Dictionary mapping question IDs to ground truth scores
        mcq_responses: Dictionary mapping question IDs to MCQ responses
        lm_matchings: Dictionary mapping question IDs to matcher scores
        matchers: Set of matcher names to include
        n_bootstrap: Number of bootstrap samples for error calculation
        normalize: Normalization method ("none", "balance", "reweight", or "scotts")
    
    Returns:
        Tuple of (DataFrame with alignment percentages and standard errors, constant baseline)
    """
    # Prepare data structures
    results = defaultdict(lambda: {"agreements": [], "total": 0})
    
    # First pass to categorize questions by ground truth
    question_ids_by_gt = {0: [], 1: []}
    
    for question_id, gt_score in ground_truth.items():
        if gt_score is not None:
            # Categorize by ground truth score
            question_ids_by_gt[gt_score].append(question_id)
    
    # Check if we have valid ground truth values
    total_valid_gt = len(question_ids_by_gt[0]) + len(question_ids_by_gt[1])
    if total_valid_gt == 0:
        print("ERROR: No valid ground truth values found (all are None or invalid).")
        # Return empty DataFrame with expected columns
        empty_df = pd.DataFrame(columns=["Source", "Agreement (%)", "Std Error", "Type", "Count"])
        return empty_df, 50.0  # Return default constant baseline
    
    # Determine majority class for constant baseline
    majority_class = 1 if len(question_ids_by_gt[1]) >= len(question_ids_by_gt[0]) else 0
    
    # Select questions and calculate weights based on normalization method
    questions_to_use, weights = select_questions_and_calculate_weights(question_ids_by_gt, normalize)
    
    # Initialize data for Scott's Pi if needed
    scotts_data = defaultdict(lambda: {"agreements": 0, "total": 0, "gt_dist": {}, "pred_dist": {}}) if normalize == "scotts" else None
    
    # Track ground truth distribution for reporting
    gt_counts = {0: 0, 1: 0}
    
    # Create constant baseline predictions (always predicting the majority class)
    constant_baseline_preds = {}
    for question_id in questions_to_use:
        gt_score = ground_truth.get(question_id)
        if gt_score is not None:
            constant_baseline_preds[question_id] = majority_class
    
    # For each question with ground truth
    for question_id in questions_to_use:
        gt_score = ground_truth[question_id]
        if gt_score is None:
            continue
            
        # Count ground truth distribution
        gt_counts[gt_score] += 1
        
        # Default weight is 1.0 if not specified
        weight = weights.get(question_id, 1.0) if normalize == "reweight" else 1.0
        
        # Check alignment with MCQ
        if question_id in mcq_responses:
            mcq_score = mcq_responses[question_id]
            if mcq_score is not None:
                results["mcq"]["total"] += 1
                agreement = int(gt_score == mcq_score)
                results["mcq"]["agreements"].append(agreement * weight)
                
                # For Scott's Pi, collect data
                if normalize == "scotts":
                    collect_scotts_pi_data(question_id, gt_score, mcq_score, scotts_data, "mcq")
        
        # Check alignment with LM matchers
        if question_id in lm_matchings:
            for matcher, matching in lm_matchings[question_id].items():
                if matcher in matchers and matching is not None:
                    results[matcher]["total"] += 1
                    agreement = int(gt_score == matching)
                    results[matcher]["agreements"].append(agreement * weight)
                    
                    # For Scott's Pi, collect data
                    if normalize == "scotts":
                        collect_scotts_pi_data(question_id, gt_score, matching, scotts_data, matcher)
        
        # Check alignment with constant baseline (majority class predictor)
        baseline_pred = constant_baseline_preds.get(question_id)
        if baseline_pred is not None:
            results["constant_baseline"]["total"] += 1
            agreement = int(gt_score == baseline_pred)
            results["constant_baseline"]["agreements"].append(agreement * weight)
            
            # For Scott's Pi, collect data
            if normalize == "scotts":
                collect_scotts_pi_data(question_id, gt_score, baseline_pred, scotts_data, "constant_baseline")
    
    # Calculate standard constant baseline (maximum frequency of 0 or 1) - just for reporting
    total_gt = sum(gt_counts.values())
    standard_constant_baseline = max(gt_counts.values()) / total_gt * 100 if total_gt > 0 else 50
    
    # Print ground truth distribution
    print(f"Ground truth distribution: {gt_counts[1]} positive, {gt_counts[0]} negative")
    print(f"Standard constant baseline (for reference): {standard_constant_baseline:.1f}%")
    
    # Calculate agreement percentages and bootstrap standard errors
    alignment_data = []
    normalized_constant_baseline = None
    
    for source, data in results.items():
        if data["total"] == 0:
            continue
            
        # Calculate agreement metric based on normalization method
        agreement_pct, std_error = calculate_agreement_metric(
            data,
            normalize,
            scotts_data[source] if normalize == "scotts" else None,
            n_bootstrap
        )
        
        # Store normalized constant baseline value
        if source == "constant_baseline":
            normalized_constant_baseline = agreement_pct
        
        # Determine source type
        if source == "mcq":
            source_type = "MCQ"
        elif source == "constant_baseline":
            source_type = "Constant Baseline"
        else:
            source_type = "Matcher"
        
        alignment_data.append({
            "Source": source,
            "Agreement (%)": agreement_pct,
            "Std Error": std_error,
            "Type": source_type,
            "Count": data["total"]
        })
    
    # Prepare predictions for error analysis
    predictions = {
        "mcq": {qid: score for qid, score in mcq_responses.items() if score is not None},
        "constant_baseline": constant_baseline_preds
    }
    
    # Add matcher predictions
    for matcher in matchers:
        matcher_preds = {}
        for qid, match_dict in lm_matchings.items():
            if matcher in match_dict and match_dict[matcher] is not None:
                matcher_preds[qid] = match_dict[matcher]
        predictions[matcher] = matcher_preds
    
    # Run error analysis
    analyze_error_types(ground_truth, predictions, questions_to_use, human_annotations)
    
    # Use the normalized constant baseline if available, otherwise use the standard one
    final_constant_baseline = normalized_constant_baseline if normalized_constant_baseline is not None else standard_constant_baseline
    
    return pd.DataFrame(alignment_data), final_constant_baseline 

def plot_mainfig(alignment_df: pd.DataFrame, ground_truth_key: str, 
                 fig_size: Tuple[int, int] = (6, 10),
                 constant_baseline: float = 50.0, output_file: Optional[str] = None,
                 normalize: str = "scotts", dataset_name: Optional[str] = None) -> None:
    """
    Create a vertical bar plot for the main figure, showing only selected models.
    Args:
        alignment_df: DataFrame with alignment data
        ground_truth_key: Key identifying the ground truth source
        fig_size: Figure size (width, height)
        constant_baseline: Value for the constant baseline (not shown)
        output_file: If provided, save the plot to this file instead of displaying it
        normalize: Normalization method used (should be 'scotts')
        dataset_name: Name of the dataset (for output filename)
    """
    # Only keep the specified sources in the new order
    keep_sources = ["mcq", "o4-mini-JUDGE", "Qwen3_4B", "deepseek-chat-v3-0324", "human_1"]
    df = alignment_df[alignment_df['Source'].isin(keep_sources)].copy()
    # Ensure the order is as specified
    df['order'] = df['Source'].apply(lambda x: keep_sources.index(x) if x in keep_sources else 999)
    df = df.sort_values('order')
    
    # Map display names using model_name_map
    def get_display_name(name):
        label = model_name_map.get(name, name)
        return label.replace(' (Judge)', '')
    
        if 'Judge' in label:
            # Remove ' (Judge)' from label if present
            return label.replace(' (Judge)', '\n(Judge)')
        
        elif 'Qwen' in label:
            return label + "\n(Matcher)"
        elif 'DeepSeek' in label:
            return label + "\n(Matcher)"
        else:
            return label
    
    df['DisplayName'] = df['Source'].apply(get_display_name)
    # Set up colors
    # human_color = '#8ac926'  # Green
    human_color = "#FCA0FF"
    # mcq_color = '#FF8300'    # Orange
    mcq_color = '#FFB761'
    # judge_color = '#FFc933'  # Yellow for judge
    judge_color = '#FC7A7A'
    # matcher_color = '#3a86ff' # Blue for Qwen3_4B and DeepSeek V3
    matcher_color = '#A2C8FD'
    matcher_color = '#81AAFF'
    
    deepseek_color = matcher_color  # DeepSeek V3 uses the same color as Qwen3_4B
    # Assign colors in the order of keep_sources
    color_map = {
        'human_1': human_color,
        'Qwen3_4B': matcher_color,
        'deepseek-chat-v3-0324': matcher_color,  # same as Qwen3_4B
        'o4-mini-JUDGE': judge_color,
        'mcq': mcq_color,
    }
    colors = [color_map.get(src, '#cccccc') for src in df['Source']]
    
    # Create figure
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Plot vertical bars (bar)
    bars = ax.bar(df['DisplayName'], df['Agreement (%)'], color=colors, width=0.6)
    
    # Rotate x-axis labels for visibility
    plt.setp(ax.get_xticklabels(), rotation=20, ha='right', fontsize=10)
    
    # xticks_labels = ax.get_xticklabels()
    # ax.set_xticklabels(xticks_labels, rotation=0, ha='center', fontsize=10)
    # ax.tick_params(axis='x', labelsize=10)
    # plt.setp(ax.get_xticklabels(), rotation=0, ha='center')
    
    # Add error bars
    for i, (_, row) in enumerate(df.iterrows()):
        ax.errorbar(
            row['DisplayName'], row['Agreement (%)'],
            yerr=row['Std Error']/2,
            color='black',
            capsize=5,
            elinewidth=1.5,
            markeredgewidth=1.5,
            fmt='none',
            zorder=10
        )
        
    # Add value labels above bars
    for i, bar in enumerate(bars):
        value = df['Agreement (%)'].iloc[i]
        error = df['Std Error'].iloc[i]
        label_text = f"{value:.2f}"
        ax.text(
            bar.get_x() + bar.get_width()/2, 
            value + error + 0.01, 
            label_text, 
            ha='center', va='bottom', fontweight='bold', fontsize=18
        )
        
    # Set y-axis limits for Scott's Pi
    min_agreement = min(df['Agreement (%)'])
    ax.set_ylim(min(min_agreement - 0.1, -0.1 if min_agreement < 0 else 0), 1.0)
    # Set axis labels and title (no Scott's pi bracket)
    ax.set_ylabel("\\textbf{Alignment} with Human 2 $\t$ (Scott's $\pi$)", fontsize=22)
    
    # ax.set_ylabel("Alignment with Human 2 (Scott's π)", fontsize=22)
    ax.set_xlabel("")
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=18)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    # Horizontal legend at the top
    from matplotlib.patches import Patch
    
    # Define your legend patches
    mcq_patch = Patch(facecolor=mcq_color, label='MCQ')
    human_patch = Patch(facecolor=human_color, label='Human Grading')
    judge_patch = Patch(facecolor=judge_color, label='LLM-as-a-Judge')
    matcher_patch = Patch(facecolor=matcher_color, label='Answer Matching')
    empty_patch = Patch(facecolor='none', edgecolor='none', label='')

    handles = [
        mcq_patch, human_patch,  # First row
        judge_patch, empty_patch,  # Second row
        matcher_patch  # Third row
    ]
    labels = [
        'MCQ', 'Human Grading',
        '',
        'LLM-as-a-Judge (without reference answer)', 
        'Answer Matching (with reference answer)' 
    ]

    
    second_legend = ax.legend(
        handles=[empty_patch, judge_patch, matcher_patch],
        labels=labels[2:],
        loc='upper center',
        bbox_to_anchor=(0.48, 1.22),
        fancybox=True,
        edgecolor='black',
        frameon=True,
        framealpha=1,
        fontsize=18,
        ncol=1,
        columnspacing=1.5,
        handletextpad=0.8,
        borderaxespad=0.3
    )
    
    first_legend = ax.legend(
        handles=handles[:2],
        loc='upper center',
        bbox_to_anchor=(0.45, 1.22),
        fancybox=True,
        edgecolor='black',
        # frameon=True,
        framealpha=1,
        fontsize=18,
        ncol=2,
        columnspacing=4.7,
        handletextpad=0.8,
        borderaxespad=0.3
    )
    ax.add_artist(second_legend)
    
    # ax.add_artist(second_legend)

    # ax.legend(
    #     handles=handles,
    #     labels=labels,
    #     loc='upper center',
    #     bbox_to_anchor=(0.5, 1.35),
    #     fancybox=True,
    #     edgecolor='black',
    #     frameon=True,
    #     framealpha=1,
    #     fontsize=18,
    #     ncol=2,
    #     columnspacing=1.5,
    #     handletextpad=0.8,
    #     borderaxespad=0.3
    # )
    
    plt.tight_layout()
    # Save or show
    if output_file is None and dataset_name is not None:
        output_file = f"plots/mainfig_{dataset_name}.png"
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved main figure to {output_file}")
    else:
        plt.show() 

def plot_accuracy(accuracy_df: pd.DataFrame, ground_truth_key: str, 
                  fig_size: Tuple[int, int] = (12, 8), output_file: Optional[str] = None,
                  dataset_name: Optional[str] = None) -> None:
    """
    Create a bar plot showing accuracy percentages with error bars.
    
    Args:
        accuracy_df: DataFrame with accuracy data
        ground_truth_key: Key identifying the ground truth source
        fig_size: Figure size (width, height)
        output_file: If provided, save the plot to this file instead of displaying it
        dataset_name: Name of the dataset (for custom plot title)
    """
    # Map display names using model_name_map
    def get_display_name(name):
        label = model_name_map.get(name, name)
        return label
        # Remove ' (Judge)' from label if present
        # return label.replace(' (Judge)', '')
    
    # Create a copy and add display names
    df = accuracy_df.copy()
    df['DisplayName'] = df['Source'].apply(get_display_name)
    
    # Handle duplicates by preferring judge versions
    # Group by display name and keep the best performing version
    def resolve_duplicates(group):
        if len(group) == 1:
            return group.iloc[0]
        
        # Prefer judge versions (sources containing 'judge' in lowercase)
        judge_rows = group[group['Source'].str.lower().str.contains('judge', na=False)]
        if not judge_rows.empty:
            # Among judge versions, pick the one with highest accuracy
            return judge_rows.loc[judge_rows['Accuracy (%)'].idxmax()]
        else:
            # Among non-judge versions, pick the one with highest accuracy
            return group.loc[group['Accuracy (%)'].idxmax()]
    
    # Group by DisplayName and resolve duplicates
    df = df.groupby('DisplayName').apply(resolve_duplicates).reset_index(drop=True)
    
    # Sort by accuracy percentage for better visualization
    df = df.sort_values("Accuracy (%)", ascending=False)
    
    # Calculate the minimum accuracy value for x-axis limits
    min_accuracy = min(df['Accuracy (%)'])
    
    # Create figure
    fig, ax = plt.subplots(figsize=fig_size)
    
    # Increase all font sizes by 4
    base_labelsize = 20
    base_titlesize = 22
    base_ticksize = 20
    
    # Define colors based on source type
    human_color = '#8ac926'  # Green
    mcq_color = '#FF8300'    # Red 
    judge_color = '#FFc933'  # Purple for judge
    cloze_color = '#ff70a6'  # Orange for MC Cloze
    verify_color = '#ffd6ba' # Yellow for MC Verify
    
    # Create a gradient for matchers
    matcher_indices = df[df['Type'] == 'Matcher'].index
    matcher_colors = plt.cm.Blues(np.linspace(0.9, 0.4, len(matcher_indices)))
    
    # Assign colors based on type
    colors = []
    matcher_idx = 0
    
    for i, source_type in enumerate(df['Type']):
        source_name = df['Source'].iloc[i]
        if source_type == 'Human':
            colors.append(human_color)
        elif source_type == 'MCQ':
            colors.append(mcq_color)
        elif source_name == 'mc_cloze':
            colors.append(cloze_color)
        elif source_name == 'mc_verify':
            colors.append(verify_color)
        else:  # Matcher
            # Check if matcher has 'judge' in its name
            if 'judge' in source_name.lower():
                colors.append(judge_color)
            else:
                colors.append(matcher_colors[matcher_idx])
            matcher_idx += 1
    
    # Create horizontal bar plot
    bars = ax.barh(df['DisplayName'], df['Accuracy (%)'], color=colors)
    
    # Add error bars
    # if dataset_name == "mmlu_pro":
    #     df['Std Error'] = df['Std Error'] / 2.0
    
    
    for i, (_, row) in enumerate(df.iterrows()):
        ax.errorbar(
            row['Accuracy (%)'], row['DisplayName'],
            xerr=row['Std Error'],
            color='black',
            capsize=5,
            elinewidth=1.5,
            markeredgewidth=1.5
        )
    
    # Add percentage labels to the bars with more spacing to avoid overlap
    for i, bar in enumerate(bars):
        value = df['Accuracy (%)'].iloc[i]
        error = df['Std Error'].iloc[i]
        count = df['Count'].iloc[i]
        
        # Determine appropriate x position for label based on value
        label_x = value + error + 2
        label_text = f"${{ {value:.1f} }}$%"
        
        ax.text(
            label_x, 
            bar.get_y() + bar.get_height()/2, 
            label_text, 
            va='center', 
            fontweight='bold',
            fontsize=base_labelsize
        )
    
        # Set title and labels based on dataset_name
    axis_labelsize = base_labelsize + 4
    # Use fontdict to force bold label
    font_dict = {'size': axis_labelsize + 4, 'weight': 'heavy'}
    ax.set_xlabel('Accuracy by the Grader (\%)', fontdict=font_dict)
    # Multiple attempts to ensure boldness
    ax.xaxis.label.set_weight('heavy')
    ax.xaxis.label.set_fontweight('bold')
    
    # Set x-axis limits
    x_min = 45 if min_accuracy > 50 else 0
    ax.set_xlim(x_min, 100)
    max_accuracy = max(df['Accuracy (%)'])
    x_max = 80 if max_accuracy < 70 else 100
    ax.set_xlim(x_min, x_max)
    
    # Customize appearance
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=base_ticksize)
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Add a legend for the color scheme
    from matplotlib.patches import Patch
    
    legend_elements = []
    if 'Human' in df['Type'].values:
        legend_elements.append(Patch(facecolor=human_color, label='Human'))
    if 'MCQ' in df['Type'].values:
        legend_elements.append(Patch(facecolor=mcq_color, label='MCQ'))
    if 'mc_cloze' in df['Source'].values:
        legend_elements.append(Patch(facecolor=cloze_color, label='MC Cloze'))
    if 'mc_verify' in df['Source'].values:
        legend_elements.append(Patch(facecolor=verify_color, label='MC Verify'))
    # Add legend element for Judge matchers if any exist
    if any('judge' in source.lower() for source in df['Source']):
        legend_elements.append(Patch(facecolor=judge_color, label='Judge'))
    if len(matcher_indices) > 0:
        legend_elements.append(Patch(facecolor=matcher_colors[0], label='Matchers'))
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved accuracy plot to {output_file}")
    else:
        plt.show()
    
    # Print accuracy statistics
    print(f"\nAccuracy Results:")
    for i, row in df.iterrows():
        print(f"{row['Source']}: {row['Accuracy (%)']: .2f}% ±{row['Std Error']: .2f}% ({row['Count']} samples)") 

def plot_accuracy_and_alignment(accuracy_df: pd.DataFrame, alignment_df: pd.DataFrame, 
                               ground_truth_key: str, fig_size: Tuple[int, int] = (20, 8),
                               show_constant_baseline: bool = True, constant_baseline: float = 50.0,
                               output_file: Optional[str] = None, normalize: str = "none", 
                               dataset_name: Optional[str] = None) -> None:
    """
    Create a combined plot showing both accuracy and alignment in side-by-side subplots.
    
    Args:
        accuracy_df: DataFrame with accuracy data
        alignment_df: DataFrame with alignment data
        ground_truth_key: Key identifying the ground truth source
        fig_size: Figure size (width, height)
        show_constant_baseline: Whether to show a line at the constant baseline (for alignment)
        constant_baseline: Value for the constant baseline (normalized to match the chosen method)
        output_file: If provided, save the plot to this file instead of displaying it
        normalize: Normalization method used ("none", "balance", "reweight", or "scotts")
        dataset_name: Name of the dataset (for custom plot title)
    """
    # Filter out the constant_baseline row from alignment_df if it exists and we want to show it as a line
    alignment_df_filtered = alignment_df.copy()
    constant_baseline_row = alignment_df_filtered[alignment_df_filtered['Source'] == 'constant_baseline']
    if not constant_baseline_row.empty:
        # Use calculated constant baseline instead of the standard one
        constant_baseline = constant_baseline_row.iloc[0]['Agreement (%)']
        # Remove from the DataFrame since we'll show it as a line
        alignment_df_filtered = alignment_df_filtered[alignment_df_filtered['Source'] != 'constant_baseline']
    
    if not show_constant_baseline:
        # Remove constant baseline row when not showing it
        alignment_df_filtered = alignment_df_filtered[alignment_df_filtered['Source'] != 'constant_baseline']
    
    # Map display names using model_name_map
    def get_display_name(name):
        label = model_name_map.get(name, name)
        return label
    
    # Process accuracy data
    acc_df = accuracy_df.copy()
    acc_df['DisplayName'] = acc_df['Source'].apply(get_display_name)
    
    # Process alignment data
    align_df = alignment_df_filtered.copy()
    align_df['DisplayName'] = align_df['Source'].apply(get_display_name)
    
    # Handle duplicates by preferring judge versions for both datasets
    def resolve_duplicates(group):
        if len(group) == 1:
            return group.iloc[0]
        
        # Prefer judge versions (sources containing 'judge' in lowercase)
        judge_rows = group[group['Source'].str.lower().str.contains('judge', na=False)]
        if not judge_rows.empty:
            # Among judge versions, pick the one with highest score
            metric_col = 'Accuracy (%)' if 'Accuracy (%)' in group.columns else 'Agreement (%)'
            return judge_rows.loc[judge_rows[metric_col].idxmax()]
        else:
            # Among non-judge versions, pick the one with highest score
            metric_col = 'Accuracy (%)' if 'Accuracy (%)' in group.columns else 'Agreement (%)'
            return group.loc[group[metric_col].idxmax()]
    
    # Group by DisplayName and resolve duplicates
    acc_df = acc_df.groupby('DisplayName').apply(resolve_duplicates).reset_index(drop=True)
    align_df = align_df.groupby('DisplayName').apply(resolve_duplicates).reset_index(drop=True)
    
    # Sort by metric for better visualization
    # print(acc_df.columns)
    acc_df = acc_df.sort_values("Accuracy (%)", ascending=False)
    align_df = align_df.sort_values("Agreement (%)", ascending=False)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=fig_size)
    
    ax1.grid(False)
    ax2.grid(False)
    
    # Font sizes
    base_labelsize = 20
    base_ticksize = 20
    axis_labelsize = base_labelsize + 4 + 4
    legend_fontsize = base_labelsize + 2
    
    # Define colors based on source type
    human_color = '#8ac926'  # Green
    mcq_color = '#FF8300'    # Orange
    judge_color = '#FFc933'  # Yellow for judge
    cloze_color = '#ff70a6'  # Pink for MC Cloze
    verify_color = '#ffd6ba' # Light orange for MC Verify
    
    def assign_colors(df):
        # Create a gradient for matchers
        matcher_indices = df[df['Type'] == 'Matcher'].index
        matcher_colors = plt.cm.Blues(np.linspace(0.9, 0.4, len(matcher_indices)))
        
        colors = []
        matcher_idx = 0
        
        for i, source_type in enumerate(df['Type']):
            source_name = df['Source'].iloc[i]
            if source_type == 'Human':
                colors.append(human_color)
            elif source_type == 'MCQ':
                colors.append(mcq_color)
            elif source_name == 'mc_cloze':
                colors.append(cloze_color)
            elif source_name == 'mc_verify':
                colors.append(verify_color)
            elif source_type == 'Constant Baseline':
                colors.append('red')
            else:  # Matcher
                # Check if matcher has 'judge' in its name
                if 'judge' in source_name.lower():
                    colors.append(judge_color)
                else:
                    colors.append(matcher_colors[matcher_idx])
                matcher_idx += 1
        
        return colors
    
    # Plot 1: Accuracy
    acc_colors = assign_colors(acc_df)
    bars1 = ax1.barh(acc_df['DisplayName'], acc_df['Accuracy (%)'], color=acc_colors)
    
    # Add error bars for accuracy
    # if dataset_name == "mmlu_pro":
    #     acc_df['Std Error'] = acc_df['Std Error'] / 2.0
    
    for i, (_, row) in enumerate(acc_df.iterrows()):
        ax1.errorbar(
            row['Accuracy (%)'], row['DisplayName'],
            xerr=row['Std Error'],
            color='black',
            capsize=5,
            elinewidth=1.5,
            markeredgewidth=1.5
        )
    
    # Add percentage labels for accuracy
    for i, bar in enumerate(bars1):
        value = acc_df['Accuracy (%)'].iloc[i]
        error = acc_df['Std Error'].iloc[i]
        label_x = value + error + 1
        label_text = f"${{ {value:.1f} }}$%"
        
        ax1.text(
            label_x, 
            bar.get_y() + bar.get_height()/2, 
            label_text, 
            va='center', 
            fontweight='bold',
            fontsize=base_labelsize
        )
    
    # Plot 2: Alignment
    align_colors = assign_colors(align_df)
    
    # Handle Scott's Pi negative values if needed
    if normalize == "scotts":
        bars2 = []
        for i, (idx, row) in enumerate(align_df.iterrows()):
            value = row['Agreement (%)']
            bar_color = align_colors[i]
            display_name = row['DisplayName']
            if value >= 0:
                bar = ax2.barh(display_name, value, color=bar_color, height=0.8)
            else:
                bar = ax2.barh(display_name, abs(value), color=bar_color, height=0.8, left=value)
            bars2.append(bar[0])
    else:
        bars2 = ax2.barh(align_df['DisplayName'], align_df['Agreement (%)'], color=align_colors)
    
    # Add error bars for alignment
    # if dataset_name == "mmlu_pro":
    #     align_df['Std Error'] = align_df['Std Error'] / 2.0
    
    
    for i, (_, row) in enumerate(align_df.iterrows()):
        ax2.errorbar(
            row['Agreement (%)'], row['DisplayName'],
            xerr=row['Std Error'],
            color='black',
            capsize=5,
            elinewidth=1.5,
            markeredgewidth=1.5
        )
    
    # Add percentage labels for alignment
    for i, bar in enumerate(bars2):
        value = align_df['Agreement (%)'].iloc[i]
        error = align_df['Std Error'].iloc[i]
        
        if normalize == "scotts":
            if value < 0:
                label_x = max(value + error, 0) - 0.02
                label_text = f"${{ {value:.2f} }}$"
            else:
                label_x = value + error + 0.02
                label_text = f"${{ {value:.2f} }}$"
        else:
            if value < 0:
                label_x = max(value + error, 0) - 2
                label_text = f"${{ {value:.1f} }}$%"
            else:
                label_x = value + error + 2
                label_text = f"${{ {value:.1f} }}$%"
        
        ax2.text(
            label_x, 
            bar.get_y() + bar.get_height()/2, 
            label_text, 
            va='center', 
            fontweight='bold',
            fontsize=base_labelsize
        )
    
    # Add constant baseline for alignment if requested
    if show_constant_baseline:
        ax2.axvline(x=constant_baseline, color='red', linestyle='--', alpha=0.7)
        
        if normalize == "scotts":
            baseline_text = f'Constant Baseline ({constant_baseline:.2f})'
        else:
            baseline_text = f'Baseline ({constant_baseline:.1f}%)'
            
        ax2.text(constant_baseline + (0.05 if normalize == "scotts" else 2.0), 
                len(align_df) * 0.05, baseline_text, 
                va='center', color='red', alpha=0.7)
    
    # Set axis labels and limits
    # Accuracy subplot
    ax1.set_xlabel('Accuracy by the Grader (\%)', fontsize=axis_labelsize, fontweight='bold')
    min_accuracy = min(acc_df['Accuracy (%)'])
    x_min_acc = 50 if min_accuracy > 50 else 0
    max_accuracy = max(acc_df['Accuracy (%)'])
    x_max_acc = 50
    while x_max_acc < max_accuracy:
        x_max_acc += 10
    ax1.set_xlim(x_min_acc, x_max_acc)
    
    # Alignment subplot
    if dataset_name == "math":
        ax2.set_xlabel("Alignment with Ground-Truth Eval" + (" (Scott's $\\pi$)" if normalize == "scotts" else " (%Agreement)"), fontsize=axis_labelsize, fontweight='bold')
    elif normalize == "scotts":
        ax2.set_xlabel("Alignment with Human 2 (Scott's $\\pi$)", fontsize=axis_labelsize, fontweight='bold')
    else:
        ax2.set_xlabel('Alignment with Human 2 (%Agreement)', fontsize=axis_labelsize, fontweight='bold')
    
    min_agreement = min(align_df['Agreement (%)'])
    if normalize == "scotts":
        x_min_align = min(min_agreement - 0.1, -0.1 if min_agreement < 0 else 0)
        ax2.set_xlim(x_min_align, 1.0)
    else:
        x_min_align = 45 if min_agreement > 50 else 0
        ax2.set_xlim(x_min_align, 100)
    
    # Customize appearance for both subplots
    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=base_ticksize)
        ax.grid(False) 
        
        # Make top and right ticks invisible
        ax.tick_params(axis='x', top=True, bottom=True)
        ax.tick_params(axis='y', left=True, right=False)
    
        # ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Create common legend at the top
    from matplotlib.patches import Patch
    
    # Collect all unique types from both dataframes
    all_types = set(acc_df['Type'].values) | set(align_df['Type'].values)
    all_sources = set(acc_df['Source'].values) | set(align_df['Source'].values)
    
    legend_elements = []
    if 'MCQ' in all_types:
        legend_elements.append(Patch(facecolor=mcq_color, label='MCQ'))
    if 'mc_cloze' in all_sources:
        legend_elements.append(Patch(facecolor=cloze_color, label='MC Cloze'))
    if 'mc_verify' in all_sources:
        legend_elements.append(Patch(facecolor=verify_color, label='MC Verify'))
    if any('judge' in source.lower() for source in all_sources):
        legend_elements.append(Patch(facecolor=judge_color, label='Judge'))
    if 'Matcher' in all_types:
        legend_elements.append(Patch(facecolor=plt.cm.Blues(0.7), label='Matchers'))
    
    if 'Human' in all_types:
        legend_elements.append(Patch(facecolor=human_color, label='Human'))
    
    if legend_elements:
        # Create legend for the main plot
        # fig.legend(
        #     handles=legend_elements,
        #     loc='upper center',
        #     bbox_to_anchor=(0.5, 0.95),
        #     fancybox=True,
        #     edgecolor='black',
        #     frameon=True,
        #     framealpha=1,
        #     fontsize=legend_fontsize,
        #     ncol=5,
        #     columnspacing=2,
        #     handletextpad=1,
        #     borderaxespad=0.5
        # )
        
        # # Create a separate figure with just the legend
        legend_fig, legend_ax = plt.subplots(figsize=(12, 2))
        legend_ax.axis('off')  # Hide the axes
        
        # Create legend in the separate figure
        legend = legend_fig.legend(
            handles=legend_elements,
            loc='center',
            fancybox=True,
            edgecolor='black',
            frameon=True,
            framealpha=1,
            fontsize=legend_fontsize,
            ncol=5,
            columnspacing=2,
            handletextpad=1
        )
        
        # Save the legend-only figure
        legend_output_file = output_file.replace('.png', '_legend.png')
        legend_output_file = output_file.replace('.pdf', '_legend.pdf')
        legend_fig.savefig(legend_output_file, dpi=300, bbox_inches='tight', pad_inches=0.1)
        print(f"Saved legend to {legend_output_file}")
        plt.close(legend_fig)  # Close the legend figure to free memory
    
    plt.tight_layout()
    # Decrease the distance between the two subplots (allow overlap if required)
    plt.subplots_adjust(top=0.83, wspace=0.34)  # wspace < 0 allows overlap
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved combined plot to {output_file}")
    else:
        plt.show()
    
    # Print statistics for both plots
    print(f"\nAccuracy Results:")
    for i, row in acc_df.iterrows():
        print(f"{row['Source']}: {row['Accuracy (%)']: .2f}% ±{row['Std Error']: .2f}% ({row['Count']} samples)")
    
    print(f"\nAlignment with {ground_truth_key}:")
    for i, row in align_df.iterrows():
        if normalize == "scotts":
            print(f"{row['Source']}: {row['Agreement (%)']: .2f} ±{row['Std Error']: .2f} ({row['Count']} samples)")
        else:
            print(f"{row['Source']}: {row['Agreement (%)']: .2f}% ±{row['Std Error']: .2f}% ({row['Count']} samples)")
    if show_constant_baseline:
        if normalize == "scotts":
            print(f"Constant Baseline: {constant_baseline: .2f}")
        else:
            print(f"Constant Baseline: {constant_baseline: .2f}%") 