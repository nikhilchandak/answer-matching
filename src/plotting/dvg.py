#!/usr/bin/env python3
import os
import json
import argparse
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import math
from typing import Dict, List, Tuple

def parse_args():
    parser = argparse.ArgumentParser(description='Create scaling plot for discrimination, verification, and generation accuracy')
    parser.add_argument('--dataset', choices=['math', 'gpqa_diamond', 'mmlu_pro'], required=True, 
                        help='Dataset to analyze')
    parser.add_argument('--thinking', action='store_true', 
                        help='Use thinking models instead of non_thinking models')
    parser.add_argument('--output', default=None,
                        help='Output plot filename')
    return parser.parse_args()

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

def get_model_size(model_name):
    """Extract model size in billions from model name."""
    # Handle both formats: Qwen3-1.7B and qwen2.5-7b
    match = re.search(r'[-_](\d+\.?\d*)B', model_name, re.IGNORECASE)
    if match:
        return float(match.group(1))
    return None

def get_model_folders(base_path, dataset, use_thinking):
    """Get list of model folders for the specified configuration."""
    if use_thinking:
        suffix = "_thinking"
    else:
        suffix = "_non_thinking"
        
    model_folders = []
    
    for subfolder in ["_mcq", "_verify", "_free"]:
        folder_path = os.path.join(base_path, f"{dataset}{subfolder}")
        # For mmlu_pro with mcq or free, add stratified_sample to the path
        if dataset == "mmlu_pro" and subfolder in ["_mcq", "_free"]:
            folder_path = os.path.join(folder_path, "stratified_sample")
        
        if os.path.exists(folder_path):
            for model_folder in os.listdir(folder_path):
                # Only include models that match the requested thinking/non-thinking type
                if (use_thinking and model_folder.endswith("_thinking") and not model_folder.endswith("_non_thinking")) or \
                   (not use_thinking and model_folder.endswith("_non_thinking")):
                    model_folders.append((model_folder, folder_path))
    
    # Get unique model names
    unique_models = set(model[0] for model in model_folders)
    return unique_models, model_folders

def compute_discrimination_accuracy(base_path, dataset, model_name):
    """Compute discrimination accuracy from MCQ data."""
    folder_path = os.path.join(base_path, f"{dataset}_mcq", model_name)
    # For mmlu_pro, add stratified_sample to the path
    if dataset == "mmlu_pro":
        folder_path = os.path.join(base_path, f"{dataset}_mcq", "stratified_sample", model_name)
    
    samples_path = os.path.join(folder_path, "samples.jsonl")
    
    if not os.path.exists(samples_path):
        return None, None
    
    total = 0
    correct = 0
    results = []
    
    with open(samples_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            total += 1
            # Check if exact_match is 1 (handle int, string, and float)
            exact_match = data.get('exact_match', '0')
            try:
                is_correct = float(exact_match) == 1.0
            except (ValueError, TypeError):
                is_correct = str(exact_match) == '1'
            
            if is_correct:
                correct += 1
            results.append(1 if is_correct else 0)
    
    if total == 0:
        return None, None
    
    acc = correct / total
    # Standard error of mean for binomial distribution
    sem = math.sqrt((acc * (1 - acc)) / total) if total > 0 else 0
    
    return acc, sem

def compute_verification_accuracy(base_path, dataset, model_name):
    """Compute verification accuracy."""
    folder_path = os.path.join(base_path, f"{dataset}_verify", model_name)
    samples_path = os.path.join(folder_path, "samples.jsonl")
    
    if not os.path.exists(samples_path):
        return None, None, None
    
    counts = defaultdict(int)
    correct = defaultdict(int)
    
    with open(samples_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            question_id = data.get('question_id')
            if question_id:
                counts[question_id] += 1
                # Check if exact_match is 1 (handle int, string, and float)
                exact_match = data.get('exact_match', '0')
                try:
                    is_correct = float(exact_match) == 1.0
                except (ValueError, TypeError):
                    is_correct = str(exact_match) == '1'
                    
                if is_correct:
                    correct[question_id] += 1
    
    # Calculate per-question accuracy
    question_results = []
    verification_predictions = {}
    for qid in counts:
        if counts[qid] > 0:
            result = 1 if correct[qid] == counts[qid] else 0
            question_results.append(result)
            verification_predictions[qid] = result
    
    if not question_results:
        return None, None, None
    
    acc = np.mean(question_results)
    # Standard error of mean
    sem = np.std(question_results, ddof=1) / np.sqrt(len(question_results)) if len(question_results) > 1 else 0
    
    return acc, sem, verification_predictions

def compute_generation_accuracy(base_path, dataset, model_name):
    """Compute generation accuracy."""
    folder_path = os.path.join(base_path, f"{dataset}_free", model_name)
    # For mmlu_pro, add stratified_sample to the path
    if dataset == "mmlu_pro":
        folder_path = os.path.join(base_path, f"{dataset}_free", "stratified_sample", model_name)
    
    samples_path = os.path.join(folder_path, "samples.jsonl")
    
    if not os.path.exists(samples_path):
        return None, None, None
    
    total = 0
    correct = 0
    results = []
    ground_truth = {}
    
    with open(samples_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            question_id = data.get('question_id')
            if not question_id:
                continue
                
            total += 1
            
            # For mmlu_pro and gpqa_diamond, use score_deepseek-chat instead of exact_match
            if dataset in ['mmlu_pro', 'gpqa_diamond']:
                score_field = 'score_deepseek-chat-v3-0324'
                score_value = data.get(score_field, '0')
            else:
                score_value = data.get('exact_match', '0')
            
            # Handle int, string, and float
            try:
                is_correct = float(score_value) == 1.0
            except (ValueError, TypeError):
                is_correct = str(score_value) == '1'
                
            if is_correct:
                correct += 1
            result = 1 if is_correct else 0
            results.append(result)
            ground_truth[question_id] = score_value
    
    if total == 0:
        return None, None, None
    
    acc = correct / total
    # Standard error of mean for binomial distribution
    sem = math.sqrt((acc * (1 - acc)) / total) if total > 0 else 0
    
    return acc, sem, ground_truth

def compute_scotts_pi_gt_vs_verify(ground_truth, verification_predictions):
    """Compute Scott's Pi between ground truth and verification predictions."""
    if not ground_truth or not verification_predictions:
        return None
    
    print("ground_truth", ground_truth)
    print("verification_predictions", verification_predictions)
    
    #make copies of ground truth and verification predictions
    ground_truth_copy = ground_truth.copy()
    verification_predictions_copy = verification_predictions.copy()
    
    #convert ground truth and verification predictions keys to int
    ground_truth_copy = {int(k): v for k, v in ground_truth_copy.items()}
    verification_predictions_copy = {int(k): v for k, v in verification_predictions_copy.items()}
    
    # Find common question IDs
    common_ids = set(ground_truth_copy.keys()) & set(verification_predictions_copy.keys())
    print("common", common_ids)
    if not common_ids:
        return None
    
    # Count agreements and distribution
    total = len(common_ids)
    agreements = 0
    gt_dist = {0: 0, 1: 0}
    verify_dist = {0: 0, 1: 0}
    
    for qid in common_ids:
        gt_result = int(ground_truth_copy[qid])
        verify_result = int(verification_predictions_copy[qid])
        
        gt_dist[gt_result] += 1
        verify_dist[verify_result] += 1
        
        if gt_result == verify_result:
            agreements += 1
    
    observed_agreement = agreements / total if total > 0 else 0
    
    # Calculate Scott's Pi
    pi_value = calculate_scotts_pi(observed_agreement, gt_dist, verify_dist, total)
    
    return pi_value, observed_agreement, total

def create_scaling_plot(results, output_file, dataset, use_thinking):
    """Create the scaling plot with 3 lines for discrimination, verification, and generation."""
    plt.figure(figsize=(10, 6))
    
    # Sort by model size
    sorted_results = sorted(results.items(), key=lambda x: x[0])
    sizes = [size for size, _ in sorted_results]
    
    # Get accuracies and standard errors
    disc_accs = [accs['discrimination'][0] for _, accs in sorted_results if accs['discrimination'][0] is not None]
    disc_sems = [accs['discrimination'][1] for _, accs in sorted_results if accs['discrimination'][0] is not None]
    disc_sizes = [size for size, accs in sorted_results if accs['discrimination'][0] is not None]
    
    ver_accs = [accs['verification'][0] for _, accs in sorted_results if accs['verification'][0] is not None]
    ver_sems = [accs['verification'][1] for _, accs in sorted_results if accs['verification'][0] is not None]
    ver_sizes = [size for size, accs in sorted_results if accs['verification'][0] is not None]
    
    gen_accs = [accs['generation'][0] for _, accs in sorted_results if accs['generation'][0] is not None]
    gen_sems = [accs['generation'][1] for _, accs in sorted_results if accs['generation'][0] is not None]
    gen_sizes = [size for size, accs in sorted_results if accs['generation'][0] is not None]
    
    # Plot the lines with error bars
    if disc_sizes and disc_accs:
        plt.errorbar(disc_sizes, disc_accs, yerr=disc_sems, fmt='o-', label='Discrimination', color='blue', capsize=4)
    
    if ver_sizes and ver_accs:
        plt.errorbar(ver_sizes, ver_accs, yerr=ver_sems, fmt='o-', label='Verification', color='green', capsize=4)
    
    if gen_sizes and gen_accs:
        plt.errorbar(gen_sizes, gen_accs, yerr=gen_sems, fmt='o-', label='Generation', color='red', capsize=4)
    
    # Set x-axis to log scale
    plt.xscale('log')
    
    # Use specific model sizes for x-ticks instead of standard log formatting
    plt.xticks(sizes, [str(size) for size in sizes])
    
    plt.xlabel('Model Size (B parameters)')
    plt.ylabel('Accuracy')
    
    # Simplify title to only mention dataset
    plt.title(f'{dataset.upper()} Dataset')
    
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add thin vertical lines at each model size
    for size in sizes:
        plt.axvline(x=size, color='gray', linestyle='--', alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Plot saved to {output_file}")

def create_gt_vs_verify_scotts_pi_plot(pi_results, output_file, dataset, use_thinking):
    """Create a plot showing Scott's Pi values between ground truth and verification across model sizes."""
    plt.figure(figsize=(10, 6))
    
    # Sort by model size
    sorted_results = sorted(pi_results.items(), key=lambda x: x[0])
    sizes = [size for size, _ in sorted_results]
    
    # Get Scott's Pi values
    pi_values = [res['gt_vs_verify'][0] for _, res in sorted_results if res['gt_vs_verify'] is not None]
    valid_sizes = [size for size, res in sorted_results if res['gt_vs_verify'] is not None]
    
    # Plot the line
    if valid_sizes and pi_values:
        plt.plot(valid_sizes, pi_values, 'o-', label='Ground Truth vs Verification', color='purple')
    
    # Set x-axis to log scale
    plt.xscale('log')
    
    # Use specific model sizes for x-ticks
    plt.xticks(sizes, [str(size) for size in sizes])
    
    plt.xlabel('Model Size (B parameters)')
    plt.ylabel('Scott\'s Pi')
    plt.title(f'Scott\'s Pi Between Ground Truth and Verification - {dataset.upper()} Dataset')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Add thin vertical lines at each model size
    for size in sizes:
        plt.axvline(x=size, color='gray', linestyle='--', alpha=0.2)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"Ground Truth vs Verification Scott's Pi plot saved to {output_file}")

def main():
    args = parse_args()
    base_path = "/is/cluster/fast/nchandak/qaevals/judge_outputs"
    
    # Get model folders
    unique_models, model_folders = get_model_folders(base_path, args.dataset, args.thinking)
    
    # Compute accuracies for each model
    results = {}
    gt_vs_verify_pi_results = {}
    
    for model_name in unique_models:
        model_size = get_model_size(model_name)
        if model_size is None:
            continue
        
        disc_acc, disc_sem = compute_discrimination_accuracy(base_path, args.dataset, model_name)
        ver_acc, ver_sem, verification_predictions = compute_verification_accuracy(base_path, args.dataset, model_name)
        gen_acc, gen_sem, ground_truth = compute_generation_accuracy(base_path, args.dataset, model_name)
        
        results[model_size] = {
            'model': model_name,
            'discrimination': (disc_acc, disc_sem),
            'verification': (ver_acc, ver_sem),
            'generation': (gen_acc, gen_sem)
        }
        
        # Compute Scott's Pi between ground truth and verification predictions
        gt_vs_verify_pi = compute_scotts_pi_gt_vs_verify(ground_truth, verification_predictions)
        gt_vs_verify_pi_results[model_size] = {
            'model': model_name,
            'gt_vs_verify': gt_vs_verify_pi
        }
        print(gt_vs_verify_pi)
        
        print(f"Model: {model_name} (Size: {model_size}B)")
        print(f"  Discrimination: {f'{disc_acc:.4f}' if disc_acc is not None else 'N/A'} ± {f'{disc_sem:.4f}' if disc_sem is not None else 'N/A'}")
        print(f"  Verification: {f'{ver_acc:.4f}' if ver_acc is not None else 'N/A'} ± {f'{ver_sem:.4f}' if ver_sem is not None else 'N/A'}")
        print(f"  Generation: {f'{gen_acc:.4f}' if gen_acc is not None else 'N/A'} ± {f'{gen_sem:.4f}' if gen_sem is not None else 'N/A'}")
        
        # Print Scott's Pi between ground truth and verification
        if gt_vs_verify_pi:
            pi_val, agreement, total = gt_vs_verify_pi
            print(f"  Ground Truth vs Verification Scott's Pi: {pi_val:.4f} (Agreement: {agreement:.4f}, N={total})")
    
    output_path = args.output
    if output_path is None:
        output_path = f"dvg_plot/{args.dataset}/{'thinking' if args.thinking else 'non_thinking'}.png"
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create scaling plot
    create_scaling_plot(results, output_path, args.dataset, args.thinking)
    
    # Create Ground Truth vs Verification Scott's Pi plot
    scotts_pi_output_path = f"dvg_plot/{args.dataset}/{'thinking' if args.thinking else 'non_thinking'}_scotts.png"
    create_gt_vs_verify_scotts_pi_plot(gt_vs_verify_pi_results, scotts_pi_output_path, args.dataset, args.thinking)

if __name__ == "__main__":
    main()
