import os
import json
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import defaultdict

# Import utility functions
from utils import (
    load_jsonl_file, 
    get_balanced_question_ids,
    plot_alignment, 
    analyze_error_types,
    calculate_alignment,
    process_list_metrics
)

class DSAlignmentAnalyzer:
    def __init__(self, 
                 mcq_file: str,
                 lm_matchings_file: str,
                 dataset: str = "mmlu_pro",
                 exclude_matchers: List[str] = None):
        """
        Initialize the DSAlignmentAnalyzer with file paths.
        
        Args:
            mcq_file: File containing model MCQ responses
            lm_matchings_file: File containing model free-form responses and LM matchings
            dataset: Dataset name (e.g., mmlu_pro, gpqa)
            exclude_matchers: List of matcher models to exclude from analysis
        """
        self.mcq_file = mcq_file
        self.lm_matchings_file = lm_matchings_file
        self.dataset = dataset
        self.exclude_matchers = exclude_matchers or ["deepseek-chat-v3-0324"]
        
        # Initialize data containers
        self.ground_truth = {}      # Will store the ground truth (from deepseek score in lm_matchings_file)
        self.mcq_responses = {}     # Will store MCQ responses
        self.lm_matchings = {}      # Will store LM matchings
        self.matchers = set()       # Will store all matcher names
        
        # Load all data
        self.load_data()
        
    def load_data(self) -> None:
        """Load data from MCQ and LM matchings files."""
        # Load LM matchings and extract ground truth
        lm_data = load_jsonl_file(self.lm_matchings_file)
        print(f"Loaded {len(lm_data)} entries from LM matchings file")
        
        # Process list-based metrics if needed
        if self.dataset == "gpqa" or self.dataset == "gpqa_diamond" or self.dataset.startswith("gpqa_"):
            print(f"Processing {self.dataset} data with list metrics...")
            lm_data = process_list_metrics(lm_data)
        
        # Track all potential matchers to help with debugging
        all_potential_matchers = set()
        
        for data in lm_data:
            question_id = data.get("question_id")
            deepseek_score = data.get("score_deepseek-chat-v3-0324")
            
            if question_id is not None and deepseek_score is not None:
                # Handle string or integer scores
                if isinstance(deepseek_score, str):
                    # Handle string scores
                    score_value = int(deepseek_score) if deepseek_score in ["0", "1"] else None
                elif isinstance(deepseek_score, int):
                    # Handle integer scores
                    score_value = deepseek_score if deepseek_score in [0, 1] else None
                elif isinstance(deepseek_score, list):
                    # Handle list scores
                    score_value = deepseek_score[0] if len(deepseek_score) > 0 else None
                else:
                    # If it's neither a string nor an integer, set to None
                    score_value = None
                    
                self.ground_truth[question_id] = score_value
                
                # Extract matcher scores
                matchings = {}
                for key, value in data.items():
                    if key.startswith("score_"):
                        matcher_name = key.replace("score_", "")
                        all_potential_matchers.add(matcher_name)
                        
                        # Skip excluded matchers
                        if self.exclude_matchers and matcher_name in self.exclude_matchers:
                            continue
                            
                        # Add matcher to the set of tracked matchers
                        self.matchers.add(matcher_name)
                        
                        # Handle string or integer scores
                        try:
                            if isinstance(value, str):
                                score_value = int(value) if value in ["0", "1"] else None
                            elif isinstance(value, int):
                                score_value = value if value in [0, 1] else None
                            else:
                                score_value = None
                            matchings[matcher_name] = score_value
                        except (ValueError, TypeError):
                            matchings[matcher_name] = None
                
                self.lm_matchings[question_id] = matchings
        
        print(f"All potential matchers found in data: {', '.join(all_potential_matchers)}")
        print(f"Matchers after exclusion: {', '.join(self.matchers)}")
        
        # Load MCQ responses
        mcq_data = load_jsonl_file(self.mcq_file)
        print(f"Loaded {len(mcq_data)} entries from MCQ file")
        
        # Process list-based metrics for MCQ data if needed
        if self.dataset == "gpqa" or self.dataset == "gpqa_diamond" or self.dataset.startswith("gpqa_"):
            print(f"Processing {self.dataset} MCQ data with list metrics...")
            mcq_data = process_list_metrics(mcq_data)
        
        for data in mcq_data:
            question_id = data.get("question_id")
            exact_match = data.get("exact_match")
            
            if question_id is not None and exact_match is not None:
                # Handle string or integer scores
                if isinstance(exact_match, str):
                    self.mcq_responses[question_id] = int(exact_match) if exact_match in ["0", "1"] else None
                elif isinstance(exact_match, int):
                    self.mcq_responses[question_id] = exact_match if exact_match in [0, 1] else None
                else:
                    self.mcq_responses[question_id] = None
        
        print(f"Processed {len(self.ground_truth)} questions with ground truth")
        print(f"Found {len(self.matchers)} matchers: {', '.join(self.matchers)}")
        if self.exclude_matchers:
            print(f"Excluded matchers: {', '.join(self.exclude_matchers)}")
    
    def calculate_alignment(self, n_bootstrap: int = 1000, normalize: str = "none") -> pd.DataFrame:
        """
        Calculate alignment between ground truth, MCQ, and LM matchers.
        
        Args:
            n_bootstrap: Number of bootstrap samples for error calculation
            normalize: Normalization method ("none", "balance", "reweight", or "scotts")
        
        Returns:
            Tuple of (DataFrame with alignment percentages and standard errors, normalized constant baseline)
        """
        return calculate_alignment(
            self.ground_truth,
            self.mcq_responses,
            self.lm_matchings,
            self.matchers,
            n_bootstrap,
            normalize
        )

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Analyze alignment between deepseek ground truth and model matchings.')
    
    # Add arguments
    parser.add_argument('--dataset', type=str, default="mmlu_pro",
                        help='Dataset name (e.g., mmlu_pro, gpqa)')
    
    parser.add_argument('--mcq-file', type=str,
                        default=None,
                        help='File containing model MCQ responses')
    
    parser.add_argument('--lm-matchings-file', type=str,
                        default=None,
                        help='File containing model free-form responses and LM matchings')
    
    parser.add_argument('--plots-dir', type=str, default=None,
                        help='Directory to save plots')
    
    parser.add_argument('--expt-name', type=str, default="",
                        help='Experiment name for creating a subdirectory within plots directory')
    
    parser.add_argument('--output-file', type=str, default=None,
                        help='If provided, save the plot to this file instead of displaying it')
    
    parser.add_argument('--no-constant-baseline', action='store_true',
                        help='Do not show the constant baseline line on the plot')
    
    parser.add_argument('--exclude-matchers', type=str, nargs='+', 
                        default=["deepseek-chat-v3-0324"],
                        help='List of matcher models to exclude from analysis')
    
    parser.add_argument('--normalize', type=str, choices=["none", "balance", "reweight", "scotts"], default="none",
                        help='Normalization method to use ("none", "balance", "reweight", or "scotts")')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set default file paths if not provided
    if args.mcq_file is None:
        if args.dataset == "mmlu_pro":
            args.mcq_file = f"/is/cluster/fast/nchandak/qaevals/judge_outputs/{args.dataset}_mcq/stratified_sample/Qwen3-4B_thinking/samples.jsonl"
        else:
            args.mcq_file = f"/is/cluster/fast/nchandak/qaevals/judge_outputs/{args.dataset}_mcq/Qwen3-4B_thinking/samples.jsonl"
    
    if args.lm_matchings_file is None:
        if args.dataset == "mmlu_pro":
            args.lm_matchings_file = f"/is/cluster/fast/nchandak/qaevals/judge_outputs/{args.dataset}_free/stratified_sample/Qwen3-4B_thinking/samples.jsonl"
        else:
            args.lm_matchings_file = f"/is/cluster/fast/nchandak/qaevals/judge_outputs/{args.dataset}_free/Qwen3-4B_thinking/samples.jsonl"
    
    # Set default plots directory if not provided
    if args.plots_dir is None:
        args.plots_dir = f"plots/{args.dataset}"
    
    # Create plots directory structure
    if args.expt_name:
        # If experiment name is provided, create a subdirectory with experiment name and normalization method
        plots_dir = os.path.join(args.plots_dir, args.expt_name, args.normalize)
    else:
        # Otherwise, just use normalization method as subdirectory
        plots_dir = os.path.join(args.plots_dir, args.normalize)
        
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
        print(f"Created plots directory: {plots_dir}")
    
    # Initialize analyzer with provided arguments
    analyzer = DSAlignmentAnalyzer(
        mcq_file=args.mcq_file,
        lm_matchings_file=args.lm_matchings_file,
        dataset=args.dataset,
        exclude_matchers=args.exclude_matchers
    )
    
    # Calculate alignment
    alignment_result = analyzer.calculate_alignment(normalize=args.normalize)
    alignment_df = alignment_result[0]
    constant_baseline = alignment_result[1]
    
    # Generate automatic output filename if not provided
    output_file = args.output_file
    if output_file is None:
        # Create filename based on parameters
        baseline = "no_baseline" if args.no_constant_baseline else "with_baseline"
        
        filename = f"ds_alignment_{baseline}_{args.normalize}.png"
        output_file = os.path.join(plots_dir, filename)
    
    # Plot results
    plot_alignment(
        alignment_df, 
        "Deepseek-v3-0324", 
        show_constant_baseline=not args.no_constant_baseline,
        constant_baseline=constant_baseline,
        output_file=output_file,
        normalize=args.normalize
    )

if __name__ == "__main__":
    main() 