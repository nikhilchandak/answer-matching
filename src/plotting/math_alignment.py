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
    calculate_alignment
)

class MathAlignmentAnalyzer:
    def __init__(self, 
                 mcq_file: str,
                 lm_matchings_file: str,
                 exclude_matchers: List[str] = None):
        """
        Initialize the MathAlignmentAnalyzer with file paths.
        
        Args:
            mcq_file: File containing model MCQ responses
            lm_matchings_file: File containing model free-form responses and LM matchings
            exclude_matchers: List of matcher models to exclude from analysis
        """
        self.mcq_file = mcq_file
        self.lm_matchings_file = lm_matchings_file
        self.exclude_matchers = exclude_matchers or ["qwen-2.5-14b-instruct"]
        
        # Initialize data containers
        self.ground_truth = {}      # Will store the ground truth (from exact_match in lm_matchings_file)
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
        
        # Track all potential matchers to help with debugging
        all_potential_matchers = set()
        
        for data in lm_data:
            question_id = data.get("question_id")
            exact_match = data.get("exact_match")
            
            if question_id is not None and exact_match is not None:
                # Extract ground truth from exact_match field (integer)
                self.ground_truth[question_id] = exact_match if exact_match in [0, 1] else None
                
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
                        
                        # Handle string scores (convert to int if possible)
                        try:
                            score_value = int(value) if value in ["0", "1"] else None
                            matchings[matcher_name] = score_value
                        except (ValueError, TypeError):
                            matchings[matcher_name] = None
                
                self.lm_matchings[question_id] = matchings
        
        print(f"All potential matchers found in data: {', '.join(all_potential_matchers)}")
        print(f"Matchers after exclusion: {', '.join(self.matchers)}")
        
        # Load MCQ responses
        mcq_data = load_jsonl_file(self.mcq_file)
        print(f"Loaded {len(mcq_data)} entries from MCQ file")
        
        for data in mcq_data:
            question_id = data.get("question_id")
            exact_match = data.get("exact_match")
            
            if question_id is not None and exact_match is not None:
                self.mcq_responses[question_id] = exact_match if exact_match in [0, 1] else None
        
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
    parser = argparse.ArgumentParser(description='Analyze alignment between ground truth and model matchings for math problems.')
    
    # Add arguments
    parser.add_argument('--mcq-file', type=str,
                        default="/is/cluster/fast/nchandak/qaevals/judge_outputs/math_mcq/qwen2.5-7b-it_non_thinking/samples.jsonl",
                        help='File containing model MCQ responses')
    
    parser.add_argument('--lm-matchings-file', type=str,
                        default="/is/cluster/fast/nchandak/qaevals/judge_outputs/math_free/qwen2.5-7b-it_non_thinking/samples.jsonl",
                        help='File containing model free-form responses and LM matchings')
    
    parser.add_argument('--plots-dir', type=str, default="plots/math",
                        help='Directory to save plots')
    
    parser.add_argument('--expt-name', type=str, default="",
                        help='Experiment name for creating a subdirectory within plots directory')
    
    parser.add_argument('--output-file', type=str, default=None,
                        help='If provided, save the plot to this file instead of displaying it')
    
    parser.add_argument('--no-constant-baseline', action='store_true',
                        help='Do not show the constant baseline line on the plot')
    
    parser.add_argument('--exclude-matchers', type=str, nargs='+', 
                        default=["qwen-2.5-14b-instruct"],
                        help='List of matcher models to exclude from analysis')
    
    parser.add_argument('--normalize', type=str, choices=["none", "balance", "reweight", "scotts"], default="none",
                        help='Normalization method to use ("none", "balance", "reweight", or "scotts")')
    
    # Parse arguments
    args = parser.parse_args()
    
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
    analyzer = MathAlignmentAnalyzer(
        mcq_file=args.mcq_file,
        lm_matchings_file=args.lm_matchings_file,
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
        
        filename = f"math_alignment_{baseline}_{args.normalize}.png"
        output_file = os.path.join(plots_dir, filename)
    
    # Plot results
    plot_alignment(
        alignment_df, 
        "Ground Truth", 
        show_constant_baseline=not args.no_constant_baseline,
        constant_baseline=constant_baseline,
        output_file=output_file,
        normalize=args.normalize
    )

if __name__ == "__main__":
    main() 