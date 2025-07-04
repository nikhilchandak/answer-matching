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
    analyze_error_types,
    calculate_alignment,
    process_list_metrics
)

# Import shared plotting utilities
from plot_all_utils import (
    extract_model_size,
    plot_alignment_by_size
)

class DSAlignmentMultiAnalyzer:
    def __init__(self, 
                 base_dir: str,
                 model_list: List[str],
                 thinking_type: str,
                 dataset: str = "mmlu_pro",
                 exclude_matchers: List[str] = None,
                 normalize: str = "none"):
        """
        Initialize the DSAlignmentMultiAnalyzer for multiple models.
        
        Args:
            base_dir: Base directory containing model outputs
            model_list: List of model names to analyze
            thinking_type: Type of thinking (thinking or non_thinking)
            dataset: Dataset name (e.g., mmlu_pro, gpqa)
            exclude_matchers: List of matcher models to exclude from analysis
            normalize: Normalization method ("none", "balance", "reweight", or "scotts")
        """
        self.base_dir = base_dir
        self.model_list = model_list
        self.thinking_type = thinking_type
        self.dataset = dataset
        self.exclude_matchers = exclude_matchers or ["deepseek-chat-v3-0324"]
        self.normalize = normalize
        
        # Results container
        self.results = {}
        
        # Track common matchers across all models
        self.common_matchers = set()
        self.first_model = True
        
        # Run analysis for each model
        self.analyze_all_models()
        
    def get_file_paths(self, model_name: str) -> Tuple[str, str]:
        """
        Generate file paths for MCQ and LM matching files for a given model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Tuple of (mcq_file_path, lm_matchings_file_path)
        """
        # MCQ file path
        if self.dataset == "mmlu_pro":
            mcq_file = os.path.join(
                f"/is/cluster/fast/nchandak/qaevals/judge_outputs/{self.dataset}_mcq/stratified_sample",
                f"{model_name}_{self.thinking_type}",
                "samples.jsonl"
            )
        else:
            mcq_file = os.path.join(
                f"/is/cluster/fast/nchandak/qaevals/judge_outputs/{self.dataset}_mcq",
                f"{model_name}_{self.thinking_type}",
                "samples.jsonl"
            )
        
        # LM matchings file path
        lm_file = os.path.join(
            self.base_dir,
            f"{model_name}_{self.thinking_type}",
            "samples.jsonl"
        )
        
        return mcq_file, lm_file
        
    def analyze_model(self, model_name: str) -> Dict[str, Any]:
        """
        Analyze alignment for a single model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary with alignment results for this model
        """
        mcq_file, lm_file = self.get_file_paths(model_name)
        
        # Check if files exist
        if not os.path.exists(mcq_file) or not os.path.exists(lm_file):
            print(f"Skipping {model_name}: Files not found {mcq_file} or {lm_file}")
            return None
        
        print(f"\nAnalyzing model: {model_name}")
        print(f"MCQ file: {mcq_file}")
        print(f"LM file: {lm_file}")
        
        # Initialize analyzer
        from ds_alignment import DSAlignmentAnalyzer
        analyzer = DSAlignmentAnalyzer(
            mcq_file=mcq_file,
            lm_matchings_file=lm_file,
            exclude_matchers=self.exclude_matchers,
            dataset=self.dataset
        )
        
        # Calculate alignment
        alignment_result = analyzer.calculate_alignment(normalize=self.normalize)
        alignment_df = alignment_result[0]
        constant_baseline = alignment_result[1]
        
        # Track available matchers
        matchers = set(alignment_df[alignment_df['Type'] == 'Matcher']['Source'])
        if self.first_model:
            self.common_matchers = matchers
            self.first_model = False
        else:
            self.common_matchers &= matchers
        
        return {
            "model_name": model_name,
            "model_size": extract_model_size(model_name),
            "alignment_df": alignment_df,
            "constant_baseline": constant_baseline
        }
        
    def analyze_all_models(self) -> None:
        """Analyze all models in the list."""
        for model_name in self.model_list:
            result = self.analyze_model(model_name)
            if result:
                self.results[model_name] = result
                
        print(f"\nCommon matchers across all models: {', '.join(self.common_matchers)}")
    
    def extract_alignment_scores(self) -> pd.DataFrame:
        """
        Extract alignment scores for all models and sources.
        
        Returns:
            DataFrame with model names, sizes, and alignment scores for each source
        """
        data = []
        
        for model_name, result in self.results.items():
            model_size = result["model_size"]
            alignment_df = result["alignment_df"]
            
            # Create a row with model info
            row = {
                "model_name": model_name,
                "model_size": model_size
            }
            
            # Add MCQ alignment if available
            mcq_row = alignment_df[alignment_df['Source'] == 'mcq']
            if not mcq_row.empty:
                row["mcq"] = mcq_row.iloc[0]['Agreement (%)']
                row["mcq_error"] = mcq_row.iloc[0]['Std Error']
            
            # Add common matcher alignments
            for matcher in self.common_matchers:
                matcher_row = alignment_df[alignment_df['Source'] == matcher]
                if not matcher_row.empty:
                    row[matcher] = matcher_row.iloc[0]['Agreement (%)']
                    row[f"{matcher}_error"] = matcher_row.iloc[0]['Std Error']
            
            data.append(row)
        
        # Create DataFrame and sort by model size
        df = pd.DataFrame(data)
        return df.sort_values("model_size")
    
    def plot_alignment_by_size(self, output_file: Optional[str] = None, 
                             show_constant_baseline: bool = True) -> None:
        """
        Create a line plot showing alignment scores by model size.
        
        Args:
            output_file: If provided, save the plot to this file
            show_constant_baseline: Whether to show the constant baseline
        """
        scores_df = self.extract_alignment_scores()
        
        # Use common plotting function
        plot_alignment_by_size(
            scores_df=scores_df, 
            common_matchers=self.common_matchers, 
            results=self.results,
            output_file=output_file, 
            show_constant_baseline=show_constant_baseline,
            task_type="deepseek",
            thinking_type=self.thinking_type,
            normalize=self.normalize,
            dataset=self.dataset
        )

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Analyze alignment across multiple models by size.')
    
    parser.add_argument('--dataset', type=str, default="mmlu_pro",
                        help='Dataset name (e.g., mmlu_pro, gpqa)')
    
    parser.add_argument('--base-dir', type=str,
                        default=None,
                        help='Base directory containing model outputs')
    
    parser.add_argument('--models', type=str, nargs='+',
                        default=["Qwen3-0.6B", "Qwen3-1.7B", "Qwen3-4B", "Qwen3-8B", "Qwen3-14B", "Qwen3-32B"],
                        help='List of model names to analyze')
    
    parser.add_argument('--thinking-type', type=str,
                        choices=["thinking", "non_thinking"],
                        default="thinking",
                        help='Type of thinking to analyze')
    
    parser.add_argument('--plots-dir', type=str, 
                        default=None,
                        help='Directory to save plots')
    
    parser.add_argument('--output-file', type=str, default=None,
                        help='If provided, save the plot to this file instead of auto-generating name')
    
    parser.add_argument('--no-constant-baseline', action='store_true',
                        help='Do not show the constant baseline line on the plot')
    
    parser.add_argument('--exclude-matchers', type=str, nargs='+', 
                        default=["deepseek-chat-v3-0324"],
                        help='List of matcher models to exclude from analysis')
    
    parser.add_argument('--normalize', type=str, choices=["none", "balance", "reweight", "scotts"], default="none",
                        help='Normalization method to use ("none", "balance", "reweight", or "scotts")')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set default base_dir if not provided
    if args.base_dir is None:
        if args.dataset == "mmlu_pro":
            args.base_dir = f"/is/cluster/fast/nchandak/qaevals/judge_outputs/{args.dataset}_free/stratified_sample"
        else:
            args.base_dir = f"/is/cluster/fast/nchandak/qaevals/judge_outputs/{args.dataset}_free"
    
    # Set default plots_dir if not provided
    if args.plots_dir is None:
        args.plots_dir = f"plots/{args.dataset}"
    
    # Create plots directory including normalization method subdirectory
    plots_dir = os.path.join(args.plots_dir, args.normalize)
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir, exist_ok=True)
        print(f"Created plots directory: {plots_dir}")
    
    # Initialize multi-analyzer
    analyzer = DSAlignmentMultiAnalyzer(
        base_dir=args.base_dir,
        model_list=args.models,
        thinking_type=args.thinking_type,
        dataset=args.dataset,
        exclude_matchers=args.exclude_matchers,
        normalize=args.normalize
    )
    
    # Generate automatic output filename if not provided
    output_file = args.output_file
    if output_file is None:
        # Create filename based on parameters
        baseline = "no_baseline" if args.no_constant_baseline else "with_baseline"
        
        filename = f"ds_alignment_by_size_{args.thinking_type}_{args.normalize}_{baseline}.png"
        output_file = os.path.join(plots_dir, filename)
    
    # Plot results
    analyzer.plot_alignment_by_size(
        output_file=output_file,
        show_constant_baseline=not args.no_constant_baseline
    )

if __name__ == "__main__":
    main() 