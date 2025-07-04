import os
import re
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any, Set

def extract_model_size(model_name: str) -> float:
    """
    Extract model size (in billions) from model name.
    
    Args:
        model_name: Name of the model (e.g., qwen2.5-7b-it or Qwen3-0.6B)
        
    Returns:
        Model size in billions
    """
    # Handle different naming patterns
    if "qwen2.5" in model_name.lower():
        match = re.search(r'(\d+(?:\.\d+)?)b', model_name)
        if match:
            return float(match.group(1))
    else:
        match = re.search(r'(\d+(?:\.\d+)?)B', model_name)
        if match:
            return float(match.group(1))
    return 0.0

def simplify_model_name(model_name: str) -> str:
    """
    Simplify model name for display in plots.
    
    Args:
        model_name: Full model name
        
    Returns:
        Simplified model name
    """
    if "qwen2.5" in model_name.lower():
        return model_name.replace("qwen2.5-", "").replace("-it", "")
    else:
        return model_name.replace("Qwen3-", "")

def plot_alignment_by_size(scores_df, common_matchers: Set[str], results: Dict[str, Any], 
                         output_file: Optional[str] = None, 
                         show_constant_baseline: bool = True,
                         task_type: str = "deepseek",
                         thinking_type: str = "thinking",
                         normalize: str = "none",
                         dataset: str = "mmlu_pro") -> None:
    """
    Create a line plot showing alignment scores by model size.
    
    Args:
        scores_df: DataFrame with model sizes and alignment scores
        common_matchers: Set of matcher names common to all models
        results: Dictionary with baseline values
        output_file: If provided, save the plot to this file
        show_constant_baseline: Whether to show the constant baseline
        task_type: Type of task (deepseek or math)
        thinking_type: Type of thinking (thinking or non_thinking)
        normalize: Normalization method ("none", "balance", "reweight", or "scotts")
        dataset: Dataset name (e.g., mmlu_pro, gpqa)
    """
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot MCQ line if available
    if "mcq" in scores_df.columns:
        plt.errorbar(
            scores_df["model_size"], 
            scores_df["mcq"],
            yerr=scores_df["mcq_error"],
            marker='o',
            markersize=8,
            linewidth=2,
            color='#FFC107',  # Yellow color for MCQ
            label="MCQ"
        )
    
    # Plot matcher lines with custom colors
    # Count non-llama matchers for color gradient
    non_llama_matchers = [m for m in common_matchers if "llama" not in m.lower()]
    llama_matchers = [m for m in common_matchers if "llama" in m.lower()]
    
    # Generate blue color gradient for non-llama matchers
    blue_colors = plt.cm.Blues(np.linspace(0.4, 0.9, max(1, len(non_llama_matchers))))
    
    # Track if we have judge matchers
    has_judge_matchers = False
    judge_color = '#E53935'  # Red for judge matchers
    
    # Plot non-llama matchers with blue gradient
    for i, matcher in enumerate(non_llama_matchers):
        if matcher in scores_df.columns:
            # Check if matcher has 'judge' in its name
            if 'judge' in matcher.lower():
                color = judge_color  # Red for judge matchers
                has_judge_matchers = True
                label = f"Judge: {matcher}"
            else:
                color = blue_colors[i]
                label = f"Matcher: {matcher}"
                
            plt.errorbar(
                scores_df["model_size"], 
                scores_df[matcher],
                yerr=scores_df[f"{matcher}_error"],
                marker='s',
                markersize=8,
                linewidth=2,
                color=color,
                label=label
            )
    
    # Plot llama matchers with a distinct color (purple)
    for matcher in llama_matchers:
        if matcher in scores_df.columns:
            # Check if matcher has 'judge' in its name
            if 'judge' in matcher.lower():
                color = judge_color  # Red for judge matchers
                has_judge_matchers = True
                label = f"Judge: {matcher}"
            else:
                color = '#9C27B0'  # Purple for llama
                label = f"Matcher: {matcher}"
                
            plt.errorbar(
                scores_df["model_size"], 
                scores_df[matcher],
                yerr=scores_df[f"{matcher}_error"],
                marker='^',  # Different marker for llama
                markersize=8,
                linewidth=2,
                color=color,
                label=label
            )
            
    # If we have judge matchers, add a legend entry for them (use a mock plot for the legend only)
    if has_judge_matchers:
        # Add a fake entry to ensure judge color appears in the legend
        plt.plot([], [], color=judge_color, label='Judge Matchers', linewidth=2, marker='s', markersize=8)
    
    # Add constant baseline if available and requested
    if show_constant_baseline and results:
        # Create a list of model sizes and their corresponding constant baselines
        baseline_data = []
        for model_name, result in results.items():
            if "constant_baseline" in result and "model_size" in result:
                baseline_data.append({
                    "model_size": result["model_size"],
                    "baseline": result["constant_baseline"]
                })
        
        # Sort by model size
        baseline_data.sort(key=lambda x: x["model_size"])
        
        # Plot the constant baselines as a line similar to MCQ
        if baseline_data:
            sizes = [item["model_size"] for item in baseline_data]
            baselines = [item["baseline"] for item in baseline_data]
            
            plt.plot(
                sizes, 
                baselines,
                marker='d',  # Diamond marker to distinguish from other lines
                markersize=7,
                linestyle='--',
                linewidth=1.5,
                color='red',
                alpha=0.7,
                label='Constant Baseline'
            )
    
    # Add labels for each point (model name)
    for i, row in scores_df.iterrows():
        # Simplify model name for display
        model_name = simplify_model_name(row["model_name"])
        
        plt.annotate(
            model_name,
            (row["model_size"], row["mcq"] if "mcq" in row else next(row[m] for m in common_matchers if m in row)),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center'
        )
    
    # Set labels and title based on task type
    plt.xlabel('Model Size (Billions)', fontsize=14)
    
    if task_type == "deepseek":
        plt.ylabel('Agreement with Deepseek-v3-0324 (%)', fontsize=14)
        dataset_name = dataset.upper().replace("_", " ")
        plt.title(f'{dataset_name} Alignment by Model Size ({thinking_type.replace("_", " ")})', 
                 fontsize=16, fontweight='bold')
    else:  # math
        plt.ylabel('Agreement with Ground Truth (%)', fontsize=14)
        dataset_name = dataset.upper().replace("_", " ")
        plt.title(f'{dataset_name} Math Alignment by Model Size ({thinking_type.replace("_", " ")})', 
                 fontsize=16, fontweight='bold')
    
    # Set y-axis limits based on normalization method and data
    if normalize == "scotts":
        # For Scott's Pi, allow negative values
        all_values = []
        # Collect all agreement values including error bars
        for column in scores_df.columns:
            if column in common_matchers or column == "mcq":
                values = scores_df[column].values
                errors = scores_df[f"{column}_error"].values if f"{column}_error" in scores_df.columns else 0
                all_values.extend(values - errors)  # Lower bounds with error bars
        
        # Also include constant baseline values
        if show_constant_baseline and baseline_data:
            all_values.extend([item["baseline"] for item in baseline_data])
        
        # Set minimum y value based on data with margin
        min_value = min(all_values) if all_values else 0
        plt.ylim(max(min_value - 0.05, -1), 1)  # Allow at least -5 for Scott's Pi
    else:
        # For other normalization methods, use standard limits
        plt.ylim(45, 100)
    
    # Apply log scale for x-axis (model size)
    plt.xscale('log')
    
    # Add grid
    plt.grid(True, alpha=0.3)
    
    # Add legend
    plt.legend(loc='lower right')
    
    plt.tight_layout()
    
    # Save or show plot
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {output_file}")
    else:
        plt.show() 