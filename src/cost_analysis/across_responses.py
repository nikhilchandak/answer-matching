import os
import json
import argparse
import logging
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import numpy as np

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
    # logger.info(f"Loading data from {file_path}")
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    # logger.info(f"Loaded {len(data)} items from {file_path}")
    return data

MODEL_COST = {
    "deepseek/deepseek-chat-v3-0324" : {
        "input": 0.30,
        "output": 0.88,
    },
    "qwen/qwen-2.5-7b-instruct" : {
        "input": 0.04,
        "output": 0.1,
    },
    "openai/o4-mini" : {
        "input": 1.1,
        "output": 4.4,
    },
    "openai/gpt-4o" : {
        "input": 2.5,
        "output": 10,
    },
    "meta-llama/llama-4-maverick" : {
        "input": 0.16,
        "output": 0.6,
    },
    "qwen/qwen3-32b" : {
        "input": 0.1,
        "output": 0.3,
    },
    "Qwen3-32B" : {
        "input": 0.1,
        "output": 0.3,
    },
    "Qwen3_4B" : {
        "input": 0.04,
        "output": 0.1,
    },
    "meta-llama/llama-4-scout" : {
        "input": 0.08,
        "output": 0.3,
    },
    "google/gemma-3-27b-it" : {
        "input": 0.1,
        "output": 0.18,
    },
}

MODEL_NAMES = {
    "deepseek/deepseek-chat-v3-0324" : "Deepseek-v3",
    "qwen/qwen-2.5-7b-instruct" : "Qwen2.5-7B",
    "Qwen3_4B" : "Qwen3-4B", 
    "openai/o4-mini" : "o4-mini",
    "openai/gpt-4o" : "gpt-4o",
    "meta-llama/llama-4-maverick" : "Llama-4-Maverick",
    "qwen/qwen3-32b" : "Qwen3-32B",
    "Qwen3-32B" : "Qwen3-32B",
    "meta-llama/llama-4-scout" : "Llama-4-Scout",
}

def save_jsonl_file(data: List[Dict[str, Any]], file_path: str) -> None:
    """
    Save data to a JSONL file.
    
    Args:
        data: List of dictionaries to save
        file_path: Path to save the JSONL file
    """
    with open(file_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    logger.info(f"Saved {len(data)} items to {file_path}")
    
def get_samples(
    input_dir: str,
) -> None:
    """
    Combine samples from different model outputs for annotation.
    
    Args:
        input_dir: Directory containing model output files
        filtered_ids_path: Path to file with filtered question IDs
        batch_size: Number of samples per model
    """
    
    models = [
        # "openai/gpt-4o"
        "openai/o4-mini",
        # "deepseek/deepseek-r1",
        # "qwen/qwen3-14b"
        # "qwen/qwen3-32b"
        # "qwen/qwen-2.5-7b-instruct",
        "Qwen3_4B",
        
        
        "deepseek/deepseek-chat-v3-0324",
        # "qwen/qwen-2.5-72b-instruct",
        
        "meta-llama/llama-4-scout",
        # "meta-llama/llama-4-maverick",
        # "meta-llama/llama-3.1-8b-instruct",
        # "meta-llama/llama-3-8b-instruct",
        
        # "qwen/qwen3-8b",
        # "qwen/qwen3-14b",
        
        # "meta-llama/llama-3.1-8b-instruct",
        # "meta-llama/llama-3.2-3b-instruct",
        # "meta-llama/llama-3.2-1b-instruct",
    ]
    
    # Initialize token usage tracking dictionary
    token_usage = {
        "free_form_input": 0,
        "free_form_output": 0,
        "mcq_input": 0,
        "mcq_output": 0
    }
    
    num_questions = {"free_form": 0.0, "mcq": 0.0}
    cost_usage = { x:0.0 for x in token_usage.keys() }
    
    matcher_usage = { model: {
                    "input": 0,
                    "output": 0,
                    } 
                   for model in models if "o4" not in model.lower()}
    
    judge_usage = { model: {
                    "input": 0,
                    "output": 0,
                    } 
                   for model in models if "llama" not in model.lower() and "qwen" not in model.lower()}
    
    # Initialize combined samples list
    combined_samples = []
    acc = []
    total = []
    processes_files = []
    mcq_questions = []
    free_form_questions = []
    
    # Process each file and extract samples in batches
    for root, _, files in os.walk(input_dir):
        if "wrong" in root or "wrong" in _ :
            continue 
        for file in files:
            model_file = os.path.join(root, file)
            # print(model_file)
            
            if "old" in model_file:
                continue 
            
            if "gen" not in model_file and "mcq" not in model_file :
                continue 
            
            # if "gpqa" not in model_file :
            #     continue 
            
            if "mmlu" not in model_file :
                continue 
            
            logger.info(f"Processing file: {model_file}")
            # Load file data
            model_data = load_jsonl_file(model_file)
            logger.info(f"Num items: {len(model_data)}")
            
            # Determine if it's MCQ or free-form based on question format
            is_mcq = "mcq" in model_file #  in item["dataset"]
                
            
            # Track token usage for each item
            for item in model_data:
                
                
                
                if "flash" in item["model"].lower() :
                    continue
                
                # Skip qwen3 
                # if "maverick" not in item["model"].lower(): 
                #     continue 
                    
                    
                # if not "qwen3" in item["model"].lower() :
                #     continue 
            
                # if not is_mcq :
                #     print(1)
                    
                flag = False 
                
                qid = item["question_id"]
                if is_mcq :
                    mcq_questions.append(qid)
                    if len(free_form_questions) > 0 :
                        assert qid in free_form_questions
                else :
                    free_form_questions.append(qid)
                    if len(mcq_questions) > 0 :
                        assert qid in mcq_questions
                
                
                
                # Process each model for this file
                for j, model in enumerate(list(matcher_usage.keys())):
                    model_name = model.split("/")[-1]
                    # logger.info(f"Processing model: {model_name}")
                    
                     
                    is_mcq = "mcq" in model_file #  in item["dataset"]
                    
                    # Check for model-specific completion tokens
                    prompt_token_field = f"prompt_tokens_{model_name}"
                    completion_token_field = f"completion_tokens_{model_name}"
                    
                    if "gen" in model_file and completion_token_field not in item:
                        continue 
                    
                    if is_mcq: 
                        continue 
                    
                    if prompt_token_field in item:
                        field = prompt_token_field
                        used = int(item[field]) if not isinstance(item[field], list) else int(item[field][0])
                        matcher_usage[model]["input"] += used
                    
                    if completion_token_field in item:
                        field = completion_token_field
                        used = int(item[field]) if not isinstance(item[field], list) else int(item[field][0])
                        
                        if is_mcq:
                            assert False
                            token_usage["mcq_output"] += used
                        else:
                            matcher_usage[model]["output"] += used
                            
                    # else :
                    #     continue 
                    
                
                # Process each model for this file
                for j, model in enumerate(list(judge_usage.keys())):
                    model_name = model.split("/")[-1]
                    # logger.info(f"Processing model: {model_name}")
                    
                     
                    is_mcq = "mcq" in model_file #  in item["dataset"]
                    
                    # Check for model-specific completion tokens
                    prompt_token_field = f"prompt_tokens_{model_name}"
                    completion_token_field = f"completion_tokens_{model_name}"
                    
                    if "gen" in model_file and completion_token_field not in item:
                        continue 
                    
                    if is_mcq: 
                        continue 
                    
                    suffix =  "-JUDGE"
                    judge_completion_token_field = f"completion_tokens_{model_name}{suffix}"
                    judge_prompt_token_field = f"prompt_tokens_{model_name}{suffix}"
                    
                    if judge_completion_token_field in item:
                        field = judge_completion_token_field
                        used = int(item[field]) if not isinstance(item[field], list) else int(item[field][0])
                        judge_usage[model]["output"] += used
                        
                    if judge_prompt_token_field in item:
                        field = judge_prompt_token_field
                        used = int(item[field]) if not isinstance(item[field], list) else int(item[field][0])
                        judge_usage[model]["input"] += used
                        
                        
                student_model = item["model"]
                
                #  "prompt_tokens" not in item or
                if "completion_tokens" not in item or "dataset" not in item :
                    continue 
                
                # Check if item has token usage information
                if "prompt_tokens" in item: # and "qwen" not in student_model.lower():
                    field = "prompt_tokens"
                    used = int(item[field]) if not isinstance(item[field], list) else int(item[field][0])
                    
                    if is_mcq:
                        token_usage["mcq_input"] += used
                        cost_usage["mcq_input"] += used * MODEL_COST[student_model]["input"]
                    else:
                        token_usage["free_form_input"] += used 
                        cost_usage["free_form_input"] += used * MODEL_COST[student_model]["input"]
                                                
                # Check for completion tokens in standard field
                if "completion_tokens" in item:
                    
                    field = "completion_tokens"
                    used = int(item[field]) if not isinstance(item[field], list) else int(item[field][0])
                    
                    if is_mcq:
                        token_usage["mcq_output"] += used
                        num_questions["mcq"] += 1
                        cost_usage["mcq_output"] += used * MODEL_COST[student_model]["output"]
                    else:
                        token_usage["free_form_output"] += used
                        num_questions["free_form"] += 1
                        cost_usage["free_form_output"] += used * MODEL_COST[student_model]["output"]
                
            processes_files.append(model_file)  
                
            logger.info(f"Free-form input tokens: {token_usage['free_form_input']}")
            logger.info(f"Free-form output tokens: {token_usage['free_form_output']}")
            logger.info(f"MCQ input tokens: {token_usage['mcq_input']}")
            logger.info(f"MCQ output tokens: {token_usage['mcq_output']}")
            logger.info("\n\n\n--------------------------------\n\n\n")
                
                
    # Check both mcq question IDs and free-form question IDs are same set of questions
    assert len(mcq_questions) == len(free_form_questions)
    assert set(mcq_questions) == set(free_form_questions)

    actual_files = list(set(processes_files))
    # Log token usage statistics
    logger.info("Token usage statistics:")
    logger.info(f"Free-form input tokens: {token_usage['free_form_input']}")
    logger.info(f"Free-form output tokens: {token_usage['free_form_output']}")
    logger.info(f"MCQ input tokens: {token_usage['mcq_input']}")
    logger.info(f"MCQ output tokens: {token_usage['mcq_output']}")
    logger.info(f"Actual files: {actual_files}")
    
    cost_usage["free_form_input"] = cost_usage["free_form_input"] / num_questions["free_form"]
    cost_usage["free_form_output"] = cost_usage["free_form_output"] / num_questions["free_form"]
    cost_usage["mcq_input"] = cost_usage["mcq_input"] / num_questions["mcq"]
    cost_usage["mcq_output"] = cost_usage["mcq_output"] / num_questions["mcq"]
    
    for model in models:
        logger.info(f"Model: {model}")
        if model in matcher_usage:
            matcher_usage[model]["input"] = matcher_usage[model]["input"] / num_questions["free_form"]
            matcher_usage[model]["output"] = matcher_usage[model]["output"] / num_questions["free_form"]
            
            logger.info(f"Input Matcher tokens: {matcher_usage[model]['input']}")
            logger.info(f"Output Matcher tokens: {matcher_usage[model]['output']}")
        if model in judge_usage:
            judge_usage[model]["input"] = judge_usage[model]["input"] / num_questions["free_form"]
            judge_usage[model]["output"] = judge_usage[model]["output"] / num_questions["free_form"]
            
            logger.info(f"Input Judge tokens: {judge_usage[model]['input']}")
            logger.info(f"Output Judge tokens: {judge_usage[model]['output']}")
        logger.info("\n\n\n--------------------------------\n\n\n")
        
    logger.info(f"Free-form input tokens: {token_usage['free_form_input']}")
    logger.info(f"Free-form output tokens: {token_usage['free_form_output']}")
    logger.info(f"MCQ input tokens: {token_usage['mcq_input']}")
    logger.info(f"MCQ output tokens: {token_usage['mcq_output']}")
    
    logger.info(f"Free-form input cost: {cost_usage['free_form_input']}")
    logger.info(f"Free-form output cost: {cost_usage['free_form_output']}")
    logger.info(f"MCQ input cost: {cost_usage['mcq_input']}")
    logger.info(f"MCQ output cost: {cost_usage['mcq_output']}")
    
    logger.info("\n\n--------------------------------\n\n")
    
    for model in models:
        logger.info(f"Model: {model}")
        if model in matcher_usage: 
            matcher_usage[model]["input"] = matcher_usage[model]["input"] * MODEL_COST[model]["input"]
            matcher_usage[model]["output"] = matcher_usage[model]["output"] * MODEL_COST[model]["output"]
            
            logger.info(f"Matcher input cost: {matcher_usage[model]['input']}")
            logger.info(f"Matcher output cost: {matcher_usage[model]['output']}")
        
        if model in judge_usage:
            judge_usage[model]["input"] = judge_usage[model]["input"] * MODEL_COST[model]["input"]
            judge_usage[model]["output"] = judge_usage[model]["output"] * MODEL_COST[model]["output"]
            
            logger.info(f"Judge input cost: {judge_usage[model]['input']}")
            logger.info(f"Judge output cost: {judge_usage[model]['output']}")
        
        logger.info("\n\n\n--------------------------------\n\n\n")
    
    logger.info(f"Num questions: {num_questions}")
    return cost_usage, token_usage, matcher_usage, judge_usage, num_questions

def plot_cost_analysis_horizontal(cost_usage, matcher_usage, judge_usage, num_questions):
    """
    Create a horizontal bar chart showing cost breakdown across different models and tasks,
    with separate input and output costs.
    
    Args:
        cost_usage: Dictionary containing MCQ and free-form costs
        matcher_usage: Dictionary containing matcher costs by model
        judge_usage: Dictionary containing judge costs by model
    """
    # Convert costs from dollars to millions of dollars, keeping input/output separate
    mcq_input_cost = cost_usage["mcq_input"] / 1e6
    mcq_output_cost = cost_usage["mcq_output"] / 1e6
    free_form_input_cost = cost_usage["free_form_input"] / 1e6
    free_form_output_cost = cost_usage["free_form_output"] / 1e6
    
    # Get list of models
    models = list(set(list(matcher_usage.keys()) + list(judge_usage.keys())))

    # Prepare data for plotting
    labels = ["MCQ"]
    
    # Add matcher and judge model labels
    for model in models:
        if model not in matcher_usage:
            continue
        
        actual_name = MODEL_NAMES[model]
        labels.append(f"Matcher\n{actual_name}")
    
    for model in models:
        if model not in judge_usage:
            continue
        
        actual_name = MODEL_NAMES[model]
        labels.append(f"Judge\n{actual_name}")
    
    # Create figure and axis with appropriate dimensions for horizontal plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Set up colors - light and dark variants for each mode
    response_input_color = '#87CEEB'   # Light blue for response input
    response_output_color = '#4682B4'  # Dark blue for response output
    matcher_input_color = '#90EE90'    # Light green for matcher input
    matcher_output_color = '#32CD32'   # Dark green for matcher output
    judge_input_color = '#FFB6C1'      # Light red for judge input
    judge_output_color = '#FF6347'     # Dark red for judge output
    
    # Set up positions for bars
    bar_height = 0.3
    y_pos = np.arange(len(labels)) * 0.8  # Decrease spacing between bars
    
    # Plot MCQ bars (separate input and output)
    ax.barh(y_pos[0], mcq_input_cost, height=bar_height, color=response_input_color, 
            label='Response Input', left=0)
    ax.barh(y_pos[0], mcq_output_cost, height=bar_height, color=response_output_color, 
            label='Response Output', left=mcq_input_cost)
    
    # Plot matcher bars
    j = 0
    for model in models:
        if model not in matcher_usage:
            continue
        
        pos = j + 1
        j += 1
        
        # Free-form response (separate input and output)
        ax.barh(y_pos[pos], free_form_input_cost, height=bar_height, color=response_input_color, 
                left=0)
        ax.barh(y_pos[pos], free_form_output_cost, height=bar_height, color=response_output_color, 
                left=free_form_input_cost)
        
        # Matcher (separate input and output)
        matcher_input_cost = matcher_usage[model]["input"] / 1e6
        matcher_output_cost = matcher_usage[model]["output"] / 1e6
        left_pos = free_form_input_cost + free_form_output_cost
        
        ax.barh(y_pos[pos], matcher_input_cost, height=bar_height, color=matcher_input_color, 
                left=left_pos)
        ax.barh(y_pos[pos], matcher_output_cost, height=bar_height, color=matcher_output_color, 
                left=left_pos + matcher_input_cost)
    
    # Plot judge bars
    j = 0
    for model in models:
        if model not in judge_usage:
            continue
        
        pos = j + 1 + len([m for m in models if m in matcher_usage])
        j += 1
        
        # Free-form response (separate input and output)
        ax.barh(y_pos[pos], free_form_input_cost, height=bar_height, color=response_input_color, 
                left=0)
        ax.barh(y_pos[pos], free_form_output_cost, height=bar_height, color=response_output_color, 
                left=free_form_input_cost)
        
        # Judge (separate input and output)
        judge_input_cost = judge_usage[model]["input"] / 1e6
        judge_output_cost = judge_usage[model]["output"] / 1e6
        left_pos = free_form_input_cost + free_form_output_cost
        
        ax.barh(y_pos[pos], judge_input_cost, height=bar_height, color=judge_input_color, 
                left=left_pos)
        ax.barh(y_pos[pos], judge_output_cost, height=bar_height, color=judge_output_color, 
                left=left_pos + judge_input_cost)
    
    # Add labels and title with larger font sizes
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=16, ha='right')
    ax.set_xlabel('Mean Cost Per Sample (USD)', fontsize=18)
    ax.tick_params(axis='x', labelsize=14)
    
    # Set x-axis to use scientific notation with proper tick spacing
    max_cost = max(
        mcq_input_cost + mcq_output_cost,
        free_form_input_cost + free_form_output_cost + 
        max([(matcher_usage[m]["input"] + matcher_usage[m]["output"]) / 1e6 for m in models if m in matcher_usage], default=0),
        free_form_input_cost + free_form_output_cost + 
        max([(judge_usage[m]["input"] + judge_usage[m]["output"]) / 1e6 for m in models if m in judge_usage], default=0)
    )
    
    ax.set_xlim(0, max_cost * 1.1)
    
    # Format the tick labels
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: f'{x*1e3:.1f}'))
    ax.text(1.01, 0.5, r'$\times 10^{-3}$', transform=ax.transAxes, fontsize=16, 
            rotation=0, va='center')
    
    # Define consistent font size for legend
    legend_fontsize = 12
    
    # Add a legend - only add one instance of each label
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, color=response_input_color, label='Response Input'),
        plt.Rectangle((0, 0), 1, 1, color=response_output_color, label='Response Output'),
        plt.Rectangle((0, 0), 1, 1, color=matcher_input_color, label='Matcher Input'),
        plt.Rectangle((0, 0), 1, 1, color=matcher_output_color, label='Matcher Output'),
        plt.Rectangle((0, 0), 1, 1, color=judge_input_color, label='Judge Input'),
        plt.Rectangle((0, 0), 1, 1, color=judge_output_color, label='Judge Output')
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.15),
              ncol=3, frameon=True, fontsize=legend_fontsize)
    
    # Add grid lines that align with the tick marks
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    
    # Create directory if it doesn't exist
    os.makedirs('plots', exist_ok=True)
    
    # Adjust layout with more padding
    plt.tight_layout(pad=2.0)
    
    # Save the figure
    plt.savefig('plots/cost_analysis_horizontal.png', dpi=300, bbox_inches='tight')
    # logger.info("Horizontal cost analysis plot saved as 'plots/cost_analysis_horizontal.png'")
    
    # Show the plot
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Combine model outputs for annotation")
    parser.add_argument("--input_dir", default="/fast/nchandak/qaevals/judge_outputs/alignment_plot/",
                        help="Directory containing model output files")
    
    args = parser.parse_args()
    
    cost_usage, token_usage, matcher_usage, judge_usage, num_questions = get_samples(
        input_dir=args.input_dir,
    )
    
    # plot_cost_analysis(cost_usage, matcher_usage, judge_usage, num_questions)
    plot_cost_analysis_horizontal(cost_usage, matcher_usage, judge_usage, num_questions)

if __name__ == "__main__":
    main()
