import os
import json
import argparse
import logging
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from matplotlib.ticker import ScalarFormatter

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
    "openai/o4-mini-high" : {
        "input": 1.10,
        "output": 4.40,
    },
    "openai/gpt-4o" : {
        "input": 2.5,
        "output": 10,
    },
    "meta-llama/llama-4-maverick" : {
        "input": 0.19,
        "output": 0.85,
    },
    "qwen/qwen3-32b" : {
        "input": 0.10,
        "output": 0.30,
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
        "output": 0.30,
    },
    "google/gemma-3-27b-it" : {
        "input": 0.1,
        "output": 0.18,
    },
    "google/gemini-2.5-pro-preview" : {
        "input": 1.25,
        "output": 10.0,
    },
    "deepseek/deepseek-r1-0528" : {
        "input": 0.50,
        "output": 2.15,
    },
    "deepseek/deepseek-r1" : {
        "input": 0.45,
        "output": 2.15,
    },
    "google/gemini-2.5-flash-preview": {
        "input": 0.15,
        "output": 0.60,
    },
    "qwen/qwen-2.5-72b-instruct": {
        "input": 0.12,
        "output": 0.39,
    },
    "qwen/qwen3-235b-a22b": {
        "input": 0.13,
        "output": 0.60,
    },
    "x-ai/grok-3-mini-beta": {
        "input": 0.30,
        "output": 0.50,
    },
    "mistralai/mistral-medium-3": {
        "input": 0.40,
        "output": 2.00,
    },
    "microsoft/phi-4": {
        "input": 0.07,
        "output": 0.14,
    },
    "deepseek/deepseek-r1-distill-llama-70b": {
        "input": 0.20,
        "output": 0.60,
    },
    "mistralai/mistral-small-24b-instruct-2501": {
        "input": 0.05,
        "output": 0.09,
    },
    "anthropic/claude-3.5-haiku": {
        "input": 0.80,
        "output": 4.00,
    },
    "openai/gpt-4.1-nano": {
        "input": 0.10,
        "output": 0.40,
    },
    "openai/gpt-4o-mini-2024-07-18": {
        "input": 0.15,
        "output": 0.60,
    },
    "microsoft/wizardlm-2-8x22b": {
        "input": 0.48,
        "output": 0.48,
    },
    "meta-llama/llama-3.3-70b-instruct": {
        "input": 0.05,
        "output": 0.17,
    },
    "qwen/qwen-2.5-72b-instruct": {
        "input": 0.12,
        "output": 0.39,
    },
    # TODO: Add costs for any additional models if found in the future.
}

MODEL_NAMES = {
    "deepseek/deepseek-chat-v3-0324" : "DeepSeek-v3",
    "qwen/qwen-2.5-7b-instruct" : "Qwen2.5-7B",
    "Qwen3_4B" : "Qwen3-4B", 
    "openai/o4-mini" : "o4-mini",
    "openai/gpt-4o" : "gpt-4o",
    "meta-llama/llama-4-maverick" : "Llama-4-Maverick",
    "qwen/qwen3-32b" : "Qwen3-32B",
    "Qwen3-32B" : "Qwen3-32B",
    "meta-llama/llama-4-scout" : "Llama-4-Scout",
}

alignment_values_map = {
    "gpqa_diamond" : {
        "deepseek/deepseek-chat-v3-0324" : 0.90,
        "meta-llama/llama-4-scout" : 0.91,
        "Qwen3_4B" : 0.87,
        "openai/o4-mini" : 0.68,
        "mcq" : 0.33,
    },
    "mmlu_pro" : {
        "deepseek/deepseek-chat-v3-0324" : 0.85,
        "meta-llama/llama-4-scout" : 0.87,
        "Qwen3_4B" : 0.81,
        "openai/o4-mini" : 0.63,
        "mcq" : 0.47,
    }
}


def get_mmlu_pro_filtered_ids(samples_path: str = "/fast/nchandak/qaevals/judge_outputs/mmlu_pro_free/stratified_sample/mmlu_pro_combined_samples_to_annotate2.jsonl") -> List[int]:
    """
    Get filtered IDs from file
    """
    # Load the combined samples file
    filtered_ids = []
    
    if os.path.exists(samples_path):
        with open(samples_path, "r") as f:
            for line in f:
                sample = json.loads(line)
                # Only keep question_ids where the model doesn't contain "gemini"
                if "model" in sample and "gemini" not in sample["model"] and "question_id" in sample:
                    filtered_ids.append(str(sample["question_id"]))
    
    return filtered_ids


def get_filtered_ids(dataset):
    """
    Get filtered IDs from file. Only returns question IDs that have rating_osq >= 4 
    and rating_multians >= 4 in ALL .jsonl files in the directory.
    """
    annotations_dir = f"/fast/nchandak/qaevals/judge_outputs/alignment_plot/annotations/{dataset}/"
    
    if not os.path.exists(annotations_dir):
        logger.warning(f"Annotations directory not found: {annotations_dir}")
        return []
    
    # Find all .jsonl files in the directory
    jsonl_files = []
    for file in os.listdir(annotations_dir):
        if file.endswith('.jsonl'):
            jsonl_files.append(os.path.join(annotations_dir, file))
    
    logger.info(f"Found {len(jsonl_files)} .jsonl files in {annotations_dir}")
    
    if not jsonl_files:
        logger.warning("No .jsonl files found in directory")
        return []
    
    # Dictionary to track ratings across files
    # Structure: {question_id: {file_path: (rating_osq, rating_multians)}}
    ratings_by_id = {}
    
    # Process each .jsonl file
    for file_path in jsonl_files:
        logger.info(f"Processing file: {file_path}")
        try:
            with open(file_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line.strip())
                        
                        # Extract required fields
                        question_id = str(data.get('question_id'))
                        rating_osq = data.get('rating_osq')
                        rating_multians = data.get('rating_multians')
                        
                        # Check if all required fields are present
                        if question_id is None or rating_osq is None or rating_multians is None:
                            continue
                        
                        # Initialize dict for this question_id if not exists
                        if question_id not in ratings_by_id:
                            ratings_by_id[question_id] = {}
                        
                        # Store ratings for this file
                        ratings_by_id[question_id][file_path] = (rating_osq, rating_multians)
                                
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse line {line_num} in {file_path}: {e}")
                        continue
                        
        except FileNotFoundError:
            logger.warning(f"File not found: {file_path}")
            continue
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            continue
    
    
    # Filter IDs that meet criteria in ALL files
    filtered_ids = []
    for question_id, file_ratings in ratings_by_id.items():
        # Check if question appears in all files
        if len(file_ratings) != len(jsonl_files):
            continue
            
        # Check if ratings meet criteria in all files
        meets_criteria = all(
            rating_osq >= 1 and rating_multians >= 1
            for rating_osq, rating_multians in file_ratings.values()
        )
        
        # print(question_id, file_ratings.values())
        if meets_criteria:
            filtered_ids.append(str(question_id))
    
    assert set(list(set(filtered_ids))) == set(filtered_ids), "Filtered IDs are not unique"
    logger.info(f"Found {len(filtered_ids)} question IDs with rating_osq >= 4 and rating_multians >= 4 in ALL files")
    return filtered_ids


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
    input_dir: str, which_dataset: str = "mmlu_pro"
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
        # "openai/o4-mini",
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
    
    
    eval_models = [
        "openai/gpt-4o",
        "deepseek/deepseek-chat-v3-0324",
        
        # "openai/o4-mini-high",
        # "google/gemini-2.5-pro-preview",
        # "deepseek/deepseek-r1-0528",
        
        # "google/gemini-2.5-flash-preview",
        "meta-llama/llama-4-maverick",
        
        # "deepseek/deepseek-r1",
        "qwen/qwen3-32b",
        # "qwen/qwen3-235b-a22b"
        
        "x-ai/grok-3-mini-beta",
        "mistralai/mistral-medium-3",
        "microsoft/phi-4",
        "deepseek/deepseek-r1-distill-llama-70b",
        "meta-llama/llama-4-scout",
        "qwen/qwen-2.5-72b-instruct",
        "google/gemma-3-27b-it",
        "openai/gpt-4.1-nano",
        "microsoft/wizardlm-2-8x22b",
        "meta-llama/llama-3.3-70b-instruct",
        "openai/gpt-4o-mini-2024-07-18",
        "mistralai/mistral-small-24b-instruct-2501",
        "anthropic/claude-3.5-haiku",
        
        # "openai/o4-mini",
        # "deepseek/deepseek-r1",
        # "qwen/qwen3-14b"
        # "qwen/qwen3-32b"
        
        # "qwen/qwen-2.5-7b-instruct",
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
    judge_usage = {}
    
    # Initialize combined samples list
    combined_samples = []
    acc = []
    total = []
    processes_files = []
    mcq_questions = []
    free_form_questions = []
    
    # which_dataset = "gpqa_diamond"
    # which_dataset = "mmlu_pro"
    
    mcq_files = list(os.walk(input_dir + f"{which_dataset}_mcq/"))
    free_form_files = list(os.walk(input_dir + f"{which_dataset}_free/"))
    
    all_files = mcq_files + free_form_files
    # relevant_ids = get_mmlu_pro_filtered_ids() if which_dataset == "mmlu_pro" else get_gpqa_filtered_ids()
    relevant_ids = get_filtered_ids(which_dataset)
    logger.info(f"Num relevant ids: {len(relevant_ids)}")
    
    # Process each file and extract samples in batches
    for root, _, files in all_files:
        if "wrong" in root or "wrong" in _ :
            continue 
        for file in files:
            model_file = os.path.join(root, file)
            
            if "samples_" not in file:
                continue 
            
            eval_model_name = file.split("_")[1].split(".")[0]
            
            eval_model_present = False 
            for model123 in eval_models:
                suffix = "_" + model123.split("/")[-1] + ".jsonl"
                # print(suffix)
                if suffix in file:
                    eval_model_present = True 
                    break
                if "qwen3-32b" in model123 and "qwen3-32b" in file and "non_think" in file:
                    eval_model_present = True 
                    break 
            
            if not eval_model_present:
                continue 
        
            if "old" in model_file:
                continue 
            
            # if "gen" not in model_file and "mcq" not in model_file :
            #     continue 
            
            
            # if "mmlu" not in model_file :
            #     continue 
            
            # print(eval_model_name)
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
                
                qid = str(item["question_id"])
                if qid not in relevant_ids:
                    continue 
                
                if is_mcq :
                    mcq_questions.append((file, qid))
                else :
                    free_form_questions.append((file, qid))
                
                
                
                
                # Process each model for this file
                for j, model in enumerate(list(matcher_usage.keys())):
                    model_name = model.split("/")[-1]
                    logger.info(f"Processing model: {model_name}")
                    
                     
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
                        # logger.info(f"Model: {model},  Used: {used}")
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
    logger.info(f"Num mcq questions: {len(mcq_questions)}")
    logger.info(f"Num free-form questions: {len(free_form_questions)}")
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

def plot_cost_analysis_vertical(cost_usage, matcher_usage, judge_usage, num_questions, which_dataset: str, ax, show_ylabel=True):
    """
    Create a vertical bar chart showing cost breakdown across different models and tasks,
    with separate input and output costs, on the provided axis.
    """
    # Convert costs from dollars to millions of dollars, keeping input/output separate
    mcq_input_cost = cost_usage["mcq_input"] / 1e6
    mcq_output_cost = cost_usage["mcq_output"] / 1e6
    free_form_input_cost = cost_usage["free_form_input"] / 1e6
    free_form_output_cost = cost_usage["free_form_output"] / 1e6

    # Get list of models
    # models = list(set(list(matcher_usage.keys()) + list(judge_usage.keys())))
    models = sorted(list(set(list(matcher_usage.keys()) + list(judge_usage.keys()))))

    # text_multiplier = 1e4 if which_dataset == "mmlu_pro" else 1e3
    text_multiplier = 1e3
    
    # Prepare data for plotting
    labels = [f"MCQ\n($\pi = {alignment_values_map[which_dataset]['mcq']}$)"]
    
    for model in models:
        if model not in matcher_usage:
            continue
        
        suffix = f"\n($\pi = {alignment_values_map[which_dataset][model]}$)"
        actual_name = MODEL_NAMES.get(model, model)
        labels.append(f"Matcher\n{actual_name}{suffix}")
        
    for model in models:
        if model not in judge_usage:
            continue
        
        suffix = f"\n($\pi = {alignment_values_map[which_dataset][model]}$)"
        actual_name = MODEL_NAMES.get(model, model)
        labels.append(f"Judge\n{actual_name}{suffix}")

    # Set up colors
    response_input_color = '#87CEEB'   # Light blue for response input
    response_output_color = '#4682B4'  # Dark blue for response output
    matcher_input_color = '#90EE90'    # Light green for matcher input
    matcher_output_color = '#32CD32'   # Dark green for matcher output
    judge_input_color = '#FFB6C1'      # Light red for judge input
    judge_output_color = '#FF6347'     # Dark red for judge output

    # Set up positions for bars
    bar_width = 0.3
    x_pos = np.arange(len(labels)) * 0.8

    # Plot MCQ bars (separate input and output)
    ax.bar(x_pos[0], mcq_input_cost, width=bar_width, color=response_input_color, label='Model Input', bottom=0)
    ax.bar(x_pos[0], mcq_output_cost, width=bar_width, color=response_output_color, bottom=mcq_input_cost, label='Model Output')
    
    # Add total cost text for MCQ
    total_mcq_cost = mcq_input_cost + mcq_output_cost
    ax.text(x_pos[0], total_mcq_cost * 1.015, f'\$ {total_mcq_cost*text_multiplier:.2f}', ha='center', va='bottom', fontsize=14, fontweight='bold')

    # Plot matcher bars
    j = 0
    for model in models:
        if model not in matcher_usage:
            continue
        pos = j + 1
        j += 1
        ax.bar(x_pos[pos], free_form_input_cost, width=bar_width, color=response_input_color, bottom=0)
        ax.bar(x_pos[pos], free_form_output_cost, width=bar_width, color=response_output_color, bottom=free_form_input_cost)
        matcher_input_cost = matcher_usage[model]["input"] / 1e6
        matcher_output_cost = matcher_usage[model]["output"] / 1e6
        bottom_pos = free_form_input_cost + free_form_output_cost
        ax.bar(x_pos[pos], matcher_input_cost, width=bar_width, color=matcher_input_color, bottom=bottom_pos)
        ax.bar(x_pos[pos], matcher_output_cost, width=bar_width, color=matcher_output_color, bottom=bottom_pos + matcher_input_cost)
        
        # Add total cost text for matcher
        total_matcher_cost = free_form_input_cost + free_form_output_cost + matcher_input_cost + matcher_output_cost
        ax.text(x_pos[pos], total_matcher_cost * 1.015, f'\$ {total_matcher_cost*text_multiplier:.2f}', ha='center', va='bottom', fontsize=14, fontweight='bold')

    # Plot judge bars
    j = 0
    for model in models:
        if model not in judge_usage:
            continue
        pos = j + 1 + len([m for m in models if m in matcher_usage])
        j += 1
        ax.bar(x_pos[pos], free_form_input_cost, width=bar_width, color=response_input_color, bottom=0)
        ax.bar(x_pos[pos], free_form_output_cost, width=bar_width, color=response_output_color, bottom=free_form_input_cost)
        judge_input_cost = judge_usage[model]["input"] / 1e6
        judge_output_cost = judge_usage[model]["output"] / 1e6
        bottom_pos = free_form_input_cost + free_form_output_cost
        ax.bar(x_pos[pos], judge_input_cost, width=bar_width, color=judge_input_color, bottom=bottom_pos)
        ax.bar(x_pos[pos], judge_output_cost, width=bar_width, color=judge_output_color, bottom=bottom_pos + judge_input_cost)
        
        # Add total cost text for judge
        total_judge_cost = free_form_input_cost + free_form_output_cost + judge_input_cost + judge_output_cost
        ax.text(x_pos[pos], total_judge_cost * 1.015, f'\${total_judge_cost*text_multiplier:.2f}', ha='center', va='bottom', fontsize=14, fontweight='bold')

    # Add labels and title
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=14, rotation=0, ha='center')
    if show_ylabel:
        ax.set_ylabel('Mean Cost Per 1000 Samples (USD)', fontsize=16)
    else:
        ax.set_ylabel("")
    ax.tick_params(axis='y', labelsize=12)
    
    new_title = "GPQA Diamond" if which_dataset == "gpqa_diamond" else "MMLU Pro"
    ax.set_title(f"{new_title}", fontsize=18)

    # Set y-axis to use scientific notation
    max_cost = max(
        mcq_input_cost + mcq_output_cost,
        free_form_input_cost + free_form_output_cost + 
        max([(matcher_usage[m]["input"] + matcher_usage[m]["output"]) / 1e6 for m in models if m in matcher_usage], default=0),
        free_form_input_cost + free_form_output_cost + 
        max([(judge_usage[m]["input"] + judge_usage[m]["output"]) / 1e6 for m in models if m in judge_usage], default=0)
    )
    
    ax.set_ylim(0, max_cost * 1.07)  # Increased to make room for text
    # Double the number of y-axis ticks
    ax.yaxis.set_major_locator(MaxNLocator(nbins=12))
    
    # Remove the last ytick if mmlu_pro
    if which_dataset == "mmlu_pro":
        print(ax.get_yticks())
        ax.set_yticks(ax.get_yticks()[:-2])
    
    # Format y-tick labels (no suffix)
    def ytick_formatter(x, pos=None):
        val = x * 1e3
        return f'{val:.2f}'
        # return f'{val:.1f}' if val - int(val) > 0.1 else f'{int(val):.0f}'
    
    ax.yaxis.set_major_formatter(plt.FuncFormatter(ytick_formatter))
    
    # yScalarFormatter = ScalarFormatter(useMathText=True)
    # yScalarFormatter.set_powerlimits((0,0))
    # ax.yaxis.set_major_formatter(yScalarFormatter)
    
    # Make top and right ticks invisible
    ax.tick_params(axis='x', top=False, bottom=False)
    # ax.tick_params(axis='y', left=False, right=False)
    
    # Increase the size of the y-axis offset text (10^x label)
    ax.yaxis.offsetText.set_fontsize(14)

    ax.grid(axis='y', linestyle='--', alpha=0.7)


def plot_both_datasets():
    """
    Plot both gpqa_diamond and mmlu_pro datasets in a single figure with two vertical subplots and a shared legend.
    """
    # Get data for both datasets
    datasets = ["gpqa_diamond", "mmlu_pro"]
    results = []
    for which_dataset in datasets:
        cost_usage, token_usage, matcher_usage, judge_usage, num_questions = get_samples(
            input_dir="/fast/nchandak/qaevals/judge_outputs/",
            which_dataset=which_dataset
        )
        results.append((cost_usage, matcher_usage, judge_usage, num_questions, which_dataset))
        # break 
    
    # return results

# def plot_cost_analysis_horizontal(cost_usage, matcher_usage, judge_usage, num_questions, which_dataset: str, ax, show_ylabel=True):
#     """
#     Create a horizontal bar chart showing cost breakdown across different models and tasks,
#     with separate input and output costs, on the provided axis.
#     """
    # Set up colors for legend
    response_input_color = '#87CEEB'   # Light blue for response input
    response_output_color = '#4682B4'  # Dark blue for response output
    matcher_input_color = '#90EE90'    # Light green for matcher input
    matcher_output_color = '#32CD32'   # Dark green for matcher output
    judge_input_color = '#FFB6C1'      # Light red for judge input
    judge_output_color = '#FF6347'     # Dark red for judge output

    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, color=response_input_color, label='Model Input'),
        plt.Rectangle((0, 0), 1, 1, color=response_output_color, label='Model Output'),
        plt.Rectangle((0, 0), 1, 1, color=matcher_input_color, label='Matcher Input'),
        plt.Rectangle((0, 0), 1, 1, color=matcher_output_color, label='Matcher Output'),
        # plt.Rectangle((0, 0), 1, 1, color=judge_input_color, label='Judge Input'),
        # plt.Rectangle((0, 0), 1, 1, color=judge_output_color, label='Judge Output'),
    ]

    # Create figure and axes
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharex=False)
    for i, (cost_usage, matcher_usage, judge_usage, num_questions, which_dataset) in enumerate(results):
        show_ylabel = (i == 0)
        plot_cost_analysis_vertical(cost_usage, matcher_usage, judge_usage, num_questions, which_dataset, axes[i], show_ylabel=show_ylabel)

    # Set y-axis offset text to '× 10⁻⁴' for both axes
    # for ax in axes:
    #     ax.yaxis.offsetText.set_text(r'$\times 10^{-3}$')

    # Shared legend below the plots
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.0),
               ncol=4, frameon=True, fontsize=20)
    plt.tight_layout(rect=[0, 0.08, 1, 1], pad=2.0)
    os.makedirs('plots', exist_ok=True)
    plt.savefig('plots/cost_analysis_both_vertical.pdf', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Combine model outputs for annotation")
    parser.add_argument("--input_dir", default="/fast/nchandak/qaevals/judge_outputs/",
                        help="Directory containing model output files")
    args = parser.parse_args()
    # plot_both_datasets will call get_samples for both datasets
    plot_both_datasets()

if __name__ == "__main__":
    main()
