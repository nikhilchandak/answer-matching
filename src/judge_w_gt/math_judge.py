#!/usr/bin/env python
# coding: utf-8

import os
import sys
import json
import asyncio
import argparse
import logging
from typing import List, Dict, Any
from pathlib import Path
from tqdm import tqdm
from functools import partial

# Add parent directory to path to find the inference module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the OpenRouter inference engine
from inference.openrouter import *

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the prompt template for the judge

JUDGE_PROMPT_TEMPLATE_WITH_GT = """
Your task is to judge whether the given response to a question matches a given ground truth answer or not. You are provided with a question, a ground truth response, and the response you need to judge.
Possible judgments:

"0": The response does not match the ground-truth answer.
"1": The response matches the ground-truth.

Question: "{question}"
Ground truth: "{target}"
Response: "{response}"

Your job is to ONLY check whether the given response matches the ground truth answer or not in the context of the question. You DO NOT NEED to assess the correctness of the response.
This is part of an automated evaluation process, therefore you MUST OUTPUT your final answer as "0" or "1" in <answer> </answer> tags. 
YOU SHOULD ALWAYS END YOUR RESPONSE WITH <answer>0</answer> OR <answer>1</answer> TAGS. /no_think"""

def get_free_judge_prompt(question, response, cot=True):
    prompt = f"""Your task is to judge whether the given response to a question is correct or not. You are given a question and the response you are judging.
Possible judgments:
"0": The response is incorrect. 
"1": The response is correct. 

Question: "{question}"
Response: "{response}"

For numeric answers, the relative error, defined as |response - ground truth| / mean(response, ground truth), must be less than 1% for the response to be judged as a correct match. Here, if the ground truth is a specific numeric quantity but the response is a range, then they don't match (even if the range contains the ground truth).

To the best of your knowledge: Does the provided response answer the question correctly? This is part of an automated evaluation process, therefore you MUST OUTPUT your final answer as "0" or "1" in <answer> </answer> tags."""
    if cot:
        prompt += "\nThink step by step and end your response with <answer>0</answer> OR <answer>1</answer> TAGS."
    else :
        prompt += "\nYOU SHOULD ALWAYS END YOUR RESPONSE WITH <answer>0</answer> OR <answer>1</answer> TAGS."
        
    return prompt

# Define the prompt templates for the judge
def get_judge_prompt_with_gt(question, target, response, incorrect_options=None, cot=True):
    """
    Generate a prompt for the judge with ground truth.
    
    Args:
        question: The question being asked
        target: The ground truth answer
        response: The response to judge
        incorrect_options: Optional string containing incorrect options
        cot: Whether to use a COT prompt
        
    Returns:
        A formatted prompt string for the judge
    """
    # The response can have more information than the ground-truth. It can be more specific (for example, "Labrador" is more specific than "dog"), or have additional possible answers. But it must cover everything mentioned in the ground-truth. It is okay if it covers it in different words, i.e. paraphrased.

    prompt = f"""Your task is to judge whether the given response to a question matches a given ground truth answer or not. You are provided with a question, a ground truth response, and the response you need to judge.
For numeric answers, the relative error, defined as |response - ground truth| / mean(response, ground truth), must be less than 1% for the response to be judged as a correct match. Here, if the ground truth is a specific numeric quantity but the response is a range, then they don't match (even if the range contains the ground truth).

Possible judgments:

"0": The response does not match the ground-truth answer.
"1": The response matches the ground-truth.

Question: "{question}"
Ground truth: "{target}"
"""

    if incorrect_options:
        prompt += f"\n{incorrect_options}"
        
    prompt += f"""Response: "{response}"

Your job is to ONLY check whether the given response matches the ground truth answer or not in the context of the question. You DO NOT NEED to assess the correctness of the response. This is part of an automated evaluation process, therefore you MUST OUTPUT your final answer as "0" or "1" in <answer> </answer> tags."""
    
    if cot:
        prompt += "\nThink step by step and end your response with <answer>0</answer> OR <answer>1</answer> TAGS."
    else :
        prompt += "\nYOU SHOULD ALWAYS END YOUR RESPONSE WITH <answer>0</answer> OR <answer>1</answer> TAGS."
        
# Think step by step and end your response with <answer>0</answer> OR <answer>1</answer> TAGS.
# YOU SHOULD ALWAYS END YOUR RESPONSE WITH <answer>0</answer> OR <answer>1</answer> TAGS.

    return prompt


WITHOUT_ANSWER_TAGS = """
Your task is to judge whether the given response to a question matches a given ground truth answer or not. You are provided with a question, a ground truth response, and the response you need to judge.
Possible judgments:

"0": The response does not match the ground-truth answer.
"1": The response matches the ground-truth.

Question: "{question}"
Ground truth: "{target}"
Response: "{response}"

Your job is to ONLY check whether the given response matches the ground truth answer or not in the context of the question. You DO NOT NEED to assess the correctness of the response.
This is part of an automated evaluation process, therefore you must only output a single word: "0" or "1". Do not justify your decision.

Evaluate (0/1):
"""


def load_existing_results(data_path: str) -> List[Dict[str, Any]]:
    """
    Load existing results from input file
    
    Args:
        data_path: Path to the input file
        
    Returns:
        List of dictionaries containing the results
    """
    if os.path.exists(data_path):
        logger.info(f"Loading existing results from {data_path}")
        try:
            with open(data_path, 'r') as f:
                if data_path.endswith('.jsonl'):
                    data = [json.loads(line) for line in f if line.strip()]
                else:  # .json or other formats
                    data = json.load(f)
            logger.info(f"Loaded {len(data)} existing results")
            return data
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse existing results from {data_path}, starting fresh")
            return []
    return []

def save_results(data: List[Dict[str, Any]], data_path: str):
    """
    Save results to input file
    
    Args:
        data: List of dictionaries containing the results
        data_path: Path to the input file
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(data_path), exist_ok=True)
    
    # Save the results
    logger.info(f"Saving {len(data)} results to {data_path}")
    with open(data_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

async def judge_responses(
    data_path: str,
    batch_size: int = 5,
    model_name: str = "google/gemini-2.0-flash-001",
    filtered_ids_path: str = "/home/nchandak/qaevals/how-to-qa/src/filtering/data/mmlu_pro/filtered_random_sample_1000.txt",
    matcher: bool = True,

):
    """
    Judge model responses against ground truth using the specified model via OpenRouter.
    
    Args:
        data_path: Path to the JSON file containing model responses and ground truth
        batch_size: Number of samples to process in a batch
        model_name: Name of the OpenRouter model to use for judgment
    """
    
    # Load the input data
    logger.info(f"Loading data from {data_path}")
    data = load_existing_results(data_path)
    
    logger.info(f"Loaded input data with {len(data)} samples")
    
    # template = WITHOUT_ANSWER_TAGS
    # template = JUDGE_PROMPT_TEMPLATE_WITH_GT
    template = get_judge_prompt_with_gt
    
    # Filter existing data to the 'data' list
    question_ids = [item.get("question_id") for item in data]
    
    filtered_ids = list(range(5001))
        
    # filtered_ids = filtered_ids[:5]
    
    # Create a mapping of question_id to existing data
    existing_items_map = {item.get("question_id"): item for item in data if "question_id" in item and item.get("question_id") in filtered_ids}
    
    # Use a shortened model name for the score field
    short_model_name = model_name.split("/")[-1] # .replace("-", "_").replace(".", "_")
    if not matcher:
        short_model_name += "-JUDGE"
        template = get_free_judge_prompt
        
    if "v3" in short_model_name.lower() or "o4" in short_model_name.lower() or "r1" in short_model_name.lower():
        template = partial(template, cot=False)
    
    score_field = f"score_{short_model_name}"
    response_field = f"response_{short_model_name}"
    tokens_field = f"completion_tokens_{short_model_name}"
    
    logger.info(f"Using score field: {score_field}")
    
    # Update data with existing evaluations and identify which samples need judgment
    samples_to_judge = []
    samples_indices = []
    
    for i, sample in enumerate(data):
        question_id = sample.get("question_id")
        
        if question_id not in filtered_ids:
            continue
        
        # If this specific model hasn't scored this item yet
        if score_field not in sample or sample.get(response_field, "") == "":
        # if True:
            samples_to_judge.append(sample)
            samples_indices.append(i)
    
    if not samples_to_judge:
        logger.info(f"All samples have already been judged by {model_name}, nothing to do")
        
        correct_count = sum(1 for item in existing_items_map.values() if int(item.get(score_field)) == 1)
        total_judged = sum(1 for item in existing_items_map.values() if score_field in item)
        reference_field = "exact_match"
        # exact_match_count = sum(1 for item in existing_map.values() if (isinstance(item.get(reference_field), int) and int(item.get(reference_field)) == 1) or (isinstance(item.get(reference_field), list) and int(item.get(reference_field)[0]) == 1))
        # alignment_count = sum(1 for item in existing_map.values() if (isinstance(item.get(score_field), int) and isinstance(item.get(reference_field), int) and ((int(item.get(score_field)) == 1 and int(item.get(reference_field)) == 1) or (int(item.get(score_field)) == 0 and int(item.get(reference_field)) == 0))))
        
        exact_match_count = 0
        alignment_count = 0
        for item in existing_items_map.values():
            if reference_field not in item:
                continue
            if isinstance(item.get(reference_field), int) or isinstance(item.get(reference_field), str):
                if int(item.get(reference_field)) == 1:
                    exact_match_count += 1
                if int(item.get(score_field)) == 1 and int(item.get(reference_field)) == 1:
                    alignment_count += 1
                if int(item.get(score_field)) == 0 and int(item.get(reference_field)) == 0:
                    alignment_count += 1
                    
            elif isinstance(item.get(reference_field), list):
                if int(item.get(reference_field)[0]) == 1:
                    exact_match_count += 1
                if int(item.get(score_field)) == 1 and int(item.get(reference_field)[0]) == 1:
                    alignment_count += 1
                if int(item.get(score_field)) == 0 and int(item.get(reference_field)[0]) == 0:
                    alignment_count += 1
        
        if total_judged > 0:
            accuracy_percentage = correct_count / total_judged * 100
            exact_match_percentage = exact_match_count / total_judged * 100
            alignment_percentage = alignment_count / total_judged * 100
            
            logger.info(f"Summary for {model_name}: {correct_count}/{total_judged} correct ({accuracy_percentage:.2f}%)")
            logger.info(f"Summary for {model_name}: {exact_match_count}/{total_judged} exact match ({exact_match_percentage:.2f}%)")
            logger.info(f"Summary for {model_name}: {alignment_count}/{total_judged} alignment ({alignment_percentage:.2f}%)")
    
        return data
    
    logger.info(f"Need to judge {len(samples_to_judge)} out of {len(existing_items_map)} samples with {model_name}")
    
    # Initialize the OpenRouter inference engine
    logger.info(f"Initializing {model_name} judge via OpenRouter")
    
    if template == WITHOUT_ANSWER_TAGS:
        judge = OpenRouterInference(model=model_name, max_tokens=64, temperature=0.0)
    else:
        if matcher:
            judge = OpenRouterInference(model=model_name, max_tokens=2048, temperature=0.0)
        else:
            judge = OpenRouterInference(model=model_name, max_tokens=16384, temperature=0.6)
    
    # Process in batches
    for batch_start in range(0, len(samples_to_judge), batch_size):
        batch_end = min(batch_start + batch_size, len(samples_to_judge))
        batch = samples_to_judge[batch_start:batch_end]
        batch_orig_indices = samples_indices[batch_start:batch_end]
        
        logger.info(f"Processing batch {batch_start//batch_size + 1}, samples {batch_start+1} to {batch_end}")
        
        # Prepare prompts for the judge
        prompts = []
        
        for y, sample in enumerate(batch):
            question = sample.get("question", "")
            target = f"${sample.get("target", "")}$"  # Ground truth answer
            # response = sample.get("resps", "")
            response = f"${sample.get("filtered_resps", "")}$"
            
            if matcher:
                prompt = template(
                    question=question,
                    target=target,
                    response=response,
                )
            else:
                prompt = template(
                    question=question,
                    response=response,
                )
            
            if "qwen3" in model_name.lower():
                prompt += " /no_think"
            
            prompts.append(prompt)
            if y < 1 :
                logger.info(f"Prompt: {prompt}")
            
        # Generate judgments
        # Generate judgments one by one since async_query_openrouter only accepts a single prompt
        raw_judgments = await judge.generate(prompts, batch_size=batch_size)
            
        # Process the judgments and update the original data
        for i, (orig_idx, ret) in enumerate(zip(batch_orig_indices, raw_judgments)):
            
            if ret is None:
                # Handle case where API request failed
                response = ""
                finish_reason = "api_error"
                prompt_tokens = 0
                completion_tokens = 0
                reasoning = ""
            else:
                # Extract fields from successful response
                response = ret.get("response", "")
                finish_reason = ret.get("finish_reason", "")
                prompt_tokens = ret.get("prompt_tokens", 0)
                completion_tokens = ret.get("completion_tokens", 0)
                reasoning = ret.get("reasoning", "")
                
            raw_judgment = response
            # print("Reasoning: ", reasoning)
            # print("Response: ", response)
            
            # Extract binary judgment (0 or 1) from response
            binary_judgment = "0"  # Default to incorrect
            
            if raw_judgment:
                # Try to extract judgment from <answer> tags
                if template == WITHOUT_ANSWER_TAGS:
                    binary_judgment = raw_judgment.strip()[0]
                    # print("Raw judgment: ", raw_judgment)
                    # print("Binary judgment: ", binary_judgment)
                else :
                    import re
                    answer_match = re.search(r'<answer>\s*(\d)\s*</answer>', raw_judgment)
                    
                    if answer_match:
                        binary_judgment = answer_match.group(1)
                    else:
                        # Fallback: look for "1" or "0" in the response
                        if "1" in raw_judgment and "0" not in raw_judgment:
                            binary_judgment = "1"
                        elif "0" in raw_judgment and "1" not in raw_judgment:
                            binary_judgment = "0"
            
            # Add judgment to the sample using existing_items_map instead of data
            question_id = data[orig_idx].get("question_id")
            if question_id in existing_items_map:
                existing_items_map[question_id][score_field] = binary_judgment
                existing_items_map[question_id][response_field] = raw_judgment
                existing_items_map[question_id][tokens_field] = completion_tokens
                
        # Save results after each batch
        if len(filtered_ids) > 100:
            save_results(list(existing_items_map.values()), data_path)
        
        # Be nice to the API
        await asyncio.sleep(1)
    
    # Calculate and print summary statistics
    # correct_count = sum(1 for sample in data if sample.get(score_field) == "1")
    # total_judged = sum(1 for sample in data if score_field in sample)
    
    correct_count = sum(1 for item in existing_items_map.values() if int(item.get(score_field)) == 1)
    total_judged = sum(1 for item in existing_items_map.values() if score_field in item)
    reference_field = "exact_match"
    # exact_match_count = sum(1 for item in existing_map.values() if (isinstance(item.get(reference_field), int) and int(item.get(reference_field)) == 1) or (isinstance(item.get(reference_field), list) and int(item.get(reference_field)[0]) == 1))
    # alignment_count = sum(1 for item in existing_map.values() if (isinstance(item.get(score_field), int) and isinstance(item.get(reference_field), int) and ((int(item.get(score_field)) == 1 and int(item.get(reference_field)) == 1) or (int(item.get(score_field)) == 0 and int(item.get(reference_field)) == 0))))
    
    exact_match_count = 0
    alignment_count = 0
    for item in existing_items_map.values():
        if reference_field not in item:
            continue
        if isinstance(item.get(reference_field), int) or isinstance(item.get(reference_field), str):
            if int(item.get(reference_field)) == 1:
                exact_match_count += 1
            if int(item.get(score_field)) == 1 and int(item.get(reference_field)) == 1:
                alignment_count += 1
            if int(item.get(score_field)) == 0 and int(item.get(reference_field)) == 0:
                alignment_count += 1
                
        elif isinstance(item.get(reference_field), list):
            if int(item.get(reference_field)[0]) == 1:
                exact_match_count += 1
            if int(item.get(score_field)) == 1 and int(item.get(reference_field)[0]) == 1:
                alignment_count += 1
            if int(item.get(score_field)) == 0 and int(item.get(reference_field)[0]) == 0:
                alignment_count += 1
    
    if total_judged > 0:
        accuracy_percentage = correct_count / total_judged * 100
        exact_match_percentage = exact_match_count / total_judged * 100
        alignment_percentage = alignment_count / total_judged * 100
        extra_info = " (matcher)" if matcher else " (free judge)"
        
        logger.info(f"Summary for {model_name}{extra_info}: {correct_count}/{total_judged} correct ({accuracy_percentage:.2f}%)")
        logger.info(f"Summary for {model_name}{extra_info}: {exact_match_count}/{total_judged} exact match ({exact_match_percentage:.2f}%)")
        logger.info(f"Summary for {model_name}{extra_info}: {alignment_count}/{total_judged} alignment ({alignment_percentage:.2f}%)")
    
    # # Avoid division by zero error
    # if total_judged > 0:
    #     accuracy_percentage = correct_count / total_judged * 100
    #     logger.info(f"Summary for {model_name}: {correct_count}/{total_judged} correct ({accuracy_percentage:.2f}%)")
    # else:
    #     logger.info(f"Summary for {model_name}: {correct_count}/{total_judged} correct (0.00%)")
    
    return data

def main():
    parser = argparse.ArgumentParser(description="Judge model responses against ground truth using an OpenRouter model")
    parser.add_argument("--input_dir", default="/fast/nchandak/qaevals/judge_outputs/math_free/qwen2.5-7b-it_non_thinking/", 
                      help="Directory containing samples.json files to process")
    parser.add_argument("--batch-size", type=int, default=501,
                      help="Number of samples to process in a batch")
    parser.add_argument("--model", default="google/gemini-2.0-flash-001",
                      help="OpenRouter model name to use for judgment")
    parser.add_argument("--filtered_ids_path", default="/home/nchandak/qaevals/how-to-qa/src/filtering/data/mmlu_pro/filtered_stratified_sample_1002.txt",
                        help="Path to the file containing filtered question ids")
    parser.add_argument("--judge", action="store_true",
                        help="Whether to use a matcher model or a free judge model")
    args = parser.parse_args()
    matcher = not args.judge
    
    models = [
        # "openai/gpt-4o"
        # "openai/o4-mini",
        # "deepseek/deepseek-r1",
        # "qwen/qwen3-14b"
        # "qwen/qwen3-32b"
        "qwen/qwen-2.5-7b-instruct",
        
        
        # "deepseek/deepseek-chat-v3-0324",
        # "qwen/qwen-2.5-72b-instruct",
        
        # "meta-llama/llama-4-scout",
        # "meta-llama/llama-4-maverick",
        # "meta-llama/llama-3.1-8b-instruct",
        # "meta-llama/llama-3-8b-instruct",
        
        # "qwen/qwen3-8b",
        # "qwen/qwen3-14b",
        
        # "meta-llama/llama-3.1-8b-instruct",
        # "meta-llama/llama-3.2-3b-instruct",
        # "meta-llama/llama-3.2-1b-instruct",
    ]
    
    k = 0
    # Walk through all files in the input directory recursively
    for root, dirs, files in os.walk(args.input_dir):
        for file in files:
            file_path = os.path.join(root, file)
            if file == "samples.jsonl":
            # if file == "samples_deepseek-chat-v3-0324.jsonl":
            # if k < 1 and file == "combined_samples_to_annotate.jsonl":
            # if "stratified_sample/samples_" in file_path and file.endswith(".jsonl"):
                logger.info(f"Processing file: {file_path}")
                
                k += 1
                for model in models:
                    # Run the async function
                    asyncio.run(judge_responses(
                        data_path=file_path,
                        batch_size=args.batch_size,
                        model_name=model,
                        filtered_ids_path=args.filtered_ids_path,
                        matcher=matcher
                    ))

if __name__ == "__main__":
    main()