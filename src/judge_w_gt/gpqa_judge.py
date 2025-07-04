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
import datasets

# Add parent directory to path to find the inference module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the OpenRouter inference engine
from inference.openrouter import *

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


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

The response should fully answer the question and must not be vague.
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
For a response to "match", it must have at least as much information as the ground-truth. 
The response can have more information than the ground-truth. It can be more specific (for example, "Labrador" is more specific than "dog"), or have additional possible correct answers. But it must cover everything mentioned in the ground-truth. It is okay if it covers it in different words, i.e. paraphrased. 
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
    # Load the additional dataset from HF
    # additional_dataset = datasets.load_dataset("nikhilchandak/gpqa-diamond-test2")
    additional_dataset = datasets.load_dataset("nikhilchandak/GPQA-diamond-free")
    
    additional_data = {item["Record ID"]: item for item in additional_dataset["train"]}
    
    if os.path.exists(data_path):
        logger.info(f"Loading existing results from {data_path}")
        try:
            data = []
            with open(data_path, 'r') as f:
                for line in f:
                    current_data = json.loads(line)
                    if current_data["question_id"] in additional_data:
                        current_data["question"] = additional_data[current_data["question_id"]]["Question"]
                    else:
                        logger.warning(f"Question ID {current_data['question_id']} not found in additional dataset")
                    data.append(current_data)
                    
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
    
    filtered_ids = list(question_ids)
    if filtered_ids_path:
        filtered_ids = []
        logger.info(f"Loading filtered ids from {filtered_ids_path}")
        if filtered_ids_path.endswith(".txt"):
            with open(filtered_ids_path, 'r') as f:
                filtered_ids = [int(line.strip()) for line in f.readlines()]
        elif filtered_ids_path.endswith(".jsonl"):
            with open(filtered_ids_path, 'r') as f:
                for line in f.readlines():
                    here = json.loads(line)
                    qid = list(here.keys())[0]
                    vals = here[qid]
                    if vals["llm_judge_fine"] >= 0:
                        filtered_ids.append(qid)

        logger.info(f"Loaded {len(filtered_ids)} filtered ids")
        
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
    prompt_token_field = f"prompt_tokens_{short_model_name}"
    
    logger.info(f"Using score field: {score_field}")
    
    # Update data with existing evaluations and identify which samples need judgment
    samples_to_judge = []
    samples_indices = []
    
    for i, sample in enumerate(data):
        question_id = sample.get("question_id")
        
        if question_id not in filtered_ids:
            continue
        
        # If this specific model hasn't scored this item yet
        if score_field not in sample or sample.get(response_field, "") == "" or prompt_token_field not in sample:
        # if True :
            samples_to_judge.append(sample)
            samples_indices.append(i)
    
    if not samples_to_judge:
        logger.info(f"All samples have already been judged by {model_name}, nothing to do")
        
        correct_count = sum(1 for item in existing_items_map.values() if int(item.get(score_field)[0]) == 1)
        total_judged = sum(1 for item in existing_items_map.values() if score_field in item)
        reference_field = "score_deepseek-chat-v3-0324"
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
                if int(item.get(score_field)[0]) == 1 and int(item.get(reference_field)[0]) == 1:
                    alignment_count += 1
                if int(item.get(score_field)[0]) == 0 and int(item.get(reference_field)[0]) == 0:
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
            temp2 = 0.3 
            if "o4" in model_name.lower() or "r1" in model_name.lower():
                temp2 = 0.6 
                
            judge = OpenRouterInference(model=model_name, max_tokens=32768, temperature=temp2)
    
    
    # Process in batches
    for batch_start in range(0, len(samples_to_judge), batch_size):
        batch_end = min(batch_start + batch_size, len(samples_to_judge))
        batch = samples_to_judge[batch_start:batch_end]
        batch_orig_indices = samples_indices[batch_start:batch_end]
        
        logger.info(f"Processing batch {batch_start//batch_size + 1}, samples {batch_start+1} to {batch_end}")
        
        # Prepare prompts for the judge
        prompts = []
        # Create a mapping to track which prompt corresponds to which sample and response index
        prompt_mapping = []  # Will store tuples of (sample_idx, response_idx)
        
        for sample_idx, sample in enumerate(batch):
            question = sample.get("question", "")
            target = sample.get("target", "")  # Ground truth answer
            options = sample.get("options", [])
            answer_index = sample.get("answer_index", -1)
            
            # Handle filtered_resps as a list of responses
            filtered_resps = sample.get("filtered_resps", [])
            if not isinstance(filtered_resps, list):
                filtered_resps = [filtered_resps]  # Convert to list if it's a single string
            
            # Generate incorrect options format
            incorrect_options_text = ""
            j = 0
            for i, option in enumerate(options):
                if i != answer_index:  # Skip the correct option
                    incorrect_options_text += f"Incorrect option ({j+1}): \"{option}\"\n"
                    j += 1
            
            # Create a prompt for each response in filtered_resps
            for resp_idx, response in enumerate(filtered_resps):
                if matcher:
                    prompt = template(
                        question=question,
                        target=target,
                        response=response,
                        # incorrect_options=incorrect_options_text
                    )
                else:
                    prompt = template(
                        question=question,
                        response=response,
                    )
                
                if "qwen3" in model_name.lower():
                    prompt += " /no_think"
                
                if sample_idx < 1 :
                    logger.info(f"Prompt: {prompt}")
                
                prompts.append(prompt)
                # logger.info(f"Prompt: {prompt}")
                prompt_mapping.append((sample_idx, resp_idx))  # Track which sample and response this prompt is for
        
        # Generate judgments
        raw_judgments = await judge.generate(prompts, batch_size=batch_size)
        
        # Process the judgments and update the original data
        # Group judgments by sample
        sample_judgments = {}  # Will store {sample_idx: {binary_judgments: [], raw_judgments: []}}
        
        for (sample_idx, resp_idx), ret in zip(prompt_mapping, raw_judgments):
            if sample_idx not in sample_judgments:
                sample_judgments[sample_idx] = {"binary_judgments": [], "raw_judgments": [], 
                                                "completion_tokens": [], "prompt_tokens": []}
            
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
            
            # Extract binary judgment (0 or 1) from response
            binary_judgment = "0"  # Default to incorrect
            
            if raw_judgment:
                # Try to extract judgment from <answer> tags
                if template == WITHOUT_ANSWER_TAGS:
                    binary_judgment = raw_judgment.strip()[0]
                    # False by default
                    if binary_judgment not in ["0", "1"]:
                        binary_judgment = "0"
                else:
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
            
            # Store the judgment for this response
            sample_judgments[sample_idx]["binary_judgments"].append(int(binary_judgment))
            sample_judgments[sample_idx]["raw_judgments"].append(raw_judgment)
            sample_judgments[sample_idx]["completion_tokens"].append(completion_tokens)
            sample_judgments[sample_idx]["prompt_tokens"].append(prompt_tokens)
            
        # Now update the original data with all judgments for each sample
        for sample_idx, judgments in sample_judgments.items():
            orig_idx = batch_orig_indices[sample_idx]
            question_id = data[orig_idx].get("question_id")
            
            
            data[orig_idx][score_field] = judgments["binary_judgments"]
            data[orig_idx][response_field] = judgments["raw_judgments"]
            data[orig_idx][tokens_field] = judgments["completion_tokens"]
            data[orig_idx][prompt_token_field] = judgments["prompt_tokens"]
            
            # if question_id in existing_items_map:
            #     existing_items_map[question_id][score_field] = judgments["binary_judgments"]
            #     existing_items_map[question_id][response_field] = judgments["raw_judgments"]
            #     existing_items_map[question_id][tokens_field] = judgments["completion_tokens"]
                
        # Save results after each batch
        if len(filtered_ids) > 100:
            save_results(data, data_path)
            # save_results(list(existing_items_map.values()), data_path)
        
        # Be nice to the API
        await asyncio.sleep(1)
    
    # Calculate and print summary statistics
    # Update to handle arrays of judgments
    correct_count = 0
    total_judged = 0
    
    for sample in data:
        if score_field in sample:
            judgments = sample.get(score_field)
            if isinstance(judgments, list):
                # correct_count += sum(1 for j in judgments if j == "1")
                # total_judged += len(judgments)
                
                correct_count += sum(judgments) / float(len(judgments))
                total_judged += 1
            else:
                # Handle legacy format (single judgment)
                correct_count += int(judgments)
                total_judged += 1
    
    # Avoid division by zero error
    if total_judged > 0:
        accuracy_percentage = correct_count / total_judged * 100
        logger.info(f"Summary for {model_name}: {correct_count}/{total_judged} correct ({accuracy_percentage:.2f}%)")
    else:
        logger.info(f"Summary for {model_name}: {correct_count}/{total_judged} correct (0.00%)")
    
    # if total_judged > 0:
        
    #     logger.info(f"All samples have already been judged by {model_name}, nothing to do")
        
    #     reference_field = "score_deepseek-chat-v3-0324"
    #     # exact_match_count = sum(1 for item in existing_map.values() if (isinstance(item.get(reference_field), int) and int(item.get(reference_field)) == 1) or (isinstance(item.get(reference_field), list) and int(item.get(reference_field)[0]) == 1))
    #     # alignment_count = sum(1 for item in existing_map.values() if (isinstance(item.get(score_field), int) and isinstance(item.get(reference_field), int) and ((int(item.get(score_field)) == 1 and int(item.get(reference_field)) == 1) or (int(item.get(score_field)) == 0 and int(item.get(reference_field)) == 0))))
        
    #     exact_match_count = 0
    #     alignment_count = 0
    #     for item in existing_items_map.values():
    #         if reference_field not in item:
    #             continue
    #         if isinstance(item.get(reference_field), int) or isinstance(item.get(reference_field), str):
    #             if int(item.get(reference_field)) == 1:
    #                 exact_match_count += 1
    #             if int(item.get(score_field)) == 1 and int(item.get(reference_field)) == 1:
    #                 alignment_count += 1
    #             if int(item.get(score_field)) == 0 and int(item.get(reference_field)) == 0:
    #                 alignment_count += 1
                    
    #         elif isinstance(item.get(reference_field), list):
    #             if int(item.get(reference_field)[0]) == 1:
    #                 exact_match_count += 1
    #             if int(item.get(score_field)[0]) == 1 and int(item.get(reference_field)[0]) == 1:
    #                 alignment_count += 1
    #             if int(item.get(score_field)[0]) == 0 and int(item.get(reference_field)[0]) == 0:
    #                 alignment_count += 1
        
    #     if total_judged > 0:
    #         accuracy_percentage = correct_count / total_judged * 100
    #         exact_match_percentage = exact_match_count / total_judged * 100
    #         alignment_percentage = alignment_count / total_judged * 100
            
    #         logger.info(f"Summary for {model_name}: {correct_count}/{total_judged} correct ({accuracy_percentage:.2f}%)")
    #         logger.info(f"Summary for {model_name}: {exact_match_count}/{total_judged} exact match ({exact_match_percentage:.2f}%)")
    #         logger.info(f"Summary for {model_name}: {alignment_count}/{total_judged} alignment ({alignment_percentage:.2f}%)")
    # else :
    #     logger.info(f"No samples to judge for {model_name}")
    
    return data

def main():
    parser = argparse.ArgumentParser(description="Judge model responses against ground truth using an OpenRouter model")
    parser.add_argument("--input_dir", default="/fast/nchandak/qaevals/judge_outputs/gpqa_diamond_free/",
                      help="Directory containing samples.json files to process")
    parser.add_argument("--batch-size", type=int, default=500,
                      help="Number of samples to process in a batch")
    parser.add_argument("--model", default="google/gemini-2.0-flash-001",
                      help="OpenRouter model name to use for judgment")
    parser.add_argument("--filtered_ids_path", default="/fast/nchandak/qaevals/filter/gpqa/GPQA_question_hash.jsonl",
                        help="Path to the file containing filtered question ids")
    parser.add_argument("--judge", action="store_true",
                        help="Whether to use a matcher model or a free judge model")
    args = parser.parse_args()
    matcher = not args.judge
    
    models = [
        # "openai/gpt-4o"
        # "openai/o4-mini",
        # "openai/o4-mini-high",
        # "deepseek/deepseek-r1",
        # "qwen/qwen3-14b"
        # "qwen/qwen3-32b"
        # "qwen/qwen-2.5-7b-instruct",
        
        
        "deepseek/deepseek-chat-v3-0324",
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
    
    annotation_models = [
        "gpt-4o",
        "deepseek-chat-v3-0324",
        "llama-4-maverick",
        "qwen3-32b",
    ]
    
    # Walk through all files in the input directory recursively
    for root, dirs, files in os.walk(args.input_dir):
        for file in files:
            file_path = os.path.join(root, file)
            # if file == "samples.jsonl":
            # if "samples" in file and "grok" in file and file.endswith(".jsonl"):
            # if file == "samples_deepseek-chat-v3-0324.jsonl":
            # if file == "gpqa_combined_samples_to_annotate.jsonl":
            if "free/samples_" in file_path and file.endswith(".jsonl"):
                # if "r1" not in file_path:
                #     continue
                
                # model_to_annotate = False 
                # for model in annotation_models:
                #     if model in file_path:
                #         model_to_annotate = True
                #         break
                
                # if not model_to_annotate:
                #     continue
                
                logger.info(f"Processing file: {file_path}")
                
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