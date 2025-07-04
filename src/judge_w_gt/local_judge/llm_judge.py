#!/usr/bin/env python
# coding: utf-8

import os
import sys
import json
import argparse
import logging
import numpy as np
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Any, Tuple
from functools import partial

# Import the models_utils file for model paths and configurations
from models_utils import get_model_name, get_n_gpus

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

# Define the prompt templates for the judge
def prompt_without_tags(question, target, response, incorrect_options=None):
    """
    Generate a prompt for the judge with ground truth.
    
    Args:
        question: The question being asked
        target: The ground truth answer
        response: The response to judge
        incorrect_options: Optional string containing incorrect options
        
    Returns:
        A formatted prompt string for the judge
    """
    prompt = f"""Your task is to judge whether the given response to a question matches a given ground truth answer or not. You are provided with a question, a ground truth response, and the response you need to judge.
For a response to "match", it must have at least as much information as the ground-truth. 
The response can have more information than the ground-truth. It can be more specific (for example, "Labrador" is more specific than "dog"), or have additional possible (correct)answers. But it must cover everything mentioned in the ground-truth. It is okay if it covers it in different words, i.e. paraphrased. 
For numeric answers, the relative error, defined as |response - ground truth| / mean(response, ground truth), must be less than 1% for the response to be judged as a correct match. Here, if the ground truth is a specific numeric quantity but the response is a range, then they don't match (even if the range contains the ground truth).

Possible judgments:

"0": The response does not match the ground-truth answer.
"1": The response matches the ground-truth.

Question: "{question}"
Ground truth: "{target}"
Response: "{response}"

Your job is to ONLY check whether the given response matches the ground truth answer or not in the context of the question. You DO NOT NEED to assess the correctness of the response.
This is part of an automated evaluation process, therefore you MUST OUTPUT your final answer as "0" or "1" and nothing else. DO NOT JUSTIFY YOUR DECISION.
You must only output a single digit: 0 or 1."""
    return prompt




JUDGE_PROMPT_TEMPLATE_WITHOUT_GT = """
Your task is to judge whether the given response to a question is correct or not. You are only given a question and the response you are judging. 
The response should be correct if it has sufficient information to answer the question. It can have more information than necessary, and as long as that additional information is correct, the response should be judged as correct. If it is missing important information ASKED in the question, it should be judged as incorrect.
For numeric answers, the relative error, defined as |response - ground truth| / mean(response, ground truth), must be less than 1% for the response to be judged as correct.

Possible judgments:
"0": The response is incorrect.
"1": The response is correct.
    
Question: "{question}"
Response: "{response}"
    
To the best of your knowledge: Does the provided response answer the question correctly? This is part of an automated evaluation process, therefore you MUST OUTPUT your final answer as "0" or "1" in <answer> </answer> tags. 
YOU SHOULD ALWAYS END YOUR RESPONSE WITH <answer>0</answer> OR <answer>1</answer> TAGS.
"""

# def load_existing_results(output_path: str) -> List[Dict[str, Any]]:
#     """
#     Load existing results from output file if it exists
    
#     Args:
#         output_path: Path to the output file
        
#     Returns:
#         List of dictionaries containing the results, or an empty list if no file exists
#     """
#     if os.path.exists(output_path):
#         logger.info(f"Loading existing results from {output_path}")
#         try:
#             data = []
#             with open(output_path, 'r') as f:
#                 for line in f:
#                     data.append(json.loads(line))
#             logger.info(f"Loaded {len(data)} existing results")
#             return data
#         except json.JSONDecodeError:
#             logger.warning(f"Failed to parse existing results from {output_path}, starting fresh")
#             return []
#     return []

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

def save_results(data: List[Dict[str, Any]], output_path: str):
    """
    Save results to output file
    
    Args:
        data: List of dictionaries containing the results
        output_path: Path to the output file
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # output_path = output_path.replace(".jsonl", "_judged.jsonl")
    
    # Save the results
    logger.info(f"Saving {len(data)} results to {output_path}")
    
    # Determine file format based on extension
    if output_path.endswith('.jsonl'):
        # Save as JSONL (one JSON object per line)
        with open(output_path, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
    else:
        # Save as regular JSON
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

def get_log_probs_vllm(model, tokenizer, prompts, token_ids_list, return_prob=False, gen_kwargs=None, max_tokens=2048):
    """
    Get log probabilities for specific token IDs using vLLM
    
    Args:
        model: vLLM model instance
        tokenizer: Tokenizer instance
        prompts: List of prompt strings
        token_ids_list: List of token IDs to get log probabilities for
        return_prob: Whether to return normalized probability in results (default: False)
        gen_kwargs: String of generation parameters (e.g. "temperature=0.7,top_p=0.9")
        max_tokens: Maximum number of tokens to generate (default: 2048)
        
    Returns:
        List of dictionaries containing generated text and optionally normalized probabilities
    """
    from vllm import SamplingParams
    import math
    
    # Default sampling parameters
    temperature = 0.0
    top_p = 0.95
    # top_k = -1
    min_p = 0.0
    do_sample = False
    n = 3  # Always generate 3 samples for each prompt by default
    
    # Parse gen_kwargs if provided
    if gen_kwargs:
        params_dict = {}
        for param in gen_kwargs.split(','):
            if '=' in param:
                key, value = param.split('=')
                key = key.strip()
                value = value.strip()
                
                if key == 'temperature':
                    temperature = float(value)
                elif key == 'top_p':
                    top_p = float(value)
                elif key == 'top_k':
                    top_k = int(value)
                elif key == 'min_p':
                    min_p = float(value)
                elif key == 'max_gen_toks':
                    max_tokens = int(value)
                elif key == 'n':
                    n = int(value)
                elif key == 'do_sample' and value.lower() == 'true':
                    do_sample = True
    
    # Use the parsed parameters
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        max_tokens=max_tokens,
        logprobs=5,
        n=n
    )
    
    logger.info(f"Using generation parameters: temperature={temperature}, top_p={top_p}, top_k={top_k}, min_p={min_p}, max_tokens={max_tokens}, do_sample={do_sample}")
    
    # Apply chat template to each prompt
    formatted_prompts = []
    for prompt in prompts:
        # Format the prompt using the model's chat template
        # Create a messages list with a single user message
        messages = [{"role": "user", "content": prompt}]
        try:
            # Apply the chat template
            formatted_prompt = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            formatted_prompts.append(formatted_prompt)
        except Exception as e:
            logger.warning(f"Failed to apply chat template: {e}. Using raw prompt.")
            formatted_prompts.append(prompt)
    
    batch_outputs = model.generate(formatted_prompts, sampling_params)
    
    results = []
    maxed_out = 0
    answer_tag_not_found = 0
    
    # Process each prompt/sample
    for i, sample_outputs in enumerate(batch_outputs):
        sample_results = []
        
        current_prompt = formatted_prompts[i]
        # Process each of the 3 generations for the current sample
        for j, output in enumerate(sample_outputs.outputs):
            # Extract logprobs from the vLLM output
            normalized_prob = None
            token_logprobs = {}
            
            # Store the full original response
            original_response = output.text if output.text else ""
            
            # Store completion tokens and prompt tokens
            completion_tokens = len(output.token_ids) if hasattr(output, 'token_ids') else None
            # Get prompt tokens from the sample_outputs (RequestOutput object)
            prompt_tokens = len(sample_outputs.prompt_token_ids) if hasattr(sample_outputs, 'prompt_token_ids') else None
            
            if completion_tokens and completion_tokens > 0:
                if completion_tokens >= max_tokens:
                    maxed_out += 1
                    
            # Check if we have logprobs in the output
            if hasattr(output, 'logprobs') and output.logprobs:
                logprobs_data = output.logprobs
                
                # Find the position of <answer> tag in the response
                answer_tag_position = -1
                answer_tag_found = False
                
                # First, find the token position of the <answer> tag
                for idx, logprob_entry in enumerate(logprobs_data):
                    for token, logprob_obj in logprob_entry.items():
                        if hasattr(logprob_obj, 'decoded_token') and '<answer>' in logprob_obj.decoded_token:
                            answer_tag_position = idx
                            answer_tag_found = True
                            break
                    if answer_tag_found:
                        break
                
                # Extract logprobs for tokens we care about (0 and 1) only after <answer> tag
                for token_id in token_ids_list:
                    # Find the token in the logprobs list, but only after the <answer> tag
                    for idx, logprob_entry in enumerate(logprobs_data):
                        if answer_tag_found and idx <= answer_tag_position:
                            continue  # Skip entries before or at the <answer> tag if it exists
                        
                        # Each entry is a dictionary mapping token_id to Logprob objects
                        for token, logprob_obj in logprob_entry.items():
                            # Check if this is one of our target tokens
                            if token == token_id or (hasattr(logprob_obj, 'decoded_token') and 
                                                    logprob_obj.decoded_token in ['0', '1']):
                                # Store using the decoded token as key
                                decoded_token = str(logprob_obj.decoded_token) if hasattr(logprob_obj, 'decoded_token') else str(token)
                                token_logprobs[decoded_token] = logprob_obj.logprob
                                # Once we find a 0 or 1 after the <answer> tag, we can stop looking
                                if answer_tag_found:
                                    break
                                    
                # Determine which token has the higher logprob and calculate its normalized probability
                if "1" in token_logprobs and "0" in token_logprobs:
                    # Convert log probabilities to probabilities
                    prob_1 = math.exp(token_logprobs["1"])
                    prob_0 = math.exp(token_logprobs["0"])
                    
                    # Determine which token has higher probability
                    if prob_1 > prob_0:
                        normalized_prob = prob_1 / (prob_1 + prob_0)
                        generation = "1"
                    else:
                        normalized_prob = prob_0 / (prob_1 + prob_0)
                        generation = "0"
                        
                elif "1" in token_logprobs:
                    normalized_prob = 1.0
                    generation = "1"
                elif "0" in token_logprobs:
                    normalized_prob = 1.0
                    generation = "0"
                else:
                    generation = original_response
                    
            elif original_response:
                generation = original_response
            else:
                generation = ""
                
            # Extract binary judgment (0 or 1) from response
            import re
            answer_matches = list(re.finditer(r'<answer>\s*(\d)\s*</answer>', original_response))
            if answer_matches:
                # Get the last occurrence
                binary_judgment = answer_matches[0].group(1)
                generation = binary_judgment
            else:
                if j < (n - 1):
                    continue # Skip this generation if <answer> tag not found but if it is the last generation, we should still return the generation
                
                answer_tag_not_found += 1
                
            # Round normalized probability to 3 decimal places if it's not None
            if normalized_prob is not None:
                normalized_prob = round(normalized_prob, 3)
                
            # Create result dictionary
            result = {
                'generation': generation,
                'full_response': original_response
            }
            
            # Add completion tokens and prompt tokens to the result
            if completion_tokens is not None:
                result['completion_tokens'] = completion_tokens
            if prompt_tokens is not None:
                result['prompt_tokens'] = prompt_tokens
            
            # Truncate full_response to only keep content up to the first </answer> tag
            if '</answer>' in original_response:
                result['full_response'] = original_response.split('</answer>')[0] + '</answer>'
            
            # Only include probability if return_prob is True
            if return_prob and normalized_prob is not None:
                result['normalized_prob'] = normalized_prob
                
            sample_results.append(result)
            
            if answer_matches:
                break 
            
            # if j == 0:
            #     print(f"Current prompt: {current_prompt}")
            #     print(f"Result: {result}")
            #     print("--------------------------------\n\n")
            
        # Count the number of 0s and 1s in the generations
        count_0 = 0
        count_1 = 0
        results_0 = []
        results_1 = []
        
        for result in sample_results:
            generation = result.get('generation', '').strip()
            binary_judgment = 1 if generation == "1" else 0
            
            if binary_judgment == 1:
                count_1 += 1
                results_1.append(result)
            else:
                count_0 += 1
                results_0.append(result)
        
        # Determine which judgment is the majority
        if count_1 > count_0:
            # 1 is the majority, pick the first result with judgment 1
            for i, result in enumerate(results_1):
                generation = result.get('generation', '').strip()
                if generation == "1":
                    best_result = result.copy()
                    break
                
            if not 'normalized_prob' in best_result and return_prob:
                best_result['normalized_prob'] = best_result.get('normalized_prob', 1.0)
        else:
            # 0 is the majority (or tie), pick the first result with judgment 0
            for i, result in enumerate(results_0):
                generation = result.get('generation', '').strip()
                if generation == "0":
                    best_result = result.copy()
                    break
                
            if not 'normalized_prob' in best_result and return_prob:
                best_result['normalized_prob'] = best_result.get('normalized_prob', 1.0)
        
        results.append(best_result)
        
    # logger.info(f"Maxed out on {maxed_out} samples")
    logger.info(f"Maxed out Percentage: {maxed_out / len(batch_outputs) * 100:.2f}%")
    logger.info(f"Answer tag not found Percentage: {answer_tag_not_found / len(batch_outputs) * 100:.2f}%")
    return results

def judge_responses_with_gt(
    data_path: str,
    judge_model_path: str,
    output_dir: str,
    use_token_logprobs: bool = True,
    batch_size: int = 32,
    n_gpus: int = 1,
    gen_kwargs: str = None,
    max_tokens: int = 2048,
    thinking: bool = False
):
    """
    Judge model responses against ground truth and calculate normalized probabilities
    
    Args:
        data_path: Path to the JSON file containing model responses and ground truth
        judge_model_path: Path or HF identifier of the judge model
        output_dir: Directory to save the judgments
        use_token_logprobs: Whether to store normalized probabilities (ignored, now always stored)
        batch_size: Number of samples to process in a batch (only used for HF)
        gen_kwargs: Generation parameters like temperature, top_p, etc. (format: "temperature=0.7,top_p=0.9")
        max_tokens: Maximum number of tokens to generate
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
        
    # Load the input data
    logger.info(f"Loading data from {data_path}")
    # Handle different file formats
    data = load_existing_results(data_path)
    
    # data = data[:20]
    logger.info(f"Loaded input data with {len(data)} samples")
    
    # Use a shortened model name for the score field
    model_name = os.path.basename(judge_model_path.rstrip("/"))
    short_model_name = model_name.replace("-", "_").replace(".", "_")
    
    
    template = partial(get_judge_prompt_with_gt, cot=True)
    # template = partial(get_judge_prompt_with_gt, cot=False)
    
    if "llama-2-70b" in judge_model_path:
        template = partial(get_judge_prompt_with_gt, cot=False)
        logger.info("Using non-COT prompt for Llama-2-70b")
        model_name = "llama-2-70b-chat-hf"
        short_model_name = "llama-2-70b"
        
    # Use the same input file for output
    output_path = data_path
    score_field = f"score_{short_model_name}"
    prob_field = f"prob_{short_model_name}"
    response_field = f"response_{short_model_name}"
    tokens_field = f"completion_tokens_{short_model_name}"
    prompt_tokens_field = f"prompt_tokens_{short_model_name}"
    
    
    # Load existing results if available
    existing_data = data
    existing_map = {item["question_id"]: item for item in existing_data if "question_id" in item}
    
    # Identify which samples need judgment
    samples_to_judge = []
    sample_question_ids = []
    
    for sample in data:
        question_id = sample.get("question_id")
        
        # Skip if already judged
        if question_id in existing_map and score_field in existing_map[question_id]:
            continue
            
        samples_to_judge.append(sample)
        sample_question_ids.append(question_id)
    
    if not samples_to_judge:
        logger.info(f"All samples have already been judged by {model_name}, nothing to do")
        
        # Count existing entries for statistics
        correct_count = sum(1 for item in existing_data if (item.get(score_field)[0] == 1 if isinstance(item.get(score_field), list) else item.get(score_field) == 1))
        total_judged = sum(1 for item in existing_data if score_field in item)
        
        if total_judged > 0:
            accuracy_percentage = correct_count / total_judged * 100
            logger.info(f"Summary for {model_name}: {correct_count}/{total_judged} correct ({accuracy_percentage:.2f}%)")
        else:
            logger.info(f"Summary for {model_name}: 0/0 correct (0.00%)")
    
        return
    
    logger.info(f"Need to judge {len(samples_to_judge)} out of {len(data)} samples with {model_name}")
    
    # Initialize the model
    from vllm import LLM
    from transformers import AutoTokenizer
    
    logger.info(f"Initializing {model_name} judge via vLLM with {n_gpus} GPUs")
    
    # Ensure max_model_len is at least as large as max_tokens + context size
    # A reasonable buffer is to multiply max_tokens by 2
    max_model_len = max(4096, max_tokens * 2)
    # max_model_len = 16384
    
    model = LLM(
        model=judge_model_path,
        tensor_parallel_size=n_gpus,
        max_model_len=max_model_len,
        dtype="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(judge_model_path)
    
    # Get token IDs for "0", "1"
    token_0 = tokenizer.encode("0", add_special_tokens=False)[0]
    token_1 = tokenizer.encode("1", add_special_tokens=False)[0]
    tokens_to_check = [token_0, token_1]
    
    # Prepare prompts for the judge
    prompts = []
    # template = prompt_without_tags
        
    for i, sample in enumerate(samples_to_judge):
        question = sample.get("question", "")
        target = sample.get("target", "")  # Ground truth answer
        response = sample.get("filtered_resps", "")[0] if isinstance(sample.get("filtered_resps"), list) else sample.get("filtered_resps", "")
        incorrect_options_text = None 
        
        if "math" in data_path:
            target = "$" + target + "$"
            response = "$" + response + "$"
        
        # Generate incorrect options format
        if "options" in sample:
            options = sample.get("options", [])
            if len(options) > 0:
                answer_index = sample.get("answer_index", -1)
                incorrect_options_text = ""
                j = 0
                for qq, option in enumerate(options):
                    if qq != answer_index:  # Skip the correct option
                        incorrect_options_text += f"Incorrect option ({j+1}): \"{option}\"\n"
                        j += 1
                    # if j >= 3:
                    #     break
        
        # Create the prompt for the judge
        # prompt = template(question=question, target=target, response=response, incorrect_options=incorrect_options_text)
        prompt = template(question=question, target=target, response=response)
        # prompt = prompt_without_tags(question, target, response, incorrect_options_text)
        if "qwen3" in short_model_name.lower():
            if not thinking:
                prompt += " /no_think"
                
        if i < 1:
            logger.info(f"Index: {i}\nPrompt: {prompt}")
                
        prompts.append(prompt)
    
    # Generate judgments and get normalized probabilities
    logger.info(f"Getting judgments and normalized probabilities for {len(prompts)} samples")
    
    # Get results with multiple generations handled internally by get_log_probs_vllm
    results = get_log_probs_vllm(model, tokenizer, prompts, tokens_to_check, return_prob=use_token_logprobs, gen_kwargs=gen_kwargs, max_tokens=max_tokens)
    
    # Process the judgments and update the existing data
    for i, (question_id, result) in enumerate(zip(sample_question_ids, results)):
        # Extract binary judgment (0 or 1) from response
        generation = result.get('generation', '').strip()
        binary_judgment = 1 if generation == "1" else 0
        
        # Get full response text and normalized probability if available
        full_response = result.get('full_response', '')
        prob = result.get('normalized_prob')
        completion_tokens = result.get('completion_tokens')
        prompt_tokens = result.get('prompt_tokens')
        
        # Update the existing entry directly
        # existing_map[question_id][score_field] = binary_judgment
        # existing_map[question_id][response_field] = full_response
        samples_to_judge[i][score_field] = [binary_judgment]
        samples_to_judge[i][response_field] = [full_response]
        
        # Add token counts if available
        if completion_tokens is not None:
            samples_to_judge[i][tokens_field] = [completion_tokens]
        if prompt_tokens is not None:
            samples_to_judge[i][prompt_tokens_field] = [prompt_tokens]
        
        # Only add probability if it was calculated and returned
        if prob is not None:
            # existing_map[question_id][prob_field] = prob
            samples_to_judge[i][prob_field] = [prob]
    
    # Save results back to the original file
    # save_results(existing_data, output_path)
    save_results(samples_to_judge, output_path)
    
    # Calculate and print summary statistics
    correct_count = sum(1 for item in samples_to_judge if item.get(score_field) == 1)
    total_judged = sum(1 for item in samples_to_judge if score_field in item)
    reference_field = "score_deepseek-chat-v3-0324"
    # exact_match_count = sum(1 for item in existing_map.values() if (isinstance(item.get(reference_field), int) and int(item.get(reference_field)) == 1) or (isinstance(item.get(reference_field), list) and int(item.get(reference_field)[0]) == 1))
    # alignment_count = sum(1 for item in existing_map.values() if (isinstance(item.get(score_field), int) and isinstance(item.get(reference_field), int) and ((int(item.get(score_field)) == 1 and int(item.get(reference_field)) == 1) or (int(item.get(score_field)) == 0 and int(item.get(reference_field)) == 0))))
    
    exact_match_count = 0
    alignment_count = 0
    for item in samples_to_judge:
        if isinstance(item.get(reference_field), int) or isinstance(item.get(reference_field), str):
            if int(item.get(reference_field)) == 1:
                exact_match_count += 1
            if item.get(score_field) == 1 and int(item.get(reference_field)) == 1:
                alignment_count += 1
            if item.get(score_field) == 0 and int(item.get(reference_field)) == 0:
                alignment_count += 1
                
        elif isinstance(item.get(reference_field), list):
            if int(item.get(reference_field)[0]) == 1:
                exact_match_count += 1
            if item.get(score_field) == 1 and int(item.get(reference_field)[0]) == 1:
                alignment_count += 1
            if item.get(score_field) == 0 and int(item.get(reference_field)[0]) == 0:
                alignment_count += 1
    
    if total_judged > 0:
        accuracy_percentage = correct_count / total_judged * 100
        exact_match_percentage = exact_match_count / total_judged * 100
        alignment_percentage = alignment_count / total_judged * 100
        
        logger.info(f"Summary for {model_name}: {correct_count}/{total_judged} correct ({accuracy_percentage:.2f}%)")
        logger.info(f"Summary for {model_name}: {exact_match_count}/{total_judged} exact match ({exact_match_percentage:.2f}%)")
        logger.info(f"Summary for {model_name}: {alignment_count}/{total_judged} alignment ({alignment_percentage:.2f}%)")
    else:
        logger.info(f"Summary for {model_name}: 0/0 correct (0.00%)")

def main():
    parser = argparse.ArgumentParser(description="Judge model responses and calculate normalized probabilities")
    parser.add_argument("--input_dir", default="/fast/nchandak/qaevals/judge_outputs/math_free/qwen2.5-1.5b-it_non_thinking/", 
                      help="Path to the directory containing samples.jsonl")
    parser.add_argument("--output_dir", default="",
                      help="Directory to save the judgments (defaults to input_dir)")
    parser.add_argument("--logprobs", action="store_true",
                      help="Store normalized probabilities")
    parser.add_argument("--no_ground_truth", action="store_true",
                      help="Judge without using ground truth answers (judge purely from knowledge)")
    parser.add_argument("--batch_size", type=int, default=32,
                      help="Number of samples to process in a batch")
    parser.add_argument('--thinking', action='store_true', 
                      help='Whether to use thinking mode generation parameters')
    parser.add_argument('--gen_kwargs', type=str, default=None, 
                      help='Generation parameters like temperature, top_p, etc. (format: "temperature=0.7,top_p=0.9")')
    parser.add_argument('--max_tokens', type=int, default=2048, 
                      help='Maximum number of tokens to generate')
    
    args = parser.parse_args()
    
    # Set generation parameters based on thinking mode
    if args.gen_kwargs is None:
        if args.thinking:
            # thinking mode parameters
            args.gen_kwargs = f"temperature=0.6,top_p=0.95,min_p=0,top_k=20,max_gen_toks={args.max_tokens},do_sample=true"
        else:
            # non-thinking mode parameters
            args.gen_kwargs = f"temperature=0.7,top_p=0.8,min_p=0,top_k=20,max_gen_toks={args.max_tokens},do_sample=true"
    # If output dir is not specified, use input dir
    output_dir = args.output_dir if args.output_dir else args.input_dir
    
    # Import required modules
    from models_utils import models, get_n_gpus
    import os
    
    # Walk through all files in the input directory recursively
    samples_files = []
    for root, dirs, files in os.walk(args.input_dir):
        for file in files:
            # if file == "samples.jsonl":
            # if "combined_samples" in file and "annotate.jsonl" in file and len(samples_files) < 1:
            if "samples_" in file[:9] and ".jsonl" in file :
                samples_files.append(os.path.join(root, file))
    
    for x in samples_files:
        print(x)
        
    if not samples_files:
        logger.error(f"Could not find any samples.jsonl files in {args.input_dir} or its subdirectories")
        sys.exit(1)
    
    # return 
    # Process each samples.jsonl file found
    for input_file in samples_files:
        logger.info(f"Processing file: {input_file}")
        # Create corresponding output directory structure
        relative_path = os.path.relpath(os.path.dirname(input_file), args.input_dir)
        file_output_dir = os.path.join(output_dir, relative_path)
        os.makedirs(file_output_dir, exist_ok=True)
        
        for model in models.keys():
            model_dir = models[model]['model_path']
            gpus = get_n_gpus(model)
            logger.info(f"Running model {model} on {input_file}")
            
            # Run the appropriate judging function
            # if args.no_ground_truth:
            #     judge_responses_without_gt(
            #         data_path=input_file,
            #         judge_model_path=model_dir,
            #         output_dir=file_output_dir,
            #         use_token_logprobs=args.logprobs,
            #         batch_size=args.batch_size,
            #         n_gpus=gpus,
            #         gen_kwargs=args.gen_kwargs,
            #         max_tokens=args.max_tokens,
            #         thinking=args.thinking
            #     )
            # else:
            judge_responses_with_gt(
                data_path=input_file,
                judge_model_path=model_dir,
                output_dir=file_output_dir,
                use_token_logprobs=args.logprobs,
                batch_size=args.batch_size,
                n_gpus=gpus,
                gen_kwargs=args.gen_kwargs,
                max_tokens=args.max_tokens,
                thinking=args.thinking
            )

if __name__ == "__main__":
    main() 