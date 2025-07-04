#!/usr/bin/env python
# coding: utf-8

import os
import sys
import json
import asyncio
import argparse
import logging
import re
from typing import List, Dict, Any
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path to find the inference module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the OpenRouter inference engine
from inference.openrouter import *

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define the prompt template for querying models
QUERY_PROMPT_TEMPLATE = """
You will be asked a multiple-choice question in {category}. Please provide your reasoning before stating your final answer.

Question: {question}

Options:
{options}

Think step by step and put your final answer in <answer> </answer> tags.
Your final answer should be a SINGLE LETTER in uppercase (A, B, C, D, etc.) corresponding to the correct option.
Your response SHOULD STRICTLY END with your answer choice in <answer> </answer> tags.
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
            data = []
            with open(data_path, 'r') as f:
                for line in f:
                    data.append(json.loads(line))
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
            
def extract_answer(response: str) -> str:
    """
    Extract answer from model response
    
    Args:
        response: Model response string
        
    Returns:
        Extracted answer or empty string if not found
    """
    if not response:
        return ""
    
    # Try to extract from <answer> tags - use findall to get all matches and take the last one
    answer_matches = re.findall(r'<answer>\s*(.*?)\s*</answer>', response, re.DOTALL)
    if answer_matches:
        # Extract just the letter from the last match
        answer = answer_matches[-1].strip()
        # If the answer contains a single letter, return it
        if re.match(r'^[A-D]$', answer):
            return answer
        # Try to extract just the letter if it's in a format like "A." or "Option A"
        letter_match = re.search(r'([A-D])[\.:\)]|[Oo]ption\s+([A-D])', answer)
        if letter_match:
            return letter_match.group(1) if letter_match.group(1) else letter_match.group(2)
    
    # Look for patterns like "The answer is A" or "I choose B"
    answer_phrases = [
        r'the answer is\s+([A-D])',
        r'I choose\s+([A-D])',
        r'answer:\s+([A-D])',
        r'final answer:?\s+([A-D])',
        r'option\s+([A-D])'
    ]
    
    for phrase in answer_phrases:
        matches = re.findall(phrase, response, re.IGNORECASE)
        if matches:
            return matches[-1].upper()  # Return the last match, ensuring it's uppercase
    
    # Last resort: look for any standalone capital letter that might be an answer
    standalone_letters = re.findall(r'(?:^|\s)([A-D])(?:$|\s|\.|\))', response)
    if standalone_letters:
        return standalone_letters[-1]
    
    return ""

async def query_model(
    data_path: str,
    output_dir: str,
    batch_size: int = 5,
    model_name: str = "google/gemini-pro",
    max_tokens: int = 2048,
    temperature: float = 0.6,
    top_p: float = 1.0,
    filtered_ids_path: str = None
):
    """
    Query model with MMLU Pro questions and save responses
    
    Args:
        data_path: Path to the JSON file containing questions
        batch_size: Number of samples to process in a batch
        model_name: Name of the OpenRouter model to use
        max_tokens: Maximum number of tokens for generation
        temperature: Temperature for generation
        top_p: Top-p value for generation
        filtered_ids_path: Path to file containing IDs to process (optional)
    """
    # Load the reference data for questions
    logger.info(f"Loading reference data from {data_path}")
    reference_data = load_existing_results(data_path)
    
    logger.info(f"Loaded reference data with {len(reference_data)} samples")
    
    # Create model-specific output file path
    model_name_simplified = model_name.split('/')[-1]
    model_specific_file = os.path.join(output_dir, f"samples_{model_name_simplified}.jsonl")
    
    # Load existing model-specific results or initialize empty list
    existing_data = load_existing_results(model_specific_file) if os.path.exists(model_specific_file) else []
    
    # Get all question IDs from reference data
    question_ids = [item.get("question_id") for item in reference_data]
    
    
    # Create a mapping of question_id to existing model-specific data
    # Only keep items where both resps and filtered_resps are not empty
    existing_items_map = {item.get("question_id"): item for item in existing_data 
                         if "question_id" in item 
                         and "resps" in item and item.get("resps") != ""
                         and "filtered_resps" in item and item.get("filtered_resps") != ""}
    
    # Create a mapping of question_id to reference data for questions and categories
    reference_items_map = {item.get("question_id"): item for item in reference_data if "question_id" in item }
    
    accuracy = 0
    total_processed = 0
    # Identify which samples need to be processed
    samples_to_query = []
    
    for question_id, ref_item in reference_items_map.items():
        # If this question hasn't been processed yet or doesn't have a response
        if question_id not in existing_items_map or "resps" not in existing_items_map[question_id] or not existing_items_map[question_id].get("resps"):
            samples_to_query.append((question_id, ref_item))
    
    if not samples_to_query:
        acc = sum(item.get("exact_match") for item in existing_data)
        total = len(existing_data)
        logger.info(f"All samples have already been processed by {model_name}, nothing to do")
        logger.info(f"Accuracy for {model_name}: {acc}/{total} ({acc/total*100:.2f}%)")
        return existing_data
    
    logger.info(f"Need to query {len(samples_to_query)} out of {len(reference_items_map)} samples with {model_name}")
    
    # Initialize the OpenRouter inference engine
    logger.info(f"Initializing {model_name} via OpenRouter")
    
    # Use the standard OpenRouterInference class and pass top_p in the data
    inference_model = OpenRouterInference(
        model=model_name,
        max_tokens=max_tokens,
        temperature=temperature
    )
    
    # Process samples in batches
    for batch_start in range(0, len(samples_to_query), batch_size):
        batch_end = min(batch_start + batch_size, len(samples_to_query))
        batch = samples_to_query[batch_start:batch_end]
        
        logger.info(f"Processing batch {batch_start//batch_size + 1}, samples {batch_start+1} to {batch_end}")
        
        # Prepare prompts for the model
        prompts = []
        batch_question_ids = []
        
        for question_id, ref_item in batch:
            question = ref_item.get("question", "")
            category = ref_item.get("category", "")
            options = ref_item.get("options", [])
            
            # Format options as a string
            options_str = ""
            for i, option in enumerate(options):
                letter = chr(65 + i)  # A, B, C, ...
                options_str += f"{letter}. {option}\n"
            
            # Create the prompt for the model
            prompt = QUERY_PROMPT_TEMPLATE.format(
                question=question,
                category=category,
                options=options_str
            )
            
            prompts.append(prompt)
            batch_question_ids.append(question_id)
        
        # Generate responses
        responses = await inference_model.generate(prompts, batch_size=batch_size)
        
        # Process the responses and update the results
        for question_id, ret in zip(batch_question_ids, responses):
            # Extract answer from response
            # Extract relevant fields from the response dictionary
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
                
            filtered_resp = extract_answer(response)
            
            # If this question_id doesn't exist in our model-specific data yet, create it
            if question_id not in existing_items_map:
                # Copy basic info from reference data
                ref_item = reference_items_map[question_id]
                answer = chr(65 + ref_item.get("answer_index")) 
                
                # Check if the extracted answer matches the correct answer
                exact_match = 1 if filtered_resp == answer else 0
                accuracy += exact_match
                total_processed += 1
                
                new_item = {
                    "question_id": question_id,
                    "question": ref_item.get("question", ""),
                    "target": ref_item.get("target", ""),
                    "options": ref_item.get("options", []),
                    "answer": answer,
                    "answer_index": ref_item.get("answer_index", -1),
                    "dataset": ref_item.get("dataset", ""),
                    "model": model_name,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "exact_match": exact_match
                }
                existing_items_map[question_id] = new_item
            else:
                # Update exact_match for existing items
                answer = reference_items_map[question_id].get("answer", "")
                existing_items_map[question_id]["exact_match"] = 1 if filtered_resp == answer else 0
            
            # Add response to the sample
            existing_items_map[question_id]["resps"] = response
            existing_items_map[question_id]["filtered_resps"] = filtered_resp
        
        # Save results after each batch
        save_results(list(existing_items_map.values()), model_specific_file)
        
        # Be nice to the API
        await asyncio.sleep(1)
    
    logger.info(f"Completed querying {model_name}")
    
    # Print completion statistics
    total_processed = sum(1 for item in existing_items_map.values() if "resps" in item and item.get("resps"))
    total_filtered = len(reference_items_map)
    
    logger.info(f"Accuracy for {model_name}: {accuracy}/{total_processed} ({accuracy/total_processed*100:.2f}%)")
    logger.info(f"Processed {total_processed}/{total_filtered} samples with {model_name}")
    
    return list(existing_items_map.values())

def find_samples_json_files(directory: str) -> List[str]:
    """
    Find all samples.jsonl files in the given directory or its subdirectories.
    
    Args:
        directory: Directory to search in
        
    Returns:
        List of paths to samples.jsonl files
    """
    logger.info(f"Searching for samples.jsonl files in {directory}")
    
    paths = []
    for root, dirs, files in os.walk(directory):
        if "samples.jsonl" in files:
            file_path = os.path.join(root, "samples.jsonl")
            paths.append(file_path)
    
    logger.info(f"Found {len(paths)} samples.jsonl files")
    return paths
def main():
    parser = argparse.ArgumentParser(description="Query models with MMLU Pro questions")
    parser.add_argument("--input_dir", default="/fast/nchandak/qaevals/judge_outputs/gpqa_diamond_mcq/", 
                      help="Directory to search for samples.jsonl files")
    parser.add_argument("--batch_size", type=int, default=100,
                      help="Number of samples to process in a batch")
    parser.add_argument("--max_tokens", type=int, default=65536,
                      help="Maximum number of tokens for generation")
    parser.add_argument("--temperature", type=float, default=0.3,
                      help="Temperature for generation")
    parser.add_argument("--top_p", type=float, default=1.0,
                      help="Top-p value for generation")
    parser.add_argument("--filtered_ids_path", type=str, default="/home/nchandak/qaevals/how-to-qa/src/filtering/data/mmlu_pro/filtered_stratified_sample_1002.txt",
                      help="Path to file containing IDs to process (optional)")
    args = parser.parse_args()
    
    # List of models to process
    
    
    models = [
        "openai/gpt-4o",
        "deepseek/deepseek-chat-v3-0324",
        
        "openai/o4-mini-high",
        # "google/gemini-2.5-pro-preview",
        "deepseek/deepseek-r1-0528",
        
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
    
    # Find all samples.jsonl files in the input directory
    file_paths = find_samples_json_files(args.input_dir)
    
    if not file_paths:
        logger.error(f"No samples.jsonl files found in {args.input_dir}")
        return
    
    # Process each file with each model
    file_path = file_paths[0]
    logger.info(f"Processing file: {file_path}")
    
    for model in models:
        temp = args.temperature
        
        if "r1" in model or "qwen3" in model or "o4" in model or "o3" in model or "2.5" in model:
            temp = 0.6
        
        # Run the async function
        asyncio.run(query_model(
            data_path=file_path,
            output_dir=args.input_dir,
            batch_size=args.batch_size,
            model_name=model,
            max_tokens=args.max_tokens,
            temperature=temp,
            top_p=args.top_p,
            filtered_ids_path=args.filtered_ids_path
        ))

if __name__ == "__main__":
    main() 