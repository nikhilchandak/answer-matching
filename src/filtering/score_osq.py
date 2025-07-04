# load dataset from huggingface

# load question, choices, ground-truth answer index in choices list

# pass to openrouter model specified in argument with prompt to check if question can be answered without choices

# parse the score from the response and store it in a dictionary with question hash as key, and another dictionary with question + choices (sorted) hash as key

# dump the dictionaries to disk

import json
import os
import argparse
import asyncio
import logging
from typing import List, Dict, Any

from tqdm import tqdm

from inference.openrouter import OpenRouterInference
from load_datasets import load_dataset_by_name
from question_hash import hash_question, hash_question_with_choices

# Define the prompts for the model
coarse_filter_prompt = """Your task is to review a series of multiple-choice questions and evaluate their ability to
be answered without the provided answer choices. For questions that begin with an incomplete
sentence (e.g., "During swallowing, ..."), use your knowledge to attempt to complete the
sentence accurately. For direct questions that ask for specific information or identification
(e.g., "Which of the following structures is part of the small intestine?"), assess whether the
question is formulated clearly enough that an informed answer can be given without seeing the
multiple-choice options. For mathematical or analytical questions (e.g., "Find all cosets of
the subgroup 4Z of 2Z"), determine if the question provides enough context and information for a
solution to be formulated without additional options.

Please follow this format for your evaluation:

QUESTION: [Insert the question here] 

VERDICT: Respond with "YES" if the question is clear and can be directly answered based on its 
content alone, or "NO" if it relies on the answer choices to be understood or answered. Your 
response should include only the verdict without any justification or reasoning."""

fine_filter_prompt = """You will assign a numerical score from 1 to 10 based on how confidently it can be answered
without the choices. The scoring criteria are as follows:

1: The question is entirely dependent on its choices for an answer, making it impossible to
answer without them. Example: 'Which of the following statements is correct?'

10: The question can be easily and confidently answered based solely on the question stem,
without any need to refer to the provided options. Example: 'What is the first law of
thermodynamics in physics?' 

Intermediate Scores:

2-4: The question stem gives very little information and is highly reliant on the choices for
context. Example: 'Which of these is a prime number?' 'The ________ perspective on sustainability 
resulted from growth models that analysed the carrying capacity of the planet, overall concluding 
that the finite capacity of the earth and_______, ________ and _______ by current and past 
generations could reduce quality of life for future generations.'

5: The question provides some context or information, that gives a moderate possibility to 
answer the question. Example: 'Which of the following best describes the structure that collects 
urine in the body?' Example: 

6: The question provides a good amount of context or information, that gives a moderate
possibility to answer the question. Example: 'Statement 1 | A factor group of a non-Abelian
group is non-Abelian. Statement 2 | If K is a normal subgroup of H and H is a normal subgroup of
G, then K is a normal subgroup of G.'

7: The question provides a good amount of context or information, that gives a high possibility
to answer the question. Example: 'The element (4, 2) of Z_12 x Z_8 has order'

8-9: The question provides a good amount of context or information, that gives a high
possibility to answer the question. Example: 'A "dished face" profile is often associated with'

ONLY GIVE THE VALUE BETWEEN 1-10 AS YOUR ANSWER. DO NOT INCLUDE ANY OTHER INFORMATION IN YOUR
RESPONSE
"""

filterosq_prompt = """
You will be given a multiple-choice question, including the question, answer choices, and the answer choice which is correct (the rest are incorrect). Note that other correct answers might also be possible.

You have to judge whether the question can be answered without the answer choices.

You will assign a numerical score from 1 to 10 based on how confident you are in your judgement.

The scoring criteria are as follows for whether the question can be answered without the answer choices:

1: The question is entirely dependent on its choices for an answer, making it impossible to
answer without them. Example: 'Which of the following statements is correct?'

10: One (not necessarily all) correct answer to the question can be provided without referring to the answer choices. Example: 'What is the first law of
thermodynamics in physics?' or 'Who is one of the founders of the Democratic Party of the United States?'

Intermediate Scores:

2-4: The question stem gives very little information and is highly reliant on the choices for
context. Example: 'Which of these is a prime number?' 'The ________ perspective on sustainability 
resulted from growth models that analysed the carrying capacity of the planet, overall concluding 
that the finite capacity of the earth and_______, ________ and _______ by current and past 
generations could reduce quality of life for future generations.'

5: The question provides some context or information, that gives a moderate possibility to 
answer the question. Example: 'Which of the following best describes the structure that collects 
urine in the body?' Example: 

6: The question provides a good amount of context or information, that gives a moderate
possibility to answer the question. Example: 'Statement 1 | A factor group of a non-Abelian
group is non-Abelian. Statement 2 | If K is a normal subgroup of H and H is a normal subgroup of
G, then K is a normal subgroup of G.'

7: The question provides a good amount of context or information, that gives a high possibility
to answer the question. Example: 'The element (4, 2) of Z_12 x Z_8 has order'

8-9: The question provides a good amount of context or information, that gives a high
possibility to answer the question. Example: 'A "dished face" profile is often associated with'

The multple-choice question is as follows:

Question: {question}
Options: {options}
Correct Answer: {answer}

Can one correct answer to the question be provided without referring to the answer choices? 
ONLY GIVE THE VALUE BETWEEN 1-10 AS YOUR ANSWER. DO NOT INCLUDE ANY OTHER INFORMATION IN YOUR RESPONSE.
"""

multiple_answer_prompt = """

You will be given the question from a multiple-choice question, without the options or final answer.

Your task is to judge whether the question has a single unique answer to the question (ignoring paraphrasing)when the answer choices are NOT provided and the question is asked in an open-ended format. You might not know the answers, but can still judge whether multiple answers are possible.

The scoring criteria are as follows for whether the question (without looking at the answer choices) has multiple semantically distinct answers:

1: The question clearly has multiple correct answers. Example: 'Who is one of the founders of the Democratic Party of the United States?', 'Provide the adjancecy list of a graph which is a tree' etc.

10: You are sure the question has a single correct answer. Example: 'What is the capital of France?', This also includes questions like 'Who is the President of the United States?' where the answer is fixed at a given point of time when the question is asked.

**Intermediate Scores**

2-4: The question probably has multiple correct answers, but you are not sure about this. It could be it has only one correct answer. 

5: You are highly uncertain whether the question has multiple semantically distinct answers or only one. Both are equally likely.

6-9: The question probably has only one correct answer, but you are not sure. 

Does the question have a unique answer or are multiple semantically or functionally distinct answers possible to the question? Higher values indicate more confidence in a single answer.
ONLY GIVE THE VALUE BETWEEN 1-10 AS YOUR ANSWER. DO NOT INCLUDE ANY OTHER INFORMATION IN YOUR RESPONSE.
"""


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def load_existing_results(q_hash_path):
    """
    Load existing results from JSONL output file if it exists
    
    Returns:
        Dictionary with question hash keys
    """
    question_hash_dict = {}
    lines_to_keep = []
    has_zero_scores = False
    
    # Load question hash dictionary if it exists
    if os.path.exists(q_hash_path):
        logger.info(f"Loading existing results from {q_hash_path}")
        with open(q_hash_path, "r") as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    entry = json.loads(line)
                    q_hash = list(entry.keys())[0]  # Each line has a single key-value pair
                    result = entry[q_hash]
                    
                    # Check if any score is 0
                    has_zero_score = False
                    if "llm_judge_fine" in result and result["llm_judge_fine"] == 0:
                        has_zero_score = True
                    if "llm_judge_unique" in result and result["llm_judge_unique"] == 0:
                        has_zero_score = True
                    if "llm_judge_coarse" in result and result["llm_judge_coarse"] == "0":
                        has_zero_score = True
                    
                    if not has_zero_score:
                        # Keep this line in the file
                        lines_to_keep.append(line)
                        # Update our dictionary with this valid result
                        question_hash_dict[q_hash] = result
                    else:
                        # Found a zero score, mark for rewriting
                        has_zero_scores = True
        
        # If there are any zero scores, rewrite the file with only the lines to keep
        if has_zero_scores:
            logger.info(f"Found entries with 0 scores to remove, rewriting file")
            rewrite_file_keeping_valid_entries(q_hash_path, lines_to_keep)
            
    logger.info(f"Loaded {len(question_hash_dict)} valid existing results")
    return question_hash_dict


def rewrite_file_keeping_valid_entries(q_hash_path, lines_to_keep):
    """
    Rewrite the JSONL file keeping only the valid entries
    
    Args:
        q_hash_path: Path to the JSONL file
        lines_to_keep: List of lines to keep in the file
    """
    if not os.path.exists(q_hash_path):
        return
    
    # Create a temporary file
    temp_path = q_hash_path + ".temp"
    
    # Write only the valid lines to the temp file
    with open(temp_path, "w") as temp:
        for line in lines_to_keep:
            temp.write(line)
    
    # Replace the original file with the temp file
    os.replace(temp_path, q_hash_path)
    logger.info(f"Rewrote file {q_hash_path} keeping {len(lines_to_keep)} valid entries")


async def process_coarse_evaluation(inference, questions, batch_size):
    """Process questions with coarse evaluation"""
    logger.info(f"Generating coarse evaluations for {len(questions)} questions")
    coarse_prompts = [
        coarse_filter_prompt + f"\n\nQuestion: {q} \n\nVerdict:" for q in questions
    ]
    return await inference.generate(coarse_prompts, batch_size=batch_size)


async def process_fine_evaluation(inference, questions, batch_size, prompt_type="filter"):
    """
    Process questions with fine evaluation
    
    Args:
        inference: OpenRouterInference instance
        questions: List of questions to evaluate
        batch_size: Number of questions to process in parallel
        prompt_type: Type of prompt to use ('filter' or 'unique')
    """
    logger.info(f"Generating fine evaluations for {len(questions)} questions using prompt type: {prompt_type}")
    
    if prompt_type == "filter":
        fine_prompts = [
            fine_filter_prompt + f"\n\nQuestion: {q} \n\nScore:" for q in questions
        ]
    elif prompt_type == "unique":
        fine_prompts = [
            multiple_answer_prompt + f"\n\nQuestion: {q} \n\nScore:" for q in questions
        ]
    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}")
        
    return await inference.generate(fine_prompts, batch_size=batch_size)


def append_new_results(new_results, q_hash_path):
    """Append new results to the JSONL output file without rewriting everything"""
    if not new_results:
        logger.info("No new results to append")
        return
    
    # Create parent directory if it doesn't exist
    os.makedirs(os.path.dirname(q_hash_path), exist_ok=True)
    
    # Append to the question hash file
    with open(q_hash_path, "a") as f:
        for q_hash, result in new_results.items():
            f.write(json.dumps({q_hash: result}) + "\n")
    
    logger.info(f"Appended {len(new_results)} new results to output file")


async def process_dataset(dataset_items, model_name, q_hash_path, batch_size=5, max_tokens=100, coarse_only=False, fine_only=False, prompt_type="filter"):
    """
    Process the dataset by evaluating questions using the OpenRouter model
    
    Args:
        dataset_items: List of dataset items containing questions and choices
        model_name: Name of the OpenRouter model to use
        q_hash_path: Path to save question hash results
        batch_size: Number of questions to process in parallel
        max_tokens: Maximum tokens for model response
        coarse_only: If True, only use the coarse evaluation prompt
        fine_only: If True, only use the fine evaluation prompt
        prompt_type: Type of prompt to use for fine evaluation ('filter' or 'unique')
        
    Returns:
        Dictionary with question hash keys
    """
    # Load existing results if any
    question_hash_dict = await load_existing_results(q_hash_path)
    
    # Filter out already processed questions
    new_items = []
    for item in dataset_items:
        q_hash = item["q_hash"]
        if q_hash not in question_hash_dict:
            new_items.append(item)
    
    logger.info(f"Processing {len(new_items)} new questions out of {len(dataset_items)} total")
    
    if not new_items:
        return question_hash_dict
    
    # Initialize the OpenRouter inference engine
    inference = OpenRouterInference(model=model_name, max_tokens=max_tokens, temperature=0.0)
    
    # Process in batches
    for i in range(0, len(new_items), batch_size):
        batch_items = new_items[i:i+batch_size]
        logger.info(f"Processing batch {i//batch_size + 1} with {len(batch_items)} questions")
        
        # Extract questions for this batch
        batch_questions = [item["question"] for item in batch_items]
        
        # Process coarse evaluations if not in fine_only mode
        coarse_results = None
        if not fine_only:
            coarse_results = await process_coarse_evaluation(inference, batch_questions, batch_size)
        
        # Process fine evaluations if not in coarse_only mode
        fine_results = None
        if not coarse_only:
            fine_results = await process_fine_evaluation(inference, batch_questions, batch_size, prompt_type)
        
        # Process results and update dictionaries
        new_results = {}
        
        for j, item in enumerate(batch_items):
            # Create result object
            result = {
                "question": item["question"],
                "choices": item["choices"],
                "answer_index": item["answer_index"],
            }
            
            # Add coarse decision if available
            if not fine_only and coarse_results and coarse_results[j] is not None:
                coarse_output = coarse_results[j]
                # Extract the actual response text from the dictionary
                if isinstance(coarse_output, dict) and 'response' in coarse_output:
                    coarse_output = coarse_output['response']
                    
                coarse_decision = coarse_output.strip()
                if coarse_decision not in ["YES", "NO"]:
                    coarse_decision = "NO"
                result["llm_judge_coarse"] = coarse_decision
                
            # Add fine decision if available
            if not coarse_only and fine_results and fine_results[j] is not None:
                fine_output = fine_results[j]
                try:
                    # Extract the actual response text from the dictionary
                    if isinstance(fine_output, dict) and 'response' in fine_output:
                        fine_output = fine_output['response']
                        
                    fine_decision = int(fine_output.strip())
                except (ValueError, AttributeError) as e:
                    logger.warning(f"Failed to parse score from '{fine_output}', defaulting to 0. Error: {e}")
                    fine_decision = 0
                if fine_decision < 1 or fine_decision > 10:
                    logger.warning(f"Score from '{fine_output}' is out of range, defaulting to 0")
                    fine_decision = 0
                
                # Store the result with a key based on the prompt type
                if prompt_type == "filter":
                    result["llm_judge_fine"] = fine_decision
                elif prompt_type == "unique":
                    result["llm_judge_unique"] = fine_decision
            
            if "llm_judge_fine" in result or "llm_judge_coarse" in result or "llm_judge_unique" in result:
                # Use pre-computed hash values
                q_hash = item["q_hash"]
                
                # Store in dictionary
                question_hash_dict[q_hash] = result
                
                # Also store in new results dictionary for appending
                new_results[q_hash] = result
        
        # Append new results after each batch
        append_new_results(new_results, q_hash_path)
    
    return question_hash_dict


async def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Score open-ended question ability of multiple-choice questions"
    )
    
    parser.add_argument(
        "--dataset_name", 
        type=str, 
        required=True,
        choices=["MMLU", "GPQA", "MMLU-Pro"],
        help="Name of the dataset to process"
    )
    parser.add_argument(
        "--model", 
        type=str, 
        default="deepseek/deepseek-chat-v3-0324",
        help="OpenRouter model name to use"
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        required=True,
        help="Path to save the results"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=5,
        help="Number of questions to process in parallel"
    )
    parser.add_argument(
        "--max_tokens", 
        type=int, 
        default=100,
        help="Maximum tokens for model response"
    )
    parser.add_argument(
        "--split", 
        type=str, 
        default="test",
        help="Dataset split to use"
    )
    parser.add_argument(
        "--subset", 
        type=str, 
        default=None,
        help="Dataset subset if applicable"
    )
    parser.add_argument(
        "--debug_limit", 
        type=int, 
        default=None,
        help="Limit the number of questions to process (for debugging)"
    )
    parser.add_argument(
        "--coarse_only", 
        action="store_true",
        help="Only use the coarse evaluation prompt (YES/NO) without the fine-grained scoring"
    )
    parser.add_argument(
        "--fine_only", 
        action="store_true",
        help="Only use the fine evaluation prompt (1-10 scoring) without the coarse evaluation"
    )
    parser.add_argument(
        "--prompt_type", 
        type=str, 
        default="filter",
        choices=["filter", "unique"],
        help="Type of prompt to use for fine evaluation ('filter' or 'unique')"
    )
    
    args = parser.parse_args()
    
    # Check for conflicting arguments
    if args.coarse_only and args.fine_only:
        parser.error("--coarse_only and --fine_only cannot be used together")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_path, exist_ok=True)
    
    # Define output path
    q_hash_path = os.path.join(args.output_path, f"{args.dataset_name}_question_hash.jsonl")
    
    # Load dataset
    logger.info(f"Loading dataset: {args.dataset_name}")
    dataset_items = load_dataset_by_name(
        name=args.dataset_name,
        split=args.split,
        subset=args.subset
    )
    
    # Apply debug limit if specified
    if args.debug_limit is not None:
        logger.info(f"Debug mode: limiting to {args.debug_limit} questions")
        dataset_items = dataset_items[:args.debug_limit]
    
    # Process dataset with checkpointing
    logger.info(f"Processing dataset with model {args.model}")
    question_hash_dict = await process_dataset(
        dataset_items=dataset_items,
        model_name=args.model,
        q_hash_path=q_hash_path,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        coarse_only=args.coarse_only,
        fine_only=args.fine_only,
        prompt_type=args.prompt_type
    )
    
    logger.info(f"Completed processing {len(question_hash_dict)} questions")
    logger.info("Done!")

if __name__ == "__main__":
    asyncio.run(main())