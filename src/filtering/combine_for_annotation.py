import os
import json
import argparse
import logging
from typing import List, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_filtered_ids(filtered_ids_path: str) -> List[int]:
    """
    Load filtered question IDs from a file.
    
    Args:
        filtered_ids_path: Path to file containing IDs
        
    Returns:
        List of question IDs
    """
    logger.info(f"Loading filtered ids from {filtered_ids_path}")
    with open(filtered_ids_path, 'r') as f:
        filtered_ids = [int(line.strip()) for line in f.readlines()]
    logger.info(f"Loaded {len(filtered_ids)} filtered ids")
    return filtered_ids

def load_jsonl_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Load data from a JSONL file.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        List of dictionaries containing the data
    """
    data = []
    logger.info(f"Loading data from {file_path}")
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    logger.info(f"Loaded {len(data)} items from {file_path}")
    return data


def get_filtered_ids(samples_path: str = "/fast/nchandak/qaevals/judge_outputs/mmlu_pro_free/stratified_sample/annotations/combined_samples_to_annotate.jsonl") -> List[int]:
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
                    filtered_ids.append(sample["question_id"])
    
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

def combine_samples_for_annotation(
    input_dir: str,
    filtered_ids_path: str = "data/mmlu_pro/filtered_stratified_sample_1002.txt",
    batch_size: int = 200
) -> None:
    """
    Combine samples from different model outputs for annotation.
    
    Args:
        input_dir: Directory containing model output files
        filtered_ids_path: Path to file with filtered question IDs
        batch_size: Number of samples per model
    """
    # Load filtered IDs
    # filtered_ids = load_filtered_ids(filtered_ids_path)
    filtered_ids = get_filtered_ids()
    
    # Find all model output files
    
    models = [
        "openai/gpt-4o",
        "deepseek/deepseek-chat-v3-0324",
        
        # "google/gemini-2.5-flash-preview",
        "meta-llama/llama-4-maverick",
        # "qwen/qwen3-32b",
        "google/gemma-3-27b-it",
        # "qwen/qwen3-235b-a22b"
    ]
    
    # model_files = []
    # for root, _, files in os.walk(input_dir):
    #     for file in files:
    #         if file.startswith("samples_") and file.endswith(".jsonl"):
    #             model_files.append(os.path.join(root, file))
    
    # logger.info(f"Found {len(model_files)} model output files")
    
    # Initialize combined samples list
    combined_samples = []
    acc = 0
    total = 0
    
    # Process each model file and extract samples in batches
    for j, model in enumerate(models):
        model_name = model.split("/")[-1]
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.startswith("samples_") and file.endswith(f"_{model_name}.jsonl") and model_name in file:
                    logger.info(f"Processing model: {model_name}")
                    
                    model_file = os.path.join(root, file)
                    # Load model data
                    model_data = load_jsonl_file(model_file)
                    
                    # Create a mapping of question_id to data
                    model_data_map = {item.get("question_id"): item for item in model_data if "question_id" in item}
                    
                    # Get batch of filtered IDs for this model
                    start_idx = j * batch_size
                    end_idx = min((j + 1) * batch_size, len(filtered_ids))
                    if j == len(models) - 1:
                        end_idx = len(filtered_ids)
                    
                    if start_idx >= len(filtered_ids):
                        logger.info(f"No more filtered IDs left for model {model_name}")
                        break
                    
                    batch_ids = filtered_ids[start_idx:end_idx]
                    logger.info(f"Processing batch {j+1}: IDs {start_idx} to {end_idx-1}")
                    
                    # Extract samples for this batch
                    batch_samples = []
                    for qid in batch_ids:
                        if qid in model_data_map:
                            sample = model_data_map[qid]
                            # sample["model"] = model_name
                            batch_samples.append(sample)
                            total += 1
                            if 'exact_match' in sample:
                                acc += sample["exact_match"]
                            else :
                                acc += int(sample["score_deepseek-chat-v3-0324"])
                        else:
                            logger.warning(f"Question ID {qid} not found in model {model_name} data")
                    
                    logger.info(f"Added {len(batch_samples)} samples from model {model_name}")
                    
                    # Add batch samples to combined list
                    combined_samples.extend(batch_samples)
                
    # Shuffle combined samples before saving
    # import random
    # logger.info(f"Shuffling {len(combined_samples)} samples")
    # random.shuffle(combined_samples)
    
    logger.info(f"Correct: {acc}, Total: {total}, Accuracy: {acc/total*100:.2f}%")
    # Save combined samples
    output_file = os.path.join(input_dir, "mmlu_pro_combined_samples_to_annotate2.jsonl")
    save_jsonl_file(combined_samples, output_file)
    logger.info(f"Combined {len(combined_samples)} samples for annotation")

def main():
    parser = argparse.ArgumentParser(description="Combine model outputs for annotation")
    parser.add_argument("--input_dir", required=True, 
                        help="Directory containing model output files")
    parser.add_argument("--filtered_ids_path", 
                        default="data/mmlu_pro/filtered_stratified_sample_1002.txt",
                        help="Path to file with filtered question IDs")
    parser.add_argument("--batch_size", type=int, default=200,
                        help="Number of samples per model")
    
    args = parser.parse_args()
    
    combine_samples_for_annotation(
        input_dir=args.input_dir,
        filtered_ids_path=args.filtered_ids_path,
        batch_size=args.batch_size
    )

if __name__ == "__main__":
    main()
