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

def load_mmlupro_ids(file_path: str) -> List[int]:
    """
    Load question IDs from a file.
    
    Args:
        file_path: Path to the file containing question IDs
        
    Returns:
        List of question IDs
    """
    fixedpath = "/fast/nchandak/qaevals/judge_outputs/mmlu_pro_free/stratified_sample/combined_samples_to_annotate.jsonl"
    filtered_ids = {}
    
    data = load_jsonl_file(fixedpath)
    for item in data:
        qid = str(item["question_id"])
        model_name = item["model"] if "qwen" not in item["model"].lower() else "qwen3-32b"
        filtered_ids[qid] = model_name.split("/")[-1]
        
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
    filtered_ids = load_mmlupro_ids(filtered_ids_path)
    print(len(filtered_ids))
    # Find all model output files
    
    models = [
        "openai/gpt-4o",
        "deepseek/deepseek-chat-v3-0324",
        
        # "google/gemini-2.5-flash-preview",
        "meta-llama/llama-4-maverick",
        "qwen/qwen3-32b",
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
    acc = []
    total = []
    
    # Process each model file and extract samples in batches
    for j, model in enumerate(models):
        model_name = model.split("/")[-1]
        for root, _, files in os.walk(input_dir):
            if "wrong" in root or "wrong" in _ or "old" in root:
                continue 
            for file in files:
                if file.startswith("samples_") and file.endswith(f"_{model_name}.jsonl") and model_name in file:
                # if file.startswith("verified_") and file.endswith(f"_{model_name}.jsonl") and model_name in file:
                    
                    logger.info(f"Processing model: {model_name}")
                    
                    model_file = os.path.join(root, file)
                    # Load model data
                    model_data = load_jsonl_file(model_file)
                    
                    # Create a mapping of question_id to data
                    model_data_map = {item.get("question_id"): item for item in model_data if "question_id" in item}
                    filtered_ids = {item.get("question_id"): model_name for item in model_data if "question_id" in item}
                    
                    # Extract samples for this batch
                    batch_samples = []
                    acc_temp = 0
                    total_temp = 0
                    for qid in filtered_ids:
                        required_model = filtered_ids[qid]
                        # print(required_model)
                        if qid in model_data_map and required_model in model_file:
                            sample = model_data_map[qid]
                            # sample["model"] = model_name
                            batch_samples.append(sample)
                            total_temp += 1
                            if 'exact_match' in sample:
                                if isinstance(sample["exact_match"], list):
                                    acc_temp += sum(sample["exact_match"]) / float(len(sample["exact_match"]))
                                else:
                                    acc_temp += sample["exact_match"]
                            else :
                                if isinstance(sample["score_deepseek-chat-v3-0324"], list):
                                    acc_temp += sum(sample["score_deepseek-chat-v3-0324"]) / float(len(sample["score_deepseek-chat-v3-0324"]))
                                else:
                                    acc_temp += sample["score_deepseek-chat-v3-0324"]
                        else:
                            pass
                            # if qid not in model_data_map:
                            #     logger.warning(f"Question ID {qid} not found in model {model_name} data")
                            
                    logger.info(f"Added {len(batch_samples)} samples from model {model_name}")
                    
                    # Add batch samples to combined list    
                    combined_samples.extend(batch_samples)
                    acc.append(acc_temp)
                    total.append(total_temp)
                    
                    print(f"Model {model_name}: Correct: {acc_temp}, Total: {total_temp}, Accuracy: {acc_temp/total_temp*100:.2f}%")

    # Shuffle combined samples before saving
    import random
    logger.info(f"Shuffling {len(combined_samples)} samples")
    random.shuffle(combined_samples)
    
    logger.info(f"Correct: {acc}, Total: {total}, Accuracy: {sum(acc)/sum(total)*100:.2f}%")
    # Save combined samples
    # output_file = os.path.join(input_dir, "combined_samples_to_annotate.jsonl")
    output_file = os.path.join(input_dir, "combined_samples.jsonl")
    
    # output_file = os.path.join(input_dir, "combined_samples.jsonl")
    save_jsonl_file(combined_samples, output_file)
    logger.info(f"Combined {len(combined_samples)} samples for annotation")

def main():
    parser = argparse.ArgumentParser(description="Combine model outputs for annotation")
    parser.add_argument("--input_dir", default="/fast/nchandak/qaevals/judge_outputs/gpqa_diamond_mcq/",
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
