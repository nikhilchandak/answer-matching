from typing import List, Dict, Any, Optional
import os
import argparse
import datasets
from datasets import Dataset
from load_datasets import load_dataset_by_name


def convert_to_verification_format(dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert an MCQ dataset into a verification format where each question-option pair
    is a separate sample.
    
    Args:
        dataset: List of MCQ samples with question, choices, and answer_index
        
    Returns:
        List of verification samples
    """
    verification_samples = []
    
    for item in dataset:
        question = item["question"]
        choices = item["choices"]
        answer_index = item["answer_index"]
        q_hash = item["q_hash"]
        subject = item.get("subject", None)
        
        # Create a separate sample for each option
        for option_id, option_value in enumerate(choices):
            is_correct = option_id == answer_index
            
            verification_samples.append({
                "q_hash": q_hash,
                "question": question,
                "option_id": option_id,
                "option_value": option_value,
                "option_correct": "A" if is_correct else "B",  # A if correct, B if wrong
                "subject": subject
            })
    
    return verification_samples


def save_as_huggingface_dataset(verification_data: List[Dict[str, Any]], 
                               save_path: str,
                               dataset_name: str,
                               split: str,
                               subset: Optional[str] = None) -> None:
    """
    Save the verification data as a Hugging Face dataset.
    
    Args:
        verification_data: List of verification samples
        save_path: Path to save the dataset
        dataset_name: Name of the original dataset
        split: Dataset split
        subset: Dataset subset if applicable
    """
    # Create HF dataset
    hf_dataset = Dataset.from_list(verification_data)
    
    # Create directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Determine dataset folder name
    folder_name = dataset_name
    if subset:
        folder_name = f"{dataset_name}_{subset}"
        
    # Save the dataset
    dataset_path = os.path.join(save_path, folder_name)
    hf_dataset.save_to_disk(dataset_path)
    
    print(f"Dataset saved to {dataset_path}")
    print(f"To push to HuggingFace Hub: datasets.load_from_disk('{dataset_path}').push_to_hub('your-username/dataset-name')")


def main():
    parser = argparse.ArgumentParser(description="Convert MCQ datasets to verification format")
    parser.add_argument("--dataset", type=str, required=True, choices=["MMLU", "GPQA", "MMLU-Pro", "MATH"], 
                        help="Dataset name")
    parser.add_argument("--split", type=str, default="test", 
                        help="Dataset split (train, validation, test)")
    parser.add_argument("--subset", type=str, default=None, 
                        help="Dataset subset if applicable")
    parser.add_argument("--save_path", type=str, 
                        default=None, 
                        help="Path to save the verification dataset")
    
    args = parser.parse_args()
    
    # Set default save path if not provided
    if args.save_path is None:
        args.save_path = f"/is/cluster/fast/nchandak/qaevals/verify/{args.dataset.lower()}"
    
    # Load the dataset
    print(f"Loading {args.dataset} dataset...")
    mcq_dataset = load_dataset_by_name(args.dataset, args.split, args.subset)
    
    # Convert to verification format
    print("Converting to verification format...")
    verification_dataset = convert_to_verification_format(mcq_dataset)
    
    # Save the dataset
    print("Saving dataset...")
    save_as_huggingface_dataset(
        verification_dataset, 
        args.save_path, 
        args.dataset, 
        args.split, 
        args.subset
    )
    
    print(f"Converted {len(mcq_dataset)} MCQ questions into {len(verification_dataset)} verification questions.")


if __name__ == "__main__":
    main()
