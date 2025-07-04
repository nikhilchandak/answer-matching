import argparse
import json
import os
from typing import Dict, Any, List
from load_datasets import load_dataset_by_name

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Filter GPQA question hash file to only include entries in the GPQA diamond dataset")
    parser.add_argument(
        "--input_file",
        type=str,
        default="/is/cluster/fast/nchandak/qaevals/filter/gpqa/GPQA_question_hash.jsonl",
        help="Path to the input JSONL file containing question hash data"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="/is/cluster/fast/nchandak/qaevals/filter/gpqa/GPQA_question_hash_filtered.jsonl",
        help="Path to save the filtered JSONL file"
    )
    
    args = parser.parse_args()
    
    # Load GPQA diamond dataset
    print("Loading GPQA diamond dataset...")
    gpqa_dataset = load_dataset_by_name(name="GPQA", subset="gpqa_diamond", split="train")
    
    # Extract q_hash values from the dataset
    gpqa_hashes = {item["q_hash"] for item in gpqa_dataset}
    print(f"Loaded {len(gpqa_dataset)} questions from GPQA diamond dataset")
    
    # Load the question hash file
    hash_data = []
    if os.path.exists(args.input_file):
        with open(args.input_file, "r") as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    hash_data.append(line.strip())
    print(f"Found {len(hash_data)} entries in the input file")
    
    # Filter hash data to only include entries in GPQA diamond
    filtered_data = []
    for line in hash_data:
        entry = json.loads(line)
        q_hash = list(entry.keys())[0]  # Each line has a single key-value pair
        if q_hash in gpqa_hashes:
            filtered_data.append(line)
    
    # Create the output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    # Write the filtered data to the output file
    with open(args.output_file, "w") as f:
        for line in filtered_data:
            f.write(line + "\n")
    
    print(f"Saved {len(filtered_data)} matching entries to {args.output_file}")

if __name__ == "__main__":
    main()