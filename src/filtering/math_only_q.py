import os
import json
import re
import argparse

from pathlib import Path


def process_samples(input_dir, output_dir, process):
    """
    Process all samples_{}.jsonl files in the input directory.
    Extract answers from responses and update exact_match field.
    Filter for Level 5 questions and save to output directory.
    
    Args:
        input_dir (str): The path to the input directory containing sample files.
        output_dir (str): The path to the output directory for saving filtered samples.
        process (str): The type of processing to apply ("mcq" or "free_response").
    """
    # Find all sample files
    sample_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.startswith("samples_") and file.endswith(".jsonl"):
                sample_files.append(os.path.join(root, file))
    
    print(f"Found {len(sample_files)} sample files")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each sample file
    for file_path in sample_files:
        print(f"Processing {file_path}")
        
        # Extract model name from file path
        model_name = file_path.split("/")[-3]
        
        timestamp = file_path.split("/")[-1].split("_")[-1].split(".")[0]
        dataset = "math_only_q"
        assert dataset in file_path, f"Dataset not found in {file_path}"
        
        print(f"Model: {model_name}, Dataset: {dataset}, Timestamp: {timestamp}")
        # Create model directory in output_dir
        model_dir = os.path.join(output_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        output_path = os.path.join(model_dir, "samples.jsonl")
        # if os.path.exists(output_path):
        #     print(f"Output file {output_path} already exists, skipping\n\n")
        #     continue
        
        # Track accuracy
        correct = 0
        total = 0
        
        # Read and process samples
        updated_samples = []
        level5_samples = []
        negative_flips = 0
        positive_flips = 0
        flipped_ids = []
        level_acc = {}
        
        with open(file_path, 'r') as f:
            for line in f:
                sample = json.loads(line)
                total += 1
                
                level = sample.get("doc", {}).get("Level")
                if level not in level_acc:
                    level_acc[level] = {"correct": 0, "total": 0}
                
                
                updated_samples.append(sample)
                correct += int(sample.get("acc_norm"))
                
                # If this is a Level 5 question, add it to level5_samples
                if level == "Level 5":
                    # Create sample in the required format
                    doc = sample.get("doc", {})
                    level5_sample = {
                        "question_id": doc.get("Question_ID"),
                        "question": doc.get("Question"),
                        "target": sample.get("target"),
                        "category": doc.get("Type"),
                        "completion_tokens": sample.get("completion_tokens"),
                        "exact_match": int(sample.get("acc_norm")),
                        "timestamp": timestamp,
                        "dataset": dataset,
                        "model": model_name,
                    }
                    if process == "mcq":
                        options = [doc.get("A"), doc.get("B"), doc.get("C"), doc.get("D")]
                        level5_sample["options"] = options
                        level5_sample["answer_index"] = ord(sample.get("Answer")) - ord("A")
                        level5_sample["answer"] = doc.get("Answer")
                        
                    level5_samples.append(level5_sample)
        
        # Calculate and print accuracy
        accuracy = correct / total * 100 if total > 0 else 0
        print(f"File: {os.path.basename(file_path)}, Accuracy: {accuracy:.2f}% ({correct}/{total})")
        print(f"Negative Flips: {negative_flips}, Positive Flips: {positive_flips}")
        print(f"Flipped IDs: {flipped_ids}")
        # for level, acc in level_acc.items():
        #     print(f"Level {level}: {acc['correct']}/{acc['total']} ({acc['correct']/acc['total']*100:.2f}%)")

        # Save Level 5 samples to output directory
        if level5_samples:
            output_path = os.path.join(model_dir, "samples.jsonl")
            print(f"Saving {len(level5_samples)} Level 5 samples to {output_path}")
            with open(output_path, 'w') as f:
                for sample in level5_samples:
                    f.write(json.dumps(sample) + '\n')
            
            print(f"Saved {len(level5_samples)} Level 5 samples to {output_path}")
        
        print("\n--------------------------------\n")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="/fast/nchandak/qaevals/outputs/math_free/", 
                        help="Path to the input directory containing sample files.")
    
    args = parser.parse_args()
    
    process = "mcq" if "mcq" in args.input_dir else "free_response"
    output_dir = args.input_dir.replace("outputs", "filtered_outputs")
    # make output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    process_samples(args.input_dir, output_dir, process)