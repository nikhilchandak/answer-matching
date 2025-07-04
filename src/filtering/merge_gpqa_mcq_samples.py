import os
import json
import argparse
from pathlib import Path


def find_sample_files(input_dir):
    """Find all sample.jsonl files recursively in the input directory."""
    sample_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            # if file == "samples.json":
            if file.endswith(".jsonl") and file.startswith("samples_") and "r1" not in file:
                print(f"Processing {file}")
                sample_files.append(os.path.join(root, file))
    return sample_files


def merge_samples(sample_files):
    """Merge all sample files into a single list of samples."""
    all_samples = []
    for file_path in sample_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():  # Skip empty lines
                        sample = json.loads(line)
                        new_sample = dict(sample)
                        
                        if not isinstance(new_sample.get("exact_match"), list):
                            new_sample["exact_match"] = [new_sample["exact_match"]]
                        else :
                            new_sample["exact_match"] = [new_sample["exact_match"][0]]
                        
                        if not isinstance(new_sample.get("resps"), list):
                            new_sample["resps"] = [new_sample["resps"]]
                        else :
                            new_sample["resps"] = [new_sample["resps"][0]]
                        
                        
                        if "thinking_finished" in new_sample:
                            if not isinstance(new_sample.get("thinking_finished"), list):
                                new_sample["thinking_finished"] = [new_sample["thinking_finished"]]
                            else :
                                new_sample["thinking_finished"] = [new_sample["thinking_finished"][0]]
                            
                        if not isinstance(new_sample.get("filtered_resps"), list):
                            new_sample["filtered_resps"] = [new_sample["filtered_resps"]]
                        else :
                            new_sample["filtered_resps"] = [new_sample["filtered_resps"][0]]
                        
                        all_samples.append(new_sample)
                        
        except json.JSONDecodeError:
            print(f"Error: Could not parse {file_path} as JSON. Skipping.")
        except Exception as e:
            print(f"Error reading {file_path}: {str(e)}. Skipping.")
    return all_samples


def main():
    parser = argparse.ArgumentParser(description="Merge all sample.jsonl files into one all_samples.jsonl file")
    parser.add_argument("--input_dir", help="Directory containing sample.jsonl files")
    args = parser.parse_args()
    
    input_dir = args.input_dir
    
    # Ensure input directory exists
    if not os.path.isdir(input_dir):
        print(f"Error: {input_dir} is not a valid directory")
        return
    
    # Find all sample.jsonl files
    sample_files = find_sample_files(input_dir)
    print(f"Found {len(sample_files)} samples.jsonl files")
    
    # Merge all samples
    all_samples = merge_samples(sample_files)
    print(f"Merged {len(all_samples)} samples")
    
    # Save merged samples as JSONL (one JSON object per line)
    output_path = os.path.join(input_dir, "all_samples.jsonl")
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in all_samples:
            f.write(json.dumps(sample) + '\n')
    
    print(f"Saved merged samples to {output_path}")


if __name__ == "__main__":
    main()
