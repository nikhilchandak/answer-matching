import os
import json
import argparse
from pathlib import Path


def find_sample_files(input_dir):
    """Find all sample.json files recursively in the input directory."""
    sample_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file == "samples.json":
                sample_files.append(os.path.join(root, file))
    return sample_files


def merge_samples(sample_files):
    """Merge all sample files into a single list of samples."""
    all_samples = []
    for file_path in sample_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                samples = json.load(f)
                if isinstance(samples, list):
                    all_samples.extend(samples)
                else:
                    print(f"Warning: {file_path} does not contain a JSON array. Skipping.")
        except json.JSONDecodeError:
            print(f"Error: Could not parse {file_path} as JSON. Skipping.")
        except Exception as e:
            print(f"Error reading {file_path}: {str(e)}. Skipping.")
    return all_samples


def main():
    parser = argparse.ArgumentParser(description="Merge all sample.json files into one all_samples.jsonl file")
    parser.add_argument("--input_dir", help="Directory containing sample.json files")
    args = parser.parse_args()
    
    input_dir = args.input_dir
    
    # Ensure input directory exists
    if not os.path.isdir(input_dir):
        print(f"Error: {input_dir} is not a valid directory")
        return
    
    # Find all sample.json files
    sample_files = find_sample_files(input_dir)
    print(f"Found {len(sample_files)} samples.json files")
    
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
