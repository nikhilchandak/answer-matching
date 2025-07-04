import json
import argparse
from typing import Dict, Any
import os
import matplotlib.pyplot as plt
import numpy as np

def load_jsonl(file_path: str) -> Dict[str, Any]:
    """Load a JSONL file and return a dictionary with hash as key."""
    data = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():  # Skip empty lines
                obj = json.loads(line)
                # Each line has a single key-value pair where key is the hash
                hash_key = list(obj.keys())[0]
                data[hash_key] = obj[hash_key]
    return data

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Compare llm_judge_fine values between two JSONL files')
    parser.add_argument('--file1', 
                        default='/is/cluster/fast/nchandak/qaevals/filter/mmlu/MMLU_question_hash.jsonl',
                        help='Path to first JSONL file')
    parser.add_argument('--file2', 
                        default='/is/cluster/fast/nchandak/qaevals/filter/mmlu_O/MMLU_question_hash.jsonl',
                        help='Path to second JSONL file')
    parser.add_argument('--output', 
                        default='/is/cluster/fast/nchandak/qaevals/filter/mmlu_QvsO/FineJudgeDiffSorted.jsonl',
                        help='Path to output JSONL file')
    parser.add_argument('--histogram', 
                        default='/is/cluster/fast/nchandak/qaevals/filter/mmlu_QvsO/ScoreDiffHistogram.png',
                        help='Path to output histogram image')
    args = parser.parse_args()

    # Load data from both files
    data1 = load_jsonl(args.file1)
    data2 = load_jsonl(args.file2)

    # Find common hashes and calculate differences
    results = []
    score_diffs = []  # For histogram
    
    for hash_key in set(data1.keys()) & set(data2.keys()):
        if 'llm_judge_fine' in data1[hash_key] and 'llm_judge_fine' in data2[hash_key]:
            score1 = data1[hash_key]['llm_judge_fine']
            score2 = data2[hash_key]['llm_judge_fine']
            
            # Skip cases where either score is 1
            if score1 == 1 or score2 == 1:
                continue
                
            diff = abs(score1 - score2)
            actual_diff = score2 - score1  # For histogram
            score_diffs.append(actual_diff)
            
            # Create a result object with the hash, both scores, and the difference
            result = {
                "hash": hash_key,
                "question": data1[hash_key]["question"],
                "choices": data1[hash_key]["choices"],
                "answer_index": data1[hash_key]["answer_index"],
                "score1": score1,
                "score2": score2,
                "diff": diff
            }
            results.append(result)

    # Sort results by difference (largest first)
    results.sort(key=lambda x: x['diff'], reverse=True)

    # Write results to output file
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        for result in results:
            hash_key = result.pop('hash')
            # Format the output to match the input format
            output_obj = {hash_key: {
                "question": result["question"],
                "choices": result["choices"],
                "answer_index": result["answer_index"],
                "score1": result["score1"],
                "score2": result["score2"],
                "diff": result["diff"]
            }}
            f.write(json.dumps(output_obj) + '\n')
    
    # Create histogram of score_2 - score_1
    plt.figure(figsize=(10, 6))
    # Use explicit bin ranges from -8 to 8 (inclusive)
    bins = np.arange(-8.5, 8.6, 1)  # -8.5 to 8.5 with step 1 creates bins centered at integers from -8 to 8
    plt.hist(score_diffs, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel('Score 2 - Score 1')
    plt.ylabel('Frequency')
    plt.title('Distribution of Score Differences (Score 2 - Score 1)')
    plt.grid(True, alpha=0.3)
    
    # Add vertical line at x=0 for reference
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    
    # Add mean and median markers
    mean_diff = np.mean(score_diffs)
    median_diff = np.median(score_diffs)
    plt.axvline(x=mean_diff, color='green', linestyle='-', alpha=0.7, label=f'Mean: {mean_diff:.3f}')
    plt.axvline(x=median_diff, color='purple', linestyle='-.', alpha=0.7, label=f'Median: {median_diff:.3f}')
    
    plt.legend()
    
    # Save histogram
    os.makedirs(os.path.dirname(args.histogram), exist_ok=True)
    plt.savefig(args.histogram, dpi=300, bbox_inches='tight')
    print(f"Histogram saved to {args.histogram}")
    
    print(f"Processed {len(results)} items (excluding cases where either score is 1)")
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()