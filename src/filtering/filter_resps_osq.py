#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Filter responses based on OSQ ratings")
    parser.add_argument(
        "--responses_path", 
        type=str, 
        default="/is/cluster/fast/nchandak/qaevals/filtered_outputs/mmlu_pro_free/all_samples.jsonl",
        help="Path to the responses JSONL file"
    )
    parser.add_argument(
        "--osq_ratings_path", 
        type=str, 
        default="/is/cluster/fast/nchandak/qaevals/filter/mmlupro/MMLU-Pro_question_hash.jsonl",
        help="Path to the OSQ ratings JSONL file"
    )
    parser.add_argument(
        "--threshold", 
        type=float, 
        default=7,
        help="Threshold value between 1 and 10 for filtering"
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        default=None,
        help="Path to save the filtered responses"
    )
    
    args = parser.parse_args()
    
    # Validate threshold
    if not 1 <= args.threshold <= 10:
        raise ValueError("Threshold must be between 1 and 10")
    
    # Set default output path if not provided
    if args.output_path is None:
        args.output_path = f"/is/cluster/fast/nchandak/qaevals/filter/mmlupro/osq_resps_{args.threshold}.jsonl"
    
    return args

def main():
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Load OSQ ratings and filter question IDs
    filtered_question_ids = set()
    with open(args.osq_ratings_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            # Each line has question ID as the key
            for question_id, question_data in data.items():
                if 'llm_judge_fine' in question_data and question_data['llm_judge_fine'] >= args.threshold:
                    filtered_question_ids.add(question_id)
    
    print(f"Found {len(filtered_question_ids)} questions with ratings >= {args.threshold}")
    
    # Filter responses
    filtered_count = 0
    with open(args.responses_path, 'r') as infile, open(args.output_path, 'w') as outfile:
        for line in infile:
            response_data = json.loads(line.strip())
            if str(response_data.get('question_id')) in filtered_question_ids:
                outfile.write(json.dumps(response_data) + '\n')
                filtered_count += 1
    
    print(f"Filtered {filtered_count} responses and saved to {args.output_path}")

if __name__ == "__main__":
    main()
