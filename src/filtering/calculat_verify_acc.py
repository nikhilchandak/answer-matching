import os
import json
import argparse
from collections import defaultdict
from pathlib import Path


def calculate_accuracy(input_dir, file_path, to_save=True):
    """
    Calculate accuracy for a single JSONL file.
    A question is considered correct if its acc adds up to 10.
    """
    question_scores = defaultdict(float)
    total_questions = set()
    total_occurences = {}
    new_samples = []
    
    file_name = file_path.name
    # Read the file and accumulate scores for each question_id
    with open(file_path, 'r') as f:
        for line in f:
            try:
                data = json.loads(line.strip())
                # print(data)
                if ('doc' in data and 'q_hash' in data['doc'] and ('acc' in data or 'exact_match' in data)) or 'question_id' in data:
                    # print(data)
                    question_id = data['doc']['q_hash'] if 'doc' in data and 'q_hash' in data['doc'] else data['question_id']
                    if question_id not in total_occurences:
                        new_samples.append(data)
                        total_occurences[question_id] = 0
                        
                    total_occurences[question_id] += 1
                    if 'acc' in data:
                        acc = float(data['acc'])
                    else:
                        acc = float(data['exact_match'])
                        
                    question_scores[question_id] += int(acc)
                    total_questions.add(question_id)
            except json.JSONDecodeError:
                print(f"Warning: Could not parse JSON in {file_path}")
                continue
            except KeyError:
                print(f"Warning: Missing required keys in a line in {file_path}")
                continue
    
    # Count questions with a total score of 10
    # print(question_scores)
    
    correct_questions = sum(1 for qid, score in question_scores.items() if score == total_occurences[qid])
    total_unique_questions = len(total_questions)
    
    if to_save:
        for i, sample in enumerate(new_samples):
            new_samples[i]["exact_match"] = 1 if question_scores[sample["question_id"]] == total_occurences[sample["question_id"]] else 0

        output_path = os.path.join(input_dir, f"verified_{file_name}")
        print(f"Saving verified samples to {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in new_samples:
                f.write(json.dumps(sample) + '\n')
        
    # Calculate accuracy
    accuracy = correct_questions / total_unique_questions if total_unique_questions > 0 else 0
    
    return accuracy, correct_questions, total_unique_questions


def process_directory(input_dir):
    """
    Process all JSONL files in the input directory recursively.
    """
    input_path = Path(input_dir)
    all_files = list(input_path.glob('**/*.jsonl'))
    
    if not all_files:
        print(f"No JSONL files found in {input_dir}")
        return
    
    print(f"Found {len(all_files)} JSONL files to process")
    
    for file_path in all_files:
        if "samples_" not in file_path.name:
            continue
        accuracy, correct, total = calculate_accuracy(input_dir, file_path)
        print(f"File: {file_path}")
        print(f"  Correct questions: {correct}/{total}")
        print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print("-" * 50)


def main():
    parser = argparse.ArgumentParser(description="Calculate verification accuracy from JSONL files")
    parser.add_argument("--input_dir", help="Directory containing JSONL files to process")
    args = parser.parse_args()
    
    if not os.path.isdir(args.input_dir):
        print(f"Error: {args.input_dir} is not a valid directory")
        return
    
    process_directory(args.input_dir)


if __name__ == "__main__":
    main()
