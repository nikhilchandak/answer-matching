from typing import List
from math_verify import parse, verify

import os
import json
import re
import argparse

from pathlib import Path


def filter_mcq_response(response, pattern_match=False):
    """
    Truncate the response at the end of the last occurrence of 'answer is' followed by
    formatted text in parentheses, asterisks, or boxed notation. If no such formatting
    is found, return the text up to the end of the sentence containing the last occurrence
    of 'answer is'.
    
    Args:
        response (str): The response string to process.
    
    Returns:
        str: The truncated or original response.
    """
    
    # # Check for answers in boxed notation: "answer is \boxed{XYZ}"
    # boxed_pattern = r"\\boxed\{([^}]+?)\}"
    # boxed_matches = list(re.finditer(boxed_pattern, response, re.IGNORECASE))
    # if boxed_matches:
    #     last_match = boxed_matches[-1]
    #     return last_match.group(1)
    
    # # Check for answers in boxed notation: "answer is \boxed{XYZ}"
    # boxed_pattern = r"\\\\boxed\{([^}]+?)\}"
    # boxed_matches = list(re.finditer(boxed_pattern, response, re.IGNORECASE))
    # if boxed_matches:
    #     last_match = boxed_matches[-1]
    #     return last_match.group(1)
    
    # # Find all occurrences of formatted answers after "answer is"
    # # Check for answers in parentheses: "answer is (XYZ)"
    # parens_pattern = r"answer is \(([^)]+?)\)"
    # parens_matches = list(re.finditer(parens_pattern, response, re.IGNORECASE))
    # if parens_matches:
    #     last_match = parens_matches[-1]
    #     return last_match.group(1)
    
    # # Check for answers in asterisks: "answer is *XYZ*" or "answer is **XYZ**"
    # asterisk_pattern = r"answer is \*{1,2}([^*]+?)\*{1,2}"
    # asterisk_matches = list(re.finditer(asterisk_pattern, response, re.IGNORECASE))
    # if asterisk_matches:
    #     last_match = asterisk_matches[-1]
    #     return last_match.group(1)
    
    final_answer = None
        
    # Look for patterns like "answer is (X)" or "the answer is X" or similar
    # Only extract A, B, C, D, E, F, G, H, I, or J as valid answers
    patterns = [
        r"\$\\boxed\{([A-J])\}\$",
        r"\\boxed\{([A-J])\}",
        r"answer is \(([A-J])\)",
        r"answer is \*([A-J])\*",
        r"answer is \*\*([A-J])\*\*",
        r"answer is \\boxed\{([A-J])\}",
        r"answer is $\\boxed\{([A-J])\}$",
        r"answer is \*{1,2}([A-J])\*{1,2}",
        r"answer is \*{1,2}(([A-J]))\*{1,2}",
        r"answer is ([A-J])[^A-J]",
        r"answer is ([A-J])$",
        r"\\text{([A-J])}",
        r"answer: ([A-J])[^A-J]",
        r"answer: ([A-J])$",
        r"option ([A-J])[^A-J]",
        r"option ([A-J])$",
        r"choice ([A-J])[^A-J]",
        r"choice ([A-J])$",
        r"<answer>([A-J])</answer>",
    ]
    if pattern_match:
        for pattern in patterns:
            matches = list(re.finditer(pattern, response, re.IGNORECASE))
            if matches:
                final_answer = matches[-1].group(1).upper()
                # print(f"Final answer: {final_answer}")
                break
    
    # If no pattern matched, look for the last capital letter A-J in the text
    if not final_answer:
        # print(f"No answer found in {response}")
        # Find all occurrences of A, B, C, D, E, F, G, H, I, or J in the text
        # all_letter_matches = list(re.finditer(r"([A-J])", response, re.IGNORECASE))
        # if all_letter_matches:
        #     # Get the last occurrence
        #     final_answer = all_letter_matches[-1].group(1).upper()
        #     print(f"Final answer: {final_answer}")
        positions = {
            'A': response.rfind('A'),
            'B': response.rfind('B'),
            'C': response.rfind('C'),
            'D': response.rfind('D'),
            'E': response.rfind('E'),
            'F': response.rfind('F'),
            'G': response.rfind('G'),
            'H': response.rfind('H'),
            'I': response.rfind('I'),
            'J': response.rfind('J')
        }
        
        # Filter out options that weren't found
        valid_positions = {letter: pos for letter, pos in positions.items() if pos != -1}
        
        if valid_positions:
            # Get the letter with the highest position (appears last in the text)
            final_answer = max(valid_positions.items(), key=lambda x: x[1])[0]
            # print(f"Final answer: {final_answer}")
            return final_answer
            
    if final_answer == None:
        # print(f"No answer found in {response}")
        return "[invalid]"
    
    return final_answer



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


def remove_thinking(response):
    """
    Remove the thinking part from a response if it exists.
    If "<think>" and "</think>" tags are present, return everything after the last "</think>".
    Otherwise, return the original response.
    
    Args:
        response (str): The response string to process.
    
    Returns:
        str: The response with thinking part removed or the original response.
    """
    if "<think>" in response:
        last_think_end = response.rfind("</think>")
        if last_think_end != -1:
            return response[last_think_end + len("</think>"):].strip(), True
        # else : # Thinking did not finish
        #     return "[invalid]"
        else :
            return response, False 
    return response, True 

def process_samples(input_dir, process):
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
    
    filter_func = filter_mcq_response if process == "mcq" else None 
    filtered_ids = get_filtered_ids()
    
    # Process each sample file
    for file_path in sample_files:
        # print(f"Processing {file_path}")
        if "non_think" not in file_path:
            continue
        
        # Extract model name from file path
        model_name = file_path.split("/")[-3]
        
        timestamp = file_path.split("/")[-1].split("_")[-1].split(".")[0]
        
        # dataset = "math_verify" if "verify" in file_path else "math_verbalize"
        dataset = file_path.split("/")[-4]
        
        assert dataset in file_path, f"Dataset not found in {file_path}"
        
        print(f"Model: {model_name}, Dataset: {dataset}") # , Timestamp: {timestamp}")
        # Create model directory in output_dir
        output_path = str(file_path)
        # output_path = str(file_path).replace(".jsonl", "_qwen3-32b.jsonl")
        
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
        correct_cnt = {}
        
        with open(file_path, 'r') as f:
            print(f"Processing {file_path}")
            for line in f:
                sample = json.loads(line)
                qid = sample.get("question_id")
                # print(qid)
                
                if qid not in filtered_ids:
                    continue
                
                total += 1
                if qid not in correct_cnt:
                    correct_cnt[qid] = 0
            
                # Get response and extract answer
                response = sample.get("resps")
                resps, finished = remove_thinking(response)
                resps = response[-1000:]
                filtered_response = filter_func(resps, pattern_match=True)
                
                if filtered_response == "[invalid]":
                    filtered_response = filter_func(resps)
                
                # Get correct answer
                # if process == "mcq":
                #     correct_answer = sample.get("doc", {}).get("Answer")
                # else:
                #     correct_answer = sample.get("target")
                
                correct_answer = sample.get("target")
                original_filtered_response = sample.get("filtered_resps")[0]
                
                # Check if extracted answer matches correct answer
                if process == "mcq":
                    is_match = filtered_response.strip().lower() == correct_answer.strip().lower()
                else:
                    assert False, f"Incorrect process: {process}"
                    
                og_field = "exact_match" 
                if og_field in sample:
                    og_match = int(sample.get(og_field))
                else:
                    og_match = 0
                    print(sample)
                    assert False
                
                correct += og_match
                
                if og_match == 1 and not is_match:
                    negative_flips += 1
                    # print(f"Filtered response: {response}, Correct answer: {correct_answer}")
                    flipped_ids.append(qid)
                
                if is_match:
                    correct += 1
                    if og_match == 0:
                        positive_flips += 1
                
                # sample[og_field] = 1 if is_match else 0
                
                # updated_samples.append(sample)
                # correct_cnt[qid] += 1 if is_match else 0
                
                
                # If this is a Level 5 question, add it to level5_samples
                # if True:
                #     accuracy = correct / total * 100 if total > 0 else 0
        
        # Calculate and print accuracy
        level5_samples = updated_samples
        accuracy = correct / total * 100 if total > 0 else 0
        print(f"File: {os.path.basename(file_path)}, Accuracy: {accuracy:.2f}% ({correct}/{total})")
        print(f"Negative Flips: {negative_flips}, Positive Flips: {positive_flips}")
        # print(f"Flipped IDs: {flipped_ids}")
        
        # Save Level 5 samples to output directory
        if level5_samples:
            # with open(output_path, 'w') as f:
            #     for sample in level5_samples:
            #         f.write(json.dumps(sample) + '\n')
            
            print(f"Saved {len(level5_samples)} Level 5 samples to {output_path}")
        
        print("\n--------------------------------\n")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="/fast/nchandak/qaevals/outputs/math_verify/", 
                        help="Path to the input directory containing sample files.")
    
    args = parser.parse_args()
    
    process = "mcq" 
    process_samples(args.input_dir, process)