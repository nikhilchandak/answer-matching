import os
import json
import re
import argparse

from pathlib import Path


def filter_response(response):
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
    # Find all occurrences of formatted answers after "answer is"
    # Check for answers in parentheses: "answer is (XYZ)"
    parens_pattern = r"answer is \(([^)]+?)\)"
    parens_matches = list(re.finditer(parens_pattern, response, re.IGNORECASE))
    if parens_matches:
        last_match = parens_matches[-1]
        return last_match.group(1)
    
    # Check for answers in asterisks: "answer is *XYZ*" or "answer is **XYZ**"
    asterisk_pattern = r"answer is \*{1,2}([^*]+?)\*{1,2}"
    asterisk_matches = list(re.finditer(asterisk_pattern, response, re.IGNORECASE))
    if asterisk_matches:
        last_match = asterisk_matches[-1]
        return last_match.group(1)
    
    # Check for answers in boxed notation: "answer is \boxed{XYZ}"
    boxed_pattern = r"answer is \\boxed\{([^}]+?)\}"
    boxed_matches = list(re.finditer(boxed_pattern, response, re.IGNORECASE))
    if boxed_matches:
        last_match = boxed_matches[-1]
        return last_match.group(1)
    
    
    # Just "\boxed{XYZ}"
    boxed_pattern = r"\\boxed\{([^}]+?)\}"
    boxed_matches = list(re.finditer(boxed_pattern, response, re.IGNORECASE))
    if boxed_matches:
        last_match = boxed_matches[-1]
        return last_match.group(1)
    
    # If no special format is found, find the last occurrence of "answer is"
    # and return everything up to the end of that sentence
    if "answer is" in response.lower():
        last_answer_pos = response.lower().rfind("answer is")
        if last_answer_pos != -1:
            # Find the end of the sentence (period, question mark, exclamation mark)
            # sentence_end = re.search(r'[.!?]', response[last_answer_pos:])
            # if sentence_end:
            #     return response[:last_answer_pos + sentence_end.end()]
            # else:
            #     # If no sentence end is found, return everything after "answer is"
            #     return response
            
            # Just return the text after "answer is"
            end_pos = last_answer_pos + len("answer is") + 1
            final_ans = response[end_pos:]
    else :
        # If "answer is" pattern is not found, return the original response
        final_ans = response
    
    # Keep the final answer brief for matcher judge. 
    if len(final_ans) > 500:
        return final_ans[:500]
    else:
        return final_ans

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
    return response, False 

def filter_and_merge_samples(input_dir, output_dir, filter_path):
    """
    Filters and merges sample files from multiple model directories.
    This function reads sample files from subdirectories within the input directory,
    filters the samples based on question IDs specified in a filter file, and merges
    the filtered samples into a single JSON file for each model. Additionally, it copies
    "results" files from the input directory to the output directory.
    Args:
        input_dir (str): The path to the input directory containing model subdirectories.
        output_dir (str): The path to the output directory where filtered samples and results will be saved.
        filter_path (str): The path to the filter file containing question IDs to filter by.
    Raises:
        FileNotFoundError: If the filter file or any sample file is not found.
        json.JSONDecodeError: If there is an error decoding a JSON sample file.
    Example:
        filter_and_merge_samples('/path/to/input', '/path/to/output', '/path/to/filter.txt')
    """
    # Load question IDs from filter file
    with open(filter_path, 'r') as f:
        filter_ids = set(int(line.strip()) for line in f if line.strip())

    dataset = input_dir.split("/")[-1]
    if dataset == "":
        dataset = input_dir.split("/")[-2]
        
    # Ensure the output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Iterate over each subdirectory (model directory) in the input directory
    for model_dir in os.listdir(input_dir):
        model_path = os.path.join(input_dir, model_dir)
        if not os.path.isdir(model_path):
            continue
        
        inside_dir = os.listdir(model_path)[0]
        model_path = os.path.join(model_path, inside_dir)
        if not os.path.isdir(model_path):
            continue
        
        model_name = model_dir.split("_")[0]
        
        dataset_dir = os.path.join(output_dir, dataset)
        Path(dataset_dir).mkdir(parents=True, exist_ok=True)
        
        # Copy "results" file to output directory
        for file_name in os.listdir(model_path):
            if file_name.startswith("results"):
                src_file = os.path.join(model_path, file_name)
                dst_file = os.path.join(dataset_dir, model_dir, file_name)
                Path(os.path.dirname(dst_file)).mkdir(parents=True, exist_ok=True)
                with open(src_file, 'r') as src, open(dst_file, 'w') as dst:
                    dst.write(src.read())

        # Combine "samples" files and filter them
        combined_samples = []
        for file_name in os.listdir(model_path):
            if file_name.startswith("samples"):
                file_path = os.path.join(model_path, file_name)
                with open(file_path, 'r') as f:
                    print(f"Reading samples from {file_path}")
                    for line in f:
                        sample = json.loads(line)
                        doc = sample.get("doc", {})
                        question_id = doc.get("question_id")
                        timestamp = file_path.split("_")[-1].split(".")[0]
                        # if question_id in filter_ids:
                        max_gen_toks = sample.get("arguments", {}).get("gen_args_0", {}).get("arg_1", {}).get("max_gen_toks", 0)
                        thinking = True if max_gen_toks >= 4096 else False
                        thinking2 = False if "non_thinking" in model_path else True
                        assert thinking == thinking2
                        
                        resps, finished = remove_thinking(sample.get("resps")[0][0])
                        if True:
                            combined_sample = {
                                "question_id": doc.get("question_id"),
                                "question": doc.get("question"),
                                "options": doc.get("options"),
                                "answer": doc.get("answer"),
                                "answer_index": doc.get("answer_index"),
                                "target": sample.get("target"),
                                # "filtered_resps": sample.get("filtered_resps")[0],
                                # "resps": filter_response(sample.get("resps")[0][0]),
                                "filtered_resps": filter_response(resps),
                                "resps": resps,
                                "category": doc.get("category"),
                                "completion_tokens": sample.get("completion_tokens"),
                                "exact_match": sample.get("exact_match"),
                                "max_gen_toks": max_gen_toks,
                                "thinking": thinking,
                                "timestamp": timestamp,
                                "dataset": dataset,
                                "model": model_name,
                                "thinking_finished": finished,
                            }
                            combined_samples.append(combined_sample)

        # Sort samples by question_id
        combined_samples.sort(key=lambda x: x.get("question_id"))
        # Print number of samples
        print(f"Model: {model_dir}, Number of samples: {len(combined_samples)}")

        # Save combined and filtered samples to a single JSONL file
        if combined_samples:
            # Create dataset folder between output_dir and model_dir
            output_file = os.path.join(dataset_dir, model_dir, "samples.jsonl")
            if os.path.exists(output_file):
                print(f"Output file {output_file} already exists, skipping")
                continue
            
            # print(f"Saving samples to {output_file}")
            # Path(os.path.dirname(output_file)).mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                for sample in combined_samples:
                    f.write(json.dumps(sample) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the input directory containing model subdirectories.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory where filtered samples and results will be saved.")
    parser.add_argument("--filter_path", type=str, required=True, help="Path to the filter file containing question IDs to filter by.")

    args = parser.parse_args()

    filter_and_merge_samples(args.input_dir, args.output_dir, args.filter_path)