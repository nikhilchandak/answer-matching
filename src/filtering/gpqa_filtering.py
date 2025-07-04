import os
import json
import re
import argparse

from pathlib import Path


def filter_free_response(response, pattern_match=False):
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
    # Check for answers in parentheses: "answer is (XYZ)" - handles nested parentheses
    parens_pattern = r"answer is \(((?:[^()]|\([^()]*\))*)\)"
    parens_matches = list(re.finditer(parens_pattern, response, re.IGNORECASE))
    if parens_matches:
        last_match = parens_matches[-1]
        return last_match.group(1)
    
    # Check for answers in asterisks: "answer is *XYZ*" or "answer is **XYZ**"
    asterisk_pattern = r"answer is \*{1,2}((?:[^*]|\*(?!\*{0,1}))*)\*{1,2}"
    asterisk_matches = list(re.finditer(asterisk_pattern, response, re.IGNORECASE))
    if asterisk_matches:
        last_match = asterisk_matches[-1]
        return last_match.group(1)
    
    # Check for answers in boxed notation: "answer is \boxed{XYZ}"
    boxed_pattern = r"answer is \\boxed\{((?:[^{}]|\{[^{}]*\})*)\}"
    boxed_matches = list(re.finditer(boxed_pattern, response, re.IGNORECASE))
    if boxed_matches:
        last_match = boxed_matches[-1]
        return last_match.group(1)
    
    
    # Just "\boxed{XYZ}"
    boxed_pattern = r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}"
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
    final_answer = None
        
    # Look for patterns like "answer is (X)" or "the answer is X" or similar
    # Only extract A, B, C, or D as valid answers
    patterns = [
        r"answer is \(([A-D])\)",
        r"answer is \*([A-D])\*",
        r"answer is \*\*([A-D])\*\*",
        r"answer is \\boxed\{([A-D])\}",
        r"answer is $\\boxed\{([A-D])\}$",
        r"answer is \*{1,2}([A-D])\*{1,2}",
        r"answer is \*{1,2}(([A-D]))\*{1,2}",
        r"answer is ([A-D])[^A-D]",
        r"answer is ([A-D])$",
        r"\$\\boxed\{([A-D])\}\$",
        r"\\boxed\{([A-D])\}",
        r"\\text{([A-D])}",
        r"answer: ([A-D])[^A-D]",
        r"answer: ([A-D])$",
        r"option ([A-D])[^A-D]",
        r"option ([A-D])$",
        r"choice ([A-D])[^A-D]",
        r"choice ([A-D])$",
        r"<answer>([A-D])</answer>",
    ]
    if pattern_match:
        for pattern in patterns:
            matches = list(re.finditer(pattern, response, re.IGNORECASE))
            if matches:
                final_answer = matches[-1].group(1).upper()
                # print(f"Final answer: {final_answer}")
                break
    
    # If no pattern matched, look for the last capital letter A-D in the text
    if not final_answer:
        # print(f"No answer found in {response}")
        # Find all occurrences of A, B, C, or D in the text
        # all_letter_matches = list(re.finditer(r"([A-D])", response, re.IGNORECASE))
        # if all_letter_matches:
        #     # Get the last occurrence
        #     final_answer = all_letter_matches[-1].group(1).upper()
        #     print(f"Final answer: {final_answer}")
        positions = {
            'A': response.rfind('A'),
            'B': response.rfind('B'),
            'C': response.rfind('C'),
            'D': response.rfind('D')
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

def filter_and_merge_samples(input_dir, output_dir, process):
    """
    Filters and merges sample files from multiple model directories.
    This function reads sample files from subdirectories within the input directory,
    filters the samples based on question IDs specified in a filter file, and merges
    the filtered samples into a single JSON file for each model. Additionally, it copies
    "results" files from the input directory to the output directory.
    Args:
        input_dir (str): The path to the input directory containing model subdirectories.
        output_dir (str): The path to the output directory where filtered samples and results will be saved.
    Raises:
        FileNotFoundError: If the filter file or any sample file is not found.
        json.JSONDecodeError: If there is an error decoding a JSON sample file.
    Example:
        filter_and_merge_samples('/path/to/input', '/path/to/output', '/path/to/filter.txt')
    """
    dataset = input_dir.split("/")[-1]
    if dataset == "":
        dataset = input_dir.split("/")[-2]
        
    # Ensure the output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    filter_func = filter_mcq_response if process == "mcq" else filter_free_response

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

        question_ids = []
        
        # Combine "samples" files and filter them
        combined_samples = []
        acc = 0
        total = 0
        for file_name in os.listdir(model_path):
            if file_name.startswith("samples"):
                file_path = os.path.join(model_path, file_name)
                with open(file_path, 'r') as f:
                    print(f"Reading samples from {file_path}")
                    for line in f:
                        sample = json.loads(line)
                        doc = sample.get("doc", {})
                        question_id = doc.get("question_id")
                        
                        # Skip repeat samples?
                        if question_id in question_ids:
                            continue
                        
                        question_ids.append(question_id)
                        timestamp = file_path.split("_")[-1].split(".")[0]
                        # if question_id in filter_ids:
                        max_gen_toks = sample.get("arguments", {}).get("gen_args_0", {}).get("arg_1", {}).get("max_gen_toks", 0)
                        thinking = True if max_gen_toks >= 4096 else False
                        thinking2 = False if "non_thinking" in model_path else True
                        assert thinking == thinking2
                        
                        # resps, finished = remove_thinking(sample.get("resps")[0][0])
                        resps, finished, filtered_resps = [], [], []
                        answer_letter = chr(ord('A') + doc.get("answer_index"))
                        matches = []
                        
                        for i, resp in enumerate(sample.get("resps")[0]):
                            # final_resp, think_finish = remove_thinking(resp)
                            # filtered_response = filter_response(final_resp)
                            # filtered_resps.append(filtered_response)
                            # resps.append(final_resp)
                            # finished.append(think_finish)
                            
                            
                            # Get response and extract answer
                            og_response = str(resp)
                            new_resp, think_finish = remove_thinking(og_response)
                            new_resp = og_response[-1000:]
                            filtered_response = filter_func(new_resp, pattern_match=True)
                            
                            if filtered_response == "[invalid]":
                                filtered_response = filter_func(new_resp, pattern_match=False)
                            
                            filtered_resps.append(filtered_response)
                            resps.append(new_resp)
                            finished.append(think_finish)
                            
                            if len(filtered_response) > 0 and filtered_response[0].lower() == answer_letter.lower():
                                matches.append(1)
                            else:
                                matches.append(0)
                            
                        options = doc.get("options") if "options" in doc else doc.get("choices")
                        actual_answer = options[doc.get("answer_index")]
                        
                        acc += sum(matches) #/float(len(matches))
                        total += len(matches)
                        
                        if True:
                            combined_sample = {
                                "question_id": doc.get("question_id"),
                                "question": doc.get("question"),
                                "options": options,
                                "answer": actual_answer,
                                "answer_index": doc.get("answer_index"),
                                "target": actual_answer,
                                
                                "filtered_resps": filtered_resps,
                                "resps": resps,
                                "category": doc.get("category"),
                                "completion_tokens": sample.get("completion_tokens"),
                                "exact_match": matches,
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
            overall_acc = float(acc) / float(total)
            print(f"Model: {model_dir}, Overall accuracy: {overall_acc}")
            
            output_file = os.path.join(dataset_dir, model_dir, "samples.jsonl")
            if os.path.exists(output_file):
                print(f"Output file {output_file} already exists, skipping")
                continue
            
            # print(f"Saving samples to {output_file}")
            # Path(os.path.dirname(output_file)).mkdir(parents=True, exist_ok=True)
            # with open(output_file, 'w') as f:
            #     for sample in combined_samples:
            #         f.write(json.dumps(sample) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Path to the input directory containing model subdirectories.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory where filtered samples and results will be saved.")
    # parser.add_argument("--filter_path", type=str, required=True, help="Path to the filter file containing question IDs to filter by.")

    args = parser.parse_args()
    process = "free" if "free" in args.input_dir else "mcq"

    filter_and_merge_samples(args.input_dir, args.output_dir, process)