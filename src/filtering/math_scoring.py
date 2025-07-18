import os
import json
import re
import argparse

from pathlib import Path

from math_verify import parse, verify
# assert version("antlr4-python3-runtime").startswith("4.11")

def filter_response(response, pattern_match=False):
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
    
    # Check for answers in boxed notation: "\boxed{XYZ}"
    boxed_pattern = r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
    boxed_matches = list(re.finditer(boxed_pattern, response, re.IGNORECASE))
    if boxed_matches:
        last_match = boxed_matches[-1]
        return last_match.group(1)
    
    # Check for answers in boxed notation: "\\boxed{XYZ}"
    boxed_pattern = r"\\\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}"
    boxed_matches = list(re.finditer(boxed_pattern, response, re.IGNORECASE))
    if boxed_matches:
        last_match = boxed_matches[-1]
        return last_match.group(1)
    
    # Find all occurrences of formatted answers after "answer is"
    # Check for answers in parentheses: "answer is (XYZ)"
    parens_pattern = r"answer is \(([^{}]*(?:\{[^{}]*\}[^{}]*)*)\)"
    parens_matches = list(re.finditer(parens_pattern, response, re.IGNORECASE))
    if parens_matches:
        last_match = parens_matches[-1]
        return last_match.group(1)
    
    # Check for answers in asterisks: "answer is *XYZ*" or "answer is **XYZ**"
    asterisk_pattern = r"answer is \*{1,2}([^{}]*(?:\{[^{}]*\}[^{}]*)*)\*{1,2}"
    asterisk_matches = list(re.finditer(asterisk_pattern, response, re.IGNORECASE))
    if asterisk_matches:
        last_match = asterisk_matches[-1]
        return last_match.group(1)
    
    # Check for LaTeX fractions and other complex expressions
    latex_pattern = r"\\frac\{([^{}]+)\}\{([^{}]+)\}"
    latex_matches = list(re.finditer(latex_pattern, response))
    if latex_matches:
        last_match = latex_matches[-1]
        return f"\\frac{{{last_match.group(1)}}}{{{last_match.group(2)}}}"
    
    return response[-100:]


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
    # Only extract A, B, C, or D as valid answers
    patterns = [
        r"\$\\boxed\{([A-D])\}\$",
        r"\\boxed\{([A-D])\}",
        r"answer is \(([A-D])\)",
        r"answer is \*([A-D])\*",
        r"answer is \*\*([A-D])\*\*",
        r"answer is \\boxed\{([A-D])\}",
        r"answer is $\\boxed\{([A-D])\}$",
        r"answer is \*{1,2}([A-D])\*{1,2}",
        r"answer is \*{1,2}(([A-D]))\*{1,2}",
        r"answer is ([A-D])[^A-D]",
        r"answer is ([A-D])$",
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
        else :
            return response, False 
    return response, True 

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
    
    filter_func = filter_mcq_response if process == "mcq" else filter_response
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Process each sample file
    for file_path in sample_files:
        print(f"Processing {file_path}")
        
        # Extract model name from file path
        model_name = file_path.split("/")[-3]
        
        timestamp = file_path.split("/")[-1].split("_")[-1].split(".")[0]
        dataset = "math_mcq" if "mcq" in file_path else "math_free"
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
                
                # Get response and extract answer
                response = sample.get("resps")[0][0]
                resps, finished = remove_thinking(response)
                resps = response[-1000:]
                filtered_response = filter_func(resps, pattern_match=True)
                
                if filtered_response == "[invalid]":
                    filtered_response = filter_func(resps)
                
                # Get correct answer
                if process == "mcq":
                    correct_answer = sample.get("doc", {}).get("Answer")
                else:
                    correct_answer = sample.get("target")
                original_filtered_response = sample.get("filtered_resps")[0]
                
                # Check if extracted answer matches correct answer
                if process == "mcq":
                    is_match = filtered_response.strip() == correct_answer.strip()
                else:
                    a = parse(f"${filtered_response}$")
                    b = parse(f"${correct_answer}$")
                    try:
                        is_match = verify(b, a, strict=False)
                    except Exception as e:
                        print(f"Error verifying: {e}")
                        is_match = False
                    
                og_field = "exact_match" if process == "mcq" else "math_verify"
                og_match = int(sample.get(og_field))
                
                if og_match == 1 and not is_match:
                    negative_flips += 1
                    flipped_ids.append(sample.get("doc", {}).get("Question_ID"))
                
                if is_match:
                    correct += 1
                    if og_match == 0:
                        positive_flips += 1
                
                    level_acc[level]["correct"] += 1
                    
                level_acc[level]["total"] += 1
                
                # Update sample with new fields
                sample["filtered_resps"] = [filtered_response]
                sample[og_field] = 1 if is_match else 0
                sample["thinking_finished"] = finished
                
                updated_samples.append(sample)
                
                # If this is a Level 5 question, add it to level5_samples
                if level == "Level 5":
                    # Extract thinking part if it exists
                    thinking = None
                    if "<think>" in response:
                        think_start = response.find("<think>")
                        think_end = response.rfind("</think>")
                        if think_end != -1:
                            thinking = response[think_start + len("<think>"):think_end].strip()
                    
                    # Create sample in the required format
                    doc = sample.get("doc", {})
                    level5_sample = {
                        "question_id": doc.get("Question_ID"),
                        "question": doc.get("Question"),
                        "target": sample.get("target"),
                        "category": doc.get("Type"),
                        "filtered_resps": filtered_response,
                        "resps": resps,
                        "completion_tokens": sample.get("completion_tokens"),
                        "exact_match": 1 if is_match else 0,
                        "max_gen_toks": sample.get("arguments", {}).get("gen_args_0", {}).get("arg_1", {}).get("max_gen_toks", 1024),
                        "thinking": thinking,
                        "timestamp": timestamp,
                        "dataset": dataset,
                        "model": model_name,
                        "thinking_finished": finished,
                    }
                    if process == "mcq":
                        options = [doc.get("A"), doc.get("B"), doc.get("C"), doc.get("D")]
                        level5_sample["options"] = options
                        level5_sample["answer_index"] = ord(sample.get("target")) - ord("A")
                        level5_sample["answer"] = doc.get("Answer")
                        
                    level5_samples.append(level5_sample)
        
        # Calculate and print accuracy
        accuracy = correct / total * 100 if total > 0 else 0
        print(f"File: {os.path.basename(file_path)}, Accuracy: {accuracy:.2f}% ({correct}/{total})")
        print(f"Negative Flips: {negative_flips}, Positive Flips: {positive_flips}")
        print(f"Flipped IDs: {flipped_ids}")
        for level, acc in level_acc.items():
            print(f"Level {level}: {acc['correct']}/{acc['total']} ({acc['correct']/acc['total']*100:.2f}%)")

        # Save Level 5 samples to output directory
        if level5_samples:
            output_path = os.path.join(model_dir, "samples.jsonl")
            print(f"Saving {len(level5_samples)} Level 5 samples to {output_path}")
            # with open(output_path, 'w') as f:
            #     for sample in level5_samples:
            #         f.write(json.dumps(sample) + '\n')
            
            # print(f"Saved {len(level5_samples)} Level 5 samples to {output_path}")
        
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