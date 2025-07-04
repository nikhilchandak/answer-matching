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

def filter_and_merge_samples(input_file_path):
    """
    Processes a JSONL file to extract final answers from responses.
    This function reads a JSONL file, extracts the final answer (option A-J) from each response,
    and updates the 'filtered_resps' field with the extracted answer.
    
    Args:
        input_file_path (str): The path to the input JSONL file.
        output_dir (str, optional): Not used in this implementation.
        filter_path (str, optional): Not used in this implementation.
    
    Returns:
        None: The function updates the JSONL file in place.
    """
    # Read the JSONL file
    with open(input_file_path, 'r') as f:
        items = [json.loads(line) for line in f]
    
    acc = 0
    original_acc = 0
    # Process each item
    for item in items:
        response = item.get("resps")
        # Extract the final answer (option A-J)
        final_answer = None
        
        # Look for patterns like "answer is (X)" or "the answer is X" or similar
        patterns = [
            r"<answer>([A-J])</answer>",
            r"answer is \(([A-J])\)",
            r"\$\\boxed\{([A-J])\}\$",
            r"answer is \*([A-J])\*",
            r"answer is \*\*([A-J])\*\*",
            r"answer is \\boxed\{([A-J])\}",
            r"answer is $\\boxed\{([A-J])\}$",
            r"\\boxed\{([A-J])\}",
            r"answer is ([A-J])[^A-J]",
            r"answer is ([A-J])$",
            r"answer: ([A-J])[^A-J]",
            r"answer: ([A-J])$",
            r"option ([A-J])[^A-J]",
            r"option ([A-J])$",
            r"choice ([A-J])[^A-J]",
            r"choice ([A-J])$",
        ]
        
        for pattern in patterns:
            matches = list(re.finditer(pattern, response, re.IGNORECASE))
            if matches:
                final_answer = matches[-1].group(1).upper()
                # print(f"Final answer: {final_answer}")
                break
        
        # If no pattern matched, look for the last standalone A-J in the text
        if not final_answer:
            standalone_matches = list(re.finditer(r"(?:^|\s)([A-J])(?:\s|$|\.|\,|\)|\])", response, re.IGNORECASE))
            if standalone_matches:
                final_answer = standalone_matches[-1].group(1).upper()
        
        original_acc += item["exact_match"]
        
        # Update the filtered_resps field
        if final_answer:
            item["filtered_resps"] = final_answer
            item["exact_match"] = final_answer == item["answer"]
            print(f"Filtered answer: {final_answer}")
            print(f"Correct answer: {item['answer']}")
            print(f"Original answer: {item["filtered_resps"]}")
        else:
            # If no answer could be extracted, keep the original filtered_resps
            # print(f"No answer could be extracted from {item['question']}")
            pass
        
        acc += final_answer == item["answer"]
        
    print(f"Accuracy: {acc * 100.0 / len(items):.2f}%")
    print(f"Original accuracy: {original_acc * 100.0 / len(items):.2f}%")
    # Write the updated items back to the file
    with open(input_file_path, 'w') as f:
        for item in items:
            f.write(json.dumps(item) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file_path", type=str, required=True, help="Path to the input directory containing model subdirectories.")
    args = parser.parse_args()

    filter_and_merge_samples(args.input_file_path)