#!/usr/bin/env python
# coding: utf-8

import json
import os
from pathlib import Path

def find_examples_with_comments():
    """
    Load the specified JSONL file and print question_ids and comments for rows
    where comments are not empty.
    """
    # file_path = "/home/nchandak/qaevals/how-to-qa/src/visualize_resps/annotation/saves/2907.jsonl"
    file_path = "/home/nchandak/qaevals/how-to-qa/src/visualize_resps/annotation/saves/strat1shash.jsonl"
    
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return
    
    print(f"Loading file: {file_path}")
    
    examples_with_comments = []
    relevant_ids = {}
    
    # Read the JSONL file line by line
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                
                if "rating_osq" in data and data["rating_osq"] >= 4 and data["rating_match"] <= 3:
                    relevant_ids[str(data["question_id"])] = data
                else :
                    continue 
                
                # Check if comments field exists and is not empty
                if "comments" in data and data["comments"]:
                    question_id = data.get("question_id", "N/A")
                    # print(f"Question ID: {question_id}")
                    # print(f"Comment: {data['comments']}")
                    # print("-" * 50)
                    
                    examples_with_comments.append({
                        "question_id": question_id,
                        "comments": data["comments"]
                    })
            except json.JSONDecodeError:
                print(f"Error parsing JSON at line {line_num}")
    
    # print(f"Found {len(examples_with_comments)} examples with comments")
    # return examples_with_comments
    print(f"Relevant IDs: {list(relevant_ids.keys())}")
    return relevant_ids


def find_incorrect_samples():
    """
    Load the specified JSONL file and print question_ids and comments for rows
    where comments are not empty.
    """
    # file_path = "/home/nchandak/qaevals/how-to-qa/src/visualize_resps/annotation/saves/2907.jsonl"
    file_path = "/fast/nchandak/qaevals/judge_outputs/alignment_plot/gen/mmlu_pro/samples.jsonl"
    
    
    print(f"Loading file: {file_path}")
    
    examples_with_comments = []
    relevant_ids = {}
    
    # Read the JSONL file line by line
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                
                # if "rating_osq" in data and data["rating_osq"] >= 4 and data["rating_match"] <= 3:
                #     relevant_ids[str(data["question_id"])] = data
                # else :
                #     continue 
                
                if int(data["score_deepseek-chat-v3-0324"]) == 0 :
                    relevant_ids[str(data["question_id"])] = data
                
                # Check if comments field exists and is not empty
                if "comments" in data and data["comments"]:
                    question_id = data.get("question_id", "N/A")
                    # print(f"Question ID: {question_id}")
                    # print(f"Comment: {data['comments']}")
                    # print("-" * 50)
                    
                    examples_with_comments.append({
                        "question_id": question_id,
                        "comments": data["comments"]
                    })
            except json.JSONDecodeError:
                print(f"Error parsing JSON at line {line_num}")
    
    # print(f"Found {len(examples_with_comments)} examples with comments")
    # return examples_with_comments
    print(f"Relevant IDs: {list(relevant_ids.keys())}")
    return relevant_ids


def find_exact_match_mcq_examples(free_form_ids):
    """
    Load MCQ data from the specified JSONL file and print questions and options
    for rows where exact_match is true.
    """
    file_path = "/fast/nchandak/qaevals/judge_outputs/alignment_plot/mcq/mmlu_pro/samples.jsonl"
    
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return
    
    print(f"Loading MCQ file: {file_path}")
    
    exact_match_examples = []
    
    # Read the JSONL file line by line
    with open(file_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                qid = str(data.get("question_id", "N/A"))
                
                # Check if exact_match field exists and is true
                if "exact_match" in data and data["exact_match"] is True and qid in free_form_ids:
                    question_id = data.get("question_id", "N/A")
                    question = data.get("question", "N/A")
                    options = data.get("options", [])
                    
                    print(f"Question ID: {question_id}")
                    print(f"Question: {question}")
                    print("Options:")
                    for i, option in enumerate(options):
                        print(f"  {chr(65 + i)}. {option}")
                    
                    print(f"Answer: {data['answer']}\n")
                    
                    print(f"\nModel's MCQ Response: {data['answer']}")
                    
                    # print(f"MCQ Response: {data['resps']}")
                    
                    response = free_form_ids[qid]['response'] if "response" in free_form_ids[qid] else free_form_ids[qid]['filtered_resps']
                    print(f"Model's Free form Response: {response}")
                    
                    # if free_form_ids[qid]["comments"] != "":
                    #     print(f"Comments: {free_form_ids[qid]['comments']}")
                        
                    print("\n\n" + "-" * 50)
                    print()
                    
                    exact_match_examples.append({
                        "question_id": question_id,
                        "question": question,
                        "options": options
                    })
            except json.JSONDecodeError:
                print(f"Error parsing JSON at line {line_num}")
    
    print(f"Found {len(exact_match_examples)} examples with exact matches")
    return exact_match_examples

if __name__ == "__main__":
    free_form_ids = find_examples_with_comments()
    free_form_ids = find_incorrect_samples()
    find_exact_match_mcq_examples(free_form_ids)
