#!/usr/bin/env python3
"""
Script to assert that the responses of "openai/gpt-4o" in *_gpqa.jsonl files 
in updated_annotation1/ folder are the same as in the reference file for the same question id.
"""
import json
import glob
import os
from typing import Dict, Any


def load_reference_responses(reference_file: str) -> Dict[str, str]:
    """
    Load the reference responses from the samples_gpt-4o.jsonl file.
    
    Args:
        reference_file: Path to the reference JSONL file
        
    Returns:
        Dictionary mapping question_id to response text
    """
    responses = {}
    with open(reference_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            if data.get('model') == 'openai/gpt-4o':
                question_id = data['question_id']
                response = data['filtered_resps']
                responses[question_id] = response
    return responses


def load_annotation_responses(annotation_file: str) -> Dict[str, str]:
    """
    Load the responses from an annotation file (ameya_gpqa.jsonl or nikhil_gpqa.jsonl).
    
    Args:
        annotation_file: Path to the annotation JSONL file
        
    Returns:
        Dictionary mapping question_id to response text for openai/gpt-4o
    """
    responses = {}
    with open(annotation_file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            models = data.get('model', [])
            if 'openai/gpt-4o' in models:
                gpt4o_index = models.index('openai/gpt-4o')
                question_id = data['question_id']
                response = data['response'][gpt4o_index]
                responses[question_id] = response
    return responses


def compare_responses(reference_responses: Dict[str, str], 
                     annotation_responses: Dict[str, str],
                     annotation_file: str) -> None:
    """
    Compare responses and assert they are the same.
    
    Args:
        reference_responses: Dictionary of reference responses
        annotation_responses: Dictionary of annotation responses
        annotation_file: Name of the annotation file being checked
    """
    print(f"\nChecking {annotation_file}...")
    
    # Check if all question IDs in annotation file exist in reference
    missing_in_reference = set(annotation_responses.keys()) - set(reference_responses.keys())
    if missing_in_reference:
        print(f"WARNING: Question IDs in {annotation_file} not found in reference: {missing_in_reference}")
    
    # Check if all question IDs in reference exist in annotation file
    missing_in_annotation = set(reference_responses.keys()) - set(annotation_responses.keys())
    if missing_in_annotation:
        print(f"WARNING: Question IDs in reference not found in {annotation_file}: {missing_in_annotation}")
    
    # Compare responses for common question IDs
    common_question_ids = set(reference_responses.keys()) & set(annotation_responses.keys())
    mismatches = []
    
    for question_id in common_question_ids:
        ref_response = reference_responses[question_id]
        ann_response = annotation_responses[question_id]
        
        if ref_response != ann_response:
            mismatches.append(question_id)
            print(f"\nMISMATCH for question_id: {question_id}")
            print(f"Reference response: {ref_response[:200]}...")
            print(f"Annotation response: {ann_response[:200]}...")
    
    if not mismatches:
        print(f"✓ All {len(common_question_ids)} responses match between reference and {annotation_file}")
    else:
        print(f"✗ Found {len(mismatches)} mismatches in {annotation_file}")
        raise AssertionError(f"Response mismatches found in {annotation_file}: {mismatches}")


def main():
    """Main function to run the comparison."""
    # File paths
    reference_file = "/fast/nchandak/qaevals/judge_outputs/gpqa_diamond_free/samples_gpt-4o.jsonl"
    # annotation_dir = "updated_annotation1"
    annotation_dir = "/fast/nchandak/qaevals/judge_outputs/alignment_plot/annotations/gpqa_diamond/"
    
    # Check if reference file exists
    if not os.path.exists(reference_file):
        raise FileNotFoundError(f"Reference file not found: {reference_file}")
    
    # Check if annotation directory exists
    if not os.path.exists(annotation_dir):
        raise FileNotFoundError(f"Annotation directory not found: {annotation_dir}")
    
    print(f"Loading reference responses from: {reference_file}")
    reference_responses = load_reference_responses(reference_file)
    print(f"Loaded {len(reference_responses)} reference responses")
    
    # Find all *_gpqa.jsonl files in the annotation directory
    pattern = os.path.join(annotation_dir, "*_gpqa.jsonl")
    annotation_files = glob.glob(pattern)
    
    if not annotation_files:
        raise FileNotFoundError(f"No *_gpqa.jsonl files found in {annotation_dir}")
    
    print(f"Found annotation files: {[os.path.basename(f) for f in annotation_files]}")
    
    # Compare each annotation file with the reference
    all_passed = True
    for annotation_file in annotation_files:
        try:
            annotation_responses = load_annotation_responses(annotation_file)
            compare_responses(reference_responses, annotation_responses, os.path.basename(annotation_file))
        except AssertionError as e:
            print(f"ERROR: {e}")
            all_passed = False
        except Exception as e:
            print(f"ERROR processing {annotation_file}: {e}")
            all_passed = False
    
    if all_passed:
        print(f"\n✓ SUCCESS: All openai/gpt-4o responses match between reference and annotation files!")
    else:
        print(f"\n✗ FAILURE: Some responses do not match!")
        exit(1)


if __name__ == "__main__":
    main()
