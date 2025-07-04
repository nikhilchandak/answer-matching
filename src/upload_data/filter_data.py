
#!/usr/bin/env python
# coding: utf-8

import os
import sys
import json
import asyncio
import argparse
import logging
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def get_filtered_ids(dataset):
    """
    Get filtered IDs from file. Only returns question IDs that have rating_osq >= 4 
    and rating_multians >= 4 in ALL .jsonl files in the directory.
    """
    annotations_dir = f"/fast/nchandak/qaevals/judge_outputs/alignment_plot/annotations/{dataset}/"
    
    if not os.path.exists(annotations_dir):
        logger.warning(f"Annotations directory not found: {annotations_dir}")
        return []
    
    # Find all .jsonl files in the directory
    jsonl_files = []
    for file in os.listdir(annotations_dir):
        if file.endswith('.jsonl'):
            jsonl_files.append(os.path.join(annotations_dir, file))
    
    logger.info(f"Found {len(jsonl_files)} .jsonl files in {annotations_dir}")
    
    if not jsonl_files:
        logger.warning("No .jsonl files found in directory")
        return []
    
    # Dictionary to track ratings across files
    # Structure: {question_id: {file_path: (rating_osq, rating_multians)}}
    ratings_by_id = {}
    
    # Process each .jsonl file
    for file_path in jsonl_files:
        logger.info(f"Processing file: {file_path}")
        try:
            with open(file_path, 'r') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line.strip())
                        
                        # Extract required fields
                        question_id = str(data.get('question_id'))
                        rating_osq = data.get('rating_osq')
                        rating_multians = data.get('rating_multians')
                        
                        # Check if all required fields are present
                        if question_id is None or rating_osq is None or rating_multians is None:
                            continue
                        
                        # Initialize dict for this question_id if not exists
                        if question_id not in ratings_by_id:
                            ratings_by_id[question_id] = {}
                        
                        # Store ratings for this file
                        ratings_by_id[question_id][file_path] = (rating_osq, rating_multians)
                                
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse line {line_num} in {file_path}: {e}")
                        continue
                        
        except FileNotFoundError:
            logger.warning(f"File not found: {file_path}")
            continue
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            continue
    
    
    # Filter IDs that meet criteria in ALL files
    filtered_ids = []
    for question_id, file_ratings in ratings_by_id.items():
        # Check if question appears in all files
        if len(file_ratings) != len(jsonl_files):
            continue
            
        # Check if ratings meet criteria in all files
        meets_criteria = all(
            rating_osq >= 4 and rating_multians >= 4
            for rating_osq, rating_multians in file_ratings.values()
        )
        
        # print(question_id, file_ratings.values())
        if meets_criteria:
            filtered_ids.append(str(question_id))
    
    assert set(list(set(filtered_ids))) == set(filtered_ids), "Filtered IDs are not unique"
    logger.info(f"Found {len(filtered_ids)} question IDs with rating_osq >= 4 and rating_multians >= 4 in ALL files")
    return filtered_ids


def main():
    dataset = "mmlu_pro"
    filtered_ids = get_filtered_ids(dataset)
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Save filtered IDs to text file
    output_file = f"data/{dataset}_human_filtered_ids.txt"
    with open(output_file, "w") as f:
        for question_id in filtered_ids:
            f.write(f"{question_id}\n")
    
    print(f"Saved {len(filtered_ids)} filtered IDs to {output_file}")

if __name__ == "__main__":
    # Load the GPQA dataset
    gpqa_path = "gpqa-diamond-freeform.csv"
    if not os.path.exists(gpqa_path):
        logger.error(f"GPQA dataset not found: {gpqa_path}")
        exit(1)
    
    gpqa_df = pd.read_csv(gpqa_path)
    logger.info(f"Loaded GPQA dataset with {len(gpqa_df)} questions")
    
    # Get human filtered IDs
    dataset = "gpqa_diamond"
    filtered_ids = get_filtered_ids(dataset)
    logger.info(f"Found {len(filtered_ids)} human filtered IDs")
    
    # Filter the dataset to keep only questions that pass human filter
    filtered_df = gpqa_df[gpqa_df['Record ID'].astype(str).isin(filtered_ids)]
    logger.info(f"Filtered dataset has {len(filtered_df)} questions")
    
    # Save the filtered dataset
    output_path = "gpqa-diamond-freeform-filtered.csv"
    filtered_df.to_csv(output_path, index=False)
    logger.info(f"Saved filtered dataset to {output_path}")