import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sys
from pathlib import Path
from collections import defaultdict

# Add the parent directory to the path to import from src.filtering
sys.path.append(str(Path(__file__).parent.parent))
from filtering.load_datasets import load_dataset_by_name

def analyze_annotations(annotation_file, save_dir, dataset_name="MMLU-Pro", dataset_split="test", 
                        unique_rating_filter=(1, 5), specific_rating_filter=(1, 5)):
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Load annotations
    annotations = []
    with open(annotation_file, 'r') as f:
        for line in f:
            annotations.append(json.loads(line))
    
    print(f"Loaded {len(annotations)} annotations from {annotation_file}")
    
    # Load dataset to get subject information
    dataset_items = load_dataset_by_name(name=dataset_name, split=dataset_split)
    print(f"Loaded {len(dataset_items)} items from {dataset_name} dataset")
    
    # Create question_id to subject mapping
    question_id_to_subject = {}
    for item in dataset_items:
        question_id = item.get("question_id")
        subject = item.get("subject")
        if question_id and subject:
            question_id_to_subject[question_id] = subject
    
    print(f"Created mapping for {len(question_id_to_subject)} questions with subject information")
    
    # Rating fields to analyze
    rating_fields = ['rating_match', 'rating_osq', 'rating_multians', 'rating_correct']
    
    # Analyze each rating field
    for field in rating_fields:
        # Count occurrences
        values = [annot.get(field) for annot in annotations]
        null_count = sum(1 for v in values if v is None)
        
        # Count ratings 1-5
        rating_counts = {}
        for i in range(1, 6):
            rating_counts[i] = sum(1 for v in values if v == i)
        
        # Create bar plot
        plt.figure(figsize=(10, 6))
        
        # Prepare data for plotting
        labels = ['null'] + [str(i) for i in range(1, 6)]
        counts = [null_count] + [rating_counts[i] for i in range(1, 6)]
        
        # Create bars
        bars = plt.bar(labels, counts, color='skyblue')
        
        # Add count labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height}', ha='center', va='bottom')
        
        # Add labels and title
        plt.xlabel('Rating Value')
        plt.ylabel('Count')
        plt.title(f'Distribution of {field}')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Save plot
        base_name = os.path.basename(annotation_file).split('.')[0]
        save_path = os.path.join(save_dir, f"{base_name}_{field}.jpg")
        plt.savefig(save_path)
        plt.close()
        
        print(f"Saved plot to {save_path}")
    
    # Analyze subject distribution for filtered questions
    analyze_subject_distribution(
        annotations, 
        question_id_to_subject, 
        save_dir, 
        os.path.basename(annotation_file).split('.')[0],
        unique_rating_filter,
        specific_rating_filter
    )

def analyze_subject_distribution(annotations, question_id_to_subject, save_dir, base_name,
                                unique_rating_filter=(1, 5), specific_rating_filter=(1, 5)):
    """
    Analyze and visualize the distribution of subjects for questions that meet the rating criteria.
    
    Args:
        annotations: List of annotation dictionaries
        question_id_to_subject: Mapping from question ID to subject
        save_dir: Directory to save plots
        base_name: Base name for output files
        unique_rating_filter: Range for filtering by rating_multians (min, max)
        specific_rating_filter: Range for filtering by rating_osq (min, max)
    """
    # Filter annotations based on rating criteria
    filtered_annotations = []
    for annot in annotations:
        question_id = annot.get("question_id")
        
        # Skip if we don't have the question ID
        if not question_id:
            continue
        
        # Check uniqueness rating (rating_multians)
        rating_multians = annot.get("rating_multians")
        if rating_multians is not None:
            if not (unique_rating_filter[0] <= rating_multians <= unique_rating_filter[1]):
                continue
        
        # Check specific rating (rating_osq)
        rating_osq = annot.get("rating_osq")
        if rating_osq is not None:
            if not (specific_rating_filter[0] <= rating_osq <= specific_rating_filter[1]):
                continue
        
        # Add subject information if available
        if question_id in question_id_to_subject:
            annot["subject"] = question_id_to_subject[question_id]
            filtered_annotations.append(annot)
    
    print(f"Found {len(filtered_annotations)} questions that meet the rating criteria")
    
    # If no filtered annotations, return
    if not filtered_annotations:
        print("No questions meet the filtering criteria. Skipping subject distribution analysis.")
        return
    
    # Count questions by subject
    subject_counts = defaultdict(int)
    for annot in filtered_annotations:
        subject = annot.get("subject", "Unknown")
        subject_counts[subject] += 1
    
    # Create DataFrame for plotting
    subject_df = pd.DataFrame({
        "subject": list(subject_counts.keys()),
        "count": list(subject_counts.values())
    })
    
    # Sort by count in descending order
    subject_df = subject_df.sort_values("count", ascending=False)
    
    # Create subject distribution bar plot
    plt.figure(figsize=(14, 10))
    bars = sns.barplot(x="count", y="subject", data=subject_df)
    
    # Add count labels
    for i, bar in enumerate(bars.patches):
        width = bar.get_width()
        plt.text(width + 0.5, bar.get_y() + bar.get_height()/2,
                f'{width:.0f}', ha='left', va='center')
    
    # Add labels and title
    multians_filter = f"multians_{unique_rating_filter[0]}-{unique_rating_filter[1]}"
    osq_filter = f"osq_{specific_rating_filter[0]}-{specific_rating_filter[1]}"
    plt.title(f'Subject Distribution for Questions with Rating Filters: {multians_filter}, {osq_filter}')
    plt.xlabel('Count')
    plt.ylabel('Subject')
    plt.tight_layout()
    
    # Save plot
    save_path = os.path.join(save_dir, f"{base_name}_subject_distribution_{multians_filter}_{osq_filter}.jpg")
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f"Saved subject distribution plot to {save_path}")
    
    # Create and save summary statistics
    summary = subject_df.copy()
    summary["percentage"] = summary["count"] / summary["count"].sum() * 100
    summary_file = os.path.join(save_dir, f"{base_name}_subject_summary_{multians_filter}_{osq_filter}.csv")
    summary.to_csv(summary_file, index=False)
    print(f"Saved subject summary statistics to {summary_file}")

def main():
    parser = argparse.ArgumentParser(description='Analyze annotation data and generate rating distribution plots')
    parser.add_argument('--annotation_file', type=str, default='visualize_resps/annotation/saves/2907.jsonl',
                        help='Path to the annotation JSONL file')
    parser.add_argument('--save_dir', type=str, default='visualize_resps/plots',
                        help='Directory to save the generated plots')
    parser.add_argument('--dataset_name', type=str, default='MMLU-Pro',
                        choices=['MMLU', 'GPQA', 'MMLU-Pro'],
                        help='Name of the dataset to load for subject information')
    parser.add_argument('--dataset_split', type=str, default='test',
                        help='Dataset split to use')
    parser.add_argument('--unique_rating_filter', type=int, nargs=2, default=[1, 5],
                        help='Range for filtering by rating_multians (min, max)')
    parser.add_argument('--specific_rating_filter', type=int, nargs=2, default=[1, 5],
                        help='Range for filtering by rating_osq (min, max)')
    
    args = parser.parse_args()
    
    analyze_annotations(
        args.annotation_file, 
        args.save_dir,
        args.dataset_name,
        args.dataset_split,
        tuple(args.unique_rating_filter),
        tuple(args.specific_rating_filter)
    )

if __name__ == "__main__":
    main()
