import json
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
import os
from datasets import load_dataset
import seaborn as sns

import scienceplots
plt.style.use('science')

def load_mmlu_pro_dataset():
    """Load the MMLU Pro dataset from Hugging Face"""
    print("Loading MMLU Pro dataset...")
    dataset = load_dataset("TIGER-Lab/MMLU-Pro")
    
    # Get the test split
    test_data = dataset['test']
    print(f"Loaded {len(test_data)} questions from MMLU Pro test set")
    
    return test_data

FIXED_ORDER = None 

def create_subject_distribution_plot(data, title="MMLU Pro Subject Distribution", save_path=None):
    global FIXED_ORDER
    """Create a pie chart of subject distribution"""
    # Count subjects
    subjects = [item['category'] for item in data]
    subject_counts = Counter(subjects)
    
    # Sort subjects by count in descending order to minimize text overlap
    sorted_subjects = sorted(subject_counts.items(), key=lambda x: x[1], reverse=True)
    subject_counts = Counter(dict(sorted_subjects))
    
    # Sort subjects by count in descending order
    sorted_subjects = sorted(subject_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Reorder subjects to minimize text overlap
    reordered_subjects = []
    left = 0
    right = len(sorted_subjects) - 1
    
    while left <= right:
        if left == right:
            reordered_subjects.append(sorted_subjects[left])
        else:
            reordered_subjects.append(sorted_subjects[left])
            reordered_subjects.append(sorted_subjects[right])
        left += 1
        right -= 1
    
    if FIXED_ORDER:
        reordered_subjects = {sub: subject_counts[sub] for sub in list(FIXED_ORDER.keys())}
    else :
        FIXED_ORDER = dict(reordered_subjects)
        
    # Create new Counter with reordered subjects
    subject_counts = Counter(dict(reordered_subjects))
    
    # Create pie chart using seaborn
    plt.figure(figsize=(14, 10))
    
    # Convert data to DataFrame for seaborn
    df = pd.DataFrame({
        'Subject': list(subject_counts.keys()),
        'Count': list(subject_counts.values())
    })
    
    # Create pie chart using seaborn
    sns.set_style("whitegrid")
    colors = sns.color_palette("Set3", n_colors=len(subject_counts))
    
    plt.pie(
        df['Count'],
        labels=df['Subject'],
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        textprops={'fontsize': 40, 'color': 'black'},
        pctdistance=0.85
    )
    
    # Improve text readability
    for text in plt.gca().texts:
        text.set_color('black')
        text.set_fontweight('bold')
        text.set_fontsize(18)
    
    plt.axis('equal')
    
    if save_path:
        # Create plots directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    # Print statistics
    print(f"\nSubject distribution ({len(subjects)} total questions):")
    for subject, count in sorted(subject_counts.items()):
        percentage = (count / len(subjects)) * 100
        print(f"{subject}: {count} ({percentage:.1f}%)")
    
    return subject_counts

def load_annotation_files(directory_path):
    """Load all JSONL files from the annotations directory, organized by file"""
    print(f"\nLoading annotation files from {directory_path}...")
    
    file_annotations = {}
    jsonl_files = [f for f in os.listdir(directory_path) if f.endswith('.jsonl')]
    
    for filename in jsonl_files:
        filepath = os.path.join(directory_path, filename)
        print(f"Loading {filename}...")
        
        file_annotations[filename] = []
        with open(filepath, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    file_annotations[filename].append(data)
                except json.JSONDecodeError as e:
                    print(f"Error parsing line in {filename}: {e}")
                    continue
        
        print(f"  Loaded {len(file_annotations[filename])} annotations from {filename}")
    
    print(f"Loaded annotations from {len(jsonl_files)} files")
    return file_annotations

def filter_high_quality_questions(file_annotations):
    """Filter questions with rating_osq and rating_multians >= 4 in ALL files"""
    print("\nFiltering questions with high ratings in ALL files...")
    
    # First, get high-quality question IDs from each file
    file_high_quality_ids = {}
    
    for filename, annotations in file_annotations.items():
        high_quality_ids = set()
        
        for annotation in annotations:
            # Check if both ratings exist and are >= 4
            rating_osq = annotation.get('rating_osq')
            rating_multians = annotation.get('rating_multians')
            
            if (rating_osq is not None and rating_multians is not None and 
                rating_osq >= 4 and rating_multians >= 4):
                
                question_id = int(annotation.get('question_id'))
                if question_id is not None:
                    high_quality_ids.add(question_id)
        
        file_high_quality_ids[filename] = high_quality_ids
        print(f"  {filename}: {len(high_quality_ids)} high-quality questions")
    
    # Find intersection - questions that appear in ALL files with high ratings
    if file_high_quality_ids:
        filtered_ids = set.intersection(*file_high_quality_ids.values())
        print(f"\nFound {len(filtered_ids)} questions with both ratings >= 4 in ALL {len(file_annotations)} files")
        
        # Print breakdown
        print("Breakdown by file:")
        for filename, ids in file_high_quality_ids.items():
            print(f"  {filename}: {len(ids)} high-quality questions")
        print(f"  Intersection (ALL files): {len(filtered_ids)} questions")
        
    else:
        filtered_ids = set()
        print("No annotation files found")
    
    return filtered_ids

def create_filtered_subject_distribution(mmlu_data, filtered_ids):
    """Create subject distribution for filtered questions"""
    print("\nCreating subject distribution for filtered questions...")
    
    # Create a mapping from question_id to subject
    id_to_subject = {}
    for item in mmlu_data:
        question_id = item.get('question_id')
        if question_id is not None:
            id_to_subject[question_id] = item['category']
    
    # Get subjects for filtered questions
    filtered_subjects = []
    missing_ids = []
    
    for question_id in filtered_ids:
        if question_id in id_to_subject:
            filtered_subjects.append(id_to_subject[question_id])
        else:
            missing_ids.append(question_id)
    
    if missing_ids:
        print(f"Warning: {len(missing_ids)} question IDs not found in MMLU Pro dataset")
    
    print(f"Creating pie chart for {len(filtered_subjects)} filtered questions")
    
    if filtered_subjects:
        # Create the filtered data structure for plotting
        filtered_data = [{'category': subject} for subject in filtered_subjects]
        subject_counts = create_subject_distribution_plot(
            filtered_data, 
            title="MMLU Pro Filtered Subject Distribution",
            save_path="plots/mmlu_pro_filtered_subject_distribution.pdf"
        )
        return subject_counts
    else:
        print("No filtered subjects found to plot")
        return {}

def main():
    """Main function to execute the analysis"""
    print("Starting MMLU Pro subject distribution analysis...")
    
    # 1. Load MMLU Pro dataset
    mmlu_data = load_mmlu_pro_dataset()
    
    # 3. Load annotation files
    annotation_dir = "/fast/nchandak/qaevals/judge_outputs/alignment_plot/annotations/mmlu_pro"
    file_annotations = load_annotation_files(annotation_dir)
    
    # 4. Filter high-quality questions (must have ratings >= 4 in ALL files)
    filtered_ids = filter_high_quality_questions(file_annotations)
    
    # 5. Create filtered subject distribution plot
    print("\n" + "="*50)
    print("FILTERED SUBJECT DISTRIBUTION (HIGH QUALITY)")
    print("="*50)
    filtered_counts = create_filtered_subject_distribution(mmlu_data, filtered_ids)
    
    # 2. Create overall subject distribution plot
    print("\n" + "="*50)
    print("OVERALL SUBJECT DISTRIBUTION")
    print("="*50)
    overall_counts = create_subject_distribution_plot(
        mmlu_data, 
        title="MMLU Pro Overall Subject Distribution",
        save_path="plots/mmlu_pro_overall_subject_distribution.pdf"
    )
    
    # 6. Compare distributions
    if filtered_counts and overall_counts:
        print("\n" + "="*50)
        print("COMPARISON SUMMARY")
        print("="*50)
        print(f"Total questions in MMLU Pro: {sum(overall_counts.values())}")
        print(f"High-quality questions (ratings ≥ 4): {sum(filtered_counts.values())}")
        print(f"Percentage of high-quality questions: {(sum(filtered_counts.values()) / sum(overall_counts.values())) * 100:.1f}%")
        
        print("\nSubject-wise comparison:")
        all_subjects = set(overall_counts.keys()) | set(filtered_counts.keys())
        for subject in sorted(all_subjects):
            overall_count = overall_counts.get(subject, 0)
            filtered_count = filtered_counts.get(subject, 0)
            if overall_count > 0:
                retention_rate = (filtered_count / overall_count) * 100
                print(f"{subject}: {filtered_count}/{overall_count} ({retention_rate:.1f}% retention)")

if __name__ == "__main__":
    main()
