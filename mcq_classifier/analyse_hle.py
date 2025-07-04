import json
import os
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset

def load_hle_dataset():
    """
    Load the Humanity Last Exam dataset from Hugging Face.
    """
    dataset = load_dataset("cais/hle", split="test")
    return dataset

def analyze_mcq_questions(data):
    """
    Analyze MCQ questions in the dataset.
    """
    mcq_count = 0
    option_counts = []
    subjects = []
    answer_types = []
    
    for item in data:
        if 'choices' in item and isinstance(item['choices'], list) and len(item['choices']) > 0:
            mcq_count += 1
            option_counts.append(len(item['choices']))
            if 'subject' in item:
                subjects.append(item['subject'])
            if 'answer_type' in item:
                answer_types.append(item['answer_type'])
    
    return {
        'total_mcq_questions': mcq_count,
        'option_counts': option_counts,
        'subjects': subjects,
        'answer_types': answer_types
    }

def plot_option_distribution(option_counts):
    """
    Plot the distribution of number of options per MCQ question.
    """
    counter = Counter(option_counts)
    options, frequencies = zip(*sorted(counter.items()))
    
    plt.figure(figsize=(10, 6))
    plt.bar(options, frequencies)
    plt.xlabel('Number of Options')
    plt.ylabel('Frequency')
    plt.title('Distribution of Options per MCQ Question')
    plt.xticks(options)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('option_distribution.png')
    plt.close()

def plot_subject_distribution(subjects):
    """
    Plot the distribution of subjects in MCQ questions.
    """
    counter = Counter(subjects)
    # Get the top 15 subjects
    top_subjects = dict(counter.most_common(15))
    
    plt.figure(figsize=(12, 8))
    plt.bar(top_subjects.keys(), top_subjects.values())
    plt.xlabel('Subject')
    plt.ylabel('Frequency')
    plt.title('Top 15 Subjects in MCQ Questions')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('subject_distribution.png')
    plt.close()

def plot_answer_type_distribution(answer_types):
    """
    Plot the distribution of answer types in MCQ questions.
    """
    counter = Counter(answer_types)
    types, frequencies = zip(*counter.most_common())
    
    plt.figure(figsize=(12, 8))
    plt.bar(types, frequencies)
    plt.xlabel('Answer Type')
    plt.ylabel('Frequency')
    plt.title('Distribution of Answer Types in MCQ Questions')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('answer_type_distribution.png')
    plt.close()

def main():
    print("Loading dataset from Hugging Face (cais/hle)...")
    data = load_hle_dataset()
    print(f"Dataset loaded with {len(data)} total questions.")
    
    # Just print the value count of the answer_type column
    print(Counter(data['answer_type']))
    
    analysis = analyze_mcq_questions(data)
    
    print(f"\nAnalysis Results:")
    print(f"Total MCQ questions: {analysis['total_mcq_questions']}")
    
    # if analysis['option_counts']:
    #     avg_options = np.mean(analysis['option_counts'])
    #     print(f"Average number of options per MCQ: {avg_options:.2f}")
    #     print(f"Min number of options: {min(analysis['option_counts'])}")
    #     print(f"Max number of options: {max(analysis['option_counts'])}")
        
    #     # Plot option distribution
    #     plot_option_distribution(analysis['option_counts'])
    #     print("Option distribution plot saved as 'option_distribution.png'")
        
    #     # Plot subject distribution if subjects are available
    #     if analysis['subjects']:
    #         plot_subject_distribution(analysis['subjects'])
    #         print("Subject distribution plot saved as 'subject_distribution.png'")
        
    #     # Print answer type value counts
    #     if analysis['answer_types']:
    #         answer_type_counts = Counter(analysis['answer_types'])
    #         print("\nAnswer Type Value Counts:")
    #         for answer_type, count in answer_type_counts.most_common():
    #             print(f"{answer_type}: {count}")
            
    #         # Plot answer type distribution
    #         plot_answer_type_distribution(analysis['answer_types'])
    #         print("Answer type distribution plot saved as 'answer_type_distribution.png'")

if __name__ == "__main__":
    main()
