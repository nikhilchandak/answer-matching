import json
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import defaultdict
from load_datasets import load_mmlu_pro, load_dataset_by_name

def load_jsonl_data(input_jsonl):
    """Load data from JSONL file with question hash as keys"""
    data = {}
    with open(input_jsonl, 'r') as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                # Each line has a single key-value pair
                q_hash = list(entry.keys())[0]
                data[q_hash] = entry[q_hash]
    return data

def main():
    parser = argparse.ArgumentParser(
        description="Create violin plot of llm_judge_unique scores by subject"
    )
    parser.add_argument(
        "--input_jsonl",
        type=str,
        default="/is/cluster/fast/nchandak/qaevals/filter/mmlupro_U/MMLU-Pro_question_hash.jsonl",
        help="Path to the input JSONL file with scores"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="plots",
        help="Directory to save the output plot"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="MMLU-Pro",
        choices=["MMLU", "GPQA", "MMLU-Pro"],
        help="Name of the dataset to process"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to use"
    )
    parser.add_argument(
        "--subset",
        type=str,
        default=None,
        help="Dataset subset if applicable"
    )
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_path, exist_ok=True)
    
    # Load the scored data
    print(f"Loading scored data from {args.input_jsonl}")
    scored_data = load_jsonl_data(args.input_jsonl)
    print(f"Loaded {len(scored_data)} scored questions")
    
    # Load the original dataset to get subject information
    print(f"Loading {args.dataset_name} dataset")
    dataset_items = load_dataset_by_name(
        name=args.dataset_name,
        split=args.split,
        subset=args.subset
    )
    print(f"Loaded {len(dataset_items)} questions from {args.dataset_name}")
    
    # Create a mapping from question hash to subject
    hash_to_subject = {}
    for item in dataset_items:
        q_hash = item["q_hash"]
        subject = item.get("subject", "Unknown")
        hash_to_subject[q_hash] = subject
    
    # Prepare data for plotting
    plot_data = []
    for q_hash, item in scored_data.items():
        if "llm_judge_unique" in item:
            subject = hash_to_subject.get(q_hash, "Unknown")
            plot_data.append({
                "subject": subject,
                "unique_score": item["llm_judge_unique"]
            })
    
    # Convert to DataFrame
    df = pd.DataFrame(plot_data)
    
    # Count questions per subject
    subject_counts = df['subject'].value_counts()
    print(f"Number of questions per subject:")
    for subject, count in subject_counts.items():
        print(f"  {subject}: {count}")
    
    # Create the violin plot
    plt.figure(figsize=(14, 10))
    ax = sns.violinplot(x="subject", y="unique_score", data=df, inner="box")
    plt.title(f"Distribution of Uniqueness Scores by Subject in {args.dataset_name}")
    plt.xlabel("Subject")
    plt.ylabel("Uniqueness Score (1-10)")
    plt.xticks(rotation=90)
    plt.tight_layout()
    
    # Add mean values as text
    for i, subject in enumerate(ax.get_xticklabels()):
        subject_name = subject.get_text()
        mean_score = df[df['subject'] == subject_name]['unique_score'].mean()
        ax.text(i, 10.5, f"{mean_score:.2f}", ha='center', va='bottom', fontweight='bold')
    
    # Save the plot
    output_file = os.path.join(args.output_path, f"{args.dataset_name.lower()}_uniqueness_by_subject.png")
    plt.savefig(output_file, dpi=300)
    print(f"Plot saved to {output_file}")
    
    # Also create a summary table
    summary = df.groupby('subject')['unique_score'].agg(['mean', 'std', 'count']).sort_values('mean', ascending=False)
    summary_file = os.path.join(args.output_path, f"{args.dataset_name.lower()}_uniqueness_summary.csv")
    summary.to_csv(summary_file)
    print(f"Summary statistics saved to {summary_file}")

if __name__ == "__main__":
    main()