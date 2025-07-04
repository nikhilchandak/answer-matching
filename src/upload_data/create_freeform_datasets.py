import os
import json
import pandas as pd
from datasets import Dataset, DatasetDict, load_dataset
from typing import Dict, List


def get_filtered_ids(dataset):
    """
    Get filtered IDs from annotation files. Only returns question IDs that have rating_osq >= 4 
    and rating_multians >= 4 in ALL .jsonl files in the directory.
    """
    annotations_dir = f"/fast/nchandak/qaevals/judge_outputs/alignment_plot/annotations/{dataset}/"
    
    if not os.path.exists(annotations_dir):
        print(f"Warning: Annotations directory not found: {annotations_dir}")
        return []
    
    # Find all .jsonl files in the directory
    jsonl_files = []
    for file in os.listdir(annotations_dir):
        if file.endswith('.jsonl'):
            jsonl_files.append(os.path.join(annotations_dir, file))
    
    print(f"Found {len(jsonl_files)} .jsonl files in {annotations_dir}")
    
    if not jsonl_files:
        print("Warning: No .jsonl files found in directory")
        return []
    
    # Dictionary to track ratings across files
    ratings_by_id = {}
    
    # Process each .jsonl file
    for file_path in jsonl_files:
        print(f"Processing file: {file_path}")
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
                        print(f"Warning: Failed to parse line {line_num} in {file_path}: {e}")
                        continue
                        
        except FileNotFoundError:
            print(f"Warning: File not found: {file_path}")
            continue
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
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
        
        if meets_criteria:
            filtered_ids.append(str(question_id))
    
    print(f"Found {len(filtered_ids)} question IDs with rating_osq >= 4 and rating_multians >= 4 in ALL files")
    return filtered_ids


def load_mmlu_pro_data_with_answers(filtered_ids: List[str]) -> Dict[str, Dict]:
    """
    Load MMLU Pro data from JSONL files and create a mapping of question_id to question/answer data.
    """
    # Paths to the JSONL files containing questions and answers
    mcq_file = "/fast/nchandak/qaevals/judge_outputs/alignment_plot/mcq/mmlu_pro/samples.jsonl"
    gen_file = "/fast/nchandak/qaevals/judge_outputs/alignment_plot/gen/mmlu_pro/samples.jsonl"
    
    question_data = {}
    
    print(f"Loading MCQ data from: {mcq_file}")
    # Load MCQ data (contains questions and options)
    try:
        with open(mcq_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    question_id = str(data.get('question_id'))
                    
                    if question_id in filtered_ids:
                        question_data[question_id] = {
                            'question': data.get('question', ''),
                            'options': data.get('options', []),
                            'answer': data.get('answer', ''),
                            'question_id': question_id
                        }
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        print(f"Error: MCQ file not found: {mcq_file}")
        return {}
    
    print(f"Loading answer data from: {gen_file}")
    # Load answer data (contains the actual text answers)
    try:
        with open(gen_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    question_id = str(data.get('question_id'))
                    
                    if question_id in question_data:
                        # Get the response/answer text
                        response = data.get('response', data.get('filtered_resps', ''))
                        if isinstance(response, list) and response:
                            response = response[0]
                        question_data[question_id]['answer_text'] = response
                        
                except json.JSONDecodeError:
                    continue
    except FileNotFoundError:
        print(f"Error: Answer file not found: {gen_file}")
        
    print(f"Loaded data for {len(question_data)} questions")
    return question_data


def get_mmlu_pro_categories():
    """
    Load the TIGER-Lab/MMLU-Pro dataset to get question_id to category mapping.
    """
    print("Loading TIGER-Lab/MMLU-Pro dataset for category mapping...")
    try:
        dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
        
        # Create mapping from question_id to category
        id_to_category = {}
        for item in dataset:
            question_id = str(item.get('question_id', ''))
            category = item.get('category', '')
            if question_id and category:
                id_to_category[question_id] = category
        
        print(f"Loaded categories for {len(id_to_category)} questions")
        return id_to_category
        
    except Exception as e:
        print(f"Error loading TIGER-Lab/MMLU-Pro dataset: {e}")
        return {}


def create_gpqa_diamond_split():
    """
    Create the GPQA diamond split using the filtered CSV data.
    """
    print("Creating GPQA diamond split...")
    
    # Load the filtered GPQA data
    gpqa_filtered_path = "gpqa-diamond-freeform-filtered.csv"
    if not os.path.exists(gpqa_filtered_path):
        print(f"Error: Filtered GPQA file not found: {gpqa_filtered_path}")
        return None
        
    gpqa_df = pd.read_csv(gpqa_filtered_path)
    print(f"Loaded {len(gpqa_df)} filtered GPQA questions")
    
    # Load the original GPQA dataset to get subdomain mapping
    print("Loading original GPQA dataset for subdomain mapping...")
    try:
        original_gpqa = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")
        
        # Create mapping from question to subdomain
        question_to_subdomain = {}
        for item in original_gpqa:
            question = item.get('Question', '')
            subdomain = item.get('Subdomain', '')
            if question and subdomain:
                question_to_subdomain[question] = subdomain
        
        print(f"Loaded subdomain mapping for {len(question_to_subdomain)} questions")
        
    except Exception as e:
        print(f"Error loading original GPQA dataset: {e}")
        question_to_subdomain = {}
    
    # Standardize column names and add category
    standardized_data = []
    for _, row in gpqa_df.iterrows():
        question = row.get('Question', '')
        subdomain = question_to_subdomain.get(question, 'unknown')
        
        standardized_row = {
            'question': question,
            'answer': row.get('Answer', ''),
            'question_id': row.get('Record ID', ''),
            'category': subdomain,
            'Canary String': row.get('Canary String', '')
        }
        standardized_data.append(standardized_row)
    
    # Convert to DataFrame and then to Dataset
    standardized_df = pd.DataFrame(standardized_data)
    gpqa_dataset = Dataset.from_pandas(standardized_df)
    return gpqa_dataset


def create_mmlu_pro_split():
    """
    Create the MMLU Pro split with question, answer, question_id, category, and empty canary_string.
    """
    print("Creating MMLU Pro split...")
    
    # Get filtered IDs for MMLU Pro
    filtered_ids = get_filtered_ids("mmlu_pro")
    if not filtered_ids:
        print("No filtered IDs found for MMLU Pro")
        return None
    
    # Load question and answer data from JSONL files
    question_data = load_mmlu_pro_data_with_answers(filtered_ids)
    if not question_data:
        print("No question data loaded for MMLU Pro")
        return None
    
    # Get category mapping from TIGER-Lab/MMLU-Pro
    id_to_category = get_mmlu_pro_categories()
    
    # Create the dataset
    dataset_rows = []
    for question_id, data in question_data.items():
        row = {
            'question': data.get('question', ''),
            'answer': data.get('answer_text', ''),
            'question_id': question_id,
            'category': id_to_category.get(question_id, 'unknown'),
            'Canary String': ''  # Empty canary string for MMLU Pro
        }
        dataset_rows.append(row)
    
    print(f"Created {len(dataset_rows)} MMLU Pro rows")
    
    # Save as CSV first
    mmlu_pro_df = pd.DataFrame(dataset_rows)
    csv_path = "mmlu_pro_filtered.csv"
    mmlu_pro_df.to_csv(csv_path, index=False)
    print(f"Saved MMLU Pro data to {csv_path}")
    
    # Convert to Hugging Face Dataset
    mmlu_pro_dataset = Dataset.from_pandas(mmlu_pro_df)
    return mmlu_pro_dataset


def main():
    """
    Main function to create the freeform datasets and push to hub.
    """
    print("Creating freeform datasets...")
    
    # Create both splits
    gpqa_dataset = create_gpqa_diamond_split()
    mmlu_pro_dataset = create_mmlu_pro_split()
    
    if gpqa_dataset is None or mmlu_pro_dataset is None:
        print("Error: Failed to create one or both datasets")
        return
    
    # Create DatasetDict with two splits
    dataset_dict = DatasetDict({
        "gpqa_diamond": gpqa_dataset,
        "mmlu_pro": mmlu_pro_dataset,
    })
    
    print(f"Dataset dict created with:")
    print(f"  - gpqa_diamond: {len(gpqa_dataset)} samples")
    print(f"  - mmlu_pro: {len(mmlu_pro_dataset)} samples")
    
    # Push to the hub
    print("Pushing to Hugging Face Hub: nikhilchandak/freeform-datasets")
    dataset_dict.push_to_hub("nikhilchandak/freeform-datasets")
    print("Successfully pushed datasets to hub!")
    
    # Push the README file
    print("Pushing README file...")
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        api.upload_file(
            path_or_fileobj="README.md",
            path_in_repo="README.md",
            repo_id="nikhilchandak/freeform-datasets",
            repo_type="dataset"
        )
        print("Successfully pushed README to hub!")
    except Exception as e:
        print(f"Error pushing README: {e}")
        print("You may need to manually upload the README.md file to the repository.")


if __name__ == "__main__":
    main() 