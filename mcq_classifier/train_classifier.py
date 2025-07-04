import pandas as pd
import numpy as np
from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding,
    TrainerCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import torch
import json
import wandb
from typing import Dict, List
from accelerate import Accelerator
from tqdm import tqdm
import os
import glob

np.random.seed(42)
torch.manual_seed(42)


# Initialize accelerator
accelerator = Accelerator()
    

NUM_OPTIONS = 10


def load_math_mc(ratio=0.5):
    """Load MMLU-Pro dataset for multiple-choice classification with 10 options."""
    # Load MMLU-Pro dataset
    dataset = load_dataset("nikhilchandak/MATH_mc")
    
    # Filter dataset to only keep Level 5 problems
    dataset = dataset.filter(lambda row: row["Level"] == "Level 5")
    print(f"Filtered MATH_mc dataset to Level 5 problems: {len(dataset['test'])} items")
    
    # Get the test set and split it 50-50 for training and testing
    test_data = dataset["test"]
    
    # Shuffle the dataset with a fixed seed for reproducibility
    full_test_data = test_data.shuffle(seed=42)
    
    # Split into train and test
    split_idx = int(len(full_test_data) * ratio)
    train_data = full_test_data.select(range(split_idx))
    test_data = full_test_data.select(range(split_idx, len(full_test_data)))
    
    # Format data for 4-way classification (A through D)
    def format_dataset(dataset):
        formatted_data = []
        for item in dataset:
            question = item['Question']
            options = [item["A"], item["B"], item["C"], item["D"]]
            
            # Get the correct answer and its index
            correct_answer_index = ord(item['Answer']) - ord('A')
            correct_answer = options.pop(correct_answer_index)
            
            # Randomly sample 3 distractors from remaining options
            remaining_options = options
            distractors = remaining_options
            
            # Create the 4 options with the correct answer included
            final_options = distractors + [correct_answer]
            
            # Shuffle the options
            shuffled_indices = np.random.permutation(len(final_options))
            shuffled_options = [final_options[i] for i in shuffled_indices]
            
            last_index = len(shuffled_indices) - 1
            # Find the new index of the correct answer
            new_correct_index = np.where(shuffled_indices == last_index)[0][0]  # 3 is the index of correct_answer in final_options
            
            # Create prompt with question and options
            # prompt = f"{question}\n"
            prompt = ""
            for i in range(len(shuffled_options)):
                prompt += f"{chr(65 + i)}. {shuffled_options[i]}\n"
            prompt = prompt.strip()  # Remove trailing newline
            
            formatted_data.append({
                'text': prompt,
                'label': new_correct_index
            })
        
        return Dataset.from_list(formatted_data)
    
    train_dataset = format_dataset(train_data)
    test_dataset = format_dataset(test_data)
    
    # print length of train and test datasets
    print(f"Train dataset length: {len(train_dataset)}")
    print(f"Test dataset length: {len(test_dataset)}")
    
    return train_dataset, test_dataset


def load_mmlu_pro_data(ratio=0.5):
    """Load MMLU-Pro dataset for multiple-choice classification with 10 options."""
    # Load MMLU-Pro dataset
    dataset = load_dataset("TIGER-Lab/MMLU-Pro")
    
    # Get the test set and split it 50-50 for training and testing
    test_data = dataset["test"]
    
    # Shuffle the dataset with a fixed seed for reproducibility
    test_data = test_data.shuffle(seed=42)
    
    # Split into train and test
    split_idx = int(len(test_data) * ratio)
    train_data = test_data.select(range(split_idx))
    test_data = test_data.select(range(split_idx, len(test_data)))
    correct_indices = {}
    # Format data for 4-way classification (A through D)
    def format_dataset(dataset):
        formatted_data = []
        for item in dataset:
            question = item['question']
            options = item['options'].copy()
            
            # Skip if we don't have enough options
            if len(options) < 10:
                print(f"Not enough options, got {len(options)}")
                continue
            
            # Get the correct answer and its index
            correct_answer_index = item['answer_index']
            correct_answer = options.pop(correct_answer_index)
            
            # Randomly sample distractors from remaining options
            remaining_options = options
            if len(remaining_options) > NUM_OPTIONS - 1:
                # Randomly select 3 distractors
                distractor_indices = np.random.choice(len(remaining_options), NUM_OPTIONS - 1, replace=False)
                distractors = [remaining_options[i] for i in distractor_indices]
            else:
                # Use all remaining options if we have exactly 3 left
                distractors = remaining_options
            
            # Create the 4 options with the correct answer included
            final_options = distractors + [correct_answer]
            
            # Shuffle the options
            shuffled_indices = np.random.permutation(len(final_options))
            shuffled_options = [final_options[i] for i in shuffled_indices]
            
            # Find the new index of the correct answer
            new_correct_index = np.where(shuffled_indices == len(final_options) - 1)[0][0]  # 3 is the index of correct_answer in final_options
            
            if new_correct_index not in correct_indices:
                correct_indices[new_correct_index] = 0
            correct_indices[new_correct_index] += 1
            
            # Create prompt with question and options
            # prompt = f"{question}\n"
            prompt = ""
            for i in range(len(final_options)):
                prompt += f"{chr(65 + i)}. {shuffled_options[i]}\n"
            prompt = prompt.strip()  # Remove trailing newline
            
            formatted_data.append({
                # 'options': options,
                # 'question_id': item['question_id'],
                'text': prompt,
                'label': new_correct_index
            })
        
        return Dataset.from_list(formatted_data)
    
    train_dataset = format_dataset(train_data)
    test_dataset = format_dataset(test_data)
    
    # print length of train and test datasets
    print(f"Train dataset length: {len(train_dataset)}")
    print(f"Test dataset length: {len(test_dataset)}")
    print(f"Correct indices distribution: {correct_indices}")
    
    return train_dataset, test_dataset


def load_super_gqpa_data(ratio=0.5):
    """Load SuperGOPQA dataset for multiple-choice classification with at most 10 options."""
    # Load SuperGOPQA dataset
    dataset = load_dataset("m-a-p/SuperGPQA")
    
    # Get the test set and split it 50-50 for training and testing
    test_data = dataset["train"]
    
    # Shuffle the dataset with a fixed seed for reproducibility
    test_data = test_data.shuffle(seed=42)
    
    # Split into train and test
    split_idx = int(len(test_data) * ratio)
    train_data = test_data.select(range(split_idx))
    test_data = test_data.select(range(split_idx, len(test_data)))
    correct_indices = {}
    # Format data for 4-way classification (A through D)
    def format_dataset(dataset):
        formatted_data = []
        for item in dataset:
            question = item['question']
            options = item['options'].copy()
            
            # Skip if we don't have enough options
            # if len(options) < 4:
            #     print(f"Not enough options, got {len(options)}")
            #     continue
            
            # Get the correct answer and its index
            correct_answer_index = ord(item['answer_letter']) - ord('A')
            correct_answer = options.pop(correct_answer_index)
            
            # Randomly sample 3 distractors from remaining options
            remaining_options = options
            if len(remaining_options) > NUM_OPTIONS - 1:
                # Randomly select 3 distractors
                distractor_indices = np.random.choice(len(remaining_options), NUM_OPTIONS - 1, replace=False)
                distractors = [remaining_options[i] for i in distractor_indices]
            else:
                # Use all remaining options if we have exactly 3 left
                distractors = remaining_options
            
            # Create the 4 options with the correct answer included
            final_options = distractors + [correct_answer]
            
            # Shuffle the options
            shuffled_indices = np.random.permutation(len(final_options))
            shuffled_options = [final_options[i] for i in shuffled_indices]
            
            # Find the new index of the correct answer
            new_correct_index = np.where(shuffled_indices == len(final_options) - 1)[0][0]  # 3 is the index of correct_answer in final_options
            
            if new_correct_index not in correct_indices:
                correct_indices[new_correct_index] = 0
            correct_indices[new_correct_index] += 1
            
            # Create prompt with question and options
            # prompt = f"{question}\n"
            prompt = ""
            for i in range(len(final_options)):
                prompt += f"{chr(65 + i)}. {shuffled_options[i]}\n"
            prompt = prompt.strip()  # Remove trailing newline
            
            formatted_data.append({
                'text': prompt,
                'label': new_correct_index
            })
        
        return Dataset.from_list(formatted_data)
    
    train_dataset = format_dataset(train_data)
    test_dataset = format_dataset(test_data)
    
    # print length of train and test datasets
    print(f"Train dataset length SUPER_GPQA: {len(train_dataset)}")
    print(f"Test dataset length SUPER_GPQA: {len(test_dataset)}")
    print(f"Correct indices distribution SUPER_GPQA: {correct_indices}")
    
    return train_dataset, test_dataset


def load_mmlu_data(split="train"):
    """Load MMLU dataset for multiple-choice classification."""
    if split == "train":
        dataset = load_dataset("cais/mmlu", "all", split="auxiliary_train")
    elif split == "validation":
        dataset = load_dataset("cais/mmlu", "all", split="validation")
    elif split == "test":
        dataset = load_dataset("cais/mmlu", "all", split="test")
    
    correct_indices = {}
    # Format data for 4-way classification (A, B, C, D)
    formatted_data = []
    if split == "train":
        incorrect_options = [] # {sub:[] for sub in validation_dataset['subject']}
        for item in dataset:
            options = item['choices']
            options.pop(item['answer'])
            incorrect_options.extend(options)
            
        # Shuffle incorrect options
        np.random.shuffle(incorrect_options)
        
        incorrect_options = list(set(incorrect_options))
        incorrect_options = incorrect_options[:10000]
        # for sub in incorrect_options:
        #     print(f"Subject: {sub}, Number of incorrect options: {len(incorrect_options[sub])}")
            
    for item in tqdm(dataset):
        question = item['question']
        choices = [item['choices'][i] for i in range(4)]
        correct_answer_idx = item['answer']
        
        if split != "train":
            options = choices.copy()
            # prompt = f"{question}\n"
            
            prompt = ""
            for i in range(4):
                prompt += f"{chr(65 + i)}. {options[i]}\n"
            prompt = prompt.strip()  # Remove trailing newline
            
            formatted_data.append({
                'text': prompt,
                'label': item['answer']  # This should be 0, 1, 2, or 3 for A, B, C, D
            })
        else:
            options = choices.copy()
            extra_options = np.random.choice(incorrect_options, size=NUM_OPTIONS - 4, replace=False)
            options.extend(extra_options)
        
            # Shuffle options
            np.random.shuffle(options)
            
            # Get the correct answer and its index
            correct_answer_index = item['answer']
            correct_answer = options.pop(correct_answer_index)
        
            distractors = options
            # Create the 4 options with the correct answer included
            final_options = distractors + [correct_answer]
            
            # Shuffle the options
            shuffled_indices = np.random.permutation(len(final_options))
            shuffled_options = [final_options[i] for i in shuffled_indices]
            
            # Find the new index of the correct answer
            new_correct_index = np.where(shuffled_indices == len(final_options) - 1)[0][0]  # 3 is the index of correct_answer in final_options
            
            if new_correct_index not in correct_indices:
                correct_indices[new_correct_index] = 0
            correct_indices[new_correct_index] += 1
            
            # Create prompt with question and options
            # prompt = f"{question}\n"
            prompt = ""
            for i in range(len(final_options)):
                prompt += f"{chr(65 + i)}. {shuffled_options[i]}\n"
            prompt = prompt.strip()  # Remove trailing newline
            
            formatted_data.append({
                'text': prompt,
                'label': new_correct_index
            })
            
    print(f"Correct indices distribution MMLU {split}: {correct_indices}")
    
    return Dataset.from_list(formatted_data)


def load_gqpa_data():
    """Load GQPA dataset for multiple-choice classification."""
    dataset = load_dataset("Idavidrein/gpqa", "gpqa_main")["train"]
    
    # Shuffle the dataset with a fixed seed for reproducibility
    dataset = dataset.shuffle(seed=42)
    
    # Format data for 4-way classification (A, B, C, D)
    formatted_data = []
    # print(dataset.column_names)
    distribution = {0:0, 1:0, 2:0, 3:0}
    for item in dataset:
        # print(item)
        question = item['Question']
        options = [item['Incorrect Answer 1'], item['Incorrect Answer 2'], item['Incorrect Answer 3'], item['Correct Answer']]
        
        # Randomly shuffle the options np.shuffle(options)
        np.random.shuffle(options)
        
        answer = item['Correct Answer']
        answer_index = options.index(answer)
        distribution[answer_index] += 1
        # Create prompt with question and options
        prompt = f"{question}\n"
        # prompt = ""
        for i, choice_letter in enumerate(["A", "B", "C", "D"]):
            prompt += f"{choice_letter}. {options[i]}\n"
        prompt = prompt.strip()  # Remove trailing newline
        
        # print(question)
        # print(prompt)
        # print(answer_index)
        # print(answer)
        # print("--------------------------------")
        
        formatted_data.append({
            'text': prompt,
            'label': answer_index
        })
    
    print("GQPA answer distribution: ", distribution)
    return Dataset.from_list(formatted_data)

def load_hle():
    """Load Humanity's Last Exam dataset for multiple-choice classification."""

    dataset = load_dataset("cais/hle", split="test")
    
    # Only keep rows with answer_type == "multipleChoice"
    dataset = dataset.filter(lambda x: x["answer_type"] == "multipleChoice")
    
    formatted_data = []
    correct_indices = {}
    
    for item in dataset:
        question_text = item['question']
        answer = item['answer']
        
        # Extract options from the question text
        # Find where the answer choices begin
        if "Answer Choices:" in question_text:
            # Split the question to get the part with answer choices
            parts = question_text.split("Answer Choices:")
            only_question = parts[0].strip()
            options_text = parts[1].strip()
            
            # Extract options (assuming they're formatted as A. Option, B. Option, etc.)
            options = []
            for line in options_text.split('\n'):
                if line.strip() and line[0].isalpha() and line[1] == '.':
                    option_text = line[2:].strip()
                    options.append(option_text)
            
            # Skip if we don't have enough options
            # if len(options) < 4:
            #     print(f"Not enough options, got {len(options)}")
            #     continue
            
            # Get the correct answer (assuming it's in format like "A" or "B")
            correct_answer_index = ord(answer.upper()) - ord('A')
            if correct_answer_index < 0 or correct_answer_index >= len(options):
                print(f"Invalid answer index: {answer}")
                continue
                
            correct_answer = options.pop(correct_answer_index)
            
            # Randomly sample 3 distractors from remaining options
            remaining_options = options
            if len(remaining_options) > NUM_OPTIONS - 1:
                # Randomly select 3 distractors
                distractor_indices = np.random.choice(len(remaining_options), NUM_OPTIONS - 1, replace=False)
                distractors = [remaining_options[i] for i in distractor_indices]
            else:
                # Use all remaining options if we have exactly 3 left
                distractors = remaining_options
            
            # Create the 4 options with the correct answer included
            final_options = distractors + [correct_answer]
            
            # Shuffle the options
            shuffled_indices = np.random.permutation(len(final_options))
            shuffled_options = [final_options[i] for i in shuffled_indices]
            
            last_index = len(final_options) - 1
            # Find the new index of the correct answer
            new_correct_index = np.where(shuffled_indices == last_index)[0][0]  # 3 is the index of correct_answer in final_options
            
            if new_correct_index not in correct_indices:
                correct_indices[new_correct_index] = 0
            correct_indices[new_correct_index] += 1
            
            # Create prompt with options only (no question)
            prompt = f"{only_question}\n"
            prompt += "Answer Choices:\n"
            # prompt = ""
            for i in range(len(final_options)):
                prompt += f"{chr(65 + i)}. {shuffled_options[i]}\n"
            prompt = prompt.strip()  # Remove trailing newline
            
            # print("Question: ", question_text)
            # print("Prompt: ", prompt)
            # print("Answer index: ", new_correct_index)
            # print("Answer: ", answer)
            # print("--------------------------------")
            
            formatted_data.append({
                'text': prompt,
                'label': new_correct_index
            })
        else :
            assert False, "Question does not contain answer choices"
    
    print(f"HLE dataset length: {len(formatted_data)}")
    print(f"HLE correct indices distribution: {correct_indices}")
    
    return Dataset.from_list(formatted_data)


def load_yourbench_data(ratio=0.5):
    """Load YourBench dataset for multiple-choice classification."""
    # Define all subjects to include
    subjects = [
        "reproduction_nutrition", 
        "reproduction_anatomy", 
        "contemporary_virology",
        "reproduction_international_law",
        "contemporary_world_religions",
        # "contemporary_social_science"
    ]
    
    # Load and combine datasets
    datasets = []
    for subject in subjects:
        dataset_name = f"sumuks/yourbench_mmlu_{subject}"
        # if subject == "nutrition":
        #     dataset_name = "sumuks/yourbench_mmlu_reproduction_nutrition"
        # elif subject == "international_law":
        #     dataset_name = "sumuks/yourbench_mmlu_reproduction_international_law"
        # elif subject == "contemporary_world_religions":
        #     dataset_name = "sumuks/yourbench_mmlu_contemporary_world_religions"
        # elif subject == "contemporary_social_science":
        #     dataset_name = "sumuks/yourbench_mmlu_contemporary_social_science"
        # else:
        #     dataset_name = f"sumuks/yourbench_mmlu_reproduction_{subject}"
            
        try:
            subject_dataset = load_dataset(dataset_name, "lighteval", split="train")
            datasets.append(subject_dataset)
            print(f"Loaded {subject} dataset with {len(subject_dataset)} examples")
        except Exception as e:
            print(f"Error loading {subject} dataset: {e}")
    
    # Combine all datasets
    dataset = concatenate_datasets(datasets)
    # Shuffle the dataset with a fixed seed for reproducibility
    full_dataset = dataset.shuffle(seed=42)
    
    # Split into train and test
    split_idx = int(len(full_dataset) * ratio)
    train_data = full_dataset.select(range(split_idx))
    test_data = full_dataset.select(range(split_idx, len(full_dataset)))
    
    def format_dataset(dataset):
        formatted_data = []
        correct_indices = {}
        num_options = {x:0 for x in range(30)}
        
        for item in dataset:
            question_text = item['question']
            answer = item['ground_truth_answer'][1]
            
            # Extract options from the question text
            # Find where the answer choices begin
            if "Options:" in question_text or "(a)" in question_text:
                # Split the question to get the part with answer choices
                if "Options:" in question_text:
                    parts = question_text.split("Options:")
                    options_text = parts[1].strip()
                else:
                    a_idx = question_text.find("(a)")
                    parts = question_text[a_idx:]
                    options_text = parts.strip()
                    
                only_question = parts[0].strip()
                
                # Extract options (assuming they're formatted as A. Option, B. Option, etc.)
                options = []
                for i, line in enumerate(options_text.split('\n')):
                    # if line.strip() and f"({chr(97 + i)})" in line:
                    #     option_text = line[4:].strip()
                    #     options.append(option_text)
                    
                    now_char = chr(97 + i + 1)
                    pos = options_text.find(f'({now_char})')
                    current_char = chr(97 + i)
                    current_char_pos = options_text.find(f'({current_char})')
                    if pos != -1:
                        option_text = options_text[current_char_pos+3:pos].strip()
                        options.append(option_text)
                    else :
                        option_text = options_text[current_char_pos+3:].strip()
                        options.append(option_text)
                
                # Skip if we don't have enough options
                # if len(options) < 4:
                #     print(f"Not enough options, got {len(options)}")
                #     continue
                
                num_options[len(options)] += 1
                
                # if len(options) > NUM_OPTIONS:
                #     print(f"More than {NUM_OPTIONS} options, got {len(options)}")
                #     continue
                
                # Get the correct answer (assuming it's in format like "A" or "B")
                correct_answer_index = ord(answer.upper()) - ord('A')
                if correct_answer_index < 0 or correct_answer_index >= len(options):
                    print(f"Invalid answer index: {answer}")
                    print(f"Options: {options}")
                    print(f"Question: {question_text}")
                    print(f"Correct answer index: {correct_answer_index}")
                    continue
                    
                correct_answer = options.pop(correct_answer_index)
                
                # Randomly sample distractors from remaining options
                remaining_options = options
                if len(remaining_options) > NUM_OPTIONS - 1:
                    # Randomly select distractors
                    distractor_indices = np.random.choice(len(remaining_options), NUM_OPTIONS - 1, replace=False)
                    distractors = [remaining_options[i] for i in distractor_indices]
                else:
                    # Use all remaining options if we don't have enough
                    distractors = remaining_options
                
                # Create the options with the correct answer included
                final_options = distractors + [correct_answer]
                
                # Shuffle the options
                shuffled_indices = np.random.permutation(len(final_options))
                shuffled_options = [final_options[i] for i in shuffled_indices]
                
                last_index = len(final_options) - 1
                # Find the new index of the correct answer
                new_correct_index = np.where(shuffled_indices == last_index)[0][0]
                
                if new_correct_index not in correct_indices:
                    correct_indices[new_correct_index] = 0
                correct_indices[new_correct_index] += 1
                
                # Create prompt with options only (no question)
                # prompt = f"{only_question}\n"
                # prompt += "Answer Choices:\n"
                prompt = ""
                for i in range(len(final_options)):
                    prompt += f"{chr(65 + i)}. {shuffled_options[i]}\n"
                prompt = prompt.strip()  # Remove trailing newline
                
                formatted_data.append({
                    'text': prompt,
                    'label': new_correct_index
                })
            else:
                print(f"Question does not contain answer choices: {item['question']}")
                print(f"Ground truth answer: {item['ground_truth_answer']}")
                # assert False, "Question does not contain answer choices"
        
        return Dataset.from_list(formatted_data), correct_indices, num_options
    
    train_dataset, train_indices, train_num_options = format_dataset(train_data)
    test_dataset, test_indices, test_num_options = format_dataset(test_data)
    
    print(f"YourBench train dataset length: {len(train_dataset)}")
    print(f"YourBench test dataset length: {len(test_dataset)}")
    print(f"YourBench train correct indices distribution: {train_indices}")
    print(f"YourBench test correct indices distribution: {test_indices}")
    
    return train_dataset, test_dataset



def load_truthfulqa_data(ratio=0.8):
    """Load TruthfulQA dataset for binary classification between best answer and best incorrect answer."""
    try:
        # Load the TruthfulQA CSV file
        df = pd.read_csv('TruthfulQA.csv')
        
        # Shuffle the dataset with a fixed seed for reproducibility
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Format data for binary classification (0: Best Answer, 1: Best Incorrect Answer)
        formatted_data = []
        
        for _, row in df.iterrows():
            question = row['Question']
            best_answer = row['Best Answer']
            best_incorrect = row['Best Incorrect Answer']
            
            # Create options array with both answers
            options = [best_answer, best_incorrect]
            
            # Shuffle the options
            shuffled_indices = np.random.permutation(2)
            shuffled_options = [options[i] for i in shuffled_indices]
            
            # Find the new index of the best answer (0 for option A, 1 for option B)
            best_answer_index = np.where(shuffled_indices == 0)[0][0]
            
            # Create prompt with question and options in MCQ format
            # prompt = f"Question: {question}\n"
            prompt = ""
            prompt += f"A. {shuffled_options[0]}\n"
            prompt += f"B. {shuffled_options[1]}"
            
            formatted_data.append({
                'text': prompt,
                'label': best_answer_index  # 0 if best answer is option A, 1 if best answer is option B
            })
        
        # Create a dataset from the formatted data
        dataset = Dataset.from_list(formatted_data)
        
        # Split into train and test
        train_test_split = dataset.train_test_split(test_size=(1-ratio), seed=42)
        train_dataset = train_test_split['train']
        test_dataset = train_test_split['test']
        
        print(f"TruthfulQA train dataset length: {len(train_dataset)}")
        print(f"TruthfulQA test dataset length: {len(test_dataset)}")
        
        return train_dataset, test_dataset
    
    except Exception as e:
        print(f"Error loading TruthfulQA dataset: {e}")
        raise


def load_hellaswag_data(split="train", use_goldenswag=False):
    """Load MMLU dataset for multiple-choice classification."""
    if split == "train":
        dataset = load_dataset("Rowan/hellaswag", split="train")
    elif split == "validation":
        dataset = load_dataset("Rowan/hellaswag", split="validation")
    elif split == "test":
        dataset = load_dataset("Rowan/hellaswag", split="test")
    
    if use_goldenswag:
        dataset = load_dataset("PleIAs/GoldenSwag", split="validation")
    
    # Shuffle the dataset
    dataset = dataset.shuffle(seed=42)
    
    # Format data for 4-way classification (A, B, C, D)
    formatted_data = []
    for item in dataset:
        choices = [item['endings'][i] for i in range(4)]
        
        if len(item['label']) < 1:
            continue 
        
        # Create prompt with question and options
        # prompt = f"{question}\n"
        # prompt = ""
        # prompt += f"A. {choices[0]}\n"
        # prompt += f"B. {choices[1]}\n"
        # prompt += f"C. {choices[2]}\n"
        # prompt += f"D. {choices[3]}"
        
        # formatted_data.append({
        #     'text': prompt,
        #     'label': int(item['label'])  # This should be 0, 1, 2, or 3 for A, B, C, D
        # })
        
        
        # Create the 4 options with the correct answer included
        final_options = choices
        
        # Shuffle the options
        shuffled_indices = np.random.permutation(len(final_options))
        shuffled_options = [final_options[i] for i in shuffled_indices]
        
        # Find the new index of the correct answer
        new_correct_index = np.where(shuffled_indices == int(item['label']))[0][0]  # 3 is the index of correct_answer in final_options
        
        prompt = ""
        for i in range(len(final_options)):
            prompt += f"{chr(65 + i)}. {shuffled_options[i]}\n"
        prompt = prompt.strip()  # Remove trailing newline
        
        formatted_data.append({
            'text': prompt,
            'label': new_correct_index
        })
            
    
    return Dataset.from_list(formatted_data)



def load_arc_data(split="train"):
    """Load ARC dataset for multiple-choice classification."""
    if split == "train":
        dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")
    elif split == "validation":
        dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="validation")
    elif split == "test":
        dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    
    # Shuffle the dataset
    dataset = dataset.shuffle(seed=42)
    
    # Format data for 4-way classification (A, B, C, D)
    formatted_data = []
    for item in dataset:
        choices = item['choices']["text"]
        # print(choices)
        options = [choices[i] for i in range(len(choices))]
        # print(options)
        
        label = item['answerKey']
        # print(label, type(label))
        if label == "1" or label == "2" or label == "3" or label == "4" or label == "0":
            label = chr(65 + int(label) - 1)
        
        label = ord(label.upper()) - ord('A')
        # print(label)
        # Create prompt with question and options
        # prompt = f"{question}\n"
        # prompt = ""
        # prompt += f"A. {choices[0]}\n"
        # prompt += f"B. {choices[1]}\n"
        # prompt += f"C. {choices[2]}\n"
        # prompt += f"D. {choices[3]}"
        
        # formatted_data.append({
        #     'text': prompt,
        #     'label': int(item['label'])  # This should be 0, 1, 2, or 3 for A, B, C, D
        # })
        
        
        # Create the 4 options with the correct answer included
        final_options = options
        
        # Shuffle the options
        shuffled_indices = np.random.permutation(len(final_options))
        shuffled_options = [final_options[i] for i in shuffled_indices]
        
        # Find the new index of the correct answer
        new_correct_index = np.where(shuffled_indices == label)[0][0]  # 3 is the index of correct_answer in final_options
        
        prompt = ""
        for i in range(len(final_options)):
            prompt += f"{chr(65 + i)}. {shuffled_options[i]}\n"
        prompt = prompt.strip()  # Remove trailing newline
        
        formatted_data.append({
            'text': prompt,
            'label': new_correct_index
        })
            
    
    return Dataset.from_list(formatted_data)




def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train_model(
    train_dataset, 
    eval_dataset, 
    model_name: str,
    token_limit: int,
    output_dir: str
) -> Dict[str, float]:
    
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=NUM_OPTIONS,  # For A, B, C, D options
        problem_type="single_label_classification",
    )

    # Add padding token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # For Llama models, ensure pad_token_id is properly set
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    
    # Enable gradient checkpointing before the model is wrapped
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    
    if hasattr(model, 'classifier'):
        model.classifier.weight.data.normal_(mean=0.0, std=0.02)
        model.classifier.bias.data.zero_()
    
    if hasattr(model, 'score'):
        model.score.weight.data.normal_(mean=0.0, std=0.02)
        
    
    # # Freeze all layers except the classification head
    # for param in model.parameters():
    #     param.requires_grad = False
    
    # # # Unfreeze only the classification head
    # for param in model.score.parameters():
    #     param.requires_grad = True
    
    # # Print number of trainable parameters
    # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"Trainable parameters: {trainable_params} ({trainable_params/total_params:.2%} of total)")

    
    def tokenize_function(examples):
        # Ensure we're returning the expected keys for the model
        return tokenizer(
            examples['text'],
            padding="max_length",
            truncation=True,
            max_length=token_limit,
        )

    # Tokenize datasets
    tokenized_train = train_dataset.map(
        tokenize_function, 
        batched=True,
        remove_columns=['text']
    )
    tokenized_eval = eval_dataset.map(
        tokenize_function, 
        batched=True,
        remove_columns=['text']
    )
    
    # Create a custom data collator
    # data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=5e-6,
        per_device_train_batch_size=64, # 512 
        per_device_eval_batch_size=64, # 512 
        gradient_accumulation_steps=1,
        num_train_epochs=100,
        weight_decay=0.1,
        eval_strategy="steps",
        eval_steps=20,
        metric_for_best_model="accuracy",
        save_total_limit=0,  # Keep only the final model
        run_name="mmlu-mcq",
        warmup_ratio=0.1,
        logging_steps=10,
        report_to="wandb" if accelerator.is_main_process else "none",
        save_strategy="steps",
        save_steps=300,
        fp16=True,
        # gradient_checkpointing=True,
        lr_scheduler_type="constant_with_warmup",
        # remove_unused_columns=False,  # Add this line
    )
    
    # Initialize optimizer with different parameters based on model architecture
    if "llama" in model_name.lower() or "qwen" in model_name.lower():
        # For Llama or qwen models
        optimizer = torch.optim.AdamW(
            [
                {"params": [p for n, p in model.named_parameters() if "score" not in n], "lr": 5e-6},  # Lower LR for pretrained layers
                {"params": [p for n, p in model.named_parameters() if "score" in n], "lr": 5e-5}  # Higher LR for classification head
            ],
            weight_decay=0.1
        )
    else :
        # For DeBERTa models
        # Initialize optimizer with different parameters
        optimizer = torch.optim.AdamW(
            [
                {"params": model.deberta.parameters(), "lr": 5e-6},  # Lower LR for pretrained layers
                {"params": model.classifier.parameters(), "lr": 5e-5}  # Higher LR for classification head
            ],
            weight_decay=0.1
        )

    
    # Create a custom callback to stop training when training loss < 0.1
    class EarlyStoppingOnTrainingLoss(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and "loss" in logs and logs["loss"] < 0.001:
                print("\n\n*** Early stopping triggered: training loss < 0.01 ***\n\n")
                control.should_training_stop = True

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, None),  # Custom optimizer, default scheduler
        # callbacks=[EarlyStoppingOnTrainingLoss()]  # Add early stopping callback for training loss
    )
    
    # # Initialize trainer
    # trainer = Trainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=tokenized_train,
    #     eval_dataset=tokenized_eval,
    #     compute_metrics=compute_metrics,
    #     optimizers=(optimizer, None)  # Custom optimizer, default scheduler
    # )

    # Train the model
    trainer.train()
    
    # Evaluate on validation set
    eval_results = trainer.evaluate()
    
    
    # Save the model
    # if accelerator.is_main_process:
    #     trainer.save_model(f"{output_dir}/final_model")
    #     print(f"Model saved to {output_dir}/final_model")
    
    
    return eval_results

def main():
    global NUM_OPTIONS
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="hellaswag")
    parser.add_argument('--token_limit', type=int, default=512)
    args = parser.parse_args()
    
    RATIO = 0.5
    # Load MMLU datasets
    # train_dataset = load_mmlu_data(split="train")
    # val_dataset = load_mmlu_data(split="validation")
    # test_dataset = load_mmlu_data(split="test")
    
    # train_dataset = train_dataset.select(range(10000))
    
    # split MMLU test into train and test
    # test_dataset = test_dataset.train_test_split(test_size=0.1, seed=42)
    
    # train_dataset = test_dataset["train"]
    # test_dataset = test_dataset["test"]
    
    # dataset_name = "super_gqpa"
    # dataset_name = "yourbench"
    # dataset_name = "mmlu_pro"
    # dataset_name = "math_mc"
    # dataset_name = "arc"
    dataset_name = args.dataset
    token_limit = args.token_limit
    model_name = "microsoft/deberta-v3-large"
    
    # We will use the SuperGOPQA dataset actually
    if dataset_name == "super_gqpa":
        train_dataset, test_dataset = load_super_gqpa_data(ratio=RATIO)
        NUM_OPTIONS = 10
        token_limit = 2048
        model_name = "/fast/nchandak/models/Qwen3-4B"
        
    elif dataset_name == "mmlu_pro":
        train_dataset, test_dataset = load_mmlu_pro_data(ratio=RATIO)
        NUM_OPTIONS = 10
        token_limit = 2048
        model_name = "/fast/nchandak/models/Qwen3-4B"
        
    elif dataset_name == "mmlu":
        train_dataset, test_dataset = load_mmlu_data(split="train")
        NUM_OPTIONS = 4
        token_limit = 2048
        
    elif dataset_name == "hle":
        train_dataset, test_dataset = load_hle(ratio=RATIO) 
        NUM_OPTIONS = 10
        token_limit = 2048
        model_name = "/fast/nchandak/models/Qwen3-4B"
        
    elif dataset_name == "math_mc":
        NUM_OPTIONS = 4
        train_dataset, test_dataset = load_math_mc(ratio=0.5)
        model_name = "/fast/nchandak/models/Qwen3-4B"
        
    elif dataset_name == "yourbench":
        NUM_OPTIONS = 4
        train_dataset, test_dataset = load_yourbench_data(ratio=RATIO)
        model_name = "/fast/nchandak/models/Qwen3-4B"
        
    elif dataset_name == "truthfulqa":
        train_dataset, test_dataset = load_truthfulqa_data(ratio=RATIO)
        NUM_OPTIONS = 2
        
    elif dataset_name == "hellaswag":
        NUM_OPTIONS = 4
        train_dataset = load_hellaswag_data(split="train")
        test_dataset = load_hellaswag_data(split="validation")
        # test_dataset = load_hellaswag_data(split="test")
        
    elif dataset_name == "goldenswag":
        NUM_OPTIONS = 4
        train_dataset = load_hellaswag_data(split="train")
        test_dataset = load_hellaswag_data(split="validation", use_goldenswag=True)
        
    elif dataset_name == "arc":
        NUM_OPTIONS = 4
        train_dataset = load_arc_data(split="train")
        test_dataset = load_arc_data(split="test")
    else:
        assert False, "Invalid dataset name"

    # Use recent models
    # model_name = "meta-llama/Meta-Llama-3.1-8B"
    # model_name = "Qwen/Qwen2.5-7B-Instruct"
    # model_name = "/fast/nchandak/models/Qwen3-4B"
    
    short_model_name = model_name.split("/")[-1]
    rr = int(RATIO*100)
    output_dir = f"/fast/nchandak/classification/{dataset_name}/{short_model_name}/{rr}percent/"
    
    
    # Initialize wandb only on the main process
    if accelerator.is_main_process:
        wandb.init(project="mcq-classifier6", name=f"{dataset_name}-try-{short_model_name}-{NUM_OPTIONS}way-noq-{RATIO*100}percent")
    
    print(f"Output directory: {output_dir}")
    
    # make output dir if not exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Train on combined train and validation data, evaluate on test
    if accelerator.is_main_process:
        print("Training model on training data...")
        
    # # First train on training data and evaluate on validation data
    test_results = train_model(
        train_dataset,
        test_dataset,
        model_name,
        token_limit,
        output_dir=output_dir
    )
    test_results = None
    
    
    # Look for checkpoint directories
    checkpoint_dirs = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    
    if checkpoint_dirs:
        # Sort by checkpoint number (extract number from directory name)
        checkpoint_dirs.sort(key=lambda x: int(x.split("-")[-1]))
        # Get the last checkpoint
        last_checkpoint = checkpoint_dirs[-1]
        print(f"Found last checkpoint: {last_checkpoint}")
        # Update output_dir to point to the last checkpoint
        output_dir = last_checkpoint
        print(f"Updated output directory: {output_dir}")
    else:
        print("No checkpoints found in the output directory.")
    
    # Log test results
    if accelerator.is_main_process:
        wandb.log({
            "test_accuracy": test_results['eval_accuracy'],
            "test_f1": test_results['eval_f1'],
            "test_precision": test_results['eval_precision'],
            "test_recall": test_results['eval_recall']
        })
        
        print("\nTest performance:")
        print(test_results)
        
        wandb.finish()
        
        # print("\nTest performance:")
        # print(test_results)
        
        # # Load the saved model and tokenizer for prediction
        # print("Loading saved model for predictions...")
        # tokenizer = AutoTokenizer.from_pretrained(f"{output_dir}")
        # model = AutoModelForSequenceClassification.from_pretrained(f"{output_dir}")
        
        # # Tokenize test dataset
        # def tokenize_function(examples):
        #     return tokenizer(
        #         examples['text'],
        #         padding="max_length",
        #         truncation=True,
        #         max_length=1024
        #     )
        
        # tokenized_test = test_dataset.map(
        #     tokenize_function, 
        #     batched=True,
        #     remove_columns=['text']
        # )
        
        # # Create a simple prediction pipeline
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # model.to(device)
        # model.eval()
        
        # # Create DataLoader for test dataset
        # from torch.utils.data import DataLoader
        # test_dataloader = DataLoader(
        #     tokenized_test, 
        #     batch_size=32,
        #     collate_fn=DataCollatorWithPadding(tokenizer=tokenizer)
        # )
        
        # # Make predictions
        # all_preds = []
        # all_labels = []
        
        # with torch.no_grad():
        #     for batch in tqdm(test_dataloader, desc="Predicting"):
        #         batch = {k: v.to(device) for k, v in batch.items()}
        #         outputs = model(**batch)
        #         predictions = outputs.logits.argmax(dim=-1)
        #         all_preds.extend(predictions.cpu().numpy())
        #         all_labels.extend(batch["labels"].cpu().numpy())
        
        # # Calculate metrics
        # accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
        # print(f"Test accuracy: {accuracy:.4f}")
        
        # # Create output file with predictions
        # output_data = []
        # for i, (example, pred) in tqdm(enumerate(zip(test_dataset, all_preds)), total=len(test_dataset), desc="Processing predictions"):
        #     # Convert the example to a dict if it's not already
        #     if not isinstance(example, dict):
        #         example = dict(example)
            
        #     # Add prediction
        #     example_with_pred = example.copy()
        #     example_with_pred["prediction"] = int(pred)
        #     example_with_pred["correct"] = (int(pred) == example["label"])
        #     output_data.append(example_with_pred)
        
        # # Save to JSONL
        # output_file = f"{output_dir}/predictions.jsonl"
        # with open(output_file, 'w') as f:
        #     for item in tqdm(output_data, desc="Saving predictions"):
        #         f.write(json.dumps(item) + '\n')
        # print(f"Saved predictions to {output_file}")
        
        # wandb.finish()
    
    return test_results

if __name__ == "__main__":
    # Enable deterministic behavior for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    test_results = main()
    print("\nTest Performance:")
    print(test_results)