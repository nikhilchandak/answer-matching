import pandas as pd
import numpy as np
from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import torch
import json
import wandb
from typing import Dict, List
from accelerate import Accelerator

# Fix seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def load_mmlu_pro_data():
    """Load MMLU-Pro dataset for multiple-choice classification with 10 options."""
    # Load MMLU-Pro dataset
    dataset = load_dataset("TIGER-Lab/MMLU-Pro")
    
    # Get the test set and split it 50-50 for training and testing
    test_data = dataset["test"]
    
    # Shuffle the dataset with a fixed seed for reproducibility
    test_data = test_data.shuffle(seed=42)
    
    # Split into train and test
    split_idx = len(test_data) // 2
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
            if len(options) < 4:
                print(f"Not enough options, got {len(options)}")
                continue
            
            # Get the correct answer and its index
            correct_answer_index = item['answer_index']
            correct_answer = options.pop(correct_answer_index)
            
            # Randomly sample 3 distractors from remaining options
            remaining_options = options
            if len(remaining_options) > 3:
                # Randomly select 3 distractors
                distractor_indices = np.random.choice(len(remaining_options), 3, replace=False)
                distractors = [remaining_options[i] for i in distractor_indices]
            else:
                # Use all remaining options if we have exactly 3 left
                distractors = remaining_options
            
            # Create the 4 options with the correct answer included
            final_options = distractors + [correct_answer]
            
            # Shuffle the options
            shuffled_indices = np.random.permutation(4)
            shuffled_options = [final_options[i] for i in shuffled_indices]
            
            # Find the new index of the correct answer
            new_correct_index = np.where(shuffled_indices == 3)[0][0]  # 3 is the index of correct_answer in final_options
            
            if new_correct_index not in correct_indices:
                correct_indices[new_correct_index] = 0
            correct_indices[new_correct_index] += 1
            
            # Create prompt with question and options
            # prompt = f"{question}\n"
            prompt = ""
            for i, choice_letter in enumerate(["A", "B", "C", "D"]):
                prompt += f"{choice_letter}. {shuffled_options[i]}\n"
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
    print(f"Correct indices distribution: {correct_indices}")
    
    return train_dataset, test_dataset


def load_super_gqpa_data():
    """Load SuperGOPQA dataset for multiple-choice classification with at most 10 options."""
    # Load SuperGOPQA dataset
    dataset = load_dataset("m-a-p/SuperGPQA")
    
    # Get the test set and split it 50-50 for training and testing
    test_data = dataset["train"]
    
    # Shuffle the dataset with a fixed seed for reproducibility
    test_data = test_data.shuffle(seed=42)
    
    # Split into train and test
    split_idx = len(test_data) // 2
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
            if len(options) < 4:
                print(f"Not enough options, got {len(options)}")
                continue
            
            # Get the correct answer and its index
            correct_answer_index = ord(item['answer_letter']) - ord('A')
            correct_answer = options.pop(correct_answer_index)
            
            # Randomly sample 3 distractors from remaining options
            remaining_options = options
            if len(remaining_options) > 3:
                # Randomly select 3 distractors
                distractor_indices = np.random.choice(len(remaining_options), 3, replace=False)
                distractors = [remaining_options[i] for i in distractor_indices]
            else:
                # Use all remaining options if we have exactly 3 left
                distractors = remaining_options
            
            # Create the 4 options with the correct answer included
            final_options = distractors + [correct_answer]
            
            # Shuffle the options
            shuffled_indices = np.random.permutation(4)
            shuffled_options = [final_options[i] for i in shuffled_indices]
            
            # Find the new index of the correct answer
            new_correct_index = np.where(shuffled_indices == 3)[0][0]  # 3 is the index of correct_answer in final_options
            
            if new_correct_index not in correct_indices:
                correct_indices[new_correct_index] = 0
            correct_indices[new_correct_index] += 1
            
            # Create prompt with question and options
            # prompt = f"{question}\n"
            prompt = ""
            for i, choice_letter in enumerate(["A", "B", "C", "D"]):
                prompt += f"{choice_letter}. {shuffled_options[i]}\n"
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
    
    # Format data for 4-way classification (A, B, C, D)
    formatted_data = []
    for item in dataset:
        question = item['question']
        choices = [item['choices'][i] for i in range(4)]
        
        # Create prompt with question and options
        # prompt = f"{question}\n"
        prompt = ""
        prompt += f"A. {choices[0]}\n"
        prompt += f"B. {choices[1]}\n"
        prompt += f"C. {choices[2]}\n"
        prompt += f"D. {choices[3]}"
        
        formatted_data.append({
            'text': prompt,
            'label': item['answer']  # This should be 0, 1, 2, or 3 for A, B, C, D
        })
    
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
        # prompt = f"{question}\n"
        prompt = ""
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
    correct_indices = {0: 0, 1: 0, 2: 0, 3: 0}
    
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
            if len(remaining_options) > 3:
                # Randomly select 3 distractors
                distractor_indices = np.random.choice(len(remaining_options), 3, replace=False)
                distractors = [remaining_options[i] for i in distractor_indices]
            else:
                # Use all remaining options if we have exactly 3 left
                distractors = remaining_options
            
            # Create the 4 options with the correct answer included
            final_options = distractors + [correct_answer]
            
            # Shuffle the options
            shuffled_indices = np.random.permutation(4)
            shuffled_options = [final_options[i] for i in shuffled_indices]
            
            # Find the new index of the correct answer
            new_correct_index = np.where(shuffled_indices == 3)[0][0]  # 3 is the index of correct_answer in final_options
            
            correct_indices[new_correct_index] += 1
            
            # Create prompt with options only (no question)
            # prompt = f"{only_question}\n"
            # prompt += "Answer Choices:\n"
            prompt = ""
            for i, choice_letter in enumerate(["A", "B", "C", "D"]):
                prompt += f"{choice_letter}. {shuffled_options[i]}\n"
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

class MultiDatasetTrainer(Trainer):
    def __init__(self, eval_datasets=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_datasets = eval_datasets or {}
        
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        # First, run evaluation on the primary dataset
        results = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        # results = {}
        
        # Then evaluate on all additional datasets
        all_results = {metric_key_prefix: results}
        
        for dataset_name, dataset in self.eval_datasets.items():
            dataset_results = super().evaluate(dataset, ignore_keys, metric_key_prefix=dataset_name)
            all_results[dataset_name] = dataset_results
            
            # Update the main results dictionary with the dataset-specific metrics
            for key, value in dataset_results.items():
                results[key] = value
        
        return results

def train_model(
    train_dataset, 
    eval_datasets: Dict[str, Dataset], 
    model_name: str,
    output_dir: str
) -> Dict[str, float]:
    
    # Initialize accelerator
    accelerator = Accelerator()
    
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=4,  # For A through D options
        problem_type="single_label_classification"
    )

    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding="max_length",
            truncation=True,
            max_length=1024
        )

    # Reinitialize the classification head for better learning
    model.classifier.weight.data.normal_(mean=0.0, std=0.02)
    model.classifier.bias.data.zero_()
    
    # Tokenize train dataset
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    
    # Tokenize all evaluation datasets
    tokenized_eval_datasets = {}
    for name, dataset in eval_datasets.items():
        tokenized_eval_datasets[name] = dataset.map(tokenize_function, batched=True)
    
    # Get the primary evaluation dataset (first one)
    primary_eval_dataset = next(iter(tokenized_eval_datasets.values())) if tokenized_eval_datasets else None

    # Calculate class weights
    labels = train_dataset['label']
    class_counts = np.bincount(labels)
    total_samples = len(labels)
    class_weights = torch.FloatTensor([total_samples / (len(class_counts) * count) for count in class_counts])

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=16,
        num_train_epochs=100,
        weight_decay=0.1,
        eval_strategy="steps",
        eval_steps=300,
        metric_for_best_model="accuracy",
        save_total_limit=1,
        run_name="mmlu-4way-100K-3",
        warmup_ratio=0.1,
        logging_steps=10,
        report_to="wandb",
        save_strategy="steps",
        save_steps=200,
        # lr_scheduler_type="cosine",
        lr_scheduler_type="constant_with_warmup"  # Changed from "linear" to "constant_with_warmup"
    )

    # Initialize trainer with multiple evaluation datasets
    trainer = MultiDatasetTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=primary_eval_dataset,
        eval_datasets=tokenized_eval_datasets,
        compute_metrics=compute_metrics,
    )
    
    # Initialize optimizer with different parameters
    optimizer = torch.optim.AdamW(
        [
            {"params": model.deberta.parameters(), "lr": 5e-6},  # Lower LR for pretrained layers
            {"params": model.classifier.parameters(), "lr": 1e-4}  # Higher LR for classification head
        ],
        weight_decay=0.1
    )

    # Initialize trainer with custom optimizer
    trainer = MultiDatasetTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=primary_eval_dataset,
        eval_datasets=tokenized_eval_datasets,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, None)  # Custom optimizer, default scheduler
    )

    # Prepare everything with accelerator
    trainer = accelerator.prepare(trainer)

    # Train the model
    trainer.train()
    
    # Save the model at the end of training
    if accelerator.is_main_process:
        model_path = f"{output_dir}/final_model"
        trainer.save_model(model_path)
        print(f"Final model saved to {model_path}")
    
    # Evaluate on all datasets
    eval_results = trainer.evaluate()
    
    return eval_results

def main():
    # Initialize accelerator
    accelerator = Accelerator()
    
    # Load MMLU datasets
    train_dataset = load_mmlu_data(split="train")
    val_dataset = load_mmlu_data(split="validation")
    test_dataset = load_mmlu_data(split="test")
    
    gqpa_test = load_gqpa_data()
    hle_test = load_hle()

    # train_dataset = train_dataset.select(range(10000))
    
    # Load MMLU-Pro datasets
    ds1, ds2 = load_mmlu_pro_data()
    # combine them into one test dataset
    mmlu_pro_test = concatenate_datasets([ds1, ds2])
    
    # Train on training data, evaluate on multiple datasets
    if accelerator.is_main_process:
        print("Training model on training data...")
    
    # Create a dictionary of evaluation datasets
    eval_datasets = {
        "mmlu_test": test_dataset,
        "mmlu_pro_test": mmlu_pro_test
    }
    
    # Eval on GQPA and HLE
    eval_datasets["gqpa_test"] = gqpa_test
    eval_datasets["hle_test"] = hle_test
    
    train2, test2 = load_super_gqpa_data()
    # combine them into one test dataset
    test3 = concatenate_datasets([train2, test2])
    
    eval_datasets["super_gqpa_test"] = test3
    
    # Use the specified output directory
    output_dir = "/fast/nchandak/classification/mmlu_aux"
    
    # Initialize wandb only on the main process
    if accelerator.is_main_process:
        wandb.init(project="mcq-classifier2", name="mmlu-4way-100K-5-only-options-try1")
    
    # Use a strong model for multiple-choice tasks
    model_name = "microsoft/deberta-v3-large"
    
    eval_results = train_model(
        train_dataset,
        eval_datasets,
        model_name,
        output_dir=output_dir
    )
    
    # Log results for all datasets
    if accelerator.is_main_process:
        # Log each dataset's results separately
        for dataset_name in eval_datasets.keys():
            prefix = dataset_name
            wandb.log({
                f"{prefix}_accuracy": eval_results[f'{prefix}_accuracy'],
                f"{prefix}_f1": eval_results[f'{prefix}_f1'],
                f"{prefix}_precision": eval_results[f'{prefix}_precision'],
                f"{prefix}_recall": eval_results[f'{prefix}_recall']
            })
        
        print("\nEvaluation performance:")
        print(eval_results)
        
        wandb.finish()
    
    return eval_results

if __name__ == "__main__":
    # Enable deterministic behavior for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    val_results = main()
    print("\nTest Performance:")
    print(val_results)