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


def load_mmlu_pro_data(ratio=0.5, subject="de"):
    """Load MMLU-Pro dataset for multiple-choice classification with 10 options."""
    # Load MMLU-Pro dataset
    dataset = load_dataset("li-lab/MMLU-ProX", subject)
    
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
            options = []
            
            for i in range(NUM_OPTIONS):
                options.append(item[f"option_{i}"])
            
            # Skip if we don't have enough options
            # if len(options) < 10:
            #     print(f"Not enough options, got {len(options)}")
            #     continue
            
            # Get the correct answer and its index
            correct_answer_index = item['answer_index']
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


def global_mmlu(ratio=0.5, subject="de"):
    """Load MMLU-Pro dataset for multiple-choice classification with 10 options."""
    # Load MMLU-Pro dataset
    dataset = load_dataset("CohereLabs/Global-MMLU", subject)
    
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
            options = [item["option_a"], item["option_b"], item["option_c"], item["option_d"]]
            
            # Get the correct answer and its index
            correct_answer_index = ord(item['answer']) - ord('A')
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
            max_length=2048,
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
        per_device_train_batch_size=32, # 512 
        per_device_eval_batch_size=32, # 512 
        gradient_accumulation_steps=2,
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
    
    # dataset_name = "mmlu_pro_de"
    dataset_name = "global_mmlu_de"
    
    if dataset_name == "mmlu_pro_de":
        train_dataset, test_dataset = load_mmlu_pro_data(ratio=RATIO, subject="de")
        NUM_OPTIONS = 10
    elif dataset_name == "mmlu_pro_hi":
        train_dataset, test_dataset = load_mmlu_pro_data(ratio=RATIO, subject="de")
        NUM_OPTIONS = 10
    elif dataset_name == "mmlu_pro_en":
        train_dataset, test_dataset = load_mmlu_pro_data(ratio=RATIO, subject="en")
        NUM_OPTIONS = 10
    elif dataset_name == "mmlu_pro_fr":
        train_dataset, test_dataset = load_mmlu_pro_data(ratio=RATIO, subject="fr")
        NUM_OPTIONS = 10
    elif dataset_name == "global_mmlu_de":
        train_dataset, test_dataset = global_mmlu(ratio=RATIO, subject="de")
        NUM_OPTIONS = 4
    elif dataset_name == "global_mmlu_en":
        train_dataset, test_dataset = global_mmlu(ratio=RATIO, subject="en")
        NUM_OPTIONS = 4
    elif dataset_name == "global_mmlu_fr":
        train_dataset, test_dataset = global_mmlu(ratio=RATIO, subject="fr")
        NUM_OPTIONS = 4
    elif dataset_name == "global_mmlu_hi":
        train_dataset, test_dataset = global_mmlu(ratio=RATIO, subject="hi")
        NUM_OPTIONS = 4
    else:
        assert False, "Invalid dataset name"

    model_name = "microsoft/deberta-v3-large"
    # Use recent models
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
        output_dir=output_dir
    )
    test_results = None
    
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
        
    
    return test_results

if __name__ == "__main__":
    # Enable deterministic behavior for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    test_results = main()
    print("\nTest Performance:")
    print(test_results)