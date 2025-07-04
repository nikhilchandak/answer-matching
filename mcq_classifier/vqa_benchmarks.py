import pandas as pd
import numpy as np
from datasets import Dataset, concatenate_datasets, load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import torch
import json
import wandb
from typing import Dict, List
from accelerate import Accelerator

# np.random.seed(42)
# torch.manual_seed(42)


# Initialize accelerator
accelerator = Accelerator()
    
NUM_OPTIONS = 10

def load_mmmu_data(ratio=0.5):
    """Load MMMU dataset for multiple-choice classification."""
    
    # dataset_name = f"suyc21/VMCBench"
    # dataset = load_dataset(dataset_name, split="test")
    mmmu_pro_vision = load_dataset("MMMU/MMMU_Pro", "vision")["test"]
    mmmu_pro_standard_4 = load_dataset("MMMU/MMMU_Pro", "standard (4 options)")["test"]
    mmmu_pro_standard_10 = load_dataset("MMMU/MMMU_Pro", "standard (10 options)")["test"]
    
    # Shuffle the dataset with a fixed seed for reproducibility
    mmmu_pro_vision = mmmu_pro_vision.shuffle(seed=42)
    mmmu_pro_standard_4 = mmmu_pro_standard_4.shuffle(seed=42)
    mmmu_pro_standard_10 = mmmu_pro_standard_10.shuffle(seed=42)
    
    combined_data = concatenate_datasets([mmmu_pro_vision, mmmu_pro_standard_10])
    combined_data = combined_data.shuffle(seed=42)
    # Split into train and test
    # train_data = mmmu_pro_vision.select(range(int(len(mmmu_pro_vision) * ratio)))
    # test_data = mmmu_pro_vision.select(range(int(len(mmmu_pro_vision) * ratio), len(mmmu_pro_vision)))
    
    train_data = combined_data.select(range(int(len(combined_data) * ratio)))
    test_data = combined_data.select(range(int(len(combined_data) * ratio), len(combined_data)))
    
    # train_data = mmmu_pro_vision
    # test_data = mmmu_pro_standard_10
    
    print(f"Train data length: {len(train_data)}")
    print(f"Test data length: {len(test_data)}")
    
    # print(f"Train data: {train_data}")
    # print(f"Test data: {test_data}")
    
    def format_dataset(dataset):
        formatted_data = []
        correct_indices = {}
        num_options = 0
        
        for item in dataset:
            # try:
            #     # Check if item is a dictionary with expected keys
            #     if isinstance(item, dict):
            #         # question_text = item['question']
            #         answer = item['answer']
            #         options = item['options']
            #     else:
            #         # Handle case where item might be a string or other type
            #         print(f"Skipping item with unexpected type: {type(item)}")
            #         print(f"Item content: {item}")
            #         continue
                
            # except (TypeError, KeyError) as e:
            #     print(f"Error processing item: {e}")
            #     print(f"Item type: {type(item)}")
            #     print(f"Item content: {item}")
            #     continue
            
            answer = item['answer']
            # Convert options from string to list if it's a string
            options = eval(item['options'])
            
            if len(options) > NUM_OPTIONS:
                print(f"Number of options {len(options)} is greater than {NUM_OPTIONS}")
                # print(f"Options: {options}")
                # print(f"Answer: {answer}")
                # print(f"Item: {item['options']}")
                continue 
            
            # assert len(options) <= NUM_OPTIONS, f"Number of options is greater than {NUM_OPTIONS}"
            
            # if len(options) < 10:
            #     print(f"Not enough options, got {len(options)}")
            #     continue
            
            num_options += len(options)
            
            # Get the correct answer (assuming it's in format like "A" or "B")
            correct_answer_index = ord(answer.upper()) - ord('A')
            if correct_answer_index < 0 or correct_answer_index >= len(options):
                print(f"Invalid answer index: {answer}")
                print(f"Options: {options}")
                print(f"Correct answer index: {correct_answer_index}")
                continue
                
            correct_answer = options.pop(correct_answer_index)
            
            # Randomly sample distractors from remaining options
            remaining_options = options
            distractors = list(remaining_options)
            
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
            
        return Dataset.from_list(formatted_data), correct_indices, num_options
    
    train_dataset, train_indices, train_num_options = format_dataset(train_data)
    # p, q = load_mmlu_pro_data()
    # train_dataset = concatenate_datasets([p, q])
    
    test_dataset, test_indices, test_num_options = format_dataset(test_data)
    avg_test_num_options = test_num_options / float(len(test_dataset))
    
    print(f"MMMU standard 10 train dataset length: {len(train_dataset)}")
    print(f"MMMU vision test dataset length: {len(test_dataset)}")
    print(f"MMMU standard 10 train num options: {train_num_options}")
    
    # print(f"MMMU standard 10 train correct indices distribution: {train_indices}")
    print(f"MMMU vision test correct indices distribution: {test_indices}")
    print(f"MMMU vision test average number of options: {avg_test_num_options}")
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
    elif hasattr(model, 'score'):
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
            max_length=4096
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
        num_train_epochs=500,
        weight_decay=0.1,
        eval_strategy="steps",
        eval_steps=20,
        metric_for_best_model="accuracy",
        save_total_limit=0,
        run_name="mmlu-mcq",
        warmup_ratio=0.1,
        logging_steps=10,
        report_to="wandb" if accelerator.is_main_process else "none",
        save_strategy="no",
        fp16=True,
        # gradient_checkpointing=True,
        lr_scheduler_type="constant_with_warmup",
        # remove_unused_columns=False,  # Add this line
    )
    
    # Initialize optimizer with different parameters based on model architecture
    if "llama" in model_name.lower() or "qwen" in model_name.lower():
        # For Llama models
        optimizer = torch.optim.AdamW(
            [
                {"params": [p for n, p in model.named_parameters() if "score" not in n], "lr": 5e-6},  # Lower LR for pretrained layers
                {"params": [p for n, p in model.named_parameters() if "score" in n], "lr": 5e-5}  # Higher LR for classification head
            ],
            weight_decay=0.1
        )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, None)  # Custom optimizer, default scheduler
    )

    # Train the model
    trainer.train()
    
    # Evaluate on validation set
    eval_results = trainer.evaluate()
    
    return eval_results

def main():
    RATIO = 0.5
    train_dataset, test_dataset = load_mmmu_data(ratio=RATIO)

    # Initialize wandb only on the main process
    if accelerator.is_main_process:
        wandb.init(project="vqa-classifier1", name=f"mmmu-pro-qwen3-{RATIO*100}percent")
    
    # Use recent models
    # model_name = "/fast/nchandak/models/Qwen3-4B"
    model_name = "Qwen/Qwen3-4B"
    
    # Train on combined train and validation data, evaluate on test
    if accelerator.is_main_process:
        print("Training model on training data...")
    
    # First train on training data and evaluate on validation data
    test_results = train_model(
        train_dataset,
        test_dataset,
        model_name,
        output_dir="./results/mmlu/validation"
    )
    
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