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

def load_super_gqpa_data():
    """Load SuperGOPQA dataset for multiple-choice classification with at most 10 options."""
    # Load SuperGOPQA dataset
    dataset = load_dataset("m-a-p/SuperGPQA")
    
    # Get the test set and split it 50-50 for training and testing
    test_data = dataset["train"]
    
    # Shuffle the dataset with a fixed seed for reproducibility
    test_data = test_data.shuffle(seed=42)
    
    # Split into train and test
    split_idx = int(len(test_data) * 0.9)
    train_data = test_data.select(range(split_idx))
    test_data = test_data.select(range(split_idx, len(test_data)))
    
    # Format data for 10-way classification (A through J)
    def format_dataset(dataset):
        formatted_data = []
        max_options = 0
        for item in dataset:
            question = item['question']
            max_options = max(max_options, len(item['options']))
            
            if len(item['options']) > 10:
                print(f"Expected 10 options, got {len(item['options'])}")
                print(item['options'])
                print(item['answer'])
                print(item['answer_letter'])
                if len(item['options']) > 10:
                    continue # skip this item
                
                while len(item['options']) < 10:
                    item['options'].append("Gibberish")
                
                continue 
            
            if len(item['options']) != 10:
                continue
            
            assert len(item['options']) == 10, f"Expected 10 options, got {len(item['options'])}"
            choices = [item['options'][i] for i in range(10)]
            
            # Shuffle the choices (TODO)
            
            # Create prompt with question and options
            # prompt = f"{question}\n"
            prompt = ""
            for i, choice_letter in enumerate(["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]):
                prompt += f"{choice_letter}. {choices[i]}\n"
            prompt = prompt.strip()  # Remove trailing newline
            
            # Convert answer letter to index (0-9 for A-J)
            answer_idx = ord(item['answer_letter']) - ord('A')
            
            formatted_data.append({
                'text': prompt,
                'label': answer_idx
            })
        
        print(f"Max options: {max_options}")
        return Dataset.from_list(formatted_data)
    
    train_dataset = format_dataset(train_data)
    test_dataset = format_dataset(test_data)
    
    # print length of train and test datasets
    print(f"Train dataset length: {len(train_dataset)}")
    print(f"Test dataset length: {len(test_dataset)}")
    
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
    
    # Initialize accelerator
    accelerator = Accelerator()
    
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=10,  # For A through J options
        problem_type="single_label_classification"
    )

    # Reinitialize the classification head for better learning
    model.classifier.weight.data.normal_(mean=0.0, std=0.02)
    model.classifier.bias.data.zero_()
    
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding="max_length",
            truncation=True,
            max_length=2048
        )

    # Tokenize datasets
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_eval = eval_dataset.map(tokenize_function, batched=True)

    # Calculate class weights
    labels = train_dataset['label']
    class_counts = np.bincount(labels)
    total_samples = len(labels)
    class_weights = torch.FloatTensor([total_samples / (len(class_counts) * count) for count in class_counts])

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=5e-6,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=32,
        num_train_epochs=100,
        weight_decay=0.01,
        eval_strategy="steps",
        eval_steps=200,
        metric_for_best_model="accuracy",
        save_total_limit=1,
        run_name="super-gqpa",
        warmup_ratio=0.1,
        logging_steps=10,
        report_to="wandb",
        save_strategy="no",
        # lr_scheduler_type="cosine",
        lr_scheduler_type="constant_with_warmup"
    )

    # Create custom trainer with weighted loss
    class WeightedTrainer(Trainer):
        def __init__(self, class_weights, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.class_weights = class_weights.to(self.args.device)

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Apply weighted cross entropy loss
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
            
            return (loss, outputs) if return_outputs else loss

    # Initialize optimizer with different parameters
    optimizer = torch.optim.AdamW(
        [
            {"params": model.deberta.parameters(), "lr": 5e-6},  # Lower LR for pretrained layers
            {"params": model.classifier.parameters(), "lr": 5e-5}  # Higher LR for classification head
        ],
        weight_decay=0.1
    )

    # Initialize trainer with custom optimizer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, None)  # Custom optimizer, default scheduler
    )

    # Prepare everything with accelerator
    trainer = accelerator.prepare(trainer)

    # Train the model
    trainer.train()
    
    # Evaluate on validation set
    eval_results = trainer.evaluate()
    
    return eval_results

def main():
    # Initialize accelerator
    accelerator = Accelerator()
    
    # Load MMLU-Pro datasets
    train_dataset, test_dataset = load_super_gqpa_data()

    # Create a validation set from the training set (20% of training data)
    # train_val_dataset = train_dataset.train_test_split(test_size=0.2, seed=42)
    # train_dataset = train_val_dataset["train"]
    # val_dataset = train_val_dataset["test"]
    
    # Initialize wandb only on the main process
    if accelerator.is_main_process:
        wandb.init(project="mcq-classifier", name="super-gqpa-10way-noq-try3")
    
    # Use a strong model for multiple-choice tasks
    model_name = "microsoft/deberta-v3-large"
    
    # Train on training data, evaluate on validation data
    if accelerator.is_main_process:
        print("Training model on training data...")
    
    val_results = train_model(
        train_dataset,
        test_dataset,
        model_name,
        output_dir="./results/super_gqpa/test"
    )
    
    # Log test results
    if accelerator.is_main_process:
        wandb.log({
            "test_accuracy": val_results['eval_accuracy'],
            "test_f1": val_results['eval_f1'],
            "test_precision": val_results['eval_precision'],
            "test_recall": val_results['eval_recall']
        })
        
        print("\Test performance:")
        print(val_results)
        
        wandb.finish()
    
    return val_results

if __name__ == "__main__":
    # Enable deterministic behavior for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    val_results = main()
    print("\nTest Performance:")
    print(val_results)