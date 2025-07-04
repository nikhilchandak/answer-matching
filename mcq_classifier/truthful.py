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
    

NUM_OPTIONS = 2


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
            max_length=512
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

def main():
    
    # Load MMLU datasets
    # train_dataset = load_mmlu_data(split="train")
    # val_dataset = load_mmlu_data(split="validation")
    # test_dataset = load_mmlu_data(split="test")
    
    # train_dataset = train_dataset.select(range(10000))
    
    # split MMLU test into train and test
    # test_dataset = test_dataset.train_test_split(test_size=0.1, seed=42)
    
    # train_dataset = test_dataset["train"]
    # test_dataset = test_dataset["test"]
    
    # We will use the SuperGOPQA dataset actually
    # train_dataset, test_dataset = load_super_gqpa_data(ratio=0.5)
    # train_dataset, test_dataset = load_mmlu_pro_data(ratio=0.5)
    # RATIO = 0.5
    # train_dataset, test_dataset = load_yourbench_data(ratio=RATIO)

    # Set NUM_OPTIONS to 2 for binary classification
    global NUM_OPTIONS
    NUM_OPTIONS = 2

    # Load TruthfulQA dataset for binary classification
    RATIO = 0.5  # 80% for training, 20% for testing
    train_dataset, test_dataset = load_truthfulqa_data(ratio=RATIO)

    # Initialize wandb only on the main process
    if accelerator.is_main_process:
        wandb.init(project="mcq-classifier4", name=f"truthfulqa-binary-{RATIO*100}percent")
    
    # Use a strong model for binary classification
    model_name = "google/flan-t5-xl"  # Alternative: "microsoft/deberta-v3-large"
    # model_name = "microsoft/deberta-v3-large"
    
    # Use recent models
    # model_name = "meta-llama/Meta-Llama-3.1-8B"
    # model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    # model_name = "/fast/nchandak/models/Qwen3-4B"
    model_name = "/fast/nchandak/models/Qwen3-4B"
    
    # Train on combined train and validation data, evaluate on test
    if accelerator.is_main_process:
        print("Training model on training data...")
    
    # Train on training data and evaluate on test data
    test_results = train_model(
        train_dataset,
        test_dataset,
        model_name,
        output_dir="./results/truthfulqa/binary_classification"
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