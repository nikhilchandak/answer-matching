import pandas as pd
import numpy as np
from datasets import Dataset, load_dataset, concatenate_datasets
from typing import Dict, List, Tuple, Union, Optional, Callable, Any
from collections import defaultdict

# Track insufficient options warnings
insufficient_options_count = defaultdict(int)

# Create a mapping from dataset names to HuggingFace dataset paths
# Create a mapping from dataset names to HuggingFace dataset paths
DATASET_PATHS = {
    "mmlu_pro": "TIGER-Lab/MMLU-Pro",
    "mmlu": "cais/mmlu",
    "hellaswag": "hellaswag",
    "arc": "ai2_arc",
    "arc_easy": "ai2_arc",
    "commonsense_qa": "commonsense_qa",
    "piqa": "piqa",
    "siqa": "social_i_qa",
    "openbookqa": "openbookqa",
}

def load_mcq_dataset(
    dataset_name: str,
    num_options: int = 4,
    train_dataset_name: Optional[str] = None,
    test_dataset_name: Optional[str] = None,
    train_ratio: float = 0.6,
    test_ratio: float = 0.3,
    seed: int = 42,
    split_test_set: bool = False,
    only_options: bool = False,
    option_sampling_strategy: str = "both"
) -> Tuple[Dataset, Dataset]:
    """
    Load and prepare multiple-choice question datasets for classification.
    
    Args:
        dataset_name: Name of the dataset in HuggingFace datasets
        num_options: Number of options for each question
        train_dataset_name: Optional separate dataset for training
        test_dataset_name: Optional separate dataset for testing
        train_ratio: Proportion to use for training if splitting a single dataset
        test_ratio: Proportion to use for testing if splitting a single dataset
        seed: Random seed for reproducibility
        split_test_set: If True, split the test set into train/test when no train set is available
        only_options: If True, include only the options in the prompt (no question)
        option_sampling_strategy: Strategy for sampling additional options ("incorrect", "correct", "both")
        
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    # Define dataset loaders
    dataset_loaders = {
        "mmlu_pro": lambda: load_dataset_with_formatter(
            DATASET_PATHS["mmlu_pro"], "test", "test",
            formatter=format_mmlu_pro_dataset, num_options=num_options, seed=seed,
            only_options=only_options, split_test=split_test_set, 
            train_ratio=train_ratio, test_ratio=test_ratio,
            option_sampling_strategy=option_sampling_strategy
        ),
        "mmlu": lambda: load_dataset_with_formatter(
            DATASET_PATHS["mmlu"], "auxiliary_train", "test", config="all",
            formatter=format_mmlu_dataset, num_options=num_options, seed=seed,
            only_options=only_options, split_test=split_test_set,
            train_ratio=train_ratio, test_ratio=test_ratio,
            option_sampling_strategy=option_sampling_strategy
        ),
        "hellaswag": lambda: load_dataset_with_formatter(
            DATASET_PATHS["hellaswag"], "train", "test",
            formatter=format_hellaswag_dataset, num_options=num_options, seed=seed,
            only_options=only_options, split_test=split_test_set,
            train_ratio=train_ratio, test_ratio=test_ratio,
            option_sampling_strategy=option_sampling_strategy
        ),
        "arc": lambda: load_dataset_with_formatter(
            DATASET_PATHS["arc"], "train", "test", config="challenge",
            formatter=format_arc_dataset, num_options=num_options, seed=seed,
            only_options=only_options, split_test=split_test_set,
            train_ratio=train_ratio, test_ratio=test_ratio,
            option_sampling_strategy=option_sampling_strategy
        ),
        "arc_easy": lambda: load_dataset_with_formatter(
            DATASET_PATHS["arc_easy"], "train", "test", config="easy",
            formatter=format_arc_dataset, num_options=num_options, seed=seed,
            only_options=only_options, split_test=split_test_set,
            train_ratio=train_ratio, test_ratio=test_ratio,
            option_sampling_strategy=option_sampling_strategy
        ),
        "commonsense_qa": lambda: load_dataset_with_formatter(
            DATASET_PATHS["commonsense_qa"], "train", "test",
            formatter=format_commonsense_qa_dataset, num_options=num_options, seed=seed,
            only_options=only_options, split_test=split_test_set,
            train_ratio=train_ratio, test_ratio=test_ratio,
            option_sampling_strategy=option_sampling_strategy
        ),
        "piqa": lambda: load_dataset_with_formatter(
            DATASET_PATHS["piqa"], "train", "validation",
            formatter=format_piqa_dataset, num_options=num_options, seed=seed,
            only_options=only_options, split_test=split_test_set,
            train_ratio=train_ratio, test_ratio=test_ratio,
            option_sampling_strategy=option_sampling_strategy
        ),
        "siqa": lambda: load_dataset_with_formatter(
            DATASET_PATHS["siqa"], "train", "validation",
            formatter=format_siqa_dataset, num_options=num_options, seed=seed,
            only_options=only_options, split_test=split_test_set,
            train_ratio=train_ratio, test_ratio=test_ratio,
            option_sampling_strategy=option_sampling_strategy
        ),
        "openbookqa": lambda: load_dataset_with_formatter(
            DATASET_PATHS["openbookqa"], "train", "test", config="main",
            formatter=format_openbookqa_dataset, num_options=num_options, seed=seed,
            only_options=only_options, split_test=split_test_set,
            train_ratio=train_ratio, test_ratio=test_ratio,
            option_sampling_strategy=option_sampling_strategy
        ),
    }
    
    # If both train and test datasets are specified (different datasets)
    if train_dataset_name and test_dataset_name:
        try:
            # Use dataset loaders to get properly formatted datasets
            train_dataset_name_lower = train_dataset_name.lower()
            test_dataset_name_lower = test_dataset_name.lower()
            
            if train_dataset_name_lower not in dataset_loaders:
                raise ValueError(f"Train dataset {train_dataset_name} not supported.")
                
            if test_dataset_name_lower not in dataset_loaders:
                raise ValueError(f"Test dataset {test_dataset_name} not supported.")
            
            # Load train dataset
            train_data, _ = dataset_loaders[train_dataset_name_lower]()
            
            # Load test dataset
            _, test_data = dataset_loaders[test_dataset_name_lower]()
            
            print(f"Train dataset ({train_dataset_name}) length: {len(train_data)}")
            print(f"Test dataset ({test_dataset_name}) length: {len(test_data)}")
            
            # Print sample examples
            print_dataset_samples(train_data, test_data)
            
            return train_data, test_data
            
        except Exception as e:
            print(f"Error loading/formatting datasets {train_dataset_name} and {test_dataset_name}: {str(e)}")
            raise
    
    # Default case: load the specified dataset
    dataset_name_lower = dataset_name.lower()
    if dataset_name_lower in dataset_loaders:
        train_data, test_data = dataset_loaders[dataset_name_lower]()
        
        # Print dataset information
        print(f"Train dataset length: {len(train_data)}")
        print(f"Test dataset length: {len(test_data)}")
        
        # Print sample examples
        print_dataset_samples(train_data, test_data)
        
        return train_data, test_data
    else:
        raise ValueError(f"Dataset {dataset_name} not found in supported datasets list.")

def load_dataset_with_formatter(
    dataset_name: str,
    train_split: str,
    test_split: str,
    formatter: Callable,
    num_options: int,
    seed: int,
    only_options: bool = False,
    config: Optional[str] = None,
    split_test: bool = False,
    train_ratio: float = 0.6,
    test_ratio: float = 0.3,
    option_sampling_strategy: str = "both"
) -> Tuple[Dataset, Dataset]:
    """
    Generic dataset loader that handles common loading patterns.
    
    Args:
        dataset_name: Name of the dataset in HuggingFace
        train_split: Name of the training split
        test_split: Name of the test split
        formatter: Function to format the dataset
        num_options: Number of options for each question
        seed: Random seed
        only_options: If True, include only options in prompt
        config: Dataset configuration name
        split_test: If True, use test data for both training and testing
        train_ratio: Proportion of data to use for training
        test_ratio: Proportion of data to use for testing
        option_sampling_strategy: Strategy for sampling additional options
        
    Returns:
        Tuple of (train_dataset, test_dataset)
    """
    try:
        dataset = load_dataset(dataset_name, config)
        
        # Get training data
        if train_split in dataset and not split_test:
            train_data = dataset[train_split]
            test_data = dataset[test_split]
            
            # Apply ratios to select subsets if needed
            if train_ratio < 1.0:
                train_data = train_data.shuffle(seed=seed)
                train_size = int(len(train_data) * train_ratio)
                train_data = train_data.select(range(train_size))
            
            if test_ratio < 1.0:
                test_data = test_data.shuffle(seed=seed)
                test_size = int(len(test_data) * test_ratio)
                test_data = test_data.select(range(test_size))
        else:
            # If we need to split the test set for both training and testing
            test_data = dataset[test_split]
            if split_test:
                # Ensure no overlap between train and test when splitting
                test_data = test_data.shuffle(seed=seed)
                total_samples = len(test_data)
                train_size = int(total_samples * train_ratio)
                test_size = int(total_samples * test_ratio)
                
                train_data = test_data.select(range(train_size))
                test_data = test_data.select(range(train_size, train_size + test_size))
            else:
                # Use test data for both if no train split available
                train_data = test_data
                
        # Format data
        train_dataset = formatter(train_data, num_options, seed, only_options, option_sampling_strategy=option_sampling_strategy)
        test_dataset = formatter(test_data, num_options, seed, only_options, option_sampling_strategy=option_sampling_strategy)
        
        return train_dataset, test_dataset
        
    except Exception as e:
        print(f"Error loading/formatting dataset {dataset_name}: {str(e)}")
        raise

def split_dataset(dataset: Dataset, train_ratio: float, test_ratio: float, 
                 seed: int = 42) -> Tuple[Dataset, Dataset]:
    """Split a dataset into train and test sets."""
    assert train_ratio + test_ratio <= 1.0, "Train and test ratios must sum to at most 1.0"
    
    # Shuffle the dataset with a fixed seed for reproducibility
    shuffled_data = dataset.shuffle(seed=seed)
    
    # Calculate split indices
    total_samples = len(shuffled_data)
    train_size = int(total_samples * train_ratio)
    test_size = int(total_samples * test_ratio)
    
    # Split into train and test
    train_data = shuffled_data.select(range(train_size))
    test_data = shuffled_data.select(range(train_size, train_size + test_size))
    
    return train_data, test_data

def build_option_pools(dataset, options_extractor, correct_index_extractor, strategy="both"):
    """
    Build pools of options from a dataset.
    
    Args:
        dataset: The dataset to extract options from
        options_extractor: Function to extract options from an item
        correct_index_extractor: Function to extract correct answer index
        strategy: "incorrect" for only incorrect options, "correct" for only correct options,
                 "both" for all options (default)
        
    Returns:
        List of options according to the strategy
    """
    option_pool = []
    
    for item in dataset:
        options = options_extractor(item)
        correct_idx = correct_index_extractor(item)
        
        if strategy == "incorrect":
            # Add all incorrect options to the pool
            for i, opt in enumerate(options):
                if i != correct_idx:
                    option_pool.append(opt)
        elif strategy == "correct":
            # Add only the correct option to the pool
            option_pool.append(options[correct_idx])
        elif strategy == "both":
            # Add all options to the pool
            option_pool.extend(options)
    
    return option_pool

def prepare_and_shuffle_options(
    options: List[str],
    correct_index: int,
    num_options: int,
    seed: int,
    example_index: int = 0,
    option_pool: Optional[List[str]] = None,
    option_sampling_strategy: str = "both"
) -> Tuple[List[str], int]:
    """
    Prepare and shuffle options for multiple choice questions.
    
    Args:
        options: All available options
        correct_index: Index of the correct answer in options
        num_options: Desired number of options
        seed: Random seed
        example_index: Index of the current example to ensure different shuffling
        option_pool: Pool of options to sample from if needed
        option_sampling_strategy: Strategy for sampling additional options ("incorrect", "correct", "both")
        
    Returns:
        Tuple of (shuffled_options, new_correct_index)
    """
    # Use a different seed for each example by combining the global seed with the example index
    np.random.seed(seed + example_index)
    
    # Extract correct answer
    correct_answer = options[correct_index]
    remaining_options = options.copy()
    remaining_options.pop(correct_index)
    
    # Sample distractors if we have more than needed
    if len(remaining_options) > (num_options - 1):
        distractor_indices = np.random.choice(len(remaining_options), num_options - 1, replace=False)
        distractors = [remaining_options[i] for i in distractor_indices]
    else:
        # Use all remaining options
        distractors = remaining_options
        
        # Sample additional options from pool if needed and available
        if option_pool and len(distractors) < (num_options - 1):
            # Calculate how many additional options we need
            additional_needed = num_options - 1 - len(distractors)
            
            # Add additional distractors by sampling until we get unique options
            additional_distractors = []
            
            # Try to add each needed distractor with up to 3 attempts per distractor
            for _ in range(additional_needed):
                if not option_pool:
                    break  # No options in pool
                
                # Make 3 attempts to find a valid option for this position
                for attempt in range(3):
                    sampled_option = np.random.choice(option_pool)
                    # Only add if it's not already in distractors and not the correct answer
                    if sampled_option not in distractors and sampled_option != correct_answer and sampled_option not in additional_distractors:
                        additional_distractors.append(sampled_option)
                        break  # Successfully added this distractor, move to next one
            
            distractors.extend(additional_distractors)
    
    # Create the options with the correct answer included
    final_options = distractors + [correct_answer]
    
    # Shuffle the options
    shuffled_indices = np.random.permutation(len(final_options))
    shuffled_options = [final_options[i] for i in shuffled_indices]
    
    # Find the new index of the correct answer
    new_correct_index = np.where(shuffled_indices == (len(final_options) - 1))[0][0]
    
    return shuffled_options, new_correct_index

def format_dataset_with_options(
    dataset: Any,
    question_extractor: Callable[[Dict], str],
    options_extractor: Callable[[Dict], List[str]],
    correct_index_extractor: Callable[[Dict], int],
    num_options: int,
    seed: int,
    only_options: bool = False,
    dataset_name: str = "dataset",
    option_sampling_strategy: str = "both"
) -> Dataset:
    """
    Generic dataset formatter for multiple-choice questions.
    
    Args:
        dataset: The dataset to format
        question_extractor: Function to extract the question from an item
        options_extractor: Function to extract the options from an item
        correct_index_extractor: Function to extract the correct answer index from an item
        num_options: Number of options for each question
        seed: Random seed
        only_options: If True, include only options in prompt
        dataset_name: Name of the dataset for tracking insufficient options
        option_sampling_strategy: Strategy for sampling additional options ("incorrect", "correct", "both")
        
    Returns:
        Formatted dataset
    """
    formatted_data = []
    skipped = 0
    
    # Build option pools
    option_pool = build_option_pools(
        dataset, 
        options_extractor, 
        correct_index_extractor, 
        strategy=option_sampling_strategy
    )
    
    for i, item in enumerate(dataset):
        question = question_extractor(item)
        options = options_extractor(item)
        
        # Get the correct answer index
        correct_answer_index = correct_index_extractor(item)
        
        # Prepare and shuffle options - pass the example index and option pool
        shuffled_options, new_correct_index = prepare_and_shuffle_options(
            options, 
            correct_answer_index, 
            num_options, 
            seed, 
            example_index=i,
            option_pool=option_pool,
            option_sampling_strategy=option_sampling_strategy
        )
        
        # Skip if we still don't have enough options (could happen if option pool is too small)
        if len(shuffled_options) < num_options:
            skipped += 1
            continue
        
        # Create prompt with question and options
        if only_options:
            prompt = ""
        else:
            prompt = f"{question}\n"
            
        for i, option_text in enumerate(shuffled_options):
            option_letter = chr(65 + i)  # A, B, C, D, ...
            prompt += f"{option_letter}. {option_text}\n"
        prompt = prompt.strip()
        
        formatted_data.append({
            'text': prompt,
            'label': new_correct_index
        })
    
    if skipped > 0:
        insufficient_options_count[dataset_name] += skipped
    
    return Dataset.from_list(formatted_data)

def format_mmlu_pro_dataset(dataset, num_options: int, seed: int, only_options: bool = False, option_sampling_strategy: str = "both") -> Dataset:
    """Format MMLU-Pro dataset for n-way classification."""
    np.random.seed(seed)
    formatted_data = []
    skipped = 0
    
    # Build option pools
    option_pool = build_option_pools(
        dataset,
        lambda item: item['options'],
        lambda item: item['answer_index'],
        strategy=option_sampling_strategy
    )
    
    for i, item in enumerate(dataset):
        question = item['question']
        options = item['options'].copy()
        
        # Get the correct answer and its index
        correct_answer_index = item['answer_index']
        
        # Prepare and shuffle options
        shuffled_options, new_correct_index = prepare_and_shuffle_options(
            options, 
            correct_answer_index, 
            num_options, 
            seed, 
            example_index=i,
            option_pool=option_pool,
            option_sampling_strategy=option_sampling_strategy
        )
        
        # Skip if we still don't have enough options
        if len(shuffled_options) < num_options:
            skipped += 1
            continue
        
        # Create prompt with question and options
        if only_options:
            prompt = ""
        else:
            prompt = f"{question}\n"
            
        for i, option_text in enumerate(shuffled_options):
            option_letter = chr(65 + i)  # A, B, C, D, ...
            prompt += f"{option_letter}. {option_text}\n"
        prompt = prompt.strip()
        
        formatted_data.append({
            'text': prompt,
            'label': new_correct_index
        })
    
    if skipped > 0:
        insufficient_options_count["mmlu_pro"] += skipped
    
    return Dataset.from_list(formatted_data)

def format_mmlu_dataset(dataset, num_options: int, seed: int, only_options: bool = False, option_sampling_strategy: str = "both") -> Dataset:
    """Format MMLU dataset for n-way classification."""
    return format_dataset_with_options(
        dataset,
        question_extractor=lambda item: item['question'],
        options_extractor=lambda item: item['choices'][:num_options],
        correct_index_extractor=lambda item: item['answer'],
        num_options=num_options,
        seed=seed,
        only_options=only_options,
        dataset_name="mmlu",
        option_sampling_strategy=option_sampling_strategy
    )

def format_hellaswag_dataset(dataset, num_options: int, seed: int, only_options: bool = False, option_sampling_strategy: str = "both") -> Dataset:
    """Format HellaSwag dataset for n-way classification."""
    return format_dataset_with_options(
        dataset,
        question_extractor=lambda item: item['ctx'],
        options_extractor=lambda item: item['endings'][:num_options],
        correct_index_extractor=lambda item: int(item['label']),
        num_options=num_options,
        seed=seed,
        only_options=only_options,
        dataset_name="hellaswag",
        option_sampling_strategy=option_sampling_strategy
    )

def format_arc_dataset(dataset, num_options: int, seed: int, only_options: bool = False, option_sampling_strategy: str = "both") -> Dataset:
    """Format ARC dataset for n-way classification."""
    return format_dataset_with_options(
        dataset,
        question_extractor=lambda item: item['question'],
        options_extractor=lambda item: item['choices']['text'][:num_options],
        correct_index_extractor=lambda item: item['choices']['label'].index(item['answerKey']),
        num_options=num_options,
        seed=seed,
        only_options=only_options,
        dataset_name="arc",
        option_sampling_strategy=option_sampling_strategy
    )

def format_commonsense_qa_dataset(dataset, num_options: int, seed: int, only_options: bool = False, option_sampling_strategy: str = "both") -> Dataset:
    """Format CommonsenseQA dataset for n-way classification."""
    return format_dataset_with_options(
        dataset,
        question_extractor=lambda item: item['question'],
        options_extractor=lambda item: item['choices']['text'][:num_options],
        correct_index_extractor=lambda item: item['choices']['label'].index(item['answerKey']),
        num_options=num_options,
        seed=seed,
        only_options=only_options,
        dataset_name="commonsense_qa",
        option_sampling_strategy=option_sampling_strategy
    )

def format_piqa_dataset(dataset, num_options: int, seed: int, only_options: bool = False, option_sampling_strategy: str = "both") -> Dataset:
    """Format PIQA dataset for n-way classification."""
    if num_options > 2:
        print(f"Warning: PIQA only has 2 options, but {num_options} were requested")
        num_options = 2
    
    return format_dataset_with_options(
        dataset,
        question_extractor=lambda item: item['goal'],
        options_extractor=lambda item: [item['sol1'], item['sol2']],
        correct_index_extractor=lambda item: item['label'],
        num_options=num_options,
        seed=seed,
        only_options=only_options,
        dataset_name="piqa",
        option_sampling_strategy=option_sampling_strategy
    )

def format_siqa_dataset(dataset, num_options: int, seed: int, only_options: bool = False, option_sampling_strategy: str = "both") -> Dataset:
    """Format SIQA dataset for n-way classification."""
    if num_options > 3:
        print(f"Warning: SIQA only has 3 options, but {num_options} were requested")
        num_options = 3
    
    return format_dataset_with_options(
        dataset,
        question_extractor=lambda item: f"{item['context']}\n{item['question']}",
        options_extractor=lambda item: [item['answerA'], item['answerB'], item['answerC']],
        correct_index_extractor=lambda item: int(item['label']) - 1,
        num_options=num_options,
        seed=seed,
        only_options=only_options,
        dataset_name="siqa",
        option_sampling_strategy=option_sampling_strategy
    )

def format_openbookqa_dataset(dataset, num_options: int, seed: int, only_options: bool = False, option_sampling_strategy: str = "both") -> Dataset:
    """Format OpenBookQA dataset for n-way classification."""
    return format_dataset_with_options(
        dataset,
        question_extractor=lambda item: item['question_stem'],
        options_extractor=lambda item: item['choices']['text'][:num_options],
        correct_index_extractor=lambda item: item['choices']['label'].index(item['answerKey']),
        num_options=num_options,
        seed=seed,
        only_options=only_options,
        dataset_name="openbookqa",
        option_sampling_strategy=option_sampling_strategy
    )

def format_dataset_by_name(dataset, dataset_name, num_options, seed, only_options=False):
    """Select and apply the appropriate formatting function based on dataset name."""
    dataset_name = dataset_name.lower()
    
    formatters = {
        "mmlu_pro": format_mmlu_pro_dataset,
        "mmlu": format_mmlu_dataset,
        "hellaswag": format_hellaswag_dataset,
        "arc": format_arc_dataset,
        "commonsense_qa": format_commonsense_qa_dataset,
        "piqa": format_piqa_dataset,
        "siqa": format_siqa_dataset,
        "openbookqa": format_openbookqa_dataset
    }
    
    for key, formatter in formatters.items():
        if key in dataset_name:
            return formatter(dataset, num_options, seed, only_options)
    
    # For unknown datasets, just return as is
    print(f"Using generic formatting for dataset {dataset_name}")
    return dataset

def print_dataset_samples(train_dataset, test_dataset, num_samples=5):
    """
    Print sample examples from training and test datasets.
    
    Args:
        train_dataset: Training dataset
        test_dataset: Test dataset
        num_samples: Number of samples to print from each dataset
    """
    print("\n=== DATASET SAMPLES ===")
    
    # Print training samples
    print("\n--- Training Sample(s) ---")
    for i in range(min(num_samples, len(train_dataset))):
        sample = train_dataset[i]
        print(f"Text:\n{sample['text']}")
        print(f"Label: {sample['label']}")
        print("-" * 50)
    
    # Print test samples
    print("\n--- Test Sample(s) ---")
    for i in range(min(num_samples, len(test_dataset))):
        sample = test_dataset[i]
        print(f"Text:\n{sample['text']}")
        print(f"Label: {sample['label']}")
        print("-" * 50)
    
    # Print insufficient options warnings
    if insufficient_options_count:
        print("\n=== INSUFFICIENT OPTIONS WARNINGS ===")
        for dataset_name, count in insufficient_options_count.items():
            if count > 0:
                print(f"Warning: {count} samples skipped in {dataset_name} due to insufficient options")