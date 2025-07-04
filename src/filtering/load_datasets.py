from typing import List, Dict, Optional, Any
import datasets
from question_hash import hash_question
import random

def load_dataset_by_name(name: str, split: str = "test", subset: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Load and format a dataset by name
    
    Args:
        name: Name of the dataset (MMLU, GPQA, MMLU-Pro, MATH)
        split: Dataset split (train, validation, test)
        subset: Dataset subset if applicable
        
    Returns:
        A list of dictionaries with question, choices, and answer_index
    """
    if name == "MMLU":
        return load_mmlu(split, subset)
    elif name == "GPQA":
        return load_gpqa(split, subset)
    elif name == "MMLU-Pro":
        return load_mmlu_pro(split, subset)
    elif name == "MATH":
        return load_math_mc(split)
    else:
        raise ValueError(f"Unknown dataset: {name}")


def load_mmlu(split: str = "test", subset: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Load and format the MMLU dataset
    
    Args:
        split: Dataset split (train, validation, test)
        subset: Subject subset to load (if None, loads all subjects)
        
    Returns:
        A list of dictionaries with question, choices, and answer_index
    """
    if subset:
        dataset = datasets.load_dataset("cais/mmlu", subset, split=split)
    else:
        # Load all subjects
        dataset = datasets.load_dataset("cais/mmlu", "all", split=split)
    
    formatted_data = []
    
    for item in dataset:
        # Extract choices directly from the 'choices' field
        choices = item["choices"]
        # Convert letter answer (A, B, C, D) to index (0, 1, 2, 3)
        answer_index = item["answer"]
        
        question = item["question"]
        
        # Compute hash value for question
        q_hash = hash_question(question)
        
        formatted_data.append({
            "question": question,
            "choices": choices,
            "answer_index": answer_index,
            "subject": item["subject"] if "subject" in item else subset,
            "q_hash": q_hash
        })
    
    return formatted_data


def load_gpqa(split: str = "test", subset: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Load and format the GPQA dataset
    
    Args:
        split: Dataset split (train, validation, test)
        subset: Subset to load (if applicable)
        
    Returns:
        A list of dictionaries with question, choices, and answer_index
    """
    if subset is None:
        subset = "gpqa_main"
        
    dataset = datasets.load_dataset("Idavidrein/gpqa", subset, split=split)
    
    formatted_data = []
    
    for item in dataset:
        # Create choices list and shuffle while tracking the correct answer
        choices = [item["Correct Answer"], item["Incorrect Answer 1"], item["Incorrect Answer 2"], item["Incorrect Answer 3"]]
        correct_answer = item["Correct Answer"]
        
        # Shuffle the choices
        random.shuffle(choices)
        
        # Find the new index of the correct answer
        answer_index = choices.index(correct_answer)
        question = item["Question"]
        
        # Use record_id if available, otherwise compute hash
        q_hash = item.get("Record ID", hash_question(question))
        
        formatted_data.append({
            "question": question,
            "choices": choices,
            "answer_index": answer_index,
            "subject": item["High-level domain"] if "High-level domain" in item else None,
            "q_hash": q_hash
        })
    
    return formatted_data


def load_mmlu_pro(split: str = "test", subset: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Load and format the MMLU-Pro dataset
    
    Args:
        split: Dataset split (train, validation, test)
        subset: Subset to load (if applicable)
        
    Returns:
        A list of dictionaries with question, choices, and answer_index
    """
    dataset = datasets.load_dataset("TIGER-Lab/MMLU-Pro", split=split)
    
    formatted_data = []
    
    for item in dataset:
        choices = item["options"]
        answer_index = item["answer_index"]
        question = item["question"]
        
        # Use question_id if available, otherwise compute hash
        q_hash = item.get("question_id", hash_question(question))
        
        formatted_data.append({
            "question": question,
            "choices": choices,
            "answer_index": answer_index,
            "question_id": item.get("question_id"),
            "subject": item.get("category"),
            "q_hash": str(q_hash)
        })
    
    return formatted_data


def load_math_mc(split: str = "test") -> List[Dict[str, Any]]:
    """
    Load and format the nikhilchandak/MATH_mc dataset
    
    Args:
        split: Dataset split (train, validation, test)
        
    Returns:
        A list of dictionaries with question, choices, and answer_index
    """
    # Load the dataset
    dataset = datasets.load_dataset("nikhilchandak/MATH_mc", split=split)
    
    formatted_data = []
    
    for item in dataset:
        # Extract choices from A, B, C, D fields
        choices = [item["A"], item["B"], item["C"], item["D"]]
        
        # Convert letter answer (A, B, C, D) to index (0, 1, 2, 3)
        answer_map = {"A": 0, "B": 1, "C": 2, "D": 3}
        answer_index = answer_map[item["Answer"]]
        
        question = item["Question"]
        
        # Use Question_ID as hash
        q_hash = str(item["Question_ID"])
        
        # Create subject from Type and Level
        subject = f"{item['Type']}_{item['Level']}"
        
        formatted_data.append({
            "question": question,
            "choices": choices,
            "answer_index": answer_index,
            "subject": subject,
            "q_hash": q_hash
        })
    
    return formatted_data 