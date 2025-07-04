import hashlib
from typing import List


def hash_question(question: str) -> str:
    """
    Create a hash for a question string
    
    Args:
        question: The question text
        
    Returns:
        A hash string for the question
    """
    # Normalize the question by removing extra whitespace and converting to lowercase
    normalized = " ".join(question.strip().lower().split())
    
    # Create a hash using SHA-256
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()


def hash_question_with_choices(question: str, choices: List[str]) -> str:
    """
    Create a hash for a question and its choices
    
    Args:
        question: The question text
        choices: List of choice texts
        
    Returns:
        A hash string for the question and choices
    """
    # Normalize the question
    normalized_q = " ".join(question.strip().lower().split())
    
    # Normalize and sort the choices to ensure consistent hashing regardless of order
    normalized_choices = sorted([" ".join(c.strip().lower().split()) for c in choices])
    
    # Combine question and choices
    content = normalized_q + "||" + "||".join(normalized_choices)
    
    # Create a hash using SHA-256
    return hashlib.sha256(content.encode('utf-8')).hexdigest() 