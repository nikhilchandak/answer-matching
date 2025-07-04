from functools import partial
import re


def transform_cot_to_binary(cot, options, correct_option_index):
    """
    Transforms a Chain-of-Thought for MCQ into one for binary True/False questions.
    Args:
        cot (str): The original Chain-of-Thought.
        options (List[str]): The multiple-choice options.
        correct_option_index (int): The index of the correct option.
    Returns:
        str: The transformed Chain-of-Thought.
    """
    # Get the correct option letter (A, B, C, etc.)
    correct_letter = chr(65 + correct_option_index)
    
    # Replace standard MCQ answer format with binary True/False answer
    pattern = r"the answer is \(?([A-Z])\)?\.?$"
    match = re.search(pattern, cot, re.IGNORECASE)
    
    if match:
        chosen_letter = match.group(1)
        if chosen_letter == correct_letter:
            # If the answer chosen is correct, replace with "A" (True)
            transformed_cot = re.sub(pattern, "the answer is A.", cot, flags=re.IGNORECASE)
        else:
            # If the answer chosen is incorrect, replace with "B" (False)
            transformed_cot = re.sub(pattern, "the answer is B.", cot, flags=re.IGNORECASE)
    else:
        # Fallback case if no "the answer is X" is found
        transformed_cot = cot.rstrip() + "\nTherefore, the answer is " + ("A." if correct_letter == "A" else "B.")
    
    return transformed_cot


def format_binary_question(question, option, options, option_index, correct_option_index):
    """
    Formats a binary True/False question from the original MCQ question and an option.
    Args:
        question (str): The original question.
        option (str): The option being evaluated.
        options (List[str]): All available options.
        option_index (int): The index of the current option.
        correct_option_index (int): The index of the correct option.
    Returns:
        dict: A dictionary with the reformatted question, options, and correct answer index.
    """
    # Format the new binary question
    binary_question = f"Question: Is the following a correct answer to the question: '{question}'?\nAnswer: {option}"
    
    # Binary options are always True and False
    binary_options = ["True", "False"]
    
    # The correct answer is True if option_index matches correct_option_index, otherwise False
    correct_binary_index = 0 if option_index == correct_option_index else 1
    
    return {
        "question": binary_question,
        "options": binary_options,
        "answer_index": correct_binary_index,
    }


def format_cot_example(example, including_answer=True, option_index=None):
    """
    Formats a Chain-of-Thought example for a binary True/False question.
    Args:
        example (dict): The original example dictionary.
        including_answer (bool): Whether to include the answer in the prompt.
        option_index (int): The index of the option to format as a binary question.
    Returns:
        str: The formatted prompt.
    """
    # Get the original question and options
    question = example["question"]
    options = example["options"]
    correct_option_index = example["answer_index"]
    
    # If option_index is not provided, default to option A (index 0)
    if option_index is None:
        option_index = 0
    
    # Get the option being evaluated
    option = options[option_index]
    
    # Format the binary question
    binary_example = format_binary_question(
        question, option, options, option_index, correct_option_index
    )
    
    # Build the prompt
    prompt = "Question:\n" + binary_example["question"] + "\n"
    prompt += "Options:\n"
    prompt += "A. True\n"
    prompt += "B. False\n"
    
    if including_answer:
        cot_content = example["cot_content"].replace(
            "A: Let's think step by step.", "Answer: Let's think step by step."
        )
        transformed_cot = transform_cot_to_binary(
            cot_content, options, correct_option_index
        )
        prompt += transformed_cot + "\n\n"
    else:
        prompt += "Answer: Let's think step by step."
    
    return prompt + " /no_think"


def binary_doc_to_text(doc):
    """
    Converts a document to a binary question prompt.
    This function will be called multiple times for each document, once for each option.
    """
    option_index = getattr(binary_doc_to_text, "current_option_index", 0)
    result = format_cot_example(doc, including_answer=False, option_index=option_index)
    
    # Update the option index for the next call, cycling through all options
    binary_doc_to_text.current_option_index = (option_index + 1) % len(doc["options"])
    
    return result


def binary_doc_to_target(doc):
    """
    Determines the target answer for a binary question.
    """
    option_index = getattr(binary_doc_to_target, "current_option_index", 0)
    correct_option_index = doc["answer_index"]
    
    # The target is "A" (for True) if this option is correct, otherwise "B" (for False)
    result = "A" if option_index == correct_option_index else "B"
    
    # Update the option index for the next call, cycling through all options
    binary_doc_to_target.current_option_index = (option_index + 1) % len(doc["options"])
    
    return result


def binary_fewshot_to_text(doc):
    """
    Formats a few-shot example for binary True/False questions.
    """
    option_index = getattr(binary_fewshot_to_text, "current_option_index", 0)
    result = format_cot_example(doc, including_answer=True, option_index=option_index)
    
    # Update the option index for the next call, cycling through all options
    binary_fewshot_to_text.current_option_index = (option_index + 1) % len(doc["options"])
    
    return result


# Initialize option indices
binary_doc_to_text.current_option_index = 0
binary_doc_to_target.current_option_index = 0
binary_fewshot_to_text.current_option_index = 0


# Export the functions
doc_to_text = binary_doc_to_text
doc_to_target = binary_doc_to_target
fewshot_to_text = binary_fewshot_to_text


def process_docs(dataset, subject):
    """
    Filters the dataset to include only documents of the specified subject.
    """
    return dataset.filter(lambda x: x["category"] == subject)


# Create subject-specific processing functions
process_biology = partial(process_docs, subject="biology")
process_business = partial(process_docs, subject="business")
process_chemistry = partial(process_docs, subject="chemistry")
process_computer_science = partial(process_docs, subject="computer science")
process_economics = partial(process_docs, subject="economics")
process_engineering = partial(process_docs, subject="engineering")
process_health = partial(process_docs, subject="health")
process_history = partial(process_docs, subject="history")
process_law = partial(process_docs, subject="law")
process_math = partial(process_docs, subject="math")
process_other = partial(process_docs, subject="other")
process_philosophy = partial(process_docs, subject="philosophy")
process_physics = partial(process_docs, subject="physics")
process_psychology = partial(process_docs, subject="psychology") 