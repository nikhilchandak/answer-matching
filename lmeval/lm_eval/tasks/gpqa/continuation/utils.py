import random
import re

import datasets


def preprocess(text):
    if text is None:
        return " "
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text


def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        # For continuation style, we need to get the answers without showing choices in context
        correct_answer = preprocess(doc["Correct Answer"])
        incorrect_answers = [
            preprocess(doc["Incorrect Answer 1"]),
            preprocess(doc["Incorrect Answer 2"]),
            preprocess(doc["Incorrect Answer 3"]),
        ]
        
        # All choices for validation
        all_choices = incorrect_answers + [correct_answer]
        
        # Shuffle choices for fairness
        random.shuffle(all_choices)
        
        out_doc = {
            "Question": doc["Question"],
            "choices": all_choices,
            "answer": correct_answer,  # The full answer text instead of just the letter
        }
        return out_doc

    return dataset.map(_process_doc) 