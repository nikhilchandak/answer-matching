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
    # Load the additional dataset from HF
    # additional_dataset = datasets.load_dataset("AmeyaPrabhu/GPQA-Test1")
    # additional_dataset = datasets.load_dataset("nikhilchandak/gpqa-diamond-test2")
    additional_dataset = datasets.load_dataset("nikhilchandak/GPQA-diamond-free")
    
    additional_data = {item["Record ID"]: item for item in additional_dataset["train"]}
    
    def _process_doc(doc):
        # Check if the Record ID exists in the additional dataset
        record_id = doc["Record ID"]
        question = doc["Question"]
        if record_id in additional_data:
            # Use the question from the additional dataset
            question = additional_data[record_id]["Question"]
        
        choices = [
            preprocess(doc["Incorrect Answer 1"]),
            preprocess(doc["Incorrect Answer 2"]),
            preprocess(doc["Incorrect Answer 3"]),
            preprocess(doc["Correct Answer"]),
        ]

        random.shuffle(choices)
        correct_answer_index = choices.index(preprocess(doc["Correct Answer"]))

        out_doc = {
            "question_id": doc["Record ID"],
            "choice1": choices[0],
            "choice2": choices[1],
            "choice3": choices[2],
            "choice4": choices[3],
            "options": [choices[0], choices[1], choices[2], choices[3]],
            "answer_index": correct_answer_index,
            "target": doc["Correct Answer"],
            "category": doc["Subdomain"],
            "question": question,
        }
        return out_doc

    return dataset.map(_process_doc)



def format_cot_example(example):
    template = f"The following is a question in {example['Subdomain']}. Think step by step and then finish your answer with \"the answer is (X)\" where X is the correct, brief answer. The final answer X should be concise and put in parentheses.\n"
    prompt = template + "Question:\n"
    question = example["question"]
    prompt += question + "\n"
        
    prompt += "Answer:" # Let's think step by step."
    # prompt += "Answer: Let's think step by step."
    # prompt += " /no_think"
    # prompt += "Please first provide your reasoning and then your final answer."
    
    return prompt 


doc_to_text = format_cot_example