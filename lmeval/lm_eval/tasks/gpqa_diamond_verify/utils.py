from functools import partial
import random
import datasets
import re

choices = [
    "A",
    "B",
]


def format_cot_example(example, including_answer=True):
    prompt = "Question:\n"
    question = example["question"]
    options = example["options"]
    prompt += question + "\n"
    prompt += "Options:\n"
    for i, opt in enumerate(options):
        prompt += "{}. {}\n".format(choices[i], opt)
    if including_answer:
        cot_content = example["cot_content"].replace(
            "A: Let's think step by step.", "Answer: Let's think step by step."
        )
        prompt += cot_content + "\n\n"
    else:
        prompt += "Answer: Let's think step by step."
    return prompt
    # return prompt + " /no_think"

ANS_TAG_PROMPT = """Think step by step and provide your final answer choice (A/B/C/...) in <answer>  </answer> tags. Your final answer choice should be a CAPITAL LETTER representing the option you think is most likely to be correct."""
MCQ_PROMPT = """Please reason step by step and then finish your answer with \"the answer is (X)\" where X is the option (A or B) you think is most likely to be correct."""

def format_zero_shot(example):
    prompt = f"You will be asked a binary choice question in the topic of {example['subject']}. {MCQ_PROMPT}" + "\n"
    # prompt = ANS_TAG_PROMPT + "\n"
    prompt += "Question:\n"
    question = example["question"]
    prompt += question + "\n"
    # prompt += "Option:\n"
    # prompt += example["option_value"] + "\n"
    
    prompt += f'Is "{example["option_value"]}" a correct answer to the above question?\nA. True\nB. False'
        
    # prompt += "\nThink step by step about the correct answer in <think> </think> tags and PROVIDE YOUR FINAL ANSWER IN <answer> A/B/C/... </answer> tags. Your final answer should be a CAPITAL LETTER representing the choice you think is most likely to be correct."
    # prompt += "\nProvide your final answer in <answer> A/B/C/... </answer> tags. Your final answer should be a CAPITAL LETTER representing the choice you think is most likely to be correct."
    # prompt += f"\n{ANS_TAG_PROMPT}"
    # prompt += "\nPlease first provide your reasoning and then your final answer."
    
    # prompt += "Answer: Let's think step by step."
    # prompt += " /no_think"
    return prompt


doc_to_text = format_zero_shot
# doc_to_text = partial(format_cot_example, including_answer=False)
fewshot_to_text = partial(format_cot_example, including_answer=True)



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
    additional_dataset = datasets.load_dataset("AmeyaPrabhu/GPQA-Test1")
    additional_data = {item["Record ID"]: item for item in additional_dataset["train"]}
    
    def _process_doc(doc):
        # Check if the Record ID exists in the additional dataset
        record_id = doc["q_hash"]
        question = doc["question"]
        
        if record_id in additional_data:
            # Use the question from the additional dataset
            question = additional_data[record_id]["Question"]
        
        # choices = [
        #     preprocess(doc["Incorrect Answer 1"]),
        #     preprocess(doc["Incorrect Answer 2"]),
        #     preprocess(doc["Incorrect Answer 3"]),
        #     preprocess(doc["Correct Answer"]),
        # ]

        # random.shuffle(choices)
        # correct_answer_index = choices.index(preprocess(doc["Correct Answer"]))

        out_doc = {
            "question_id": doc["q_hash"],
            # "choice1": choices[0],
            # "choice2": choices[1],
            # "choice3": choices[2],
            # "choice4": choices[3],
            # "options": [choices[0], choices[1], choices[2], choices[3]],
            # "answer_index": correct_answer_index,
            "target": doc["option_value"],
            "category": doc["subject"],
            "question": question,
        }
        return out_doc

    return dataset.map(_process_doc)


# def process_docs(dataset):
#     # Filter to only keep Level 5 problems
#     # dataset = dataset.select(range(100))
#     return dataset

