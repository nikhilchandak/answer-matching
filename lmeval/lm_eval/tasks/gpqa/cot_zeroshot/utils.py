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
            "choices": [choices[0], choices[1], choices[2], choices[3]],
            "answer_index": correct_answer_index,
            "target": doc["Correct Answer"],
            "category": doc["Subdomain"],
            "question": doc["Question"],
        }
        return out_doc

    return dataset.map(_process_doc)


# choices = ["A", "B", "C", "D"]
# ANS_TAG_PROMPT = """Think step by step and provide your final answer choice (A/B/C/...) in <answer>  </answer> tags. Your final answer choice should be a CAPITAL LETTER representing the option you think is most likely to be correct."""
# MCQ_PROMPT = """Think step by step and then finish your answer with \"the answer is (X)\" where X is the option (A/B/C/...) you think is most likely to be correct."""

# def format_zero_shot(example):
#     template = f"The following is a question in {example['Subdomain']}. {MCQ_PROMPT}\n"
#     # prompt = ANS_TAG_PROMPT + "\n"
#     prompt = template + "Question:\n"
#     question = example["Question"]
    
#     options = example["options"]
    
#     prompt += question + "\n"
#     prompt += "Options:\n"
#     for i, opt in enumerate(options):
#         prompt += "{}. {}\n".format(choices[i], opt)
        
#     # prompt += "\nThink step by step about the correct answer in <think> </think> tags and PROVIDE YOUR FINAL ANSWER IN <answer> A/B/C/... </answer> tags. Your final answer should be a CAPITAL LETTER representing the choice you think is most likely to be correct."
#     # prompt += "\nProvide your final answer in <answer> A/B/C/... </answer> tags. Your final answer should be a CAPITAL LETTER representing the choice you think is most likely to be correct."
#     # prompt += f"\n{ANS_TAG_PROMPT}"
#     # prompt += "\nPlease first provide your reasoning and then your final answer."
    
#     # prompt += "Answer: Let's think step by step."
#     # prompt += " /no_think"
#     return prompt


# doc_to_text = format_zero_shot