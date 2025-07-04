from functools import partial
import datasets
import random

choices = [
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
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
MCQ_PROMPT = """Think step by step and then finish your answer with \"the answer is (X)\" where X is the option (A/B/C/...) you think is most likely to be correct."""

def format_zero_shot(example): 
    prompt = f"You will be asked a question in the topic of {example['field']}. {MCQ_PROMPT}" + "\n"
    
    # prompt = MCQ_PROMPT + "\n"
    # prompt = ANS_TAG_PROMPT + "\n"
    prompt += "Question:\n"
    question = example["question"]
    options = example["options"]
    prompt += question + "\n"
    prompt += "Options:\n"
    for i, opt in enumerate(options):
        prompt += "{}. {}\n".format(choices[i], opt)
        
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


def process_docs(dataset):
    return dataset



def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    def _process_doc(doc):
        
        question_text = doc['question']
        answer = doc['ground_truth_answer'][1]
        
        out_doc = {
            "choice1": choices[0],
            "choice2": choices[1],
            "choice3": choices[2],
            "choice4": choices[3],
            "choices": [choices[0], choices[1], choices[2], choices[3]],
        }
        return out_doc

    return dataset.map(_process_doc)