from functools import partial
import json
import os


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
]



def fewshot_only_question(example):
    prompt = "Question:\n"
    question = example["question"]
    prompt += question + "\n"
    ans = example["options"][example["answer_index"]]
    prompt += "Answer: " + ans + "\n\n"
    return prompt


def only_question_prompt(example):
    prompt = ""
    prompt += "Question:\n"
    question = example["question"]
    prompt += question + "\n"
    prompt += "Answer:"
    return prompt


def fewshot_both_qa(example, is_few_shot=True):
    prompt = ""
    prompt += "Question:\n"
    question = example["question"]
    prompt += question + "\n"

    options = example["options"]
    prompt += "Options:\n"
    for i, opt in enumerate(options):
        prompt += "{}. {}\n".format(choices[i], opt)
    
    ans = f"{example["answer"]}. {example["options"][example["answer_index"]]}"
    prompt += "Answer:" 

    if is_few_shot:
        prompt += f" {ans}"
        prompt += "\n\n"
    
    return prompt

both_qa = partial(fewshot_both_qa, is_few_shot=False)

ANS_TAG_PROMPT = """Think step by step and provide your final answer choice (A/B/C/...) in <answer>  </answer> tags. Your final answer choice should be a CAPITAL LETTER representing the option you think is most likely to be correct."""
MCQ_PROMPT = """Please reason step by step and then finish your answer with \"the answer is (X)\" where X is the option (A or B) you think is most likely to be correct."""

def format_zero_shot(example):
    prompt = f"You will be asked a question in the topic of {example['subject']}. Please reason step by step, and put your final answer within \\boxed{{}}." + "\n"
    # prompt = ANS_TAG_PROMPT + "\n"
    prompt += "Question:\n"
    question = example["question"]
    prompt += question + "\n"
    # prompt += "Option:\n"
    # prompt += example["option_value"] + "\n"
    
    prompt += f'Is "{example["option_value"]}" a correct answer to the above question?\nPlease provide your probability of the answer being correct inside \\boxed{{}}.'
       
    
    # prompt += "\nThink step by step about the correct answer in <think> </think> tags and PROVIDE YOUR FINAL ANSWER IN <answer> A/B/C/... </answer> tags. Your final answer should be a CAPITAL LETTER representing the choice you think is most likely to be correct."
    # prompt += "\nProvide your final answer in <answer> A/B/C/... </answer> tags. Your final answer should be a CAPITAL LETTER representing the choice you think is most likely to be correct."
    # prompt += f"\n{ANS_TAG_PROMPT}"
    # prompt += "\nPlease first provide your reasoning and then your final answer."
    
    # prompt += "Answer: Let's think step by step."
    # prompt += " /no_think"
    return prompt



def create_prompt(example, include_question=True, include_options=True):
    prompt = ""
    if include_question:
        prompt = f"You will be asked a question in the topic of {example['category']}.\n"
    elif include_options:
        prompt = f"You will be given options to a question in the topic of {example['category']}. You have to choose the most likely option.\n"
        prompt += f"Think step by step about each of the options and then finish your answer with \"the answer is (X)\" where X is the option (A/B/C...) you think is most likely to be correct.\n"
    
    if include_question:
        prompt += "Question:\n"
        question = example["question"]
        prompt += question + "\n"
    
    if include_options:
        options = example["options"]
        prompt += "Options:\n"
        for i, opt in enumerate(options):
            prompt += "{}. {}\n".format(choices[i], opt)
    
    prompt += "Answer:"

    return prompt

def get_choices(example):
    options = example["options"]
    prompt = "Options:\n"
    new_options = []
    for i, opt in enumerate(options):
        new_options.append("{}. {}\n".format(choices[i], opt))
    return new_options


# both_qa = partial(create_prompt, include_question=True, include_options=True)
# only_question = partial(create_prompt, include_question=True, include_options=False)
only_question = only_question_prompt
only_options = partial(create_prompt, include_question=False, include_options=True)
no_qa = partial(create_prompt, include_question=False, include_options=False)

def process_docs(dataset, few_shot=False):
    # Filter to only keep Level 5 problems
    # id_path = "/home/nchandak/qaevals/how-to-qa/src/filtering/data/mmlu_pro/filtered_ids.txt"
    # with open(id_path, "r") as f:
    #     filtered_ids = [int(line.strip()) for line in f.readlines()]
    # filtered_dataset = dataset.filter(lambda example: int(example["q_hash"]) in filtered_ids)
    
    if few_shot or len(dataset) < 100000:
        return dataset
    
    # Load the combined samples file
    samples_path = "/fast/nchandak/qaevals/judge_outputs/mmlu_pro_free/stratified_sample/annotations/combined_samples_to_annotate.jsonl"
    filtered_ids = []
    
    if os.path.exists(samples_path):
        with open(samples_path, "r") as f:
            for line in f:
                sample = json.loads(line)
                # Only keep question_ids where the model doesn't contain "gemini"
                if "model" in sample and "gemini" not in sample["model"] and "question_id" in sample:
                    filtered_ids.append(sample["question_id"])
    
    # Filter the dataset to keep only rows with q_hash in filtered_ids
    filtered_dataset = dataset.filter(lambda example: int(example["question_id"]) in filtered_ids)
    
    # print(filtered_ids)
    # print(len(dataset))
    # print(len(filtered_dataset))
    
    # filtered_dataset = filtered_dataset.select(range(100))
    return filtered_dataset


doc_to_text = format_zero_shot
fewshot_process_docs = partial(process_docs, few_shot=True)
