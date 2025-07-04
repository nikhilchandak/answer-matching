from functools import partial
import json
import os

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

def both_verify_prompt(example):
    subject = example["subject"]
    
    # prompt = f"You will be asked a binary verification question in the topic of {subject}. {MCQ_PROMPT}" + "\n"
    prompt = f"You will be provided a question and response in the topic of {subject} and you have to check whether the response is a correct answer to the question." + "\n"
    
    
    # prompt = ANS_TAG_PROMPT + "\n"
    prompt += "Question:\n"
    question = example["question"]
    prompt += question + "\n"
    # prompt += "Option:\n"
    # prompt += example["option_value"] + "\n"
    
    prompt += f'Is "{example["option_value"]}" a correct answer to the above question?\nYou must respond only with "True" or "False". Do not output any other text.'
        
    # prompt += "\nThink step by step about the correct answer in <think> </think> tags and PROVIDE YOUR FINAL ANSWER IN <answer> A/B/C/... </answer> tags. Your final answer should be a CAPITAL LETTER representing the choice you think is most likely to be correct."
    # prompt += "\nProvide your final answer in <answer> A/B/C/... </answer> tags. Your final answer should be a CAPITAL LETTER representing the choice you think is most likely to be correct."
    # prompt += f"\n{ANS_TAG_PROMPT}"
    # prompt += "\nPlease first provide your reasoning and then your final answer."
    
    # prompt += "Answer: Let's think step by step."
    prompt += "\nAnswer:"
    # prompt += " /no_think"
    return prompt

def get_answer_index(example):
    idx = ord(example["option_correct"]) - ord("A")
    return idx


def format_zero_shot(example):
    subject = example["subject"]
    
    prompt = f"You will be asked a binary verification question in the topic of {subject}. {MCQ_PROMPT}" + "\n"
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


def process_docs(dataset):
    # Filter to only keep Level 5 problems
    # id_path = "/home/nchandak/qaevals/how-to-qa/src/filtering/data/mmlu_pro/filtered_ids.txt"
    # with open(id_path, "r") as f:
    #     filtered_ids = [int(line.strip()) for line in f.readlines()]
    # filtered_dataset = dataset.filter(lambda example: int(example["q_hash"]) in filtered_ids)
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
    filtered_dataset = dataset.filter(lambda example: int(example["q_hash"]) in filtered_ids)
    
    
    # filtered_dataset = filtered_dataset.select(range(100))
    return filtered_dataset

