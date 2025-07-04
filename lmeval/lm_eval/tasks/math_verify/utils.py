from functools import partial


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
    category = example["subject"]
    underscore = category.rfind("_")
    subject = category[:underscore]
    level = category[underscore+1:]
    
    prompt = f"You will be asked a binary choice question in the topic of {subject}. {MCQ_PROMPT}" + "\n"
    # prompt = ANS_TAG_PROMPT + "\n"
    prompt += "Question:\n"
    question = example["question"]
    prompt += question + "\n"
    # prompt += "Option:\n"
    # prompt += example["option_value"] + "\n"
    
    prompt += f'Is ${example["option_value"]}$ the correct answer to the above question?\nA. True\nB. False'
        
    # prompt += "\nThink step by step about the correct answer in <think> </think> tags and PROVIDE YOUR FINAL ANSWER IN <answer> A/B/C/... </answer> tags. Your final answer should be a CAPITAL LETTER representing the choice you think is most likely to be correct."
    # prompt += "\nProvide your final answer in <answer> A/B/C/... </answer> tags. Your final answer should be a CAPITAL LETTER representing the choice you think is most likely to be correct."
    # prompt += f"\n{ANS_TAG_PROMPT}"
    # prompt += "\nPlease first provide your reasoning and then your final answer."
    
    # prompt += "Answer: Let's think step by step."
    prompt += "\nAnswer: "
    # prompt += " /no_think"
    return prompt


doc_to_text = format_zero_shot
# doc_to_text = partial(format_cot_example, including_answer=False)
fewshot_to_text = partial(format_cot_example, including_answer=True)


def get_answer_index(example):
    idx = ord(example["option_correct"]) - ord("A")
    return idx

def process_docs(dataset):
    # Filter to only keep Level 5 problems
    def extract_level(subject):
        underscore = subject.rfind("_")
        if underscore != -1:
            return subject[underscore+1:]
        return subject
    
    
    def extract_subject(subject):
        underscore = subject.rfind("_")
        if underscore != -1:
            return subject[:underscore]
        return subject
    
    # Add a new column for Level
    dataset = dataset.map(lambda example: {"level": extract_level(example["subject"])})
    # dataset = dataset.map(lambda example: {"subject": extract_subject(example["subject"])})
    # Filter to only keep Level 5 problems
    filtered_dataset = dataset.filter(lambda example: example["level"] == "Level 5")
    # filtered_dataset = filtered_dataset.select(range(100))
    return filtered_dataset

