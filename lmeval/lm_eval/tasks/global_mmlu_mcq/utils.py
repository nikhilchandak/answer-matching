from functools import partial


choices = [
    "A",
    "B",
    "C",
    "D",
]


def fewshot_only_question(example):
    prompt = "Question:\n"
    question = example["problem"]
    prompt += question + "\n"
    ans = example["solution"] + "\n\n"
    prompt += ans
    return prompt



def only_q_prompt(example):
    prompt = "Question:\n"
    question = example["Question"]
    prompt += question + "\n"
    prompt += "Answer:"
    
    # prompt += " /no_think"
    return prompt


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
MCQ_PROMPT = """Please reason step by step and then finish your answer with \"the answer is (X)\" where X is the option (A/B/C/...) you think is most likely to be correct."""

def format_zero_shot(example, keep_options=True, cot=True):
    prompt = f"You will be asked a question in the topic of {example['subject_category']}. {MCQ_PROMPT}" + "\n"
    # prompt = ANS_TAG_PROMPT + "\n"
    prompt += "Question:\n"
    question = example["question"]
    options = [example["option_a"], example["option_b"], example["option_c"], example["option_d"]]
    
    prompt += question + "\n"
    if keep_options:
        prompt += "Options:\n"
        for i, opt in enumerate(options):
            prompt += "{}. {}\n".format(choices[i], opt)
        
    # prompt += "\nThink step by step about the correct answer in <think> </think> tags and PROVIDE YOUR FINAL ANSWER IN <answer> A/B/C/... </answer> tags. Your final answer should be a CAPITAL LETTER representing the choice you think is most likely to be correct."
    # prompt += "\nProvide your final answer in <answer> A/B/C/... </answer> tags. Your final answer should be a CAPITAL LETTER representing the choice you think is most likely to be correct."
    # prompt += f"\n{ANS_TAG_PROMPT}"
    # prompt += "\nPlease first provide your reasoning and then your final answer."
    
    # if keep_options and cot:
    #     prompt += "Answer: Let's think step by step."
    # else :
    #     prompt += "Answer: "
        
    prompt += "Answer: "
    
    # prompt += " /no_think"
    return prompt


def get_options(example):
    return [example["option_a"], example["option_b"], example["option_c"], example["option_d"]]

def get_choices(example):
    options = [example["option_a"], example["option_b"], example["option_c"], example["option_d"]]
    options_str = []
    for i, opt in enumerate(options):
        options_str.append("{}. {}".format(choices[i], opt))
    return options_str

def get_answer_index(example):
    index = ord(example["answer"]) - ord("A")
    return index

doc_to_text = format_zero_shot
# doc_to_text = partial(format_cot_example, including_answer=False)
fewshot_to_text = partial(format_cot_example, including_answer=True)

only_question = partial(format_zero_shot, keep_options=False, cot=False)
both_qa = partial(format_zero_shot, keep_options=True, cot=False)

def process_docs(dataset):
    # Filter to only keep Level 5 problems
    # dataset = dataset.select(range(100))
    return dataset

