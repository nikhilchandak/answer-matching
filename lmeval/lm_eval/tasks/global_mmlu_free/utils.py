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
    prompt = f"You will be asked a question in the topic of {example['subject_category']}. Think step by step and then finish your answer with \"the answer is (X)\" where X is the correct, brief answer. The final answer X should be in the language asked in the question and concise and put in parentheses.\n"    # prompt = ANS_TAG_PROMPT + "\n"
    prompt += "Question:\n"
    
    prompt = "आपसे एक प्रश्न पूछा जाएगा. चरण दर चरण सोचें और फिर अपना उत्तर \"उत्तर है (X)\" के साथ समाप्त करें जहां X सही, संक्षिप्त उत्तर है। अंतिम उत्तर X प्रश्न में पूछी गई भाषा में और संक्षिप्त तथा कोष्ठक में होना चाहिए।\n"
    prompt += "सवाल:\n"
    
    question = example["question"]
    prompt += question + "\n"
    # prompt += "Answer: "
    prompt += "उत्तर: "
    
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

def doc_to_target(example):
    index = get_answer_index(example)
    options = get_options(example)
    return options[index]



doc_to_text = format_zero_shot
# doc_to_text = partial(format_cot_example, including_answer=False)
fewshot_to_text = partial(format_cot_example, including_answer=True)

only_question = partial(format_zero_shot, keep_options=False, cot=False)
both_qa = partial(format_zero_shot, keep_options=True, cot=False)

def process_docs(dataset):
    # Filter to only keep Level 5 problems
    # dataset = dataset.select(range(100))
    return dataset

