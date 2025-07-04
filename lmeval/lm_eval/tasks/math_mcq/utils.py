from functools import partial


choices = [
    "A",
    "B",
    "C",
    "D",
]


def list_fewshot_samples() -> list[dict]:
    return [
        {
            "problem": "Find the domain of the expression  $\\frac{\\sqrt{x-2}}{\\sqrt{5-x}}$.}",
            "solution": "Answer: $[2,5)$.",
            "few_shot": "1",
        },
        {
            "problem": "If $\\det \\mathbf{A} = 2$ and $\\det \\mathbf{B} = 12,$ then find $\\det (\\mathbf{A} \\mathbf{B}).$",
            "solution": "Answer: $24$.",
            "few_shot": "1",
        },
        {
            "problem": "Terrell usually lifts two 20-pound weights 12 times. If he uses two 15-pound weights instead, how many times must Terrell lift them in order to lift the same total weight?",
            "solution": "Answer: $16$.",
            "few_shot": "1",
        },
        {
            "problem": "If the system of equations\n\n\\begin{align*}\n6x-4y&=a,\\\n6y-9x &=b.\n\\end{align*}has a solution $(x, y)$ where $x$ and $y$ are both nonzero,\nfind $\\frac{a}{b},$ assuming $b$ is nonzero.",
            "solution": "Answer: $-\\frac{2}{3}$.",
            "few_shot": "1",
        },
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
    prompt = f"You will be asked a question in the topic of {example['Type']}. {MCQ_PROMPT}" + "\n"
    # prompt = ANS_TAG_PROMPT + "\n"
    prompt += "Question:\n"
    question = example["Question"]
    options = [example["A"], example["B"], example["C"], example["D"]]
    
    prompt += question + "\n"
    if keep_options:
        prompt += "Options:\n"
        for i, opt in enumerate(options):
            prompt += "{}. {}\n".format(choices[i], opt)
        
    # prompt += "\nThink step by step about the correct answer in <think> </think> tags and PROVIDE YOUR FINAL ANSWER IN <answer> A/B/C/... </answer> tags. Your final answer should be a CAPITAL LETTER representing the choice you think is most likely to be correct."
    # prompt += "\nProvide your final answer in <answer> A/B/C/... </answer> tags. Your final answer should be a CAPITAL LETTER representing the choice you think is most likely to be correct."
    # prompt += f"\n{ANS_TAG_PROMPT}"
    # prompt += "\nPlease first provide your reasoning and then your final answer."
    
    if keep_options and cot:
        prompt += "Answer: Let's think step by step."
    else :
        prompt += "Answer: "
    
    # prompt += " /no_think"
    return prompt


def get_options(example):
    return [f"${example["A"]}$", f"${example["B"]}$", f"${example["C"]}$", f"${example["D"]}$"]

def get_choices(example):
    options = [example["A"], example["B"], example["C"], example["D"]]
    options_str = []
    for i, opt in enumerate(options):
        options_str.append("{}. {}".format(choices[i], opt))
    return options_str

def get_answer_index(example):
    index = ord(example["Answer"]) - ord("A")
    return index

doc_to_text = format_zero_shot
# doc_to_text = partial(format_cot_example, including_answer=False)
fewshot_to_text = partial(format_cot_example, including_answer=True)

only_question = partial(format_zero_shot, keep_options=False, cot=False)
both_qa = partial(format_zero_shot, keep_options=True, cot=False)

def process_docs(dataset):
    # Filter to only keep Level 5 problems
    dataset = dataset.filter(lambda example: example["Level"] == "Level 5")
    # dataset = dataset.select(range(100))
    return dataset

