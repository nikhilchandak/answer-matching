import os
import json
import re
import random
from datasets import load_dataset
from vllm import LLM, SamplingParams

# Model and output config
model_name = "/fast/nchandak/models/Qwen3-4B/"
output_dir = "data/generation/full_prompt"
os.makedirs(output_dir, exist_ok=True)
filename = os.path.join(output_dir, "hellaswag_qwen3_4B_results.json")
random.seed(42)

def preprocess(text):
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    text = text.replace("  ", " ")
    return text

def process_docs(dataset, eval_type='full'):
    def _process_doc(doc):
        if eval_type == 'full':
            ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
        else:
            ctx = doc["ctx_b"].capitalize()
        choices = [preprocess(ending) for ending in doc["endings"]]
        gold_index = int(doc["label"])
        indices = list(range(len(choices)))
        random.shuffle(indices)
        shuffled_choices = [choices[i] for i in indices]
        new_gold_index = indices.index(gold_index)
        if eval_type == 'full':
            query = preprocess(doc["activity_label"] + ": " + ctx)
        else:
            query = preprocess(ctx)
        out_doc = {
            "query": query,
            "choices": shuffled_choices,
            "gold": new_gold_index,
        }
        return out_doc
    return dataset.map(_process_doc)

def construct_prompt(context, endings):
    prompt = (
        "You are given a situation followed by four possible endings. "
        "Choose the most appropriate ending by selecting the corresponding number. "
        "Respond only with the number of the correct answer.\n\n"
        f"Context: {context}\n"
    )
    for i, ending in enumerate(endings):
        prompt += f"{i + 1}. {ending}\n"
    prompt += "\nAnswer: "
    return prompt

def extract_number_answer(response):
    """Robustly extract the answer number (1-4) from the model's response."""
    # Look for the first digit 1-4 after 'Answer:' or at the start
    patterns = [
        r"Answer[:\s]*([1-4])",
        r"^([1-4])$",
        r"^([1-4])\.\s",
        r"option\s*([1-4])",
        r"number\s*([1-4])",
        r"([1-4])"
    ]
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            return match.group(1)
    # fallback: find first digit 1-4 in the response
    for c in response:
        if c in '1234':
            return c
    return None

def main():
    eval_type = 'full'
    # dataset = load_dataset("hellaswag", split="validation")
    dataset = load_dataset("PleIAs/GoldenSwag", split="validation")
    dataset = process_docs(dataset, eval_type=eval_type)

    prompts = []
    golds = []
    contexts = []
    options = []
    for example in dataset:
        context = example["query"]
        endings = example["choices"]
        correct_answer = str(example["gold"] + 1)  # Labels are 0-indexed
        prompt = construct_prompt(context, endings)
        prompts.append(prompt)
        golds.append(correct_answer)
        contexts.append(context)
        options.append(endings)

    # vLLM setup
    llm = LLM(model=model_name, tensor_parallel_size=1)
    sampling_params = SamplingParams(
        temperature=0.6,
        max_tokens=8192,
        top_p=0.95,
        n=1
    )

    # vLLM will handle batching automatically
    results = []
    correct = 0
    total = len(prompts)
    outputs = llm.generate(prompts, sampling_params)
    generations = [o.outputs[0].text.strip() for o in outputs]

    for i in range(total):
        generated_answer = extract_number_answer(generations[i])
        result = {
            "context": contexts[i],
            "options": options[i],
            "correct_answer": golds[i],
            "generated_answer": generated_answer if generated_answer is not None else generations[i]
        }
        results.append(result)
        if golds[i] == generated_answer:
            correct += 1

    accuracy = correct / total
    print(f"Initial Accuracy: {accuracy:.2%}")

    with open(filename, "w") as f:
        json.dump(results, f, indent=4)

    def compute_real_accuracy(filename):
        with open(filename, "r") as f:
            data = json.load(f)
        correct = 0
        for question in data:
            digits = [a for a in str(question['generated_answer']) if a in '1234']
            if len(digits) == 0:
                print('ERROR:', question)
            else:
                correct += int(digits[0] == question['correct_answer'])
        return correct / len(data)

    real_accuracy = compute_real_accuracy(filename)
    print(f"Real Accuracy: {real_accuracy:.2%}")

if __name__ == "__main__":
    main()