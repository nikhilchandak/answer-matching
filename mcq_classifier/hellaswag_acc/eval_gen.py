import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
import json
import re
import datasets
import random

model_name = "/fast/nchandak/models/Qwen3-4B/"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).half().cuda()
random.seed(42)


def preprocess(text):
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    text = text.replace("  ", " ")
    return text

def process_docs(dataset: datasets.Dataset, eval_type='full') -> datasets.Dataset:
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

def generate_answer(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_length=inputs.input_ids.shape[1] + 2,
            temperature=0.6,
            top_p=0.95
        )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
    return answer.split("Answer:")[-1].strip()

eval_type = 'full'
# dataset = load_dataset("hellaswag", split="validation")
dataset = load_dataset("PleIAs/GoldenSwag", split="validation")
dataset = process_docs(dataset, eval_type=eval_type)

correct = 0
total = len(dataset)
results = []

for example in dataset:
    context = example["query"]
    endings = example["choices"]
    correct_answer = str(example["gold"] + 1)  # Labels are 0-indexed
    prompt = construct_prompt(context, endings)
    generated_answer = generate_answer(prompt)

    results.append({
        "context": context,
        "options": endings,
        "correct_answer": correct_answer,
        "generated_answer": generated_answer
    })

    if correct_answer == generated_answer:
        correct += 1

accuracy = correct / total
print(f"Initial Accuracy: {accuracy:.2%}")

filename = f"data/generation/{eval_type}_prompt/hellaswag_qwen3_4B_results.json"

with open(filename, "w") as f:
    json.dump(results, f, indent=4)


def compute_real_accuracy(filename):
    with open(filename, "r") as f:
        data = json.load(f)

    correct = 0
    for question in data:
        digits = [a for a in question['generated_answer'] if a in '1234']
        if len(digits) == 0:
            print('ERROR:', question)
        else:
            correct += int(digits[0] == question['correct_answer'])

    return correct / len(data)


real_accuracy = compute_real_accuracy(filename)
print(f"Real Accuracy: {real_accuracy:.2%}")