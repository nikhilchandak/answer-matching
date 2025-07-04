import re
import datasets
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


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
        if eval_type == 'full':
            query = preprocess(doc["activity_label"] + ": " + ctx)
        elif eval_type == 'zero':
            query = preprocess(ctx)
        elif eval_type == 'placeholder':
            query = 'Lorem ipsum dolor sit amet, consectetur adipiscing elit. Morbi vel venenatis dui. Pellentesque sed cursus massa. ' + preprocess(ctx)
        else:
            raise NotImplementedError(eval_type + " is not implemented")
        out_doc = {
            "query": query,
            "choices": [preprocess(ending) for ending in doc["endings"]],
            "gold": int(doc["label"]),
        }
        return out_doc

    return dataset.map(_process_doc)


def run_likelihood_requests(dataset: datasets.Dataset, model, tokenizer) -> datasets.Dataset:
    def _classify_doc(doc):
        query = doc['query']
        choices = doc['choices']

        inputs = [tokenizer(query + ' ' + choice, return_tensors='pt') for choice in choices]

        likelihoods = []
        for input_ids in inputs:
            with torch.no_grad():
                input_ids_ = {k: v.cuda() for k, v in input_ids.items()}
                outputs = model(**input_ids_, labels=input_ids_['input_ids'])
                loss = outputs.loss
                likelihoods.append(-loss.item())

        return dict(zip(['nll_0', 'nll_1', 'nll_2', 'nll_3'], likelihoods))

    return dataset.map(_classify_doc)


def run_ll_requests(llm_name, eval_type='full'):
    llm = AutoModelForCausalLM.from_pretrained(llm_name).half().cuda()
    llm_tokenizer = AutoTokenizer.from_pretrained(llm_name)
    ds_ = run_likelihood_requests(ds, llm, llm_tokenizer)
    result_df = ds_.to_pandas()
    result_df[['nll_0', 'nll_1', 'nll_2', 'nll_3']].to_csv(f'data/log_likelihood/{eval_type}_prompt/{llm_name.replace("/", "--")}.csv',
                                                           index=False)


if __name__ == '__main__':
    eval_type = 'full'
    ds = load_dataset('hellaswag')['validation']
    ds = process_docs(ds, eval_type=eval_type)
    print(ds[0])
    for name in ['/fast/nchandak/models/Qwen3-4B/']:
        run_ll_requests(name, eval_type=eval_type)