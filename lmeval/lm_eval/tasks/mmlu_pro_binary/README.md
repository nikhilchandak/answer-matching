# MMLU Pro Binary

This is a modified version of the MMLU-Pro benchmark that transforms multiple-choice questions into binary True/False questions presented as letter options (A or B).

## Description

The MMLU-Pro Binary benchmark converts each original multiple-choice question into a series of binary True/False questions. For each original question, we create a separate binary question for each possible answer option, asking whether that option is the correct answer to the original question.

For example, if the original question is:
> What is the capital of France?
> A. London
> B. Paris
> C. Berlin
> D. Rome

This gets transformed into four binary questions:
1. "Is London the correct answer to the question: 'What is the capital of France?'"
   > A. True
   > B. False
   (Answer: B)

2. "Is Paris the correct answer to the question: 'What is the capital of France?'"
   > A. True
   > B. False
   (Answer: A)

3. "Is Berlin the correct answer to the question: 'What is the capital of France?'"
   > A. True
   > B. False
   (Answer: B)

4. "Is Rome the correct answer to the question: 'What is the capital of France?'"
   > A. True
   > B. False
   (Answer: B)

## Implementation Details

- Each original question is expanded into multiple binary questions, one for each option
- The binary questions all have the same options: "A. True" and "B. False"
- The correct answer to each binary question is:
  - "A" if the option being evaluated is the correct answer to the original question
  - "B" otherwise
- This implementation uses 5-shot learning, where the model sees 5 examples with answers before being asked to answer a question
- Models are expected to follow the format:
  ```
  Answer: Let's think step by step.
  [reasoning steps]
  the answer is (A|B).
  ```
- The model's response is evaluated by extracting whether it answered "A" or "B"

## Evaluation Process

1. For each document in the dataset, we cycle through each option to create a binary question
2. For each binary question:
   - We present the question: "Is [option] the correct answer to the question: [original question]?"
   - We provide options: "A. True" and "B. False"
   - We provide 5 few-shot examples with similar binary questions and their step-by-step reasoning
   - We expect the model to reason and conclude with "the answer is A" or "the answer is B"
   - We extract the model's answer using a regex pattern that looks for this exact phrase
   - We compare the extracted answer with the ground truth

## Usage

To run an evaluation using MMLU-Pro Binary:

```
lm_eval --model <model_name> --tasks mmlu_pro_binary
```

Or for specific subjects:

```
lm_eval --model <model_name> --tasks mmlu_pro_binary_biology,mmlu_pro_binary_math
```

## Advantages

- Tests a model's ability to evaluate individual answer options independently
- May better assess a model's understanding of specific answer options
- Creates a more challenging evaluation by requiring the model to consider each option in isolation
- Provides more granular insights into model performance on different types of answer options
- Uses step-by-step reasoning to encourage better explanation of the model's decision process 