# MCQ Classifier Training

This script trains a multiple-choice question (MCQ) classifier by using choices-only data using HuggingFace Transformers and Accelerate. It supports a variety of MCQ datasets with different numbers of answer options.

## Supported Lnaguage Datasets
- **super_gqpa**: SuperGOPQA (up to 10 options)
- **mmlu_pro**: MMLU-Pro (up to 10 options)
- **yourbench**: YourBench MMLU (4 options)
- **mmlu**: MMLU (4 options, uses `auxilary_train` split in the original dataset for training)
- **truthfulqa**: TruthfulQA v2 (2 options)
- **hellaswag**: HellaSwag (4 options)
- **goldenswag**: GoldenSwag (4 options, uses GoldenSwag for validation and HellaSwag train split for training)
- **arc**: ARC Challenge (4 options)

Most datasets don't have a train split, we split the test set 50-50 (randomized) and train our classifier on first half and test on the second half (held-out).

## Usage (MCQ Classifier)

To train the classifier, run:

```
accelerate launch --config_file zero3_config.yaml --num_processes {NUM_GPUS} train_classifier.py --dataset {DATASET_NAME} --token_limit {TOKEN_LIMIT}
```

- Replace `{NUM_GPUS}` with the number of GPUs to use.
- Replace `{DATASET_NAME}` with one of the supported datasets above.
- `{TOKEN_LIMIT}` is optional (default is 512; some datasets use 2048).

## VQA Benchmarks (MMMU Pro)

The `vqa_benchmarks.py` script trains a classifier on the MMMU Pro(Vision MCQ) benchmark using only the answer choices (no question text or images). It combines the "vision" and "standard (10 options)" splits, shuffles, and splits 50-50 for training and testing.

To run the VQA benchmark:

```
accelerate launch --config_file zero3_config.yaml --num_processes {NUM_GPUS} vqa_benchmarks.py
```

- The default model path is set to `/fast/nchandak/models/Qwen3-4B` in the script. Change this in the script if needed.
- Results and logs are saved to `./results/mmlu/validation` by default.

## Notes
- For most benchmarks with 4 options, DeBerta-v3-large is used for training (since it achieves similar performance as language models). For other benchmarks (with 10 options, like MMLU Pro), Qwen3-4B is used for training.
- Best results (reported in our paper) were obtained on a node with **8x H100 GPUs** and the training configs have been set accordingly. Training time can very from few mins (for example, for TruthfulQA v2) to hours (for SuperGPQA). - If you face memory issues, please decrease batch-size and increase gradient accumulation steps. 
