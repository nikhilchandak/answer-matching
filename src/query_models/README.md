# Query Models

This directory contains scripts for querying various language models with different types of questions and evaluating their responses.

## Files Overview


### `gpqa.py` and `gpqa_mcq.py`
Scripts for querying models with general-purpose QA tasks:
- `gpqa.py`: Handles open-ended questions
- `gpqa_mcq.py`: Handles multiple-choice questions

### `mmlu_pro.py`, `mmlu_pro_mcq.py`, and `mmlu_pro_verify.py`
Scripts for working with MMLU Pro dataset:
- `mmlu_pro.py`: Base implementation for MMLU Pro questions
- `mmlu_pro_mcq.py`: Handles multiple-choice MMLU Pro questions
- `mmlu_pro_verify.py`: Verifies and validates model responses for MMLU Pro

## Usage

All scripts follow a similar command-line interface:

```bash
python <script_name>.py [options]

Common options:
--input_dir: Directory containing input files (default varies by script)
--batch_size: Number of samples to process in a batch (default: 100)
--max_tokens: Maximum tokens for generation (default: 16384)
--temperature: Temperature for generation (default: 0.3)
```

Example:
```bash
python gpqa_mcq.py --input_dir /path/to/input --batch_size 50 --temperature 0.6
```

## Requirements

These scripts require:
- Python 3.7+
- OpenRouter API access
- Required Python packages (install via pip):
  - asyncio
  - tqdm
  - logging
  - json 