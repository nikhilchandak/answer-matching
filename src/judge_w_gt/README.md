# LLM Response Judging Framework

This directory contains a framework for evaluating language model responses using other language models as judges. The framework supports both ground-truth-based evaluation and free-form judging.

## Directory Structure

### Core Judging Scripts
- `gpqa_judge.py`: Evaluates responses to general-purpose question-answering tasks
- `judge_responses.py`: Base implementation of the judging framework
- `math_judge.py`: Specialized judge for mathematical responses and calculations

### Additional Components
- `local_judge/`: Implementation for running judges locally
- `notebooks/`: Analysis and visualization notebooks
- `run_multiple_judges.sh`: Utility script for running multiple judge models in parallel

## Features

- Ground-truth based evaluation
- Free-form response judging
- Support for multiple question types:
  - Multiple choice questions
  - Open-ended questions
  - Mathematical problems
- Batch processing for efficient evaluation
- Configurable judging criteria and prompts
- Extensible architecture for adding new judge types

## Usage

Basic usage pattern:

```bash
python gpqa_judge.py [options]

Common options:
--input_dir: Directory containing response files to judge
--batch_size: Number of samples to process in batch (default: 500)
--model: Judge model to use
--filtered_ids_path: Path to file containing filtered question IDs
--judge: Use free-form judging instead of matcher model
```

Example:
```bash
# Run ground-truth based evaluation
python gpqa_judge.py --input_dir /path/to/responses --batch_size 100

# Run free-form judging
python gpqa_judge.py --input_dir /path/to/responses --judge
```

## Judging Criteria

The framework supports two main judging approaches:

1. **Ground Truth Matching**
   - Compares responses against known correct answers
   - Supports partial credit and semantic matching
   - Configurable matching thresholds

2. **Free-form Judging**
   - Evaluates response quality without ground truth
   - Considers factors like:
     - Completeness
     - Correctness
     - Relevance
     - Clarity

## Requirements

- Python 3.7+
- Required packages:
  - asyncio
  - tqdm
  - logging
  - json
