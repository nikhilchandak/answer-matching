# Freeform Datasets

This repository contains two carefully curated datasets for evaluating large language models on human-filtered subset of popular benchmarks which are suitable for evaluation in freeform (open-ended) format.

## Dataset Structure

The repository contains two splits:

### 1. `gpqa_diamond` Split
- **Source**: Filtered subset of [Idavidrein/gpqa](https://huggingface.co/datasets/Idavidrein/gpqa) (gpqa_diamond config)
- **Format**: Questions from GPQA Diamond dataset with freeform answers
- **Filtering**: Human-curated questions that meet quality criteria (questions specific enough to be answerable in free-form format and have a unique answer)
- **Count**: 126 questions

### 2. `mmlu_pro` Split  
- **Source**: Filtered subset of [TIGER-Lab/MMLU-Pro](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro)
- **Format**: MMLU-Pro questions converted to freeform format with text answers
- **Filtering**: Human-curated questions that meet quality criteria (questions specific enough to be answerable in free-form format and have a unique answer)
- **Count**: 493 questions

## Schema

Both splits share the same schema for consistency:

| Column | Type | Description |
|--------|------|-------------|
| `question` | string | The question text |
| `answer` | string | The expected answer in freeform text |
| `question_id` | string | Unique identifier for the question (GPQA: `Record ID`, MMLU Pro: `question_id`) |
| `category` | string | Subject category (GPQA: `subdomain`, MMLU-Pro: `category`) |
| `Canary String` | string | Special marker (present in GPQA, empty in MMLU-Pro) |

## Data Sources and Processing

### GPQA Diamond Processing
1. Started with filtered GPQA Diamond questions from human annotations
2. Mapped questions to their original subdomains from the source dataset
3. Preserved original answer format and canary strings
4. Standardized column names for consistency

### MMLU-Pro Processing  
1. Identified high-quality questions through human annotation filtering
2. Extracted questions from MCQ format JSONL files
3. Retrieved corresponding freeform answers from generative evaluation files
4. Mapped question IDs to subject categories from the original TIGER-Lab/MMLU-Pro dataset
5. Created standardized CSV format before conversion to dataset

## Usage

```python
from datasets import load_dataset

# Load the entire dataset
dataset = load_dataset("nikhilchandak/freeform-datasets")

# Load specific splits
gpqa_data = load_dataset("nikhilchandak/freeform-datasets", split="gpqa_diamond")
mmlu_data = load_dataset("nikhilchandak/freeform-datasets", split="mmlu_pro")

# Example usage
for example in gpqa_data:
    question = example["question"]
    answer = example["answer"] 
    category = example["category"]
    print(f"Q: {question}")
    print(f"A: {answer}")
    print(f"Category: {category}\n")
```

## Categories

### GPQA Diamond Categories (Subdomains)
- Physics subdomains (e.g., quantum mechanics, thermodynamics)
- Chemistry subdomains (e.g., organic chemistry, physical chemistry)  
- Biology subdomains (e.g., molecular biology, genetics)

### MMLU-Pro Categories
- Various academic subjects from the MMLU-Pro taxonomy
- Includes STEM fields, humanities, social sciences, etc.

## Citation

If you use this dataset, please cite the original sources:

**For GPQA Diamond:**
```bibtex
@article{rein2023gpqa,
  title={GPQA: A Graduate-Level Google-Proof Q\&A Benchmark},
  author={Rein, David and Hou, Betty Li and Stickland, Asa Cooper and Petty, Jackson and Pang, Richard Yuanzhe and Dirani, Julien and Michael, Julian and Bowman, Samuel R},
  journal={arXiv preprint arXiv:2311.12022},
  year={2023}
}
```

**For MMLU-Pro:**
```bibtex
@article{wang2024mmlu,
  title={MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark},
  author={Wang, Yubo and Ma, Xueguang and Zhang, Ge and Ni, Yuansheng and Chandra, Abhranil and Guo, Shiguang and Ren, Weiming and Arulraj, Aaran and He, Xuan and Jiang, Ziyan and others},
  journal={arXiv preprint arXiv:2406.01574},
  year={2024}
}
```

## License

This dataset follows the licensing terms of the original datasets. Please refer to the original dataset pages for specific license information.

## Contact

For questions or issues regarding this dataset, please contact the repository maintainer. 