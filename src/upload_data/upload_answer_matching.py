import os
import json
from pathlib import Path
from datasets import Dataset
from huggingface_hub import HfApi, create_repo
import time
from typing import List, Any

def load_jsonl_file(file_path: str) -> List[Any]:
    """
    Load a JSONL file and return list of dictionaries, converting bools to ints and question_id to str.
    """
    def convert_bools_to_ints(obj):
        if isinstance(obj, dict):
            return {k: convert_bools_to_ints(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_bools_to_ints(v) for v in obj]
        elif isinstance(obj, bool):
            return int(obj)
        else:
            return obj
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    obj = json.loads(line.strip())
                    obj = convert_bools_to_ints(obj)
                    if isinstance(obj, dict) and 'question_id' in obj:
                        obj['question_id'] = str(obj['question_id'])
                    data.append(obj)
                except json.JSONDecodeError as e:
                    print(f"Warning: Error parsing line {line_num} in {file_path}: {e}")
                    continue
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
    return data


def create_annotations_split(annotation_dir: Path, annotator_files: List[str], split_name: str) -> Dataset:
    combined_data = []
    for jsonl_file in annotator_files:
        file_path = annotation_dir / jsonl_file
        if file_path.exists():
            data = load_jsonl_file(str(file_path))
            if data:
                for item in data:
                    item['__source_file__'] = jsonl_file
                    item['__source_path__'] = str(annotation_dir)
                    item['__annotator__'] = jsonl_file.replace('_gpqa.jsonl', '').replace('.jsonl', '')
                combined_data.extend(data)
    if not combined_data:
        raise ValueError(f"No data found for split {split_name}")
    print(f"Created split '{split_name}' with {len(combined_data)} samples")
    return Dataset.from_list(combined_data)


def upload_directory_files(base_path: Path, repo_id: str, exclude_paths: List[Path]):
    api = HfApi()
    for root, dirs, files in os.walk(base_path):
        for file in files:
            file_path = Path(root) / file
            rel_path = file_path.relative_to(base_path)
            # Exclude annotation files that are part of splits
            if any(str(file_path).startswith(str(exclude)) for exclude in exclude_paths):
                continue
            print(f"Uploading file: {rel_path}")
            api.upload_file(
                path_or_fileobj=str(file_path),
                path_in_repo=str(rel_path),
                repo_id=repo_id,
                repo_type="dataset"
            )
            time.sleep(0.5)  # avoid rate limit


def describe_directory_structure(base_path: Path) -> str:
    description = []
    for root, dirs, files in os.walk(base_path):
        rel_root = Path(root).relative_to(base_path)
        if rel_root == Path('.'):
            rel_root = ''
        if files:
            description.append(f"### `{rel_root}`\n")
            for file in files:
                description.append(f"- `{file}`")
            description.append('')
    return '\n'.join(description)


def create_dataset_card(repo_id: str, base_path: Path) -> str:
    dir_description = describe_directory_structure(base_path)
    card_content = f"""---
license: mit
task_categories:
- question-answering
- text-classification
language:
- en
tags:
- evaluation
- answer-matching
- alignment
- human-annotations
- model-evaluation
size_categories:
- 10K<n<100K
---

# Answer Matching Dataset

This dataset contains a single split for human annotation analysis:

- **gpqa_diamond_annotations**: Combined GPQA Diamond annotations from all annotators (Ameya + Nikhil)

All other evaluation files are available in the "Files and versions" tab, preserving the original directory structure.

## Directory Structure and Data Overview

{dir_description}

## 🚀 Quick Start

```python
from datasets import load_dataset
# Load the default split (GPQA Diamond annotations)
gpqa_annotations = load_dataset('{repo_id}', split='gpqa_diamond_annotations')
```

## Data Schema
- **question_id**: Unique identifier for each question
- **__annotator__**: Name of the human annotator
- **__source_file__**: Original JSONL filename
- **__source_path__**: Original directory path
- Other fields depend on the annotation file

## License
MIT. Please also respect the licensing terms of the original GPQA and MMLU Pro datasets.
"""
    return card_content


def create_dataset_config():
    config = {
        "dataset_info": {
            "features": {
                "__source_file__": {"dtype": "string"},
                "__source_path__": {"dtype": "string"},
                "__annotator__": {"dtype": "string"},
                "question_id": {"dtype": "string"}
            },
            "splits": {
                "gpqa_diamond_annotations": {
                    "name": "gpqa_diamond_annotations",
                    "num_bytes": 0,
                    "num_examples": 0
                }
            },
            "download_size": 0,
            "dataset_size": 0
        },
        "default_split": "gpqa_diamond_annotations"
    }
    return config


def main():
    base_path = Path("/fast/nchandak/qaevals/hf_release")
    repo_id = "nikhilchandak/answer-matching"
    print(f"Starting upload of {base_path} to {repo_id}")
    if not base_path.exists():
        print(f"Error: Directory {base_path} does not exist")
        return
    print("Creating repository (if needed)...")
    try:
        create_repo(repo_id, repo_type="dataset", exist_ok=True)
    except Exception as e:
        print(f"Error creating repository: {e}")
        return
    # GPQA Diamond annotations
    gpqa_ann_dir = base_path / "alignment_plot" / "annotations" / "gpqa_diamond"
    gpqa_ann_files = ["ameya_gpqa.jsonl", "nikhil_gpqa.jsonl"]
    gpqa_split = create_annotations_split(gpqa_ann_dir, gpqa_ann_files, "gpqa_diamond_annotations")
    print("Uploading gpqa_diamond_annotations split...")
    gpqa_split.push_to_hub(repo_id, split="gpqa_diamond_annotations")
    # Upload all other files (excluding annotation files used above)
    print("Uploading all other files to repo (Files and versions tab)...")
    exclude_paths = [gpqa_ann_dir / f for f in gpqa_ann_files]
    upload_directory_files(base_path, repo_id, exclude_paths)
    # Upload README and config
    print("Uploading README and config...")
    api = HfApi()
    readme_content = create_dataset_card(repo_id, base_path)
    api.upload_file(
        path_or_fileobj=readme_content.encode('utf-8'),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset"
    )
    config = create_dataset_config()
    api.upload_file(
        path_or_fileobj=json.dumps(config, indent=2).encode('utf-8'),
        path_in_repo="dataset_infos.json",
        repo_id=repo_id,
        repo_type="dataset"
    )
    print(f"\n🎉 Upload complete! Only one split (gpqa_diamond_annotations) is available. All other files are in Files and versions tab.")

if __name__ == "__main__":
    main()