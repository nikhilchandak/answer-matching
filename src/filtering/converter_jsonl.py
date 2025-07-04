#!/usr/bin/env python
# coding: utf-8

import os
import json
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_json_to_jsonl(json_file_path, jsonl_file_path):
    """
    Convert a JSON file to JSONL format.
    
    Args:
        json_file_path: Path to the input JSON file
        jsonl_file_path: Path to the output JSONL file
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            logger.warning(f"{json_file_path} does not contain a JSON array. Skipping.")
            return False
        
        with open(jsonl_file_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        
        logger.info(f"Converted {json_file_path} to {jsonl_file_path}")
        return True
    
    except json.JSONDecodeError:
        logger.error(f"Error: Could not parse {json_file_path} as JSON. Skipping.")
        return False
    except Exception as e:
        logger.error(f"Error processing {json_file_path}: {str(e)}. Skipping.")
        return False

def process_directory(input_dir):
    """
    Walk through the directory and convert all JSON files to JSONL format.
    
    Args:
        input_dir: Directory containing JSON files
    """
    input_dir = Path(input_dir)
    
    if not input_dir.is_dir():
        logger.error(f"Error: {input_dir} is not a valid directory")
        return
    
    json_files = []
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.json'):
                json_files.append(Path(root) / file)
    
    logger.info(f"Found {len(json_files)} JSON files")
    
    success_count = 0
    for json_file in json_files:
        jsonl_file = json_file.with_suffix('.jsonl')
        if convert_json_to_jsonl(json_file, jsonl_file):
            success_count += 1
    
    logger.info(f"Successfully converted {success_count} out of {len(json_files)} files")

def main():
    parser = argparse.ArgumentParser(description="Convert JSON files to JSONL format")
    parser.add_argument("--input_dir", required=True, help="Directory containing JSON files")
    args = parser.parse_args()
    
    process_directory(args.input_dir)

if __name__ == "__main__":
    main()
