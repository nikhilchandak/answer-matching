import os
import json
from pathlib import Path

def clean_annotation_files(directory_path):
    """
    Recursively go through all .jsonl files in the directory and remove 'comments' field from each line.
    
    Args:
        directory_path (str): Path to the directory containing .jsonl files
    """
    directory = Path(directory_path)
    
    # Find all .jsonl files recursively
    jsonl_files = list(directory.rglob("*.jsonl"))
    
    print(f"Found {len(jsonl_files)} .jsonl files to process")
    
    for file_path in jsonl_files:
        print(f"Processing: {file_path}")
        
        # Read the file
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Process each line
        cleaned_lines = []
        for line_num, line in enumerate(lines, 1):
            try:
                data = json.loads(line.strip())
                
                # Remove 'comments' field if it exists
                if 'comments' in data:
                    del data['comments']
                    print(f"  Removed 'comments' field from line {line_num}")
                
                # Write back the cleaned data
                cleaned_lines.append(json.dumps(data, ensure_ascii=False) + '\n')
                
            except json.JSONDecodeError as e:
                print(f"  Warning: Could not parse JSON on line {line_num}: {e}")
                # Keep the original line if it can't be parsed
                cleaned_lines.append(line)
        
        # Write the cleaned data back to the file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.writelines(cleaned_lines)
        
        print(f"  Completed processing {file_path}")

if __name__ == "__main__":
    directory_path = "/fast/nchandak/qaevals/hf_release/alignment_plot/annotations"
    
    if not os.path.exists(directory_path):
        print(f"Error: Directory {directory_path} does not exist")
    else:
        clean_annotation_files(directory_path)
        print("Finished cleaning all annotation files")
