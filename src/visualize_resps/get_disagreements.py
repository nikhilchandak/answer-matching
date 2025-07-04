import json
import csv
import os

def load_annotations(file_path):
    """Load annotation data from JSONL file."""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        item = json.loads(line)
                        data.append(item)
                    except json.JSONDecodeError as e:
                        print(f"Error parsing line in {file_path}: {e}")
                        continue
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return []
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return []
    
    return data

def extract_rating(item, field="rating_osq"):
    """Extract OSQ rating from annotation item."""
    rating = item.get(field)
    
    # Handle list format (take first element)
    if isinstance(rating, list) and len(rating) > 0:
        return rating[0]
    elif isinstance(rating, (int, float)):
        return rating
    else:
        return None

def find_osq_disagreements():
    """Find OSQ rating disagreements between Ameya and Nikhil."""
    # File paths
    ameya_file = "annotation/saves/ameya_gpqa.jsonl"
    nikhil_file = "annotation/saves/nikhil_gpqa.jsonl"
    
    
    ameya_file = "updated_annotation/ameya_gpqa.jsonl"
    nikhil_file = "updated_annotation/nikhil_gpqa.jsonl"
    
    output_file = "osq_disagreements.csv"
    
    # Load data
    ameya_data = load_annotations(ameya_file)
    nikhil_data = load_annotations(nikhil_file)
    
    if not ameya_data or not nikhil_data:
        print("Failed to load annotation data")
        return
    
    # Create lookup dictionaries
    ameya_lookup = {item['question_id']: item for item in ameya_data}
    nikhil_lookup = {item['question_id']: item for item in nikhil_data}
    
    # Find common question IDs
    common_ids = set(ameya_lookup.keys()) & set(nikhil_lookup.keys())
    
    disagreements = []
    agreements = []
    
    for question_id in common_ids:
        ameya_item = ameya_lookup[question_id]
        nikhil_item = nikhil_lookup[question_id]
        
        ameya_osq = extract_rating(ameya_item, field="rating_osq")
        nikhil_osq = extract_rating(nikhil_item, field="rating_osq")
        
        ameya_multians = extract_rating(ameya_item, field="rating_multians")
        nikhil_multians = extract_rating(nikhil_item, field="rating_multians")
        
        # Skip if either rating is missing
        if ameya_osq is None or nikhil_osq is None:
            continue
        
        # Check for disagreement conditions:
        # 1. Difference >= 2
        # 2. One <= 3 and other >= 4
        difference = abs(ameya_osq - nikhil_osq)
        condition1 = difference >= 2
        condition2 = (ameya_osq <= 3 and nikhil_osq >= 4) or (ameya_osq >= 4 and nikhil_osq <= 3)
        
        if condition1 or condition2:
            disagreements.append({
                'question_id': question_id,
                'ameya_osq': ameya_osq,
                'nikhil_osq': nikhil_osq,
                'difference': difference
            })
            
        if ameya_osq >= 4 and nikhil_osq >= 4 and ameya_multians >= 4 and nikhil_multians >= 4:
            agreements.append({
                'question_id': question_id,
                'ameya_osq': ameya_osq,
                'nikhil_osq': nikhil_osq,
                'difference': difference
            })
    
    # Sort by difference (largest first)
    disagreements.sort(key=lambda x: x['difference'], reverse=True)
    print(f"Found {len(agreements)} questions with OSQ rating agreements")
    
    # Write to CSV
    try:
        # with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        #     fieldnames = ['question_id', 'ameya_osq', 'nikhil_osq', 'difference']
        #     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
        #     writer.writeheader()
        #     for disagreement in disagreements:
        #         writer.writerow(disagreement)
        
        # print(f"Successfully created {output_file} with {len(disagreements)} disagreements")
        print(f"Found {len(disagreements)} questions with OSQ rating disagreements")
        
        # Print summary statistics
        if disagreements:
            max_diff = max(d['difference'] for d in disagreements)
            min_diff = min(d['difference'] for d in disagreements)
            avg_diff = sum(d['difference'] for d in disagreements) / len(disagreements)
            
            print(f"Difference range: {min_diff} - {max_diff}")
            print(f"Average difference: {avg_diff:.2f}")
            
            # Show first few examples
            print("\nFirst 5 disagreements:")
            for i, d in enumerate(disagreements[:5]):
                print(f"  {d['question_id']}: Ameya={d['ameya_osq']}, Nikhil={d['nikhil_osq']}, Diff={d['difference']}")
    
    except Exception as e:
        print(f"Error writing CSV file: {e}")

if __name__ == "__main__":
    find_osq_disagreements()