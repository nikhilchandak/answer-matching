import json
import os

def convert_lists_to_single_values(data):
    """Convert all list fields to single values by taking the first element."""
    if isinstance(data, dict):
        result = {}
        for key, value in data.items():
            if isinstance(value, list) and len(value) > 0:
                # Take the first element of the list
                result[key] = value[0]
            elif isinstance(value, list) and len(value) == 0:
                # Handle empty lists
                result[key] = None
            elif isinstance(value, dict):
                # Recursively process nested dictionaries
                result[key] = convert_lists_to_single_values(value)
            else:
                # Keep non-list values as they are
                result[key] = value
        return result
    elif isinstance(data, list):
        # If the top-level data is a list, process each item
        return [convert_lists_to_single_values(item) for item in data]
    else:
        return data

def process_nikhil_gpqa():
    """Process the nikhil_gpqa.jsonl file and create nikhil_gpqa2.jsonl."""
    input_file = "annotation/saves/ameya_gpqa.jsonl"
    output_file = "annotation/saves/ameya_gpqa2.jsonl"
    
    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}")
        return
    
    processed_count = 0
    
    try:
        with open(input_file, 'r', encoding='utf-8') as infile, \
             open(output_file, 'w', encoding='utf-8') as outfile:
            
            for line_num, line in enumerate(infile, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    # Parse the JSON line
                    data = json.loads(line)
                    
                    # Convert lists to single values
                    processed_data = convert_lists_to_single_values(data)
                    
                    # Write the processed data to the output file
                    json.dump(processed_data, outfile, ensure_ascii=False)
                    outfile.write('\n')
                    
                    processed_count += 1
                    
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON on line {line_num}: {e}")
                    continue
                except Exception as e:
                    print(f"Error processing line {line_num}: {e}")
                    continue
    
    except Exception as e:
        print(f"Error processing file: {e}")
        return
    
    print(f"Successfully processed {processed_count} rows")
    print(f"Output saved to: {output_file}")

if __name__ == "__main__":
    process_nikhil_gpqa()