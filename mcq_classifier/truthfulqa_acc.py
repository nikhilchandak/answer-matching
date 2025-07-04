import json
import re

def calculate_accuracy():
    # Load the JSON data
    with open('truthfulqa_results.json', 'r') as file:
        data = json.load(file)
    
    total_questions = len(data['choices'])
    correct_answers = 0
    
    # Process each question
    for item in data['choices']:
        model_response = item['model_response']
        correct_choice = item['correct_choice']
        
        # Extract the first occurrence of A or B
        match = re.search(r'(?:Answer:)?[^\w]*(A|B)\b', model_response, re.IGNORECASE)
        
        if match:
            extracted_choice = match.group(1).upper()
            is_correct = extracted_choice == correct_choice
            
            # For debugging
            if not is_correct:
                print(f"Question {item['question_idx']}: '{item['raw_prompt']}'")
                print(f"Extracted: {extracted_choice}, Correct: {correct_choice}")
                print(f"Model response: {model_response[:100]}...\n")
            
            if is_correct:
                correct_answers += 1
        else:
            print(f"No choice found for question {item['question_idx']}: '{item['question']}'")
    
    # Calculate accuracy
    accuracy = correct_answers / total_questions if total_questions > 0 else 0
    
    print(f"\nResults:")
    print(f"Total questions: {total_questions}")
    print(f"Correct answers: {correct_answers}")
    print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Compare with the reported accuracy in the file
    reported_accuracy = data.get('accuracy', None)
    if reported_accuracy is not None:
        print(f"Reported accuracy in file: {reported_accuracy:.4f} ({reported_accuracy*100:.2f}%)")
        print(f"Difference: {abs(accuracy - reported_accuracy):.4f}")

if __name__ == "__main__":
    calculate_accuracy()