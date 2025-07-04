import os
import json
from collections import defaultdict

# Paths
ANNOTATION_DIR = "updated_annotation0"
SAMPLES_DIR = "/fast/nchandak/qaevals/judge_outputs/gpqa_diamond_free/"
OUTPUT_DIR = "updated_annotation1"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Find all *_gpqa.jsonl files
gpqa_files = [f for f in os.listdir(ANNOTATION_DIR) if f.endswith('_gpqa.jsonl')]

for gpqa_file in gpqa_files:
    print(f"Processing {gpqa_file}...")
    
    ANNOTATION_PATH = os.path.join(ANNOTATION_DIR, gpqa_file)
    OUTPUT_PATH = os.path.join(OUTPUT_DIR, gpqa_file)
    
    # Helper: map annotation model names to samples file names
    MODEL_TO_FILE = {}
    
    # Step 1: Collect all model names from the annotation file
    model_names = set()
    with open(ANNOTATION_PATH, 'r') as f:
        for line in f:
            entry = json.loads(line)
            for m in entry["model"]:
                model_names.add(m)
    
    # Step 2: Map model names to sample file names
    # Heuristic: samples_{model_name.replace('/', '-').replace('.', '-')}.jsonl
    for m in model_names:
        base = m.split('/')[-1] if '/' in m else m
        candidates = [
            f for f in os.listdir(SAMPLES_DIR)
            if f.startswith("samples_") and (base in f or m.replace('/', '-') in f or m.replace('/', '.') in f)
        ]
        if not candidates:
            dash_name = m.replace('/', '-').replace('.', '-')
            print("Name:", dash_name)
            candidates = [f for f in os.listdir(SAMPLES_DIR) if f"samples_{dash_name}.jsonl" in f]
        if candidates:
            print("Candidate found:", candidates)
            MODEL_TO_FILE[m] = candidates[0]
        else:
            raise ValueError(f"No samples file found for model: {m}")
    
    # Step 3: Load all responses and questions from each samples file, mapping (question_id -> response, question)
    model_responses = defaultdict(dict)  # model -> {question_id: response}
    model_questions = defaultdict(dict)  # model -> {question_id: question}
    model_filtered_resps = defaultdict(dict)  # model -> {question_id: response}
    actual_answers = defaultdict(dict)  # {question_id: answer}
    for m, fname in MODEL_TO_FILE.items():
        path = os.path.join(SAMPLES_DIR, fname)
        with open(path, 'r') as f:
            for line in f:
                obj = json.loads(line)
                qid = obj.get("question_id")
                # Try to find the response field
                if "resps" in obj:
                    resp = obj["resps"]
                elif "response" in obj:
                    resp = obj["response"]
                elif "answer" in obj:
                    resp = obj["answer"]
                else:
                    resp = next((v for k, v in obj.items() if k != "question_id" and isinstance(v, str)), None)
                model_responses[m][qid] = resp
                # Try to find the question field
                if "question" in obj:
                    question = obj["question"]
                elif "question_text" in obj:
                    question = obj["question_text"]
                else:
                    question = None
                model_questions[m][qid] = question
                
                if "filtered_resps" in obj:
                    filtered_resps = obj["filtered_resps"]
                    if qid == "recNu3MXkvWUzHZr9":
                        print(m, filtered_resps, path)
                    model_filtered_resps[m][qid] = filtered_resps
                
                if "answer" in obj:
                    if qid not in actual_answers:
                        actual_answers[qid] = obj["answer"]
                    else:
                        if actual_answers[qid].strip().lower() != obj["answer"].strip().lower():
                            print(f"Model {m} Question {qid} has multiple answers: {actual_answers[qid]} and {obj['answer']}")
                        assert actual_answers[qid].strip().lower() == obj["answer"].strip().lower()
        
        print(f"Model {m} has {len(actual_answers)} answers")
        
    # Step 4: Update annotation file
    with open(ANNOTATION_PATH, 'r') as fin, open(OUTPUT_PATH, 'w') as fout:
        for line in fin:
            entry = json.loads(line)
            models = entry["model"]
            qid = entry["question_id"]
            new_resps = []
            new_filtered_resps = []
            for m in models:
                resp = model_responses[m].get(qid, "")
                new_resps.append(resp)
                filtered_resp = model_filtered_resps[m].get(qid, "")
                new_filtered_resps.append(filtered_resp)
            
            entry["full_response"] = new_resps
            entry["response"] = new_filtered_resps
            entry["answer"] = actual_answers[qid]
                
            assert len(entry["answer"]) > 0 
            
            # Update question_text from the first model's sample file if available
            first_model = models[0]
            new_question = model_questions[first_model].get(qid)
            if new_question:
                entry["question_text"] = new_question
            fout.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"Updated annotation file written to {OUTPUT_PATH}")

print("All GPQA files processed successfully!")
