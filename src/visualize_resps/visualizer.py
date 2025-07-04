import json
import os
import argparse
import glob
from collections import defaultdict
from flask import Flask, render_template, request, jsonify
from datasets import load_dataset

def create_app(input_path, annotator_id, multi_model=False, debug_limit=None):
    app = Flask(__name__, 
                template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates'))
    
    # Load HuggingFace dataset if annotator_id contains 'ameya'
    hf_dataset = None
    if multi_model and 'ameya' in annotator_id.lower():
        try:
            hf_dataset = load_dataset("AmeyaPrabhu/GPQA-Anno2")
            print("Successfully loaded HuggingFace dataset for Ameya")
            
            # Calculate and print score frequencies
            osq_scores = defaultdict(int)
            multians_scores = defaultdict(int)
            
            for item in hf_dataset['train']:
                # Map "Is non-MCQ?" to rating_osq (1-5 scale)
                is_non_mcq = item.get('Is non-MCQ?')
                if is_non_mcq is not None:
                    try:
                        osq_score = int(is_non_mcq)
                        if not (1 <= osq_score <= 5):
                            raise ValueError(f"OSQ score {osq_score} out of range 1-5")
                        osq_scores[osq_score] += 1
                    except (ValueError, TypeError) as e:
                        print(f"Invalid OSQ value: {is_non_mcq} - {e}")
                
                # Map "Is Unique?" to rating_multians (1-5 scale)
                is_unique = item.get('Is Unique?')
                if is_unique is not None:
                    try:
                        multians_score = int(is_unique)
                        if not (1 <= multians_score <= 5):
                            raise ValueError(f"Multians score {multians_score} out of range 1-5")
                        multians_scores[multians_score] += 1
                    except (ValueError, TypeError) as e:
                        print(f"Invalid Unique value: {is_unique} - {e}")
            
            print("Rating OSQ (Is non-MCQ?) score frequencies:")
            for score in range(1, 6):
                print(f"  Score {score}: {osq_scores[score]} items")
            
            print("Rating Multians (Is Unique?) score frequencies:")
            for score in range(1, 6):
                print(f"  Score {score}: {multians_scores[score]} items")
                
        except Exception as e:
            print(f"Error loading HuggingFace dataset: {e}")
    
    def load_data():
        data = []
        
        # Load data from JSONL file
        with open(input_path, 'r') as f:
            for i, line in enumerate(f):
                # If in debug mode, only load limited number of items
                if debug_limit is not None and i >= debug_limit:
                    break
                    
                if line.strip():
                    try:
                        item = json.loads(line)
                        # Always store question_id as string unless it's an int in the input
                        if not isinstance(item["question_id"], int):
                            item["question_id"] = str(item["question_id"])
                        data.append(item)
                    except Exception as e:
                        print(f"Error processing line: {e}")
        
        # If multi_model is enabled, group data by question_id
        if multi_model:
            grouped_data = defaultdict(list)
            for item in data:
                qid = item["question_id"]
                grouped_data[qid].append(item)
            
            processed_data = []
            for qid, items in grouped_data.items():
                # Get common data from first item
                base_item = items[0].copy()
                base_item["models"] = []
                base_item["responses"] = []
                base_item["thinking_list"] = []
                
                # Extract model-specific data
                for item in items:
                    base_item["models"].append(item.get("model", ""))
                    base_item["responses"].append(item.get("filtered_resps", ""))
                    base_item["thinking_list"].append(item.get("thinking", ""))
                
                processed_data.append(base_item)
                
            return processed_data
        else:
            return data
    
    def load_annotations():
        annotations = {}
        save_dir = os.path.join("annotation", "saves")
        os.makedirs(save_dir, exist_ok=True)
        
        save_path = os.path.join(save_dir, f"{annotator_id}.jsonl")
        if os.path.exists(save_path):
            try:
                with open(save_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            try:
                                item = json.loads(line)
                                key = item["question_id"]
                                if not isinstance(key, int):
                                    key = str(key)
                                annotations[key] = item
                            except Exception as e:
                                print(f"Error loading annotation: {e}")
            except Exception as e:
                print(f"Error loading annotations file: {e}")
        
        return annotations
    
    # Cache data loading
    all_data = load_data()
    annotations = load_annotations()
    
    @app.route('/')
    def index():
        question_ids = [str(item["question_id"]) for item in all_data]
        return render_template('index.html', 
                               question_ids=question_ids,
                               annotator_id=annotator_id,
                               multi_model=multi_model,
                               use_hf_dataset=hf_dataset is not None)
    
    @app.route('/api/completed')
    def completed_questions():
        # Get list of all completed questions
        completed = []
        for qid, annotation in annotations.items():
            required_fields = ["rating_osq", "rating_multians"]
            if multi_model:
                has_all_match_ratings = (
                    "rating_match" in annotation and 
                    isinstance(annotation["rating_match"], list) and
                    len(annotation["rating_match"]) == len(annotation.get("model", []))
                )
                if has_all_match_ratings and all(key in annotation and annotation[key] is not None for key in required_fields):
                    completed.append(str(qid))
            else:
                required_fields.append("rating_match")
                match_val = annotation.get("rating_match", 0)
                multians_val = annotation.get("rating_multians", 0)
                needs_correctness = (1 <= match_val <= 3) and (1 <= multians_val <= 3)
                if all(key in annotation and annotation[key] is not None for key in required_fields):
                    if not needs_correctness or annotation.get("rating_correct") is not None:
                        completed.append(str(qid))
        return jsonify({"completed_questions": completed})
    
    @app.route('/api/question/<question_id>')
    def get_question(question_id):
        # Always treat question_id as string unless explicitly int in data
        qid = str(question_id)
        question = next((item for item in all_data if str(item["question_id"]) == qid), None)
        if question is None:
            return jsonify({"error": "Question not found"}), 404
        
        question_annotations = annotations.get(qid, {})
        
        # If using HuggingFace dataset and this question isn't already annotated
        if hf_dataset is not None and multi_model:
            try:
                # Find matching record in HF dataset
                record = next((item for item in hf_dataset['train'] if str(item['Record ID']) == qid), None)
                if record:
                    # Map values from HF dataset to our annotation format
                    if 'rating_osq' not in question_annotations:
                        # Map "Is non-MCQ?" to rating_osq (1-5 scale)
                        is_non_mcq = record.get('Is non-MCQ?')
                        if is_non_mcq is not None:
                            try:
                                osq_score = int(is_non_mcq)
                                if not (1 <= osq_score <= 5):
                                    raise ValueError(f"OSQ score {osq_score} out of range 1-5")
                                question_annotations['rating_osq'] = osq_score
                                # print(f"Loaded osq value for {qid}: {is_non_mcq} -> {question_annotations['rating_osq']}")
                            except (ValueError, TypeError) as e:
                                print(f"Invalid OSQ value for question {qid}: {is_non_mcq} - {e}")
                    
                    if 'rating_multians' not in question_annotations:
                        # Map "Is Unique?" to rating_multians (1-5 scale)
                        is_unique = record.get('Is Unique?')
                        if is_unique is not None:
                            try:
                                multians_score = int(is_unique)
                                if not (1 <= multians_score <= 5):
                                    raise ValueError(f"Multians score {multians_score} out of range 1-5")
                                question_annotations['rating_multians'] = multians_score
                                # print(f"Loaded multians value for {qid}: {is_unique} -> {question_annotations['rating_multians']}")
                            except (ValueError, TypeError) as e:
                                print(f"Invalid Unique value for question {qid}: {is_unique} - {e}")
            except Exception as e:
                print(f"Error loading HF dataset values for question {qid}: {e}")
        
        return jsonify({
            "question": question,
            "annotations": question_annotations,
            "multi_model": multi_model
        })
    
    @app.route('/api/questions')
    def get_questions():
        question_ids = [str(item["question_id"]) for item in all_data]
        return jsonify({"question_ids": question_ids})
    
    @app.route('/api/save', methods=['POST'])
    def save_annotations():
        data = request.json
        question_id = data.get("question_id")
        qid_str = str(question_id)
        if question_id:
            if multi_model:
                annotation = {
                    "question_id": question_id,
                    "model": data.get("model", []),
                    "thinking": data.get("thinking", []),
                    "question_text": data.get("question_text", ""),
                    "answer": data.get("answer", ""),
                    "response": data.get("response", []),
                    "rating_match": data.get("rating_match", []),
                    "rating_osq": data.get("rating_osq"),
                    "rating_multians": data.get("rating_multians"),
                    "comments": data.get("comments", "")
                }
            else:
                annotation = {
                    "question_id": question_id,
                    "model": data.get("model", ""),
                    "thinking": data.get("thinking", ""),
                    "question_text": data.get("question_text", ""),
                    "answer": data.get("answer", ""),
                    "response": data.get("response", ""),
                    "rating_match": data.get("rating_match"),
                    "rating_osq": data.get("rating_osq"),
                    "rating_multians": data.get("rating_multians"),
                    "rating_correct": data.get("rating_correct"),
                    "comments": data.get("comments", "")
                }
            annotations[qid_str] = annotation
            save_dir = os.path.join("annotation", "saves")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"{annotator_id}.jsonl")
            with open(save_path, 'w') as f:
                for qid, anno in annotations.items():
                    f.write(json.dumps(anno) + '\n')
            return jsonify({"success": True})
        return jsonify({"error": "Invalid question ID"}), 400
    
    return app

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Annotation Interface')
    parser.add_argument('--input_path', type=str, 
                        default='/is/cluster/fast/nchandak/qaevals/filter/mmlupro/annotate.jsonl',
                        help='Path to the jsonl file')
    parser.add_argument('--annotator_id', type=str, required=True, 
                        help='Unique identifier for the annotator')
    parser.add_argument('--port', type=int, default=5000, 
                        help='Port to run the server on')
    parser.add_argument('--debug', type=int, default=None,
                        help='Load only specified number of responses for debugging (default: None)')
    parser.add_argument('--multi_model', action='store_true',
                        help='Enable multi-model mode for annotating multiple models per question')
    
    args = parser.parse_args()
    
    app = create_app(args.input_path, args.annotator_id, args.multi_model, args.debug)
    app.run(host='0.0.0.0', port=args.port, debug=True)
