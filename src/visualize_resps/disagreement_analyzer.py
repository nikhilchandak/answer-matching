import json
import os
import argparse
from flask import Flask, render_template, request, jsonify

def create_app(annotator1_path, annotator2_path, model_responses_path, rating_type):
    app = Flask(__name__, 
                template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates'))
    
    def load_annotations(file_path):
        annotations = {}
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            try:
                                item = json.loads(line)
                                annotations[str(item["question_id"])] = item
                            except Exception as e:
                                print(f"Error loading annotation: {e}")
            except Exception as e:
                print(f"Error loading annotations file: {e}")
        return annotations
    
    def load_model_responses():
        responses = {}
        if os.path.exists(model_responses_path):
            try:
                with open(model_responses_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            try:
                                item = json.loads(line)
                                responses[str(item["question_id"])] = item
                            except Exception as e:
                                print(f"Error loading model response: {e}")
            except Exception as e:
                print(f"Error loading model responses file: {e}")
        else:
            print(f"Model responses file not found: {model_responses_path}")
        return responses
    
    def find_disagreements(annotations1, annotations2, rating_key):
        disagreements = []
        
        # Find questions that are in both annotation sets
        common_questions = set(annotations1.keys()).intersection(set(annotations2.keys()))
        
        for qid in common_questions:
            rating1 = annotations1[qid].get(rating_key)
            rating2 = annotations2[qid].get(rating_key)
            
            # Skip if either rating is missing
            if rating1 is None or rating2 is None:
                continue
            
            # Check if there's significant disagreement (>=2)
            disagreement = abs(rating1 - rating2)
            if disagreement >= 2 or (rating1 <= 3 and rating2 > 3) or (rating1 > 3 and rating2 <= 3):
                disagreements.append({
                    # "question_id": int(qid),
                    "question_id": str(qid),
                    "annotation1": annotations1[qid],
                    "annotation2": annotations2[qid],
                    "disagreement_score": disagreement
                })
        
        # Sort by disagreement score (descending)
        disagreements.sort(key=lambda x: x["disagreement_score"], reverse=True)
        return disagreements
    
    # Load data
    annotator1_data = load_annotations(annotator1_path)
    annotator2_data = load_annotations(annotator2_path)
    model_responses = load_model_responses()
    
    # Find disagreements
    disagreements = find_disagreements(annotator1_data, annotator2_data, rating_type)
    
    @app.route('/')
    def index():
        annotator1_name = os.path.basename(annotator1_path).split('.')[0]
        annotator2_name = os.path.basename(annotator2_path).split('.')[0]
        
        return render_template('disagreement.html', 
                              disagreement_count=len(disagreements),
                              annotator1_name=annotator1_name,
                              annotator2_name=annotator2_name,
                              rating_type=rating_type)
    
    @app.route('/api/disagreements')
    def get_disagreements():
        return jsonify({"disagreements": [d["question_id"] for d in disagreements]})
    
    @app.route('/api/disagreement/<string:question_id>')
    def get_disagreement(question_id):
        # Find the disagreement
        disagreement = next((d for d in disagreements if d["question_id"] == str(question_id)), None)
        
        if disagreement is None:
            return jsonify({"error": "Disagreement not found"}), 404
        
        # Get model response data
        model_data = model_responses.get(str(question_id), {})
        
        # Print model_data to terminal for debugging
        # print("\n=== DEBUG MODEL DATA ===")
        # print(f"Question ID: {question_id}")
        # print(json.dumps(model_data, indent=2))
        # print("=== END DEBUG MODEL DATA ===\n")
        
        return jsonify({
            "disagreement": disagreement,
            "model_data": model_data
        })
    
    return app

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Annotation Disagreement Analysis Interface')
    parser.add_argument('--annotator1', type=str, 
                        default='/is/sg2/sgoel/evals/how-to-qa/src/visualize_resps/annotation/saves/strat1shash.jsonl',
                        help='Path to the first annotator\'s JSONL file')
    parser.add_argument('--annotator2', type=str, 
                        default='/is/sg2/sgoel/evals/how-to-qa/src/visualize_resps/annotation/saves/2907.jsonl',
                        help='Path to the second annotator\'s JSONL file')
    parser.add_argument('--model_responses', type=str,
                        default='/is/sg2/sgoel/evals/how-to-qa/combined_samples_to_annotate.jsonl',
                        help='Path to the model responses JSONL file')
    parser.add_argument('--rating_type', type=str, default='rating_match',
                        choices=['rating_match', 'rating_osq', 'rating_multians', 'rating_correct'],
                        help='Which rating to analyze for disagreements')
    parser.add_argument('--port', type=int, default=5000, 
                        help='Port to run the server on')
    
    args = parser.parse_args()
    
    app = create_app(args.annotator1, args.annotator2, args.model_responses, args.rating_type)
    app.run(host='0.0.0.0', port=args.port, debug=True) 