import json
import os
import argparse
import math
from flask import Flask, render_template, request, jsonify
import glob
from collections import defaultdict

def create_app(folder1, folder2):
    app = Flask(__name__, 
                template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates'))
    
    model1_name = os.path.basename(os.path.normpath(folder1))
    model2_name = os.path.basename(os.path.normpath(folder2))
    
    def load_data():
        data = {}
        domains = set()
        
        # Load data from both folders
        for folder, model_name in [(folder1, model1_name), (folder2, model2_name)]:
            data[folder] = {}  # Use full path as key instead of model name
            jsonl_files = glob.glob(os.path.join(folder, "samples_*.jsonl"))
            
            for file_path in jsonl_files:
                with open(file_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            try:
                                item = json.loads(line)
                                question_key = item['doc_hash']  # Using doc_hash as unique identifier
                                
                                # Extract normalized probabilities
                                logits = [float(resp[0]) for resp in item['filtered_resps']]
                                probas = softmax(logits)
                                
                                # Get model prediction (index of max probability)
                                prediction = probas.index(max(probas))
                                
                                # Check if prediction is correct
                                correct = prediction == item['doc']['answer']
                                
                                # Get the actual subject from the JSON data
                                subject = item['doc']['subject']
                                # Add subject to domains set
                                domains.add(subject)
                                
                                # Store relevant information
                                if question_key not in data[folder]:
                                    data[folder][question_key] = {
                                        'question': item['doc']['question'],
                                        'choices': item['doc']['choices'],
                                        'answer': item['doc']['answer'],
                                        'domain': subject,
                                        'probas': probas,
                                        'prediction': prediction,
                                        'correct': correct
                                    }
                            except Exception as e:
                                print(f"Error processing line in {file_path}: {e}")
        
        # Combine data from both models
        combined_data = []
        for q_key in set(data[folder1].keys()).intersection(set(data[folder2].keys())):
            item1 = data[folder1][q_key]
            item2 = data[folder2][q_key]
            
            combined_item = {
                'question': item1['question'],
                'choices': item1['choices'],
                'answer': item1['answer'],
                'domain': item1['domain'],
                'model1': {
                    'name': model1_name,
                    'probas': item1['probas'],
                    'prediction': item1['prediction'],
                    'correct': item1['correct']
                },
                'model2': {
                    'name': model2_name,
                    'probas': item2['probas'],
                    'prediction': item2['prediction'],
                    'correct': item2['correct']
                },
                'comparison': get_comparison_category(item1['correct'], item2['correct'])
            }
            combined_data.append(combined_item)
        
        return combined_data, sorted(domains)
    
    def softmax(logits):
        exp_logits = [math.exp(x) for x in logits]
        sum_exp = sum(exp_logits)
        return [x / sum_exp for x in exp_logits]
    
    def get_comparison_category(correct1, correct2):
        if correct1 and correct2:
            return "correct-correct"
        elif correct1 and not correct2:
            return "correct-wrong"
        elif not correct1 and correct2:
            return "wrong-correct"
        else:
            return "wrong-wrong"
    
    # Cache data loading
    all_data, all_domains = load_data()
    
    @app.route('/')
    def index():
        return render_template('index.html', 
                               model1_name=model1_name, 
                               model2_name=model2_name,
                               domains=all_domains)
    
    @app.route('/api/questions')
    def get_questions():
        comparison_filter = request.args.get('comparison', 'all')
        domain_filter = request.args.get('domain', 'all')
        page = int(request.args.get('page', 0))
        per_page = int(request.args.get('per_page', 10))
        
        # Apply filters
        filtered_data = all_data
        if comparison_filter != 'all':
            filtered_data = [item for item in filtered_data if item['comparison'] == comparison_filter]
        if domain_filter != 'all':
            filtered_data = [item for item in filtered_data if item['domain'] == domain_filter]
        
        # Paginate
        start = page * per_page
        end = start + per_page
        paginated_data = filtered_data[start:end]
        
        return jsonify({
            'questions': paginated_data,
            'total': len(filtered_data),
            'page': page,
            'per_page': per_page
        })
    
    return app

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare model predictions')
    parser.add_argument('--folder1', type=str, required=True, 
                        help='Path to the first model results folder')
    parser.add_argument('--folder2', type=str, required=True, 
                        help='Path to the second model results folder')
    parser.add_argument('--port', type=int, default=5000, 
                        help='Port to run the server on')
    
    args = parser.parse_args()
    
    app = create_app(args.folder1, args.folder2)
    app.run(host='0.0.0.0', port=args.port, debug=True)