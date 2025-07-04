import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import seaborn as sns

# Base directories
BASE_FREE_DIR = "/fast/nchandak/qaevals/judge_outputs/math_free/"
BASE_MCQ_DIR = "/fast/nchandak/qaevals/judge_outputs/math_mcq/"

# Function to load data from a jsonl file
def load_jsonl(file_path):
    data = []
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        print(f"Error parsing JSON in {file_path}: {e}")
    return data

# Function to calculate agreement between exact_match and other judges
def calculate_agreement(data):
    """Calculate agreement percentage between exact_match and other judges."""
    reference_field = "exact_match"
    
    # Find all score fields
    score_fields = [field for field in data[0].keys() if field.startswith("score_")] if data else []
    
    # Calculate agreement percentages and accuracy for each model
    agreements = {}
    accuracies = {}
    agreement_counts = {}  # Store counts for error calculation
    
    # Calculate accuracy for reference model
    ref_correct = sum(1 for item in data if reference_field in item and item[reference_field] == 1)
    ref_total = sum(1 for item in data if reference_field in item)
    if ref_total > 0:
        accuracies["exact_match"] = (ref_correct / ref_total) * 100
    
    for field in score_fields:
        if field == reference_field:
            continue
            
        model_name = field.replace("score_", "")
        agreement_count = 0
        valid_count = 0
        
        # Calculate accuracy for this model
        correct_count = sum(1 for item in data if field in item and item[field] == "1")
        total_count = sum(1 for item in data if field in item)
        if total_count > 0:
            accuracies[model_name] = (correct_count / total_count) * 100
        
        for item in data:
            # Only count if both scores exist
            if reference_field in item and field in item and item[reference_field] and item[field]:
                valid_count += 1
                if int(item[reference_field]) == int(item[field]):
                    agreement_count += 1
        
        if valid_count > 0:
            agreements[model_name] = (agreement_count / valid_count) * 100
            agreement_counts[model_name] = (agreement_count, valid_count)  # Store for error calculation
    
    return agreements, agreement_counts, accuracies

# Calculate standard errors using bootstrapping
def calculate_std_errors(agreement_counts):
    np.random.seed(42)  # For reproducibility
    n_bootstrap = 10000
    std_errors = {}

    for model, (agree_count, total_count) in agreement_counts.items():
        # Create binary array representing agreements (1) and disagreements (0)
        binary_data = np.concatenate([
            np.ones(agree_count),
            np.zeros(total_count - agree_count)
        ])
        
        # Bootstrap resampling
        bootstrap_estimates = []
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(binary_data, size=total_count, replace=True)
            bootstrap_agreement = 100 * np.mean(bootstrap_sample)  # Convert to percentage
            bootstrap_estimates.append(bootstrap_agreement)
        
        # Calculate standard error from bootstrap distribution
        std_error = np.std(bootstrap_estimates)
        std_errors[model] = std_error
    
    return std_errors

# Get all model folders
model_folders = [os.path.basename(folder) for folder in glob.glob(f"{BASE_FREE_DIR}*")]
print(f"Found {len(model_folders)} model folders: {model_folders}")

# Prepare data structure for plotting
all_results = []

# Process each model
for model_name in model_folders:
    print(f"\nProcessing model: {model_name}")
    
    # Paths for this model
    free_path = os.path.join(BASE_FREE_DIR, model_name, "samples.jsonl")
    mcq_path = os.path.join(BASE_MCQ_DIR, model_name, "samples.jsonl")
    
    # Load free-response data
    free_data = load_jsonl(free_path)
    if free_data:
        print(f"Loaded {len(free_data)} samples from {free_path}")
    else:
        print(f"No data found in {free_path}")
        continue
    
    # Load MCQ data
    mcq_data = load_jsonl(mcq_path)
    if mcq_data:
        print(f"Loaded {len(mcq_data)} samples from {mcq_path}")
    else:
        print(f"No data found in {mcq_path}")
        continue
    
    # Calculate MCQ accuracy
    mcq_correct = sum(1 for item in mcq_data if "exact_match" in item and item["exact_match"] == 1)
    mcq_total = len(mcq_data)
    mcq_accuracy = (mcq_correct / mcq_total * 100) if mcq_total > 0 else 0
    print(f"MCQ accuracy: {mcq_correct}/{mcq_total} ({mcq_accuracy:.2f}%)")
    
    # Add MCQ scores to free-response data
    question_id_to_acc = {item["question_id"]: item.get("exact_match", 0) for item in mcq_data}
    for item in free_data:
        question_id = item.get("question_id")
        if question_id in question_id_to_acc:
            item["score_mcq"] = str(int(question_id_to_acc[question_id]))
    
    # Calculate agreement between judges
    agreement_percentages, agreement_counts, accuracies = calculate_agreement(free_data)
    std_errors = calculate_std_errors(agreement_counts)
    
    # Store results for this model
    model_results = {
        'model_name': model_name,
        'mcq_accuracy': mcq_accuracy,
        'agreements': agreement_percentages,
        'std_errors': std_errors,
        'accuracies': accuracies
    }
    all_results.append(model_results)
# Prepare data for plotting
plot_data = []

for result in all_results:
    model_name = result['model_name']
    
    # Add MCQ accuracy
    plot_data.append({
        'Model': model_name,
        'Metric': 'MCQ Accuracy',
        'Value': result['mcq_accuracy'],
        'Std_Error': 0  # We don't have std error for MCQ accuracy
    })
    
    # Add judge agreements
    for judge, agreement in result['agreements'].items():
        plot_data.append({
            'Model': model_name,
            'Metric': f'Judge: {judge}',
            'Value': agreement,
            'Std_Error': result['std_errors'].get(judge, 0)
        })
# Convert to DataFrame
plot_df = pd.DataFrame(plot_data)

# Create the plot
plt.figure(figsize=(15, 10))
ax = plt.subplot(111)

# Set up colors for different metrics
metrics = plot_df['Metric'].unique()
colors = plt.cm.tab20(np.linspace(0, 1, len(metrics)))
color_map = dict(zip(metrics, colors))

# Group by model
models = plot_df['Model'].unique()
y = np.arange(len(models))  # Using y instead of x for horizontal bars
width = 0.8 / len(metrics)  # Width of bars

# Plot each metric as a group of horizontal bars
for i, metric in enumerate(metrics):
    metric_data = plot_df[plot_df['Metric'] == metric]
    metric_values = []
    metric_errors = []
    
    # Ensure data is in the same order as models
    for model in models:
        model_data = metric_data[metric_data['Model'] == model]
        if not model_data.empty:
            metric_values.append(model_data['Value'].values[0])
            metric_errors.append(model_data['Std_Error'].values[0])
        else:
            metric_values.append(0)
            metric_errors.append(0)
    
    # Plot horizontal bars for this metric
    bars = ax.barh(y + (i - len(metrics)/2 + 0.5) * width, metric_values, 
                   width, label=metric, color=color_map[metric])
    
    # Add error bars (horizontal)
    ax.errorbar(metric_values, y + (i - len(metrics)/2 + 0.5) * width, 
                xerr=metric_errors, fmt='none', color='black', capsize=3)
ax.set_xlabel('Percentage (%)', fontsize=14)
ax.set_title('Model Performance and Judge Agreement', fontsize=16, fontweight='bold')
ax.set_yticks(y)
ax.set_yticklabels(models)
ax.legend(loc='upper right', bbox_to_anchor=(1.05, 1))  # Moved legend to top right corner
ax.set_xlim(40, 100)  # x-axis starts at 50
ax.grid(axis='x', linestyle='--', alpha=0.7)

# Add percentage labels to the bars
for bar in ax.patches:
    width = bar.get_width()
    if width > 0:
        # Get the bar's metric from the original data
        # Matplotlib bar patches don't store labels directly
        
        # Find the corresponding error value
        bar_idx = list(ax.patches).index(bar)
        model_idx = bar_idx // len(metrics)
        metric_idx = bar_idx % len(metrics)
        model = models[model_idx]
        metric = metrics[metric_idx]
        
        metric_name = str(metric)
        print(metric_name)
        
        # Set error_value to 0 for MCQ Accuracy, otherwise get from plot_df
        if 'mcq' in metric_name.lower():
            error_value = 0
        else:
            # Check if the data exists before accessing it
            matching_rows = plot_df[(plot_df['Model'] == model) & (plot_df['Metric'] == metric)]
            if not matching_rows.empty:
                error_value = matching_rows['Std_Error'].values[0]
            else:
                error_value = 0
        
        # Determine offset based on metric type
        offset = 2.5 if 'mcq' in metric_name.lower() else 1.5
        
        ax.text(width + offset, bar.get_y() + bar.get_height()/2,
                f'{width:.1f}% ±{error_value:.1f}%', ha='left', va='center', fontsize=12)  # Increased font size

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
# plt.show()

# Print summary of results
print("\nSummary of Results:")
for result in all_results:
    print(f"\nModel: {result['model_name']}")
    print(f"MCQ Accuracy: {result['mcq_accuracy']:.2f}%")
    print("Judge Agreements:")
    for judge, agreement in sorted(result['agreements'].items(), key=lambda x: x[1], reverse=True):
        std_error = result['std_errors'].get(judge, 0)
        print(f"  {judge}: {agreement:.2f}% ±{std_error:.2f}%")