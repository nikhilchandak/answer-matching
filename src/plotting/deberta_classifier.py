import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import scienceplots
plt.style.use('science')

# Data from the table
# datasets = ['TruthfulQA\nv2', 'MMLU', 'Yourbench\n- MMLU', 'GPQA', 'MMLU Pro', 'SuperGPQA']
datasets = ['TruthfulQA\nv2', 'MMLU', 'MMLU\nPro', 'Yourbench\nMMLU', 'GPQA', 'SuperGPQA', 'HellaSwag', 'GoldenSwag']
baseline_accuracy = [50, 25, 10, 25, 25, 10, 25, 25]  # in percentage
classifier_accuracy = [83.5, 38.6, 10.4, 34, 27.2, 10.1, 87.2, 93.7]  # in percentage

benchmark_size = [800 * 0.5, 16000, 12000 * 0.5, 2200 * 0.5, 400 * 0.5, 22000 * 0.5, 10000, 1525]

# Calculate the additional accuracy from the classifier (difference)
classifier_improvement = [cls - base for cls, base in zip(classifier_accuracy, baseline_accuracy)]

# Calculate error bars using bootstrap
def bootstrap_confidence_interval(accuracy, sample_size, n_bootstrap=1000, confidence=0.95):
    # For a binomial distribution, we can simulate bootstrap samples
    successes = int(accuracy * sample_size / 100)  # Convert percentage to count
    p = successes / sample_size
    
    # Generate bootstrap samples
    bootstrap_samples = np.random.binomial(sample_size, p, n_bootstrap) / sample_size * 100
    
    # Calculate confidence interval
    lower = np.percentile(bootstrap_samples, (1 - confidence) * 100 / 2)
    upper = np.percentile(bootstrap_samples, 100 - (1 - confidence) * 100 / 2)
    
    return accuracy - lower, upper - accuracy

# Calculate error bars for each dataset
error_bars = [bootstrap_confidence_interval(acc, size) for acc, size in zip(classifier_accuracy, benchmark_size)]
error_lower = [err[0] for err in error_bars]
error_upper = [err[1] for err in error_bars]
yerr = [error_lower, error_upper]

# Create figure and axis
fig, ax = plt.subplots(figsize=(12, 6))

# Bar positions
x = np.arange(len(datasets))
width = 0.5

# Create the stacked bars with better colors
baseline_bars = ax.bar(x, baseline_accuracy, width, color='#2E8B57', label='Expected Chance (Random Baseline)')  # Sea Green
# classifier_bars = ax.bar(x, classifier_improvement, width, bottom=baseline_accuracy, color='#FF6B6B', label='Choice-only Shortcut (Classifier)')  # Light Coral
classifier_bars = ax.bar(x, classifier_improvement, width, bottom=baseline_accuracy, color='#f23838', label='Choice-only Shortcut (Classifier)')  # Light Coral
# classifier_bars = ax.bar(x, classifier_improvement, width, bottom=baseline_accuracy, color='#D32F2F', label='Choice-only Shortcut (Classifier)')  # Darker red

# Add error bars
ax.errorbar(x, classifier_accuracy, yerr=yerr, fmt='none', color='black', capsize=5)

# Add horizontal black lines between the baseline and classifier parts
for i, bar in enumerate(baseline_bars):
    ax.hlines(y=baseline_accuracy[i], xmin=bar.get_x(), xmax=bar.get_x() + width, color='black', linewidth=1.5)

# Add dashed lines for actual_acc above each bar
acc_color = '#4DA6FF'  # A good shade of light blue

# Customize the plot
# ax.set_ylabel('Accuracy without the Question (%)')

# Import matplotlib.ticker for percentage formatting
import matplotlib.ticker as mtick

ax.set_ylabel('Accuracy without the Question', labelpad=10, fontsize=22)
ax.yaxis.set_major_formatter(mtick.PercentFormatter())
# Increase font size of y-axis tick labels
ax.tick_params(axis='y', labelsize=16)

# ax.set_ylabel('Accuracy without the Question (%)', labelpad=10)
# ax.set_title('Baseline vs DeBERTa Classifier Accuracy Across Datasets')
ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=16)
ax.set_ylim(0, 100)  # Set y-axis from 0 to 100%

# Add a legend entry for the actual accuracy dashed lines
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, frameon=True, fontsize=16)

# Display the total accuracy values on top of each bar, above the error bars
for i, value in enumerate(classifier_accuracy):
    # Position the text above the error bars
    ax.text(i, value + yerr[1][i] + 0.5, f'{value}%', ha='center', va='bottom', fontsize=20, fontweight='bold')

# Display the baseline accuracy values in the middle of the baseline part
for i, value in enumerate(baseline_accuracy):
    ax.text(i, value/2, f'{value}%', ha='center', va='center', color='white', fontsize=20, fontweight='bold')

plt.tight_layout()
plt.savefig('plots/deberta_classifier.png', dpi=300)
plt.savefig('plots/deberta_classifier.pdf', dpi=300)
plt.show()
