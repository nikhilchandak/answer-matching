import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import scienceplots
plt.style.use('science')

# Data from the table
datasets = ['TruthfulQA\nv2', 'MMLU',  'MMLU Pro', 'Yourbench\n- MMLU', 'GPQA', 'SuperGPQA'][::-1]
datasets = ['TruthfulQA\nv2', 'MMLU',  'MMLU Pro', 'Yourbench\n- MMLU'][::-1]
# datasets = ['TruthfulQA v2', 'MMLU', 'MMLU Pro', 'Yourbench MMLU', 'GPQA', 'SuperGPQA']

baseline_accuracy = [50, 25, 10, 25, 25, 10][::-1]  # in percentage
baseline_accuracy = [50, 25, 10, 25][::-1]  # in percentage

classifier_accuracy = [83, 36, 35, 61, 28, 27][::-1]  # in percentage
classifier_accuracy = [83, 36, 35, 61][::-1]  # in percentage

actual_acc = [82, 74.7, 65, 72.5, 58, 35.48][::-1]
actual_acc = [82, 74.7, 65, 72.5][::-1]

benchmark_size = [800 * 0.5, 16000 * 0.4, 12000 * 0.4, 2200 * 0.4, 400 * 0.4, 22000 * 0.4][::-1]
benchmark_size = [800 * 0.5, 16000 * 0.4, 12000 * 0.4, 2200 * 0.4][::-1]
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
xerr = [error_lower, error_upper]  # Changed from yerr to xerr for horizontal bars

# Create figure and axis
fig, ax = plt.subplots(figsize=(5, 3.5))  # Reduced height from 6 to 4.5 (or 3.8)

# Bar positions
y = np.arange(len(datasets))  # Changed from x to y
height = 0.6  # Increase height to reduce white space between bars

# Create the stacked bars with better colors (horizontal bars)
baseline_bars = ax.barh(y, baseline_accuracy, height, color='#2E8B57', label='Expected Chance')  # Sea Green
classifier_bars = ax.barh(y, classifier_improvement, height, left=baseline_accuracy, color='#f23838', label='Choice-only Shortcut')  # Light Coral

# Add error bars (horizontal)
ax.errorbar(classifier_accuracy, y, xerr=xerr, fmt='none', color='black', capsize=5)

# Add vertical black lines between the baseline and classifier parts
for i, bar in enumerate(baseline_bars):
    ax.vlines(x=baseline_accuracy[i], ymin=bar.get_y(), ymax=bar.get_y() + height, color='black', linewidth=1.5)

# Add dashed lines for actual_acc to the right of each bar
# acc_color = '#4DA6FF'  # A good shade of light blue
# for i, bar in enumerate(baseline_bars):
#     ax.vlines(x=actual_acc[i], ymin=bar.get_y(), ymax=bar.get_y() + height, 
#               color=acc_color, linewidth=2, linestyle='--')

# Import matplotlib.ticker for percentage formatting
import matplotlib.ticker as mtick

# Customize the plot
ax.set_xlabel('Accuracy (without the Question)', labelpad=8, fontsize=14)
ax.xaxis.set_major_formatter(mtick.PercentFormatter())
ax.tick_params(axis='x', labelsize=12)

ax.set_yticks(y)
ax.set_yticklabels(datasets, fontsize=12)
ax.set_xlim(0, 100)  # Set x-axis from 0 to 100%

# Add a legend entry for the actual accuracy dashed lines
handles, labels = ax.get_legend_handles_labels()
# from matplotlib.lines import Line2D
# handles.append(Line2D([0], [0], color=acc_color, linewidth=2, linestyle='--'))
# labels.append('Actual Accuracy (with Question)')

# ax.legend(handles, labels, frameon=True, fontsize=10)
# Place legend on top of the figure with 3 columns
ax.legend(handles, labels, frameon=True, fontsize=11, loc='upper center', 
          bbox_to_anchor=(0.5, 1.2), ncol=2)

# Display the total accuracy values to the right of each bar, after the error bars
for i, value in enumerate(classifier_accuracy):
    # Position the text to the right of the error bars
    ax.text(value + xerr[1][i] + 1.5, i, f'{value}%', ha='left', va='center', fontsize=12, fontweight='bold')

# Display the baseline accuracy values in the middle of the baseline part
for i, value in enumerate(baseline_accuracy):
    ax.text(value/2, i, f'{value}%', ha='center', va='center', color='white', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('plots/accuracy_comparison_vertical.png', dpi=300)
plt.savefig('plots/accuracy_comparison_vertical.pdf', dpi=300)
plt.show()
