import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import scienceplots
plt.style.use('science')

# Data from the table
# datasets = ['TruthfulQA\nv2', 'MMLU', 'MMLU Pro', 'Yourbench\nMMLU', 'GPQA', 'Hellaswag'][::-1]
# baseline_accuracy = [50, 25, 10, 25, 25, 25][::-1]  # in percentage
# classifier_accuracy = [83, 39, 41, 61, 28, 87][::-1]  # in percentage
# actual_acc = [82, 74.7, 65, 72.5, 58, 89][::-1]
# benchmark_size = [800 * 0.5, 16000 * 0.5, 12000 * 0.5, 2200 * 0.5, 400 * 0.5, 10000 * 0.5][::-1]


datasets = ['TruthfulQA\nv2', 'Hellaswag', 'MMLU', 'MMLU Pro', 'GPQA', 'MMMU Pro'][::-1]
baseline_accuracy = [50, 25, 25, 10, 25, 10][::-1]  # in percentage
classifier_accuracy = [83, 87, 39, 41, 28, 51][::-1]  # in percentage
actual_acc = [82, 89, 74.7, 65, 58, None][::-1]
benchmark_size = [800 * 0.5, 10000, 16000 * 0.5, 12000 * 0.5, 400 * 0.5, 3460 * 0.5][::-1]

# Calculate the additional accuracy from the classifier (difference)
classifier_improvement = [cls - base for cls, base in zip(classifier_accuracy, baseline_accuracy)]

# Calculate error bars using bootstrap
def bootstrap_confidence_interval(accuracy, sample_size, n_bootstrap=1000, confidence=0.95):
    successes = int(accuracy * sample_size / 100)
    p = successes / sample_size
    bootstrap_samples = np.random.binomial(sample_size, p, n_bootstrap) / sample_size * 100
    lower = np.percentile(bootstrap_samples, (1 - confidence) * 100 / 2)
    upper = np.percentile(bootstrap_samples, 100 - (1 - confidence) * 100 / 2)
    return accuracy - lower, upper - accuracy

# Calculate error bars for each dataset
error_bars = [bootstrap_confidence_interval(acc, size) for acc, size in zip(classifier_accuracy, benchmark_size)]
error_lower = [err[0] for err in error_bars]
error_upper = [err[1] for err in error_bars]
xerr = [error_lower, error_upper]  # Changed from yerr to xerr for horizontal bars

# Create figure and axis
fig, ax = plt.subplots(figsize=(13, 8))

# Bar positions
y = np.arange(len(datasets))
height = 0.5

# Create the stacked bars horizontally
baseline_bars = ax.barh(y, baseline_accuracy, height, color='#2E8B57', label='Random Baseline')
classifier_bars = ax.barh(y, classifier_improvement, height, left=baseline_accuracy, color='#f23838', label='Choice-only Shortcut (Classifier)')

# Add error bars horizontally
ax.errorbar(classifier_accuracy, y, xerr=xerr, fmt='none', color='black', capsize=5)

# Add vertical black lines between the baseline and classifier parts
for i, bar in enumerate(baseline_bars):
    ax.vlines(x=baseline_accuracy[i], ymin=bar.get_y(), ymax=bar.get_y() + height, color='black', linewidth=1.5)

# Add dashed lines for actual_acc
acc_color = '#4DA6FF'
for i, bar in enumerate(baseline_bars):
    ax.vlines(x=actual_acc[i], ymin=bar.get_y(), ymax=bar.get_y() + height, 
              color=acc_color, linewidth=2.5, linestyle='--')

# Customize the plot
import matplotlib.ticker as mtick
ax.set_xlabel('Accuracy without the Question', labelpad=10, fontsize=22)
ax.xaxis.set_major_formatter(mtick.PercentFormatter())
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=22)  # Increased y-axis label fontsize from 16 to 22

ax.set_yticks(y)
ax.set_yticklabels(datasets, fontsize=22)  # Added fontsize parameter to yticklabels
ax.set_xlim(0, 100)

# Add legend above the plot
handles, labels = ax.get_legend_handles_labels()
from matplotlib.lines import Line2D
handles.append(Line2D([0], [0], color=acc_color, linewidth=2.5, linestyle='--'))
labels.append('Actual Accuracy (w/ Question)')
ax.legend(handles, labels, frameon=True, fontsize=16, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=3)

# Display the baseline accuracy values
for i, value in enumerate(baseline_accuracy):
    ax.text(value/2, i, f'{value}%', ha='center', va='center', color='white', fontsize=22, fontweight='bold')

# Display the total accuracy values
for i, value in enumerate(classifier_accuracy):
    ax.text(value + xerr[1][i] + 1.5, i, f'{value}%', ha='left', va='center', fontsize=22, fontweight='bold')

plt.tight_layout()
plt.savefig('plots/accuracy_comparison_horizontal_stanford.png', dpi=300, bbox_inches='tight')
# plt.savefig('plots/accuracy_comparison_horizontal2.pdf', dpi=300, bbox_inches='tight')
plt.show()
