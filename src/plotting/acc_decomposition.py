import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import scienceplots
plt.style.use('science')

# Data from the table
data_dict = {
    "MCQ": 65.3,
    "Only\nOptions": 25.6,
    "Multiple Choice\nVerification": 47.1,
    "MCQ\nCloze": 27.6,
    "No Question\nCloze": 9.8,
    # "Additional Format": 27  # This was in the original list but not in the datasets list
}

# Create the respective lists from the dictionary
datasets = list(data_dict.keys())
model_acc = list(data_dict.values())  # in percentage

benchmark_size = [12000] * 5

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
error_bars = [bootstrap_confidence_interval(acc, size) for acc, size in zip(model_acc, benchmark_size)]
error_lower = [err[0] for err in error_bars]
error_upper = [err[1] for err in error_bars]
yerr = [error_lower, error_upper]

# Create figure and axis
fig, ax = plt.subplots(figsize=(4, 6))

# Define bar positions with spacing between groups
# Group 1: MCQ and Only Options
# Group 2: MCQ Cloze and No Question Cloze
# Group 3: MCQ Verification
group_width = 0.5  # Width of each bar
group_spacing = 0.075  # Space between groups

# Define positions for each bar
positions = [
    0,                     # MCQ
    0 + group_width,        # Only Options
    2 + 4 * group_spacing,      # MCQ Verification
    4 + 2 * group_spacing,   # MCQ Cloze
    4 + group_spacing + group_width,  # No Question Cloze
]

# Define colors for each bar
colors = ['#4CAF50',  # Medium green for MCQ 
          '#FF6B6B',  # Red for Only Options
          
          '#FFA500',  # Orange for MCQ Verification
          
          '#2E8B57',  # Sea Green for MCQ Cloze
        #   '#90EE90',  # Light green for MCQ Cloze
          '#FFCCCB',  # Light red for No Question Cloze
]

# Set tick positions
x_tick_positions = [0, 0 + group_width, 2 + 4 * group_spacing, 4 + 2 * group_spacing, 4 + 2*group_spacing + group_width]
ax.set_xticks(x_tick_positions)
ax.set_xticklabels(datasets)

# Adjust alignment of tick labels for better readability
for i, label in enumerate(ax.get_xticklabels()):
    
    if i == 0:
        label.set_ha('right')
    elif i == 1:
        # label.set_ha('left')
        label.set_ha('center')
    elif i == 2:
        # label.set_ha('right')
        label.set_ha('center')
    elif i == 3:
        label.set_ha('center')
        # label.set_ha('center')
    elif i == 4:
        label.set_ha('left')
        # label.set_ha('right')
    
    # if i % 2 == 1:  # Even indices (0, 2, 4)
    #     # Use 'center' alignment with slight offset instead of 'left'
    #     label.set_ha('center')
    #     # Move label 75% of the way to the left
    #     label.set_position((x_tick_positions[i] - 0.75 * group_width, label.get_position()[1]))
    # else:  # Odd indices (1, 3)
    #     # Use 'center' alignment with slight offset instead of 'right'
    #     label.set_ha('center')
    #     # Move label 75% of the way to the right
    #     label.set_position((x_tick_positions[i] + 0.75 * group_width, label.get_position()[1]))

# Create the bars
bars = ax.bar(positions, model_acc, group_width, color=colors)

# Add error bars
for i, pos in enumerate(positions):
    ax.errorbar(pos, model_acc[i], yerr=[[error_lower[i]], [error_upper[i]]], 
                fmt='none', color='black', capsize=5)

# Customize the plot
ax.set_ylabel('Accuracy')
ax.set_title('Qwen3-8B Performance in different formats')
ax.set_ylim(0, 100)  # Set y-axis from 0 to 100%

import matplotlib.ticker as mtick

ax.yaxis.set_major_formatter(mtick.PercentFormatter())

# Add a blue dashed line at 10% with a label
ax.axhline(y=10, color='blue', linestyle='--', linewidth=1, label='Expected Chance')

# Add a legend with colored squares
legend_elements = [
    plt.Rectangle((0, 0), 1, 1, color=colors[0], label='MCQ'),
    plt.Rectangle((0, 0), 1, 1, color=colors[1], label='Only Options'),
    plt.Rectangle((0, 0), 1, 1, color=colors[2], label='MCQ Verification'),
    plt.Rectangle((0, 0), 1, 1, color=colors[3], label='MCQ Cloze'),
    plt.Rectangle((0, 0), 1, 1, color=colors[4], label='No Question Cloze'),
    plt.Line2D([0], [0], color='blue', linestyle='--', linewidth=1, label='Expected Chance')
]
ax.legend(handles=legend_elements, loc='upper right')
# ax.legend(frameon=True) # loc='upper left')

# Add gridlines for better readability
ax.grid(axis='y', linestyle='--', alpha=0.7)

# Display the total accuracy values on top of each bar, above the error bars
for i, value in enumerate(model_acc):
    # Position the text above the error bars
    if i == 1 : 
        ax.text(positions[i] + 0.1, value + error_upper[i] + 1, f'{value}%', ha='center', va='bottom')
    elif i == 3: 
        ax.text(positions[i] + 0.05, value + error_upper[i] + 1, f'{value}%', ha='center', va='bottom')
    else:
        ax.text(positions[i], value + error_upper[i] + 1, f'{value}%', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('plots/mmlu_pro_acc_decomposition.png', dpi=300)
plt.show()
