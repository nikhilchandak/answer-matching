import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem
import numpy.random as random

# Set larger font sizes
plt.rcParams.update({'font.size': 16})
plt.rcParams['axes.titlesize'] = 18
plt.rcParams['axes.labelsize'] = 16
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['legend.fontsize'] = 15

# Data
papers = [
    'Spurious\nRewards',
    'RL with\n1 Example',
    'Verifree',
    'Entropy\nMinimization',
    'Can Large Models\nSelf-Train?'
]

papers = [
    'Spurious Rewards\n(Qwen2.5-7B)',
    'RL with\n1 Example\n(R1-Distill-Qwen-1.5B)',
    'Verifree\n(Qwen3-4B)',
    'Entropy\nMinimization\n(Qwen2.5-7B)',
    'Can Large Reasoning\nModels Self-Train?\n(Qwen 2.5-7B-Math)',
]


reported_rl = [70.1, 78.0, 74.8, 70.8, 80.0]
reported_pre_rl = [41.6, 71.9, 73.4, 43.8, 42.0]
actual_pre_rl = [64.6, 84.9, np.nan, 64.6, 64.3]

# Calculate gains
reported_gain = [rl - pre for rl, pre in zip(reported_rl, reported_pre_rl)]
actual_gain = [rl - pre if not np.isnan(pre) else np.nan 
               for rl, pre in zip(reported_rl, actual_pre_rl)]

# Create figure and axis
plt.figure(figsize=(12, 6))

# Set width of bars and positions
x = np.arange(len(papers))
width = 0.35


# Create bars
bars1 = plt.bar(x - width/2, reported_gain, width, label='Reported Gain', 
                color='#e74c3c', capsize=5)
bars2 = plt.bar(x + width/2, actual_gain, width, label='Actual Gain', 
                color='#2ecc71', capsize=5)

# Add value labels on top of bars
def autolabel(bars):
    for bar in bars:
        height = bar.get_height()
        if np.isnan(height):
            plt.text(bar.get_x() + bar.get_width()/2., 2,
                    '?',
                    ha='center', va='bottom', fontsize=20)
        else:
            if height > 0:
                delta = 1
            else :
                delta = -3
                
            plt.text(bar.get_x() + bar.get_width()/2., height + delta,
                    f'{height:+.1f}',  # Added + sign for gains
                    ha='center', va='bottom', fontsize=14)

autolabel(bars1)
autolabel(bars2)

# Customize the plot
plt.ylabel('Accuracy Gain on MATH500 (%)', fontsize=16)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)

# Legend in top left, inside figure
plt.legend(loc='upper left', bbox_to_anchor=(0.0, 1.0),
          ncol=2, frameon=True)

# Remove x-axis ticks but keep labels
plt.xticks(x, papers, rotation=0, ha='center')
plt.tick_params(axis='x', length=0)

# Set y-axis limits to show gains properly
plt.ylim(-10, 45)  # Adjusted to show both positive and negative gains

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot
plt.savefig('plots/math500_results.png', bbox_inches='tight', dpi=300)
plt.close()
