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
    'RL with 1 Example\n(R1-Distill-Qwen-1.5B)',
    'Verifree\n(Qwen3-4B)',
    'Entropy\nMinimization\n(Qwen2.5-7B)',
    'Can Large Reasoning\nModels Self-Train?\n(Qwen 2.5-7B-Math)',
]


reported_rl = [70.1, 78.0, 74.8, 70.8, 80.0]
reported_pre_rl = [41.6, 71.9, 73.4, 43.8, 42.0]
actual_pre_rl = [64.6, 84.9, np.nan, 64.6, 64.3]

# Bootstrap error calculation (n=500 problems)
def bootstrap_ci(acc, n_samples=1000, n_problems=500):
    if np.isnan(acc):
        return 0
    successes = int(acc * n_problems / 100)  # Convert percentage to counts
    data = np.array([1] * successes + [0] * (n_problems - successes))
    means = []
    for _ in range(n_samples):
        sample = random.choice(data, size=n_problems, replace=True)
        means.append(100 * np.mean(sample))
    return np.std(means)

errors_rl = [bootstrap_ci(x) for x in reported_rl]
errors_pre_rl = [bootstrap_ci(x) for x in reported_pre_rl]
errors_actual = [bootstrap_ci(x) for x in actual_pre_rl]

# Create figure and axis
plt.figure(figsize=(12, 6))

# Set width of bars and positions
x = np.arange(len(papers))
width = 0.25



# Create bars
bars1 = plt.bar(x - width, reported_rl, width, label='Reported RL Acc.', 
                color='#2ecc71', capsize=5)
bars2 = plt.bar(x, reported_pre_rl, width, label='Reported Pre-RL Acc.', 
                color='#e74c3c', capsize=5)
bars3 = plt.bar(x + width, actual_pre_rl, width, label='Actual Pre-RL Acc.', 
                color='#3498db', capsize=5)

# Add value labels on top of bars with more vertical offset
def autolabel(bars):
    for bar in bars:
        height = bar.get_height()
        if np.isnan(height):
            plt.text(bar.get_x() + bar.get_width()/2., 2,  # Small offset from bottom
                    '?',
                    ha='center', va='bottom', fontsize=12)
        else:
            plt.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{height:.1f}',
                    ha='center', va='bottom', fontsize=12)

autolabel(bars1)
autolabel(bars2)
autolabel(bars3)

# Customize the plot
plt.ylabel('Accuracy on MATH500 (%)', fontsize=16)
plt.grid(True, axis='y', linestyle='--', alpha=0.7)

# Legend in top left, inside figure
plt.legend(loc='upper left', bbox_to_anchor=(0.0, 1.0),
          ncol=3, frameon=True)

# Remove x-axis ticks but keep labels
plt.xticks(x, papers, rotation=0, ha='center')
plt.tick_params(axis='x', length=0)

# Set y-axis limit to 100
plt.ylim(0, 100)

# Adjust layout to prevent label cutoff
plt.tight_layout()

# Save the plot
plt.savefig('plots/math500_results.png', bbox_inches='tight', dpi=300)
plt.close()
