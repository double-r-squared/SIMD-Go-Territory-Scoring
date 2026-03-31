import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Configure Seaborn for better aesthetic defaults
sns.set_theme(style="whitegrid")

# Read the CSV
try:
    df = pd.read_csv('benchmark_results.csv')
except FileNotFoundError:
    print("Error: benchmark_results.csv not found. Please ensure the benchmark is run first.")
    exit()

# Define the precise order of tests for plotting (CRITICAL for sorting)
TARGET_ORDER = ['ST-test', 'MT-test', 'arm-optimized']

# Filter and set the target order
df = df[df['test_name'].isin(TARGET_ORDER)]

# ==== 1. CALCULATE STATISTICS ON FULL DATASET (OUTLIERS INCLUDED) ====
df_raw = df.copy()

# Calculate statistics for each test (on ALL data)
stats = df_raw.groupby('test_name')['per_pair_ns'].agg([
    ('mean', 'mean'),
    ('median', 'median'),
    ('std', 'std'),
    ('min', 'min'),
    ('max', 'max'),
]).reset_index()

# Sort the stats DataFrame explicitly by the TARGET_ORDER
stats['test_name'] = pd.Categorical(stats['test_name'], categories=TARGET_ORDER, ordered=True)
stats = stats.sort_values('test_name').reset_index(drop=True)

# Calculate Coefficient of Variation (CV) 
stats['CV'] = (stats['std'] / stats['mean']) * 100 # Percentage

# Prettier names for plots and labels
test_name_map = {
    'ST-test': 'Single-threaded (1 thread)',
    'MT-test': 'Multi-threaded (2 threads)',
    'arm-optimized': 'ARM Aggressive (4 threads)'
}
stats['display_name'] = stats['test_name'].map(test_name_map)
df_raw['display_name'] = df_raw['test_name'].map(test_name_map)

# Define consistent colors and plot positions
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1'] # Match the ST, MT, ARM order
x_pos = np.arange(len(stats))
benchmark_title = 'Go Territory Scoring Benchmark: Full Performance Analysis (100 Runs, Outliers Included)'

# === CALCULATE SPEEDUP % FOR PLOT C (Tiered Comparison) ===
mean_st_test = stats[stats['test_name'] == 'ST-test']['mean'].iloc[0]
mean_mt_test = stats[stats['test_name'] == 'MT-test']['mean'].iloc[0]
mean_arm_test = stats[stats['test_name'] == 'arm-optimized']['mean'].iloc[0]

# Speedup 1: MT vs ST (Baseline: ST)
mt_vs_st_speedup = ((mean_st_test - mean_mt_test) / mean_st_test) * 100

# Speedup 2: ARM vs MT (Baseline: MT)
arm_vs_mt_speedup = ((mean_mt_test - mean_arm_test) / mean_mt_test) * 100

# Data structure for the new Plot C
speedup_data = pd.DataFrame({
    'comparison': ['MT vs ST', 'ARM vs MT'],
    'speedup_percent': [mt_vs_st_speedup, arm_vs_mt_speedup],
    'color': [colors[1], colors[2]], 
})


# === DATA FILTERING FOR PLOT B (MT and ARM only) ===
FAST_TESTS = ['MT-test', 'arm-optimized']
stats_fast = stats[stats['test_name'].isin(FAST_TESTS)].reset_index(drop=True)
x_pos_fast = np.arange(len(stats_fast))
# The display names are already mapped in the stats_fast DataFrame


# ==============================================================================
# ==== CREATE FINAL COMPREHENSIVE FIGURE (5 PLOTS) ====
# ==============================================================================
# Grid: 3 rows and 2 columns. Increased figsize width for landscape format (slides)
fig = plt.figure(figsize=(24, 12)) 
gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.2) 

# ---- Plot 1 (Row 1, Col 1): Mean Time with Standard Deviation Error Bars (A) ----
ax1 = fig.add_subplot(gs[0, 0])
bars1 = ax1.bar(x_pos, stats['mean'], 
                yerr=stats['std'],
                color=colors,
                capsize=7,
                alpha=0.85,
                edgecolor='black',
                linewidth=1.5)

ax1.set_ylabel('Mean Time per Pair (ns)', fontsize=13, fontweight='bold')
ax1.set_title('A) Absolute Performance (Mean $\\pm$ Std Dev)', fontsize=15, fontweight='bold')
ax1.set_xticks(x_pos)
ax1.set_xticklabels(stats['display_name'], fontsize=11) 
ax1.grid(axis='y', alpha=0.5, linestyle='--')

for bar, mean, std in zip(bars1, stats['mean'], stats['std']):
    height = bar.get_height()
    # Annotation centered inside the bar (horizontally and vertically)
    ax1.text(bar.get_x() + bar.get_width()/2., height / 2, 
             f'Avg: {mean:.0f}ns\n$\\sigma$: {std:.0f}',
             ha='center', va='center', fontsize=10, fontweight='bold', color='white',
             bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2'))


# ---- Plot 2 (Row 1, Col 2): Min/Mean/Max Range (B) - Optimized Tests Only ----
ax2 = fig.add_subplot(gs[0, 1])

# Calculate yerr for plotting the min/max range using stats_fast
yerr_lower = stats_fast['mean'] - stats_fast['min']
yerr_upper = stats_fast['max'] - stats_fast['mean']

# Plotting the Mean with full Min/Max error bars
ax2.errorbar(x_pos_fast, stats_fast['mean'], 
             yerr=[yerr_lower, yerr_upper], 
             fmt='o', # Plot the mean as a circle
             capsize=7, 
             color='black', 
             ecolor='gray', 
             linewidth=2, 
             markersize=8,
             label='Mean $\\pm$ Range')

# Plotting the Median as a marker
ax2.scatter(x_pos_fast, stats_fast['median'], 
            marker='D', 
            color='gold', 
            edgecolor='black', 
            s=50, 
            zorder=3, # Bring median marker to the front
            label='Median')

ax2.set_ylabel('Time per Pair (ns)', fontsize=13, fontweight='bold')
ax2.set_title('B) Optimized Performance Range (Min, Mean, Median, Max)', fontsize=15, fontweight='bold')
ax2.set_xticks(x_pos_fast)
ax2.set_xticklabels(stats_fast['display_name'], fontsize=11)
ax2.grid(axis='y', alpha=0.5, linestyle='--')
ax2.legend(loc='upper right', fontsize=10)


# ---- Plot 3 (Row 2, Col 1): Tiered Percentage Speedup (C) ----
ax3 = fig.add_subplot(gs[1, 0])
x_pos_speedup = np.arange(len(speedup_data))

bars3 = ax3.bar(x_pos_speedup, speedup_data['speedup_percent'], 
                color=speedup_data['color'],
                alpha=0.85,
                edgecolor='black',
                linewidth=1.5)

ax3.set_ylabel('Speedup (%)', fontsize=13, fontweight='bold')
ax3.set_title('C) Speedup Comparison (Tiered Baselines)', fontsize=15, fontweight='bold')
ax3.set_xticks(x_pos_speedup)
ax3.set_xticklabels([
    'MT vs ST (Baseline: ST)', 
    'ARM vs MT (Baseline: MT)'
], fontsize=11)
ax3.grid(axis='y', alpha=0.5, linestyle='--')
ax3.axhline(0, color='gray', linestyle='--', linewidth=1) # Baseline at 0%

for bar, speedup, comparison in zip(bars3, speedup_data['speedup_percent'], speedup_data['comparison']):
    height = bar.get_height()
    # Annotation centered inside the bar (horizontally and vertically)
    ax3.text(bar.get_x() + bar.get_width()/2., height / 2, 
             f'+{speedup:.1f}%',
             ha='center', va='center', fontsize=11, fontweight='bold', color='white',
             bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2'))


# ---- Plot 4 (Row 2, Col 2): Coefficient of Variation (CV) (D) ----
ax4 = fig.add_subplot(gs[1, 1])
bars4 = ax4.bar(x_pos, stats['CV'], 
                color=colors, # Use ST, MT, ARM colors
                alpha=0.85,
                edgecolor='black',
                linewidth=1.5)

ax4.set_ylabel('Coefficient of Variation (%)', fontsize=13, fontweight='bold')
ax4.set_title('D) Relative Consistency (CV, Lower = More Stable)', fontsize=15, fontweight='bold')
ax4.set_xticks(x_pos)
ax4.set_xticklabels(stats['display_name'], fontsize=11)
ax4.grid(axis='y', alpha=0.5, linestyle='--')

for bar, cv_val in zip(bars4, stats['CV']):
    height = bar.get_height()
    # Annotation centered inside the bar (horizontally and vertically)
    ax4.text(bar.get_x() + bar.get_width()/2., height / 2,
             f'{cv_val:.2f}%',
             ha='center', va='center', fontsize=11, fontweight='bold', color='white',
             bbox=dict(facecolor='black', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2'))


# ---- Plot 5 (Row 3, Spanning): Raw Performance Over Iterations (E) ----
ax5 = fig.add_subplot(gs[2, :]) # This spans all columns in the third row

# Plot each test type using the raw data, ensuring they are plotted in order
for (test_name, color) in zip(TARGET_ORDER, colors):
    test_data = df_raw[df_raw['test_name'] == test_name]
    ax5.plot(test_data['run_number'], test_data['per_pair_ns'], 
             label=test_name_map[test_name], 
             color=color, 
             linewidth=2,
             alpha=0.9)

ax5.set_xlabel('Iteration Number (1 to 100)', fontsize=13, fontweight='bold')
ax5.set_ylabel('Time per Territory Pair (ns)', fontsize=13, fontweight='bold')
ax5.set_title('E) Raw Performance Trace', fontsize=15, fontweight='bold')
ax5.legend(fontsize=11, loc='upper right')
ax5.grid(True, alpha=0.4, linestyle='--')
ax5.tick_params(axis='both', which='major', labelsize=10)


fig.suptitle(benchmark_title, 
             fontsize=18, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig('final_benchmark_figure.png', dpi=300, bbox_inches='tight')
plt.show()