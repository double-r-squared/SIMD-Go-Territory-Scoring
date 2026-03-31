import pandas as pd
import sys

try:
    df = pd.read_csv('benchmark_results.csv')
except FileNotFoundError:
    print("Error: benchmark_results.csv not found. Run 'make benchmark' first.")
    sys.exit(1)

TARGET_ORDER = ['ST-test', 'MT-test', 'arm-optimized']
DISPLAY_NAMES = {
    'ST-test':        'Single-threaded (1 thread)',
    'MT-test':        'Multi-threaded (2 threads)',
    'arm-optimized':  'ARM Optimized (4 P-cores)',
}

df = df[df['test_name'].isin(TARGET_ORDER)]

stats = df.groupby('test_name')['per_pair_ns'].agg(
    mean='mean', median='median', std='std', min='min', max='max'
).reindex(TARGET_ORDER)

stats['cv_pct']      = (stats['std'] / stats['mean']) * 100
stats['speedup_vs_st'] = stats.loc['ST-test', 'mean'] / stats['mean']

# MT-tier speedup: MT vs ST, ARM vs MT
st_mean  = stats.loc['ST-test',       'mean']
mt_mean  = stats.loc['MT-test',       'mean']
arm_mean = stats.loc['arm-optimized', 'mean']

mt_vs_st_pct  = (st_mean  - mt_mean)  / st_mean  * 100
arm_vs_mt_pct = (mt_mean  - arm_mean) / mt_mean  * 100
arm_vs_st_pct = (st_mean  - arm_mean) / st_mean  * 100

# ── Terminal table ──────────────────────────────────────────────────────────
SEP = '-' * 100
print(SEP)
print(f"{'Test':<28} {'Mean':>10} {'Median':>10} {'Std':>8} {'Min':>10} {'Max':>10} {'CV%':>6} {'Speedup vs ST':>14}")
print(SEP)
for name in TARGET_ORDER:
    r = stats.loc[name]
    print(f"{DISPLAY_NAMES[name]:<28} "
          f"{r['mean']:>10.1f} "
          f"{r['median']:>10.1f} "
          f"{r['std']:>8.1f} "
          f"{r['min']:>10.1f} "
          f"{r['max']:>10.1f} "
          f"{r['cv_pct']:>6.2f} "
          f"{r['speedup_vs_st']:>14.2f}x")
print(SEP)

print()
print(f"Tiered speedup:")
print(f"  MT  vs ST  : +{mt_vs_st_pct:.1f}%  ({st_mean:.0f} → {mt_mean:.0f} ns)")
print(f"  ARM vs MT  : +{arm_vs_mt_pct:.1f}%  ({mt_mean:.0f} → {arm_mean:.0f} ns)")
print(f"  ARM vs ST  : +{arm_vs_st_pct:.1f}%  ({st_mean:.0f} → {arm_mean:.0f} ns)")
print()

# ── Markdown table ──────────────────────────────────────────────────────────
print("Markdown summary (paste into README):")
print()
print("| Version | Threads | Mean (ns) | Median (ns) | Min (ns) | Std Dev | Speedup vs ST |")
print("|---|---|---|---|---|---|---|")
THREADS = {'ST-test': 1, 'MT-test': 2, 'arm-optimized': 4}
for name in TARGET_ORDER:
    r = stats.loc[name]
    speedup = f"**{r['speedup_vs_st']:.1f}×**" if r['speedup_vs_st'] > 1 else "1.0×"
    print(f"| {DISPLAY_NAMES[name]} | {THREADS[name]} | "
          f"{r['mean']:,.0f} | {r['median']:,.0f} | {r['min']:,.0f} | "
          f"{r['std']:.0f} | {speedup} |")
