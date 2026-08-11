from Brownian_SImulation_Model import simulate_brownian_motion   # <--- THE LINK
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# ---- Part A: Plot 10 paths (just for visual) ----
# time_plot, W_plot, T = simulate_brownian_motion(
#     T=1.0, N=252, n_paths=10, plot=True, savefig='brownian_motion_paths.png'
# )

# ---- Part B: Statistical verification with 10,000 paths ----
time_large, W_large, T = simulate_brownian_motion(
    T=1.0, N=252, n_paths=10000, plot=False
)
terminal_vals = W_large[-1, :]
mean_T = np.mean(terminal_vals)
std_T = np.std(terminal_vals)
print(f"Mean of W(T) across 10000 paths: {mean_T:.4f} (expected ~0)")
print(f"Std of W(T) across 10000 paths: {std_T:.4f} (expected ~{np.sqrt(T):.4f})")

# Histogram of terminal values
plt.figure(figsize=(8, 5))
plt.hist(terminal_vals, bins=50, density=True, alpha=0.6, label='Simulated W(T)')
x = np.linspace(min(terminal_vals), max(terminal_vals), 100)
plt.plot(x, stats.norm.pdf(x, 0, np.sqrt(T)), 'r-', linewidth=2, label='Theoretical N(0,1)')
plt.title('Terminal Distribution of W(T)')
plt.xlabel('W(T)')
plt.ylabel('Density')
plt.legend()
plt.savefig('terminal_distribution.png', dpi=150)
plt.show()