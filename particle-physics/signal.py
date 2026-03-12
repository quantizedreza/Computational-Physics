import scipy.stats as stats
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def confidence_limits(observed_count, confidence_level):
    alpha = 1 - confidence_level
    lower_limit = stats.poisson.ppf(alpha / 2, observed_count)
    upper_limit = stats.poisson.ppf(1 - alpha / 2, observed_count)
    return lower_limit, upper_limit

assert confidence_limits(100, 0.68) == (90.0, 110.0)
assert confidence_limits(100, 0.95) == (81.0, 120.0)

def background(E_min, E_max, n_bins, B):
    bin_edges = np.linspace(E_min, E_max, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bkg_counts = np.full(n_bins, B / n_bins)
    return bin_centers, bkg_counts

E_min = 100
E_max = 300
n_bins = 40
B = 2000

energies, bkg_counts = background(E_min, E_max, n_bins, B)
assert len(energies) == n_bins
assert math.isclose(np.sum(bkg_counts), B, abs_tol=1.0)

def signal(E_min, E_max, n_bins, E_0, sigma, S):
    bin_edges = np.linspace(E_min, E_max, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    signal_counts = S * np.exp(-0.5 * ((bin_centers - E_0) / sigma) ** 2)
    signal_counts /= (sigma * np.sqrt(2 * np.pi))
    signal_counts *= (E_max - E_min) / n_bins
    return bin_centers, signal_counts

E_0 = 250
sigma = 10
S = 50

energies, signal_counts = signal(E_min, E_max, n_bins, E_0, sigma, S)
assert len(energies) == n_bins
assert math.isclose(np.sum(signal_counts), S, abs_tol=0.1)

plt.figure(figsize=(7, 5))
plt.plot(energies, bkg_counts, alpha=0.7, color='royalblue', label='Background')
plt.plot(energies, signal_counts, alpha=0.7, color='darkorange', label='Signal')
plt.plot(energies, signal_counts + bkg_counts, alpha=0.7, color='red', label='Total')
plt.xlabel('Energy [GeV]', fontsize=14)
plt.ylabel('Events / bin', fontsize=14)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.legend(frameon=False, fontsize=14)
plt.xlim(100, 300)
plt.ylim(0, 100)
plt.show()

def generate_toy_mc(E_min, E_max, n_bins, E_0, sigma, S, B):
    bin_edges = np.linspace(E_min, E_max, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = (E_max - E_min) / n_bins
    signal_counts = S * np.exp(-0.5 * ((bin_centers - E_0) / sigma) ** 2)
    signal_counts *= bin_width / (np.sqrt(2 * np.pi) * sigma)
    background_counts = np.full(n_bins, B / n_bins)
    signal_observed = np.random.poisson(signal_counts)
    background_observed = np.random.poisson(background_counts)
    toy_data = signal_observed + background_observed
    return toy_data

np.random.seed(42)
observed_counts = generate_toy_mc(E_min, E_max, n_bins, E_0, sigma, S, B)
expected_counts = signal_counts + bkg_counts
assert observed_counts[-1] == 64
assert np.sum(observed_counts) == 2087

plt.figure(figsize=(7, 5))
plt.plot(energies, bkg_counts, alpha=0.7, color='royalblue', label='Background')
plt.plot(energies, signal_counts, alpha=0.7, color='darkorange', label='Signal')
plt.plot(energies, expected_counts, alpha=0.7, color='red', label='Total')
plt.errorbar(energies, observed_counts, yerr=np.sqrt(observed_counts), fmt='o', linestyle='None', alpha=0.7, color='black', label='Observed')
plt.xlabel('Energy [GeV]', fontsize=14)
plt.ylabel('Events / bin', fontsize=14)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)
plt.legend(loc='upper left', frameon=False, fontsize=14)
plt.xlim(100, 300)
plt.ylim(0, 120)
plt.show()

def chi2(S_fit, B_fit):
    _, signal_counts = signal(E_min, E_max, n_bins, E_0, sigma, S_fit)
    _, bkg_counts = background(E_min, E_max, n_bins, B_fit)
    expected_counts = signal_counts + bkg_counts
    chi2_value = np.sum((observed_counts - expected_counts) ** 2 / expected_counts)
    return chi2_value

S_init = 100
B_init = np.sum(observed_counts)

def objective(params):
    S_fit, B_fit = params
    return chi2(S_fit, B_fit)

res = minimize(objective, x0=[S_init, B_init], bounds=[(0, None), (0, None)], method='L-BFGS-B')
best_fit_S, best_fit_B = res.x
chi2_min = res.fun
print(f"Best-fit S: {best_fit_S:.2f}")
print(f"Best-fit B: {best_fit_B:.2f}")
print(f"Best-fit chi-square: {chi2_min:.2f}")

chi2_test = chi2(50, 2000)
assert math.isclose(chi2_test, 48.4744, abs_tol=1e-2)

S_scan = np.linspace(0, 80, 200)
q_values = []
for s in S_scan:
    def objective_fixed(params):
        B_fit = params[0]
        return chi2(s, B_fit)
    res_fixed = minimize(objective_fixed, x0=[best_fit_B], bounds=[(0, None)], method='L-BFGS-B')
    chi2_fixed = res_fixed.fun
    q = chi2_fixed - chi2_min
    q_values.append(q)

plt.figure(figsize=(7, 5))
plt.plot(S_scan, q_values, linewidth=2)
plt.xlabel("Signal Rate", fontsize=14)
plt.ylabel("q", fontsize=14)
plt.grid()
plt.show()

closest_diff = float('inf')
closest_rate = float('inf')
for i in range(len(S_scan)):
    if abs(q_values[i]-2.71) < closest_diff and S_scan[i]>best_fit_S:
        closest_diff = abs(q_values[i]-2.71)
        closest_rate = S_scan[i]

print(f"Upper-limit on Signal Rate: {closest_rate:.2f}")
assert math.isclose(closest_rate, 52.66, abs_tol=1e-1)

plt.figure(figsize=(7, 5))
plt.plot(S_scan, q_values, color='black', linewidth=2)
plt.hlines(2.71, 0, closest_rate, linestyle='--', color='red', alpha=0.8)
plt.vlines(closest_rate, 0, 2.71, linestyle='--', color='red', alpha=0.8)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("Signal Rate", fontsize=14)
plt.ylabel("q", fontsize=14)
plt.grid()
plt.show()
