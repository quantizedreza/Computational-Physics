import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# ────────────────────────────────────────────────
#  SETTINGS & HYPERPARAMETERS
# ────────────────────────────────────────────────
RANDOM_SEED       = 2025
N_ITER            = 80_000
BURN_IN           = 20_000
THIN              = 5

PROPOSAL_SCALE = {
    'intercept': 0.50,
    'slope':     0.004,
    'sigma':     0.12
}

np.random.seed(RANDOM_SEED)

# ────────────────────────────────────────────────
#  1. Load data & compute national annual means
# ────────────────────────────────────────────────
df = pd.read_csv("data.csv")
df["Date"] = pd.to_datetime(df[["Year", "Month"]].assign(day=1))
df = df.sort_values(["State", "Date"])

annual = df.groupby("Year")["Temperature"].mean().reset_index()
annual = annual.rename(columns={"Temperature": "T_avg"})

years = annual["Year"].values
y     = annual["T_avg"].values

t = years - years[0]
n = len(y)

print(f"Years: {years.min()} – {years.max()}    (n = {n})")
print(f"Mean temperature: {y.mean():.2f} °F\n")

# ────────────────────────────────────────────────
#  2. Log posterior
# ────────────────────────────────────────────────
def log_posterior(intercept, slope, log_sigma, t, y):
    sigma = np.exp(log_sigma)
    if sigma <= 0:
        return -np.inf

    mu = intercept + slope * t
    log_lik = stats.norm.logpdf(y, loc=mu, scale=sigma).sum()

    log_prior_a     = stats.norm.logpdf(intercept,  52, 10)
    log_prior_b     = stats.norm.logpdf(slope,     0.04, 0.08)
    log_prior_logsig = stats.norm.logpdf(log_sigma, np.log(1.2), 1.0)

    return log_lik + log_prior_a + log_prior_b + log_prior_logsig

# ────────────────────────────────────────────────
#  3. Metropolis-Hastings sampler
# ────────────────────────────────────────────────
def metropolis_hastings():
    # Initial guess (rough OLS)
    slope_init    = np.polyfit(t, y, 1)[0]
    intercept_init = y.mean() - slope_init * t.mean()
    resid = y - (intercept_init + slope_init * t)
    sigma_init = np.std(resid, ddof=1)
    log_sigma_init = np.log(max(sigma_init, 0.3))

    current = np.array([intercept_init, slope_init, log_sigma_init])
    samples = []

    n_accepted = 0

    for i in range(N_ITER):
        proposal = current + np.random.normal(0, [
            PROPOSAL_SCALE['intercept'],
            PROPOSAL_SCALE['slope'],
            PROPOSAL_SCALE['sigma']
        ])

        log_p_curr = log_posterior(*current, t, y)
        log_p_prop = log_posterior(*proposal, t, y)

        log_alpha = log_p_prop - log_p_curr

        if np.log(np.random.rand()) < log_alpha:
            current = proposal
            n_accepted += 1

        if i >= BURN_IN and (i - BURN_IN) % THIN == 0:
            samples.append(current.copy())

        if (i + 1) % 5000 == 0 or (i + 1 <= 20000 and (i + 1) % 2000 == 0):
            acc_rate = n_accepted / (i + 1)
            current_sigma = np.exp(current[2])
            print(f"Iter {i+1:6d} | acc = {acc_rate:5.1%} | "
                  f"b = {current[1]:+.5f} | σ = {current_sigma:.3f}")

    samples = np.array(samples)
    final_acc = n_accepted / N_ITER
    print(f"\nFinal acceptance rate: {final_acc:.1%}  ({n_accepted:,} / {N_ITER:,})")
    return samples

# ────────────────────────────────────────────────
#  Run MCMC
# ────────────────────────────────────────────────
print("Running MCMC ...\n")
posterior_samples = metropolis_hastings()

intercept_samples = posterior_samples[:, 0]
slope_samples     = posterior_samples[:, 1]
sigma_samples     = np.exp(posterior_samples[:, 2])

# ────────────────────────────────────────────────
#  Summary statistics
# ────────────────────────────────────────────────
print("\n" + "═"*65)
print("POSTERIOR SUMMARY  (after burn-in & thinning)")
print("═"*65)
print(f"Intercept     {np.mean(intercept_samples):8.2f} ± {np.std(intercept_samples):.2f}")
print(f"Slope (°F/yr) {np.mean(slope_samples):8.5f} ± {np.std(slope_samples):.5f}")
print(f"Sigma         {np.mean(sigma_samples):8.2f} ± {np.std(sigma_samples):.2f}")
print()

p_warming = (slope_samples > 0).mean()
ci95 = np.percentile(slope_samples, [2.5, 97.5])
print(f"P(slope > 0)  = {p_warming:.4f}")
print(f"95% CrI slope = [{ci95[0]:+.5f}, {ci95[1]:+.5f}] °F/year")
print("═"*65 + "\n")

# ────────────────────────────────────────────────
#  Separate slope posterior plot (mean ±1σ + 95% CrI)
# ────────────────────────────────────────────────
fig_slope, ax_slope = plt.subplots(figsize=(9, 5.5))

ax_slope.hist(slope_samples, bins=65, density=True, color='lightsteelblue',
              edgecolor='steelblue', alpha=0.92, label='Posterior distribution')

mean_slope = np.mean(slope_samples)
sd_slope   = np.std(slope_samples)
ci_low, ci_high = np.percentile(slope_samples, [2.5, 97.5])

ax_slope.axvline(mean_slope, color='darkred', lw=2.4, zorder=5,
                 label=f'Mean = {mean_slope:+.5f} °F/yr')

ax_slope.axvline(mean_slope + sd_slope, color='darkred', lw=1.8, ls='--', alpha=0.9,
                 label=f'+1σ = {mean_slope + sd_slope:+.5f}')
ax_slope.axvline(mean_slope - sd_slope, color='darkred', lw=1.8, ls='--', alpha=0.9,
                 label=f'−1σ = {mean_slope - sd_slope:+.5f}')

ax_slope.axvspan(ci_low, ci_high, color='goldenrod', alpha=0.18, zorder=2,
                 label='95% CrI')

ax_slope.axvline(0, color='black', lw=0.9, alpha=0.5, ls='-', zorder=1)

ax_slope.set_xlabel("Slope (°F per year)")
ax_slope.set_ylabel("Posterior density")
ax_slope.set_title("Posterior Distribution of Warming Rate\n(mean ±1σ and 95% credible interval)",
                   fontsize=13, pad=12)

# ── Changed: legend now on the left side ────────────────────────────
ax_slope.legend(loc='upper left', frameon=True, fontsize=10.5)

ax_slope.grid(True, alpha=0.3, ls='--')

# Numerical annotation (still centered on the mean)
ymax = ax_slope.get_ylim()[1]
ax_slope.text(mean_slope, ymax * 0.92,
              f"Mean: {mean_slope:+.5f}\n"
              f"σ:  {sd_slope:.5f}\n"
              f"95% CrI: [{ci_low:+.5f}, {ci_high:+.5f}]",
              ha='center', va='top', fontsize=11, family='monospace',
              bbox=dict(facecolor='white', edgecolor='0.7', boxstyle='round,pad=0.4', alpha=0.95))

plt.tight_layout()
plt.show()

# ────────────────────────────────────────────────
#  Main visualisation: data + posterior samples + trace
# ────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9), height_ratios=[3, 1.2])

# ── Main plot: data + posterior lines ───────────────────────────────
ax1.scatter(years, y, c='C0', s=60, alpha=0.9, label="Annual national mean")

for i in np.random.choice(len(slope_samples), 120, replace=False):
    a, b = intercept_samples[i], slope_samples[i]
    ax1.plot(years, a + b * t, color='salmon', alpha=0.07, lw=1.1, zorder=1)

a_mean = np.mean(intercept_samples)
b_mean = np.mean(slope_samples)
ax1.plot(years, a_mean + b_mean * t, 'k', lw=2.4,
         label=f"Posterior mean trend  ({b_mean:+.4f} °F/yr)")

ax1.set_title("U.S. National Annual Mean Temperature\nBayesian Linear Trend (Metropolis-Hastings MCMC)",
              fontsize=14, pad=12)
ax1.set_ylabel("Temperature (°F)")
ax1.set_xlabel("Year")
ax1.legend(loc="upper left", frameon=True)
ax1.grid(True, alpha=0.35)

# ── Trace plot (slope) ──────────────────────────────────────────────
ax2.plot(slope_samples, lw=0.8, color='C0', alpha=0.9)
ax2.set_ylabel("Slope (°F/yr)")
ax2.set_xlabel("MCMC sample index (post burn-in & thinning)")
ax2.grid(True, alpha=0.3)

# ── Summary box on main plot ────────────────────────────────────────
ci95 = np.percentile(slope_samples, [2.5, 97.5])
p_positive = (slope_samples > 0).mean()

summary_text = (
    "Posterior Summary\n"
    f"Intercept       {np.mean(intercept_samples):.2f} ± {np.std(intercept_samples):.2f} °F\n"
    f"Slope           {np.mean(slope_samples):+.5f} ± {np.std(slope_samples):.5f} °F/yr\n"
    f"Sigma           {np.mean(sigma_samples):.2f} ± {np.std(sigma_samples):.2f} °F\n"
    f"P(slope > 0)    {p_positive:.4f}\n"
    f"95% CrI slope   [{ci95[0]:+.5f}, {ci95[1]:+.5f}] °F/yr"
)

ax1.text(0.98, 0.98, summary_text,
         transform=ax1.transAxes,
         fontsize=11,
         verticalalignment='top',
         horizontalalignment='right',
         bbox=dict(facecolor='white', edgecolor='0.6', boxstyle='round,pad=0.5', alpha=0.92),
         family='monospace')

plt.tight_layout()
plt.show()
