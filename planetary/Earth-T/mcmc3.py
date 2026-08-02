import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(2025)

# ────────────────────────────────────────────────
#  SETTINGS
# ────────────────────────────────────────────────
N_ITER         = 35000
BURN_IN        =  7000
THIN           =     5
PROPOSAL_SCALE = {'intercept': 1.4, 'slope': 0.020, 'sigma': 0.45}

# ────────────────────────────────────────────────
#  1. Load & annual means
# ────────────────────────────────────────────────
df = pd.read_csv("data.csv")
annual = df.groupby(['State', 'Year'])['Temperature'].mean().reset_index()
annual = annual.rename(columns={'Temperature': 'T_avg'})

# Quick OLS ranking – strongest positive trends
trends = []
for state, group in annual.groupby('State'):
    if len(group) < 15: continue
    yrs = group['Year'].values
    tmp = group['T_avg'].values
    slope, _ = np.polyfit(yrs, tmp, 1)
    trends.append((state, slope, len(yrs), tmp.mean()))

trends_df = pd.DataFrame(trends, columns=['State','ols_slope','nyears','mean_T'])
trends_df = trends_df.sort_values('ols_slope', ascending=False)

# Select top warming states
focus_states = trends_df['State'].head(10).tolist()   # ← change number if desired
print("States selected for MCMC (strongest warming trends):")
print(trends_df.head(10)[['State','ols_slope','nyears']].round(4).to_string(index=False))
print()

# ────────────────────────────────────────────────
#  2. Log-posterior & MCMC sampler (same as before)
# ────────────────────────────────────────────────
def log_posterior(intercept, slope, log_sigma, t, y):
    sigma = np.exp(log_sigma)
    if sigma <= 0: return -np.inf
    mu = intercept + slope * t
    log_lik = stats.norm.logpdf(y - mu, scale=sigma).sum()
    log_prior = (
        stats.norm.logpdf(intercept,  45, 15) +
        stats.norm.logpdf(slope,     0.06, 0.12) +
        stats.norm.logpdf(log_sigma, np.log(1.4), 1.3)
    )
    return log_lik + log_prior


def run_mcmc(y, t):
    slope_init, intercept_init = np.polyfit(t, y, 1)
    resid = y - (intercept_init + slope_init * t)
    sigma_init = np.std(resid, ddof=1)
    log_sigma_init = np.log(max(sigma_init, 0.01))

    current = np.array([intercept_init, slope_init, log_sigma_init])
    samples = []
    n_accepted = 0

    for i in range(N_ITER):
        prop = current + np.random.normal(0, [
            PROPOSAL_SCALE['intercept'],
            PROPOSAL_SCALE['slope'],
            PROPOSAL_SCALE['sigma']
        ])

        logp_curr = log_posterior(*current, t, y)
        logp_prop = log_posterior(*prop,     t, y)

        if np.log(np.random.rand()) < (logp_prop - logp_curr):
            current = prop
            n_accepted += 1

        if i >= BURN_IN and (i - BURN_IN) % THIN == 0:
            samples.append(current.copy())

    samples = np.array(samples)
    acc = n_accepted / N_ITER
    return samples, acc


# ────────────────────────────────────────────────
#  3. Run & plot
# ────────────────────────────────────────────────
results = {}
fig, axes = plt.subplots(len(focus_states)//2 + (len(focus_states)%2), 2,
                         figsize=(13, 3.3 * ((len(focus_states)+1)//2)),
                         sharex=True, squeeze=False)
fig.suptitle("States with strongest warming trends\n(Bayesian posterior samples & mean trend)", y=0.98)

for idx, state in enumerate(focus_states):
    sub = annual[annual['State'] == state].sort_values('Year')
    years = sub['Year'].values
    temps = sub['T_avg'].values
    t = years - years[0]

    print(f"Running MCMC → {state}  ({len(years)} years)")
    samples, acc = run_mcmc(temps, t)
    if len(samples) == 0:
        print("  → failed")
        continue

    intercept_s = samples[:,0]
    slope_s     = samples[:,1]
    sigma_s     = np.exp(samples[:,2])

    results[state] = {
        'slope_mean': slope_s.mean(),
        'slope_std':  slope_s.std(),
        'slope_ci95': np.percentile(slope_s, [2.5,97.5]),
        'p_positive': (slope_s > 0).mean(),
        'acc_rate':   acc
    }

    ax = axes[idx//2, idx%2]
    ax.scatter(years, temps, c='C0', s=55, alpha=0.9, label='Annual mean')

    for i in np.random.choice(len(slope_s), 90, replace=False):
        ax.plot(years, intercept_s[i] + slope_s[i] * t,
                c='darkorange', alpha=0.07, lw=1.1, zorder=1)

    ax.plot(years, intercept_s.mean() + slope_s.mean() * t,
            'k', lw=2.5, label=f'mean trend  {slope_s.mean():+.4f}')

    ax.set_title(f"{state}    {slope_s.mean():+.4f} ± {slope_s.std():.4f} °F/yr")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

for ax in axes.flat:
    ax.set_ylabel("°F")
axes[-1,0].set_xlabel("Year")
if len(focus_states) % 2 == 1:
    axes[-1,1].set_visible(False)
plt.tight_layout(rect=[0,0,1,0.96])
plt.show()

# ────────────────────────────────────────────────
#  Summary table
# ────────────────────────────────────────────────
summary = pd.DataFrame.from_dict(results, orient='index')
summary = summary[['slope_mean','slope_std','p_positive','acc_rate']]
summary.columns = ['slope_mean','slope_std','P(slope>0)','accept_rate']
print("\nPosterior slope estimates – strongest warming states:")
print(summary.sort_values('slope_mean', ascending=False).round(4).to_string())
