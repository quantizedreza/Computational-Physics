import numpy as np
import mathplotlib.pyplot as plt

proposals = np.random.normal(0, 1, n_samples) 

weights = target_pdf(proposals) / proposal_pdf(proposals)  # These are unnormalized weights

weights = np.maximum(weights, 0)       # guard against tiny negatives
weights /= weights.sum() # normalize
ess = 1.0 / np.sum(weights**2)          # this is effective sample size
resampled = np.random.choice(proposals, size=n_samples, p=weights) # resample according to weights
# the algorithm above computes importance sampling with effective sample size (ESS) calculation. 


# Metropolis-Hastings (reuse mh_laplace defined earlier)
samples_mh = mh_laplace(n_samples, sigma=2, c=2)

# Plot both with the target pdf
bins = np.linspace(-10, 10, 121)
plt.figure(figsize=(8,5))
plt.hist(resampled, bins=bins, density=True, alpha=0.5, label=f'Importance (ESS={ess:.0f})')
plt.hist(samples_mh, bins=bins, density=True, alpha=0.5, label='Metropolis-Hastings')
d = np.linspace(-10, 10, 1000)
plt.plot(d, target_pdf(d), 'k-', lw=2, label='Target PDF (Laplace)')
plt.xlabel('x')
plt.ylabel('Density')
plt.legend()
plt.title('Importance Sampling vs Metropolis-Hastings (Laplace)')
plt.show()
