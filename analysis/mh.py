#Metropolis-Hastings Algorithm for Laplace Distribution

import numpy as np
import matplotlib.pyplot as plt

# Metropolis-Hastings example for p(d) = (1/2*c) exp(-|d|/c)
def mh_laplace(n_samples, sigma, c):
    samples = [0]
    for _ in range(n_samples - 1):
        current = samples[-1] # last element in samples
        proposal = current + np.random.normal(0, sigma) 
        p_current = (1/(2*c)) * np.exp(-np.abs(current)/c)
        p_proposal = (1/(2*c)) * np.exp(-np.abs(proposal)/c)
        if np.random.uniform(0,1) < p_proposal / p_current:
            samples.append(proposal)
        else:
            samples.append(current)
    return np.array(samples)

samples = mh_laplace(1000, sigma=2, c=2)
plt.hist(samples, bins=50, density=True, alpha=0.7)
x = np.linspace(-10, 10, 1000)
p = (1/4) * np.exp(-np.abs(x)/2)  # c=2
plt.plot(x, p, 'r-')
plt.title('Metropolis-Hastings for Laplace Distribution')
plt.show()
