import numpy as np
import matplotlib.pyplot as plt

def log_posterior(x, y):
    return -0.5 * ((y - x**3)**2 + x**2)

def rwm_sampler(y, num_samples, proposal_std, initial_x=0.0):

    samples = np.zeros(num_samples)
    x_current = initial_x
    accept_count = 0
    
    for i in range(num_samples):
        x_proposal = np.random.normal(x_current, proposal_std)
        
        log_alpha = log_posterior(x_proposal, y) - log_posterior(x_current, y)
        alpha = np.exp(log_alpha)
        
        if np.random.rand() < alpha:
            x_current = x_proposal
            accept_count += 1
        
        samples[i] = x_current
    
    acceptance_rate = accept_count / num_samples
    return samples, acceptance_rate

def importance_sampler(y, num_samples):
    samples = np.random.normal(0, 1, num_samples)
    
    log_weights = -0.5 * ((y - samples**3)**2 + samples**2)
    
    log_proposal = -0.5 * samples**2
    
    log_importance_weights = log_weights - log_proposal
    importance_weights = np.exp(log_importance_weights - np.max(log_importance_weights))  
    
    return samples, importance_weights

y = -2  
num_samples = 10000  
proposal_std = 1 
initial_x = 0.0  

samples_mcmc, acceptance_rate = rwm_sampler(y, num_samples, proposal_std, initial_x)
print(f"MCMC Acceptance rate: {acceptance_rate:.2f}")
importance_samples, importance_weights = importance_sampler(y, num_samples)
normalized_weights = importance_weights / np.sum(importance_weights)
plt.figure(figsize=(12, 6))
plt.hist(samples_mcmc, bins=100, density=True, alpha=0.3, label="MCMC Samples")
plt.hist(importance_samples, bins=100, weights=normalized_weights, density=True, alpha=0.4, label="Importance Samples")
plt.title("MCMC vs Importance Sampling")
plt.xlabel("x")
plt.ylabel("Density")
plt.legend()
plt.show()
