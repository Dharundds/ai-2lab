import numpy as np
from scipy.stats import beta

data = np.random.binomial(n=1,p=0.7,size=100)


alpha_prior = 1
beta_prior = 1

alpha_posterior = alpha_prior + np.sum(data)
beta_posterior = beta_prior + len(data) - np.sum(data)

posterior = beta(alpha_posterior, beta_posterior)

samples = posterior.rvs(size=1000)

mean = np.mean(samples)
credible_interval = np.percentile(samples, [2.5, 97.5])

print("Posterior mean:", mean)
print("Credible interval (95%):", credible_interval)
