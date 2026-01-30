# Uniform vs transformed vs direct Gaussian Realizations 

uniform = np.random.uniform(0, 1, 100) # Uniform distribution 
m_trans = norm.ppf(uniform, 5, 1) #transformed 

m_direct = np.random.normal(5, 1, 100) # direct random variables

plt.hist(m_trans, bins=20, density=True, alpha=0.5, label='Transformed')
plt.hist(m_direct, bins=20, density=True, alpha=0.5, label='Direct')
x = np.linspace(0, 10, 1000)
plt.plot(x, norm.pdf(x, 5, 1), 'k-', label='Exact')
plt.legend()
plt.title('Gaussian Realizations: Transformed, Direct, and Exact')
plt.show()
