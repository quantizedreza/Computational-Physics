import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# Assuming columns: [0: ignored, 1: ignored, 2: creep rate, 3: latitude, 4: uncertainty]
data = np.loadtxt('creep_data.txt', delimiter=',')
X = data[:, 0].reshape(-1, 1)  # Latitude (1st column)
y = data[:, 2]                 # Creep rate (3rd column)
dy = data[:, 3]                # Uncertainty (4th column) 

# Define the kernel: Constant kernel * RBF kernel
kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=10.0, length_scale_bounds=(1e-2, 1e2))

# Initialize Gaussian Process Regressor
# alpha is the variance of the noise (dy^2)
gp = GaussianProcessRegressor(kernel=kernel, alpha=dy**2, n_restarts_optimizer=10)

# Fit the model
gp.fit(X, y)

# Generate points for prediction
X_pred = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
y_pred, sigma = gp.predict(X_pred, return_std=True)

# Plot the results
plt.figure(figsize=(10, 5))
plt.errorbar(X.ravel(), y, dy, fmt='r.', markersize=10, label='Observations')
plt.plot(X_pred, y_pred, 'b-', label='Prediction')
plt.fill_between(X_pred.ravel(), y_pred - 1.96 * sigma, y_pred + 1.96 * sigma,
                 alpha=0.2, color='blue', label='95% confidence interval')
plt.xlabel('Latitude')
plt.ylabel('Creep Rate')
plt.title('Gaussian Process Regression on Creep Rate vs Latitude')
plt.legend(loc='upper right')
plt.grid(True)
plt.show()

# Print optimized kernel parameters
print("Optimized kernel parameters:", gp.kernel_)
