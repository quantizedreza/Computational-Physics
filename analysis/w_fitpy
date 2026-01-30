import matplotlib.pyplot as plt
import numpy as np

data_file = "data.txt"
x, y, y_err = np.loadtxt(data_file, delimiter=',', unpack=True)

weights = 1 / y_err**2

n_rows = 4
n_cols = 5
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 16))
fig.suptitle("Weighted Polynomial Fits from Degree 1 to 20", fontsize=16)

degrees = range(1, 21)  # Polynomial degrees
x_fit = np.linspace(min(x), max(x), 500)  # Smooth x values for plotting

print("Weighted RMSE for each polynomial degree:")
for i, degree in enumerate(degrees):
    # Fit the polynomial (weighted)
    coefficients = np.polyfit(x, y, degree, w=weights)  # Weighted least squares
    polynomial = np.poly1d(coefficients)
    
    # Calculating fitted values and weighted RMSE
    y_pred = polynomial(x)  # Predicted y values at original x points
    weighted_rmse = np.sqrt(np.sum(weights * (y - y_pred) ** 2) / np.sum(weights))
    print(f"Degree {degree}: Weighted RMSE = {weighted_rmse:.4f}")
    
    y_fit = polynomial(x_fit)
    
    row, col = divmod(i, n_cols)
    ax = axes[row, col]
    
    ax.errorbar(x, y, yerr=y_err, fmt='o', label='Data', capsize=5)
    
    ax.plot(x_fit, y_fit, color='red', linewidth=2.5, label=f'Degree {degree}')
    
    ax.set_title(f'Degree {degree}\nWeighted RMSE = {weighted_rmse:.2f}')
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.grid(True)
    ax.legend()
    
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('twenty_weighted.png')
plt.show()
