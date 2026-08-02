#This code produces the random forest fit to the temperature data 
# The oscillations resemble ENSO.

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor

# Load data
try:
    df = pd.read_csv("data.csv")
except FileNotFoundError:
    print("Error: data.csv not found")
    exit(1)

# Quick national annual means
national = df.groupby("Year")["Temperature"].mean().reset_index()

years = national["Year"].values
temps = national["Temperature"].values

if len(years) < 2:
    print("Not enough data points")
    exit(1)

print(f"Years: {years.min()}–{years.max()}  (n={len(years)})")
print(f"Latest observed: {temps[-1]:.3f} °F\n")

# Prepare data for smooth prediction
X = years.reshape(-1, 1)
X_smooth = np.linspace(years.min(), years.max(), 300).reshape(-1, 1)

# Random Forest
rf = RandomForestRegressor(n_estimators=150, random_state=42)
rf.fit(X, temps)
rf_pred = rf.predict(X_smooth)

latest_rf = rf.predict([[years[-1]]])[0]

# Summary
print("Most recent year:")
print(f"  Observed     : {temps[-1]:.3f} °F")
print(f"  Random Forest smoothed prediction: {latest_rf:.3f} °F\n")

# Plot
plt.figure(figsize=(11, 6))

plt.scatter(years, temps, c='darkred', edgecolor='black', s=70, label='Observed annual mean')
plt.axhline(temps.mean(), color='gray', ls='--', alpha=0.5,
            label=f'Overall mean = {temps.mean():.2f} °F')

plt.plot(X_smooth, rf_pred, color='#e67e22', lw=2.5,
         label='Random Forest smoothed fit')

plt.title("U.S. Annual Mean Temperature – Random Forest Fit")
plt.xlabel("Year")
plt.ylabel("Temperature (°F)")
plt.grid(alpha=0.3)
plt.legend(loc='upper left', fontsize=10)

plt.tight_layout()
plt.show()
