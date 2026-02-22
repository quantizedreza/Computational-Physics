import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Load data (use sep=r'\s+' to handle multiple spaces)
data_1  = pd.read_csv('pi0s_1.txt',  sep=r'\s+', header=0, names=['event', 'px', 'py', 'pz', 'E'])
data_5  = pd.read_csv('pi0s_5.txt',  sep=r'\s+', header=0, names=['event', 'px', 'py', 'pz', 'E'])
data_25 = pd.read_csv('pi0s_25.txt', sep=r'\s+', header=0, names=['event', 'px', 'py', 'pz', 'E'])

def get_all_pair_masses(df):
    masses = []
    for _, g in df.groupby('event'):
        if len(g) < 2:
            continue
        E  = g['E'].values
        px = g['px'].values
        py = g['py'].values
        pz = g['pz'].values
        i, j = np.triu_indices(len(E), k=1)
        tot_E  = E[i] + E[j]
        tot_px = px[i] + px[j]
        tot_py = py[i] + py[j]
        tot_pz = pz[i] + pz[j]
        m2 = tot_E**2 - tot_px**2 - tot_py**2 - tot_pz**2
        masses.extend(np.sqrt(np.maximum(0, m2)))
    return np.array(masses)

m1  = get_all_pair_masses(data_1)
m5  = get_all_pair_masses(data_5)
m25 = get_all_pair_masses(data_25)

# Quick histograms
plt.figure(figsize=(15,4))
for i, (data, lbl) in enumerate(zip([m1, m5, m25], ['N=1', 'N=5', 'N=25']), 1):
    plt.subplot(1,3,i)
    plt.hist(data, bins=80, range=(0.08, 0.22), histtype='step', lw=1.5)
    plt.title(lbl)
    plt.xlabel('Invariant mass [GeV]')
    plt.ylabel('Pairs')
plt.tight_layout()
plt.show()

# Gaussian + linear background fit
def model(x, A, mu, sigma, a, b):
    return A * np.exp(-0.5 * ((x - mu)/sigma)**2) + a*x + b

def fit_peak(masses, label):
    hist, edges = np.histogram(masses, bins=100, range=(0.1, 0.17))
    centers = (edges[:-1] + edges[1:]) / 2
    width = edges[1] - edges[0]

    # Initial guess
    peak_bin = np.argmax(hist)
    p0 = [hist.max() * width * 2.5, centers[peak_bin], 0.008, -500, 100]

    popt, _ = curve_fit(model, centers, hist, p0=p0, sigma=np.sqrt(hist + 1e-6))

    A, mu, sigma, a, b = popt
    N_pions = A * sigma * np.sqrt(2 * np.pi)           # ≈ total signal yield
    region = (centers >= mu - 3*sigma) & (centers <= mu + 3*sigma)
    S_region = np.sum(model(centers[region], *popt) - (a*centers[region] + b))
    B_region = np.sum(a*centers[region] + b)
    sb = S_region / B_region if B_region > 0 else float('inf')

    print(f"{label}:")
    print(f"  Estimated pions = {N_pions:.0f}")
    print(f"  S/B in ±3σ     = {sb:.1f}")
    print(f"  Peak at        = {mu:.3f} ± ~{sigma:.3f} GeV\n")

    # Plot
    xfine = np.linspace(0.1, 0.17, 300)
    plt.hist(masses, bins=100, range=(0.1,0.17), alpha=0.6, label='Data')
    plt.plot(xfine, model(xfine, *popt), 'r-', lw=2, label='Fit')
    plt.plot(xfine, a*xfine + b, 'g--', label='BG')
    plt.title(f"{label} – pions ≈ {N_pions:.0f}, S/B ≈ {sb:.1f}")
    plt.xlabel('Invariant mass [GeV]')
    plt.ylabel('Pairs')
    plt.legend()
    plt.show()

fit_peak(m1,  'N = 1')
fit_peak(m5,  'N = 5')
fit_peak(m25, 'N = 25')
