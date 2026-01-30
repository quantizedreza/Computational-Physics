"""
Gaussian Covariance Matrix & Spectral Analysis
- Heatmap of the covariance matrix
- Log-scale plot of sorted eigenvalues
- Clean, modern visualization style
"""

import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
#  PARAMETERS
# =============================================================================
N = 180               # Number of grid points
L = 1.0               # Domain length [0, L]
ell = 0.12            # Characteristic length scale (correlation length)

dx = L / (N - 1)      # Grid spacing
x = np.linspace(0, L, N)

# =============================================================================
#  BUILD COVARIANCE MATRIX  (vectorized - fast)
# =============================================================================
# C_ij = exp( - (||x_i - x_j||²) / (2 * ell²) )
dist = np.abs(x[:, None] - x[None, :])
C = np.exp(-0.5 * (dist / ell) ** 2)

# =============================================================================
#  EIGEN-DECOMPOSITION
# =============================================================================
# Use eigh for symmetric real matrix → faster & more stable
eigvals, eigvecs = np.linalg.eigh(C)

# Sort descending
idx = np.argsort(eigvals)[::-1]
eigvals = eigvals[idx]
# eigvecs = eigvecs[:, idx]   # uncomment if you need the eigenvectors later

# =============================================================================
#  VISUALIZATION - Beautiful side-by-side layout
# =============================================================================
plt.style.use('ggplot')  # nice modern look (you can also try 'bmh', 'seaborn-v0_8')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5),
                               gridspec_kw={'width_ratios': [1, 1.15]})

# ─── Left: Covariance Matrix Heatmap ────────────────────────────────────────
im = ax1.imshow(C,
                cmap='inferno',          # 'viridis', 'magma', 'plasma', 'inferno', 'turbo'
                interpolation='nearest',
                extent=[0, L, 0, L],
                origin='lower',
                vmin=0, vmax=1)

cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
cbar.set_label('Covariance value', fontsize=11)

ax1.set_title(f'Gaussian Covariance Matrix\n'
              f'$\\ell = {ell:.2f}$,   $N = {N}$,   domain $[0, {L}]$',
              fontsize=13, pad=12)
ax1.set_xlabel('Position', fontsize=11)
ax1.set_ylabel('Position', fontsize=11)
ax1.set_aspect('equal')

# ─── Right: Eigenvalues (log scale) ─────────────────────────────────────────
ax2.semilogy(np.arange(1, N+1), eigvals,
             color='#1f77b4', linewidth=2.4, alpha=0.9, marker='o',
             markersize=4, markevery=10)

ax2.set_title('Sorted Eigenvalues (log scale)', fontsize=13, pad=12)
ax2.set_xlabel('Mode number', fontsize=11)
ax2.set_ylabel('Eigenvalue λ', fontsize=11)
ax2.grid(True, which='both', ls='-', alpha=0.25, zorder=0)

# Add some informative annotations
total_var = np.sum(eigvals)
n_90 = np.argmax(np.cumsum(eigvals)/total_var >= 0.90) + 1
n_99 = np.argmax(np.cumsum(eigvals)/total_var >= 0.99) + 1

ax2.axvline(n_90, color='darkred', ls='--', alpha=0.6, lw=1.4,
            label=f'90% variance ({n_90} modes)')
ax2.axvline(n_99, color='darkorange', ls='--', alpha=0.6, lw=1.4,
            label=f'99% variance ({n_99} modes)')

ax2.legend(frameon=True, fontsize=10, loc='upper right',
           facecolor='white', framealpha=0.92)

# Final touches
plt.suptitle('Gaussian Process Covariance & Spectral Decay', fontsize=15, y=1.02)
plt.tight_layout()

# Save in good quality
plt.savefig("gaussian_covariance_eigenvalues.png",
            dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

# Quick console summary
print(f"Domain: [0, {L}],  N = {N},  ℓ = {ell}")
print(f"Strongest eigenvalue:    {eigvals[0]:.4g}")
print(f"Number of modes for:")
print(f"  • ≥ 90% variance → {n_90}")
print(f"  • ≥ 99% variance → {n_99}")
