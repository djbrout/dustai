"""Generate before/after plots for Uchuu 6600 SNe comparison."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm

# Load data
data = np.load('uchuu_6600_comparison.npz')
gauss_fs8 = data['gauss_fs8']
gauss_lo = data['gauss_lo']
gauss_hi = data['gauss_hi']
st_fs8 = data['studentt_fs8']
st_lo = data['studentt_lo']
st_hi = data['studentt_hi']
fid = float(data['fsigma8_fid'])
seeds = data['seeds']
n_mocks = len(gauss_fs8)

resid = np.load('uchuu_residuals.npz')
delta_mu = resid['delta_mu']
colors = resid['colors']

# ============================================================
# Figure 1: Hubble residual distribution from Uchuu P23
# ============================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

# Left: delta_mu histogram
bins = np.linspace(-0.5, 0.8, 100)
ax1.hist(delta_mu, bins=bins, density=True, alpha=0.7, color='steelblue',
         label=f'Uchuu $S_{{\\rm P23}}$')
x = np.linspace(-0.5, 0.8, 300)
ax1.plot(x, norm.pdf(x, loc=np.mean(delta_mu), scale=np.std(delta_mu)),
         'k--', lw=1.5, label='Gaussian')
from scipy.stats import skew, kurtosis
gamma = skew(delta_mu)
kappa = kurtosis(delta_mu, fisher=True)
ax1.text(0.95, 0.95, f'Skewness = {gamma:.2f}\nKurtosis = {kappa:.2f}',
         transform=ax1.transAxes, ha='right', va='top', fontsize=11,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
ax1.set_xlabel(r'$\Delta\mu$ (mag)', fontsize=12)
ax1.set_ylabel('Density', fontsize=12)
ax1.set_title('Hubble residuals (Uchuu + P23)', fontsize=13)
ax1.legend(fontsize=10)
ax1.set_xlim(-0.5, 0.8)

# Right: delta_mu vs color
ax2.hexbin(colors, delta_mu, gridsize=50, cmap='Blues', mincnt=1)
ax2.set_xlabel('SALT color $c$', fontsize=12)
ax2.set_ylabel(r'$\Delta\mu$ (mag)', fontsize=12)
ax2.set_title('Residual vs.\ color', fontsize=13)
ax2.set_xlim(-0.3, 0.3)
ax2.set_ylim(-0.6, 0.8)
ax2.axhline(0, color='k', ls='--', lw=0.8)

plt.tight_layout()
plt.savefig('fig_uchuu_residuals.pdf', dpi=150, bbox_inches='tight')
plt.savefig('fig_uchuu_residuals.png', dpi=150, bbox_inches='tight')
print('Saved fig_uchuu_residuals')

# ============================================================
# Figure 2: Per-mock fsigma8 comparison (main result figure)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5))

mock_idx = np.arange(1, n_mocks + 1)

# Gaussian: points with error bars
gauss_err_lo = gauss_fs8 - gauss_lo
gauss_err_hi = gauss_hi - gauss_fs8
ax.errorbar(mock_idx - 0.15, gauss_fs8,
            yerr=[gauss_err_lo, gauss_err_hi],
            fmt='o', color='steelblue', markersize=8, capsize=4,
            label='Gaussian likelihood', zorder=5)

# Student-t: points with error bars
st_err_lo = st_fs8 - st_lo
st_err_hi = st_hi - st_fs8
ax.errorbar(mock_idx + 0.15, st_fs8,
            yerr=[st_err_lo, st_err_hi],
            fmt='s', color='indianred', markersize=8, capsize=4,
            label='Student-$t$ likelihood', zorder=5)

# Fiducial line
ax.axhline(y=fid, color='k', linestyle='--', lw=1.5,
           label=f'Fiducial $f\\sigma_8 = {fid:.3f}$')
ax.axhline(y=0, color='gray', linestyle=':', lw=0.5)

# Mean lines
ax.axhline(y=np.mean(gauss_fs8), color='steelblue', ls=':', lw=1, alpha=0.5)
ax.axhline(y=np.mean(st_fs8), color='indianred', ls=':', lw=1, alpha=0.5)

ax.set_xlabel('Mock index', fontsize=13)
ax.set_ylabel(r'$f\sigma_8$', fontsize=14)
ax.set_title(r'$f\sigma_8$ recovery: 6600 SNe $\times$ 8 Uchuu mocks with $S_{\rm P23}$',
             fontsize=14)
ax.legend(fontsize=11, loc='upper left')
ax.set_xlim(0.3, n_mocks + 0.7)
ax.set_xticks(mock_idx)

plt.tight_layout()
plt.savefig('fig_uchuu_fsigma8.pdf', dpi=150, bbox_inches='tight')
plt.savefig('fig_uchuu_fsigma8.png', dpi=150, bbox_inches='tight')
print('Saved fig_uchuu_fsigma8')

# ============================================================
# Figure 3: Summary bar chart
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))

methods = ['True vel.\n(C25)', 'Gaussian\n(C25)', 'BBC\n(C25)',
           'Gaussian\n(this work)', 'Student-$t$\n(this work)']
values = [0.988, 0.808, 0.743, np.mean(gauss_fs8)/fid, np.mean(st_fs8)/fid]
errors = [0.013/fid, 0.036/fid, 0.022/fid,
          np.std(gauss_fs8)/fid, np.std(st_fs8)/fid]
bar_colors = ['green', 'steelblue', 'steelblue', 'steelblue', 'indianred']
hatches = ['', '', '//', '', '']

bars = ax.bar(range(len(methods)), values, yerr=errors, capsize=5,
              color=bar_colors, alpha=0.7, edgecolor='black', linewidth=0.8)
for bar, h in zip(bars, hatches):
    bar.set_hatch(h)

ax.axhline(y=1.0, color='k', linestyle='--', lw=1.5, label='Fiducial')
ax.set_xticks(range(len(methods)))
ax.set_xticklabels(methods, fontsize=11)
ax.set_ylabel(r'$\langle f\sigma_8 \rangle / (f\sigma_8)_{\rm fid}$', fontsize=13)
ax.set_ylim(0, 1.4)
ax.set_title(r'$S_{\rm P23}$ bias comparison', fontsize=14)

# Add bias labels
for i, (v, e) in enumerate(zip(values, errors)):
    bias = (v - 1) * 100
    ax.text(i, v + e + 0.03, f'{bias:+.0f}%', ha='center', fontsize=10,
            fontweight='bold' if i >= 3 else 'normal')

plt.tight_layout()
plt.savefig('fig_uchuu_summary.pdf', dpi=150, bbox_inches='tight')
plt.savefig('fig_uchuu_summary.png', dpi=150, bbox_inches='tight')
print('Saved fig_uchuu_summary')

# Print summary table
print(f'\n=== Summary ===')
print(f'Gaussian: mean={np.mean(gauss_fs8):.4f} std={np.std(gauss_fs8):.4f} '
      f'bias={np.mean(gauss_fs8)/fid-1:+.1%}')
print(f'Student-t: mean={np.mean(st_fs8):.4f} std={np.std(st_fs8):.4f} '
      f'bias={np.mean(st_fs8)/fid-1:+.1%}')
print(f'Fiducial: {fid:.4f}')
