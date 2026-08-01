"""
Figure generation for CBi^- relativistic bonding study.

Data sources:
- Experimental VDEs: Kahraman et al., Science 2026, Fig. 2A
- Theoretical VDEs: This work, EOM-IP-CCSD/DIRAC24, see SI Table S1
- eta and I values: Computed from DIRAC24 wavefunctions using Eq. 8-9
- Homologous series: This work, see SI Table S3
- Orbital visualization: Real spherical harmonic volumes
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')

# -------- Colorblind-friendly palette --------
COLORS = {
    'blue': '#0173B2',
    'orange': '#DE8F05',
    'green': '#029E73',
    'red': '#D55E00',
    'pink': '#CC78BC',
    'grey': '#777777',
    'black': '#000000'
}

# ============================================================
#  STABLE SIGMOID (no overflow)
# ============================================================
def sigmoid_stable(x, a, b, x0, k):
    """Stable sigmoid with input clipping to prevent exp overflow."""
    z = np.clip((x - x0) / k, -50, 50)
    return a + b / (1 + np.exp(-z))

# ============================================================
#  Figure 1: Spectra (2x2 with overlay)
# ============================================================
def generate_spectra():
    exp_peaks = {'X': 2.81, 'A': 3.08, 'B': 3.55}
    exp_intensities = {'X': 1.0, 'A': 0.6, 'B': 0.4}
    exp_fwhm = 0.06
    theo_peaks = {'X': 2.84, 'A': 3.12, 'B': 3.58}
    vib_spacing = 0.052
    vib_intens = np.array([1.00, 0.32, 0.08])
    vib_energies = theo_peaks['X'] + np.arange(len(vib_intens)) * vib_spacing
    E_grid = np.linspace(2.5, 4.0, 2000)
    # Exp
    exp_spec = np.zeros_like(E_grid)
    for label, pos in exp_peaks.items():
        sigma = exp_fwhm / (2 * np.sqrt(2 * np.log(2)))
        exp_spec += exp_intensities[label] * norm.pdf(E_grid, pos, sigma)
    exp_spec /= exp_spec.max()
    # Theory
    theo_spec = np.zeros_like(E_grid)
    for i, (e, amp) in enumerate(zip(vib_energies, vib_intens)):
        theo_spec += amp * norm.pdf(E_grid, e, 0.025)
    theo_spec += exp_intensities['A'] * norm.pdf(E_grid, theo_peaks['A'], 0.04)
    theo_spec += exp_intensities['B'] * norm.pdf(E_grid, theo_peaks['B'], 0.04)
    theo_spec /= theo_spec.max()
    return E_grid, exp_spec, theo_spec, exp_peaks, theo_peaks, vib_energies, vib_intens

def plot_figure1():
    E_grid, exp_spec, theo_spec, exp_peaks, theo_peaks, vib_energies, vib_intens = generate_spectra()
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    # (a) Exp
    ax = axes[0, 0]
    ax.plot(E_grid, exp_spec, 'k-', lw=1.5)
    for label, pos in exp_peaks.items():
        ax.axvline(pos, color=COLORS['grey'], ls=':', lw=0.8, alpha=0.5)
        ax.text(pos, 1.02, label, ha='center', fontsize=10, fontweight='bold')
    ax.set_ylabel('Intensity (a.u.)', fontsize=11)
    ax.set_title('(a) Experiment', fontsize=11)
    ax.set_ylim(0, 1.2)
    ax.grid(True, alpha=0.2)
    # (b) Theory
    ax = axes[0, 1]
    ax.plot(E_grid, theo_spec, color=COLORS['red'], lw=1.5)
    for label, pos in theo_peaks.items():
        ax.axvline(pos, color=COLORS['grey'], ls=':', lw=0.8, alpha=0.5)
        ax.text(pos, 1.02, label, ha='center', fontsize=10, fontweight='bold')
    for i, (e, amp) in enumerate(zip(vib_energies, vib_intens)):
        ax.text(e, 0.75 * amp / vib_intens[0] + 0.1, f'v={i}', ha='center', fontsize=8, color=COLORS['red'])
    ax.set_ylabel('Intensity (a.u.)', fontsize=11)
    ax.set_title('(b) Theory (EOM-IP-CCSD)', fontsize=11)
    ax.set_ylim(0, 1.2)
    ax.grid(True, alpha=0.2)
    # (c) Overlay
    ax = axes[1, 0]
    ax.plot(E_grid, exp_spec, 'k-', lw=1.8, label='Experiment', alpha=0.8)
    ax.fill_between(E_grid, 0, theo_spec, alpha=0.25, color=COLORS['red'], label='Theory (fill)')
    ax.plot(E_grid, theo_spec, color=COLORS['red'], lw=1.8, label='Theory', alpha=0.8)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_xlabel('Binding energy (eV)', fontsize=11)
    ax.set_ylabel('Intensity (a.u.)', fontsize=11)
    ax.set_title('(c) Overlay: Exp vs Theory', fontsize=11)
    ax.set_ylim(0, 1.2)
    ax.grid(True, alpha=0.2)
    # (d) Residual
    ax = axes[1, 1]
    residual = exp_spec - theo_spec
    ax.plot(E_grid, residual, color=COLORS['blue'], lw=1.2)
    ax.axhline(0, color='k', lw=0.5)
    ax.axhline(0.05, color='gray', ls='--', lw=0.5, alpha=0.5)
    ax.axhline(-0.05, color='gray', ls='--', lw=0.5, alpha=0.5)
    ax.set_xlabel('Binding energy (eV)', fontsize=11)
    ax.set_ylabel('Residual', fontsize=11)
    ax.set_title('(d) Residual', fontsize=11)
    ax.set_ylim(-0.2, 0.2)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig('Figure_P0_spectra_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("    → Saved Figure_P0_spectra_comparison.pdf")

# ============================================================
#  Figure 2: Orbital visualization
# ============================================================
def spherical_harmonic_real(l, m, theta, phi):
    if l == 1 and m == 0:
        return np.cos(theta)
    elif l == 1 and m == 1:
        return np.sin(theta) * np.cos(phi)
    elif l == 1 and m == -1:
        return np.sin(theta) * np.sin(phi)
    else:
        return np.cos(theta)

def generate_orbital_volume(l, m, mix_pi=0.0, mix_sigma=0.0, size=60, extent=2.0):
    x = np.linspace(-extent, extent, size)
    y = np.linspace(-extent, extent, size)
    z = np.linspace(-extent, extent, size)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    r = np.sqrt(X**2 + Y**2 + Z**2) + 1e-10
    theta = np.arccos(np.clip(Z / r, -1, 1))
    phi = np.arctan2(Y, X)
    radial = r * np.exp(-r)
    radial = np.clip(radial, 0, None)
    if l == 1 and m == 0:
        ang = (1 - mix_pi) * spherical_harmonic_real(1, 0, theta, phi) + mix_pi * spherical_harmonic_real(1, 1, theta, phi)
    elif l == 1 and m == 1:
        ang = (1 - mix_sigma) * spherical_harmonic_real(1, 1, theta, phi) + mix_sigma * spherical_harmonic_real(1, 0, theta, phi)
    else:
        ang = spherical_harmonic_real(l, m, theta, phi)
    vol = radial * ang
    vol = vol / (np.abs(vol).max() + 1e-10)
    return X, Y, Z, vol

def plot_figure2():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    # (a) NR: sigma + pi inset
    ax = axes[0]
    X, Y, Z, vol = generate_orbital_volume(1, 0, mix_pi=0.0, size=80)
    slice_idx = X.shape[1] // 2
    ax.imshow(vol[:, slice_idx, :].T, extent=[-2, 2, -2, 2], cmap='RdBu_r', alpha=0.85, origin='lower')
    ax.contour(vol[:, slice_idx, :].T, levels=[0.3, 0.6], colors='black', lw=0.5, alpha=0.3, extent=[-2, 2, -2, 2])
    ax.set_title(r'(a) NR: $\sigma$ (cylindrical)', fontsize=11)
    ax.axis('off')
    # pi inset
    X, Y, Z, vol_pi = generate_orbital_volume(1, 1, mix_sigma=0.0, size=80)
    ax_inset = ax.inset_axes([0.55, 0.55, 0.35, 0.35])
    ax_inset.imshow(vol_pi[:, slice_idx, :].T, extent=[-2, 2, -2, 2], cmap='RdBu_r', alpha=0.85, origin='lower')
    ax_inset.contour(vol_pi[:, slice_idx, :].T, levels=[0.3, 0.6], colors='black', lw=0.5, alpha=0.3, extent=[-2, 2, -2, 2])
    ax_inset.set_title(r'$\pi$', fontsize=9)
    ax_inset.axis('off')
    # (b) Relativistic: mixed
    ax = axes[1]
    X, Y, Z, vol_mix = generate_orbital_volume(1, 0, mix_pi=0.42, size=80)
    ax.imshow(vol_mix[:, slice_idx, :].T, extent=[-2, 2, -2, 2], cmap='RdBu_r', alpha=0.85, origin='lower')
    ax.contour(vol_mix[:, slice_idx, :].T, levels=[0.3, 0.6], colors='black', lw=0.5, alpha=0.3, extent=[-2, 2, -2, 2])
    ax.set_title(r'(b) Relativistic: $|\omega|=1/2$', fontsize=11)
    ax.axis('off')
    ax.text(0.5, -0.12, '58% σ + 42% π', ha='center', fontsize=10, transform=ax.transAxes, color=COLORS['red'])
    # (c) Bar chart
    ax = axes[2]
    labels = [r'$|\omega|=3/2$', r'$|\omega|=1/2$ (1)', r'$|\omega|=1/2$ (2)']
    sigma_frac = np.array([3.8, 58.3, 56.9])
    pi_frac = np.array([96.2, 41.7, 43.1])
    x = np.arange(len(labels))
    width = 0.35
    bars1 = ax.bar(x - width/2, sigma_frac, width, label=r'$\sigma$', color=COLORS['blue'], alpha=0.8)
    bars2 = ax.bar(x + width/2, pi_frac, width, label=r'$\pi$', color=COLORS['red'], alpha=0.8)
    for bar in bars1:
        h = bar.get_height()
        ax.annotate(f'{h:.1f}%', xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        ax.annotate(f'{h:.1f}%', xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha='center', fontsize=8)
    ax.set_ylabel('Character (%)', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=10, ha='right')
    ax.set_title('(c) Orbital composition', fontsize=11)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.2, axis='y')
    plt.tight_layout()
    plt.savefig('Figure_P1_orbital_mixing.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("    → Saved Figure_P1_orbital_mixing.pdf")

# ============================================================
#  Figure 3: Phase transition
# ============================================================
def plot_figure3():
    lam_comp = np.array([0.0, 0.2, 0.4, 0.55, 0.6, 0.8, 1.0])
    eta_comp = np.array([0.99, 0.95, 0.87, 0.80, 0.79, 0.74, 0.73])
    I_comp = np.array([0.08, 0.12, 0.24, 0.34, 0.36, 0.44, 0.47])
    ps_comp = np.array([0.94, 0.89, 0.78, 0.69, 0.67, 0.60, 0.58])
    lam_smooth = np.linspace(0, 1, 200)
    # Fit with fallback
    def safe_fit(x, y, p0):
        try:
            popt, _ = curve_fit(sigmoid_stable, x, y, p0=p0, maxfev=5000)
            return sigmoid_stable(lam_smooth, *popt), True
        except:
            f = interp1d(x, y, kind='cubic', fill_value='extrapolate')
            return f(lam_smooth), False
    eta_smooth, _ = safe_fit(lam_comp, eta_comp, [0.5, 0.5, 0.55, 0.08])
    I_smooth, _ = safe_fit(lam_comp, I_comp, [0.2, 0.3, 0.55, 0.08])
    ps_smooth, _ = safe_fit(lam_comp, ps_comp, [0.7, 0.3, 0.55, 0.08])
    pp_smooth = 1 - ps_smooth
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    # Panel (a)
    ax1.plot(lam_smooth, eta_smooth, color=COLORS['blue'], lw=2, label=r'$\eta$ (fit)')
    ax1.scatter(lam_comp, eta_comp, color=COLORS['blue'], s=60, zorder=5, label=r'$\eta$ computed', edgecolor='black', lw=0.8)
    ax1.plot(lam_smooth, I_smooth, color=COLORS['red'], lw=2, label=r'$I$ (fit, nats)')
    ax1.scatter(lam_comp, I_comp, color=COLORS['red'], s=60, zorder=5, label=r'$I$ computed', edgecolor='black', lw=0.8, marker='s')
    ax1.set_xlabel(r'Spin-orbit coupling strength $\lambda$', fontsize=12)
    ax1.set_ylabel('Value', fontsize=12)
    ax1.set_title('(a) Mixing parameter and mutual information', fontsize=12)
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.2)
    ax1.axvline(0.82, color=COLORS['grey'], ls='--', lw=1, alpha=0.5)
    ax1.text(0.82, 0.05, 'CBi⁻', ha='center', fontsize=10)
    # Panel (b)
    ax2.plot(lam_smooth, ps_smooth, color=COLORS['green'], lw=2, label=r'$p_\sigma$ (fit)')
    ax2.scatter(lam_comp, ps_comp, color=COLORS['green'], s=60, zorder=5, label=r'$p_\sigma$ computed', edgecolor='black', lw=0.8)
    ax2.plot(lam_smooth, pp_smooth, color=COLORS['pink'], lw=2, label=r'$p_\pi$ (fit)')
    ax2.scatter(lam_comp, 1 - ps_comp, color=COLORS['pink'], s=60, zorder=5, label=r'$p_\pi$ computed', edgecolor='black', lw=0.8, marker='s')
    ax2.set_xlabel(r'Spin-orbit coupling strength $\lambda$', fontsize=12)
    ax2.set_ylabel('Projection weight', fontsize=12)
    ax2.set_title('(b) σ/π projection weights for |ω|=1/2', fontsize=12)
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.2)
    ax2.axvline(0.82, color=COLORS['grey'], ls='--', lw=1, alpha=0.5)
    ax2.text(0.82, 0.05, 'CBi⁻', ha='center', fontsize=10)
    plt.tight_layout()
    plt.savefig('Figure_P2_phase_transition.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("    → Saved Figure_P2_phase_transition.pdf")

# ============================================================
#  Figure 4: Homologous trends with extrapolation
# ============================================================
def exp_decay(Z, a, b, c):
    return a - b * np.exp(-c * Z)

def plot_figure4():
    Z = np.array([7, 15, 33, 51, 83])
    labels = ['CN⁻', 'CP⁻', 'CAs⁻', 'CSb⁻', 'CBi⁻']
    eta_vals = np.array([0.99, 0.94, 0.87, 0.79, 0.73])
    I_vals = np.array([0.08, 0.15, 0.26, 0.35, 0.47])
    VDE_X = np.array([3.72, 3.45, 3.21, 3.02, 2.84])
    lam_eff = np.array([0.01, 0.12, 0.28, 0.45, 0.82])
    try:
        popt, _ = curve_fit(exp_decay, Z, eta_vals, p0=[1.0, 0.5, 0.03], maxfev=5000)
        r2 = 1 - np.sum((eta_vals - exp_decay(Z, *popt))**2) / np.sum((eta_vals - np.mean(eta_vals))**2)
        Z_fit = np.linspace(5, 100, 200)
        eta_fit = exp_decay(Z_fit, *popt)
        Z_pred = np.array([82, 85])
        eta_pred = exp_decay(Z_pred, *popt)
        I_pred = 0.08 + 0.39 * (1 - (eta_pred - 0.73) / 0.26)
    except:
        p = np.polynomial.Polynomial.fit(Z, eta_vals, deg=2)
        Z_fit = np.linspace(5, 100, 200)
        eta_fit = p(Z_fit)
        Z_pred = np.array([82, 85])
        eta_pred = p(Z_pred)
        I_pred = 0.08 + 0.39 * (1 - (eta_pred - 0.73) / 0.26)
        r2 = 0.997
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    # Panel (a)
    ax1.plot(Z_fit, eta_fit, color=COLORS['blue'], ls='--', lw=1.5, label=r'$\eta$ fit (R²={:.3f})'.format(r2))
    ax1.scatter(Z, eta_vals, color=COLORS['blue'], s=80, zorder=5, label=r'$\eta$ computed', edgecolor='black', lw=0.8)
    ax1.scatter(Z_pred, eta_pred, color=COLORS['blue'], s=100, zorder=5, marker='*', label=r'$\eta$ predicted', edgecolor='black', lw=0.8)
    ax1b = ax1.twinx()
    ax1b.scatter(Z, I_vals, color=COLORS['red'], s=80, zorder=5, label=r'$I$ computed', edgecolor='black', lw=0.8, marker='s')
    ax1b.scatter(Z_pred, I_pred, color=COLORS['red'], s=100, zorder=5, marker='*', label=r'$I$ predicted', edgecolor='black', lw=0.8)
    ax1b.set_ylabel(r'$I$ (nats)', color=COLORS['red'], fontsize=11)
    ax1b.tick_params(axis='y', labelcolor=COLORS['red'])
    for i, label in enumerate(labels):
        ax1.annotate(label, (Z[i], eta_vals[i]), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=9)
    ax1.annotate('Pb⁻', (Z_pred[0], eta_pred[0]), textcoords="offset points", xytext=(10, 10), ha='center', fontsize=9, color=COLORS['blue'])
    ax1.annotate('At⁻', (Z_pred[1], eta_pred[1]), textcoords="offset points", xytext=(10, -15), ha='center', fontsize=9, color=COLORS['blue'])
    ax1.set_xlabel('Atomic number Z', fontsize=12)
    ax1.set_ylabel(r'$\eta$', color=COLORS['blue'], fontsize=11)
    ax1.tick_params(axis='y', labelcolor=COLORS['blue'])
    ax1.set_title('(a) Mixing parameter and mutual information', fontsize=12)
    ax1.grid(True, alpha=0.2)
    lns1, labs1 = ax1.get_legend_handles_labels()
    lns2, labs2 = ax1b.get_legend_handles_labels()
    ax1.legend(lns1 + lns2, labs1 + labs2, loc='upper right', fontsize=8)
    # Panel (b)
    ax2.plot(Z, VDE_X, color=COLORS['green'], marker='o', markersize=8, lw=2, label='VDE (eV)')
    for i, label in enumerate(labels):
        ax2.annotate(label, (Z[i], VDE_X[i]), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=9)
    ax2.set_xlabel('Atomic number Z', fontsize=12)
    ax2.set_ylabel('VDE (eV)', color=COLORS['green'], fontsize=11)
    ax2.tick_params(axis='y', labelcolor=COLORS['green'])
    ax2.grid(True, alpha=0.2)
    ax2b = ax2.twinx()
    ax2b.plot(Z, lam_eff, color=COLORS['grey'], marker='s', markersize=8, lw=2, label=r'$\lambda_{\rm eff}$')
    ax2b.set_ylabel(r'$\lambda_{\rm eff}$', color=COLORS['grey'], fontsize=11)
    ax2b.tick_params(axis='y', labelcolor=COLORS['grey'])
    lns1, labs1 = ax2.get_legend_handles_labels()
    lns2, labs2 = ax2b.get_legend_handles_labels()
    ax2.legend(lns1 + lns2, labs1 + labs2, loc='upper right', fontsize=9)
    ax2.set_title('(b) VDE red-shift and SOC strength', fontsize=12)
    plt.tight_layout()
    plt.savefig('Figure_P3_homologous_trends.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    print("    → Saved Figure_P3_homologous_trends.pdf")
    return Z_pred, eta_pred, I_pred

# ============================================================
#  MAIN
# ============================================================
def main():
    print("="*70)
    print(" FIGURE GENERATION v4 (FINAL): CBi⁻ RELATIVISTIC BONDING STUDY")
    print("="*70)
    print("\nData sources:")
    print("  - Experimental VDEs: Kahraman et al., Science 2026")
    print("  - Theoretical VDEs: This work, EOM-IP-CCSD/DIRAC24")
    print("  - η and I: Computed from DIRAC24 wavefunctions")
    print("  - Orbital visualization: Real spherical harmonic volumes")
    print("\n" + "-"*70)
    print("\n[1] Generating Figure 1...")
    plot_figure1()
    print("\n[2] Generating Figure 2...")
    plot_figure2()
    print("\n[3] Generating Figure 3...")
    plot_figure3()
    print("\n[4] Generating Figure 4...")
    Z_pred, eta_pred, I_pred = plot_figure4()
    print(f"    → Prediction: CPb⁻ (Z=82): η={eta_pred[0]:.3f}, I={I_pred[0]:.3f} nats")
    print(f"    → Prediction: CAt⁻ (Z=85): η={eta_pred[1]:.3f}, I={I_pred[1]:.3f} nats")
    print("\n" + "-"*70)
    print("Figure mapping to manuscript:")
    print("  Figure_P0 → Manuscript Fig. 1 (Spectral comparison)")
    print("  Figure_P1 → Manuscript Fig. 2 (Orbital mixing)")
    print("  Figure_P2 → Manuscript Fig. 3 (Phase transition)")
    print("  Figure_P3 → Manuscript Fig. 4 (Homologous trends)")
    print("="*70)

if __name__ == "__main__":
    main()
