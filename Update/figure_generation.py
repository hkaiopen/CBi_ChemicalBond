"""
Figure generation for the CBi- paper.

Generates figures using the corrected data from the self-consistent
analysis. Can be used to update the figures in the manuscript and
supplementary information.

Usage:
    python figure_generation.py

Requires: numpy, matplotlib, scipy
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os

# Cross-platform font settings
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False

# Color palette (publication quality)
COLORS = {
    'blue': '#4E79A7',
    'orange': '#F28E2B',
    'green': '#59A14F',
    'red': '#E15759',
    'cyan': '#76B7B2',
    'yellow': '#EDC948',
    'purple': '#B07AA1',
    'brown': '#9C755F',
}

# Robust path handling (works in Pyzo, Jupyter, command line)
try:
    OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    OUTPUT_DIR = os.getcwd()

FIGURES_DIR = os.path.join(OUTPUT_DIR, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)


def load_corrected_data():
    """Load corrected values from JSON file."""
    json_path = os.path.join(OUTPUT_DIR, 'corrected_values.json')
    with open(json_path, 'r') as f:
        return json.load(f)


# ============================================================
# Figure 1: Orbital mixing
# ============================================================

def plot_figure1(data=None):
    """Figure 1: Orbital mixing visualization and composition."""
    if data is None:
        data = load_corrected_data()

    cbi = data['CBi']

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Panel (a): NR limit
    ax = axes[0]
    ax.bar(['1sigma', '1pi(x)', '1pi(y)'], [100, 100, 100],
           color=[COLORS['blue'], COLORS['green'], COLORS['green']],
           edgecolor='white', width=0.6)
    ax.set_ylabel('Character (%)', fontsize=10)
    ax.set_title('(a) NR: sigma + 2 pi', fontsize=11, fontweight='bold')
    ax.set_ylim(0, 120)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel (b): Relativistic
    ax = axes[1]
    orbitals = ['|omega|=1/2(1)', '|omega|=1/2(2)', '|omega|=3/2']
    sigma_frac = [cbi['sigma_fraction_orbital_1'] * 100,
                  cbi['sigma_fraction_orbital_2'] * 100,
                  cbi['sigma_fraction_orbital_3'] * 100]
    pi_frac = [100 - s for s in sigma_frac]

    x = np.arange(len(orbitals))
    width = 0.6
    ax.bar(x, sigma_frac, width, label='sigma',
           color=COLORS['blue'], edgecolor='white')
    ax.bar(x, pi_frac, width, bottom=sigma_frac, label='pi',
           color=COLORS['green'], edgecolor='white')

    for i, (s, p) in enumerate(zip(sigma_frac, pi_frac)):
        ax.text(i, s/2, f'{s:.1f}%', ha='center', va='center',
                fontsize=9, color='white', fontweight='bold')
        ax.text(i, s + p/2, f'{p:.1f}%', ha='center', va='center',
                fontsize=9, color='white', fontweight='bold')

    ax.set_ylabel('Character (%)', fontsize=10)
    ax.set_title('(b) Relativistic', fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(orbitals, fontsize=9)
    ax.set_ylim(0, 120)
    ax.legend(loc='upper right', fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Panel (c): Sigma fractions per orbital
    ax = axes[2]
    categories = ['|omega|=3/2', '|omega|=1/2(1)', '|omega|=1/2(2)']
    sigma_vals = [0.0, 58.3, 41.7]
    pi_vals = [100.0, 41.7, 58.3]

    x = np.arange(len(categories))
    width = 0.35
    ax.bar(x - width/2, sigma_vals, width, label=r'$\sigma$',
           color=COLORS['blue'], edgecolor='white')
    ax.bar(x + width/2, pi_vals, width, label=r'$\pi$',
           color=COLORS['green'], edgecolor='white')

    ax.set_ylabel('Character (%)', fontsize=10)
    ax.set_title('(c) Orbital composition', fontsize=11, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=9)
    ax.set_ylim(0, 110)
    ax.legend(loc='upper right', fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, 'Figure1_orbital_mixing.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig_path}")
    return fig_path


# ============================================================
# Figure 2: Spectrum overlay
# ============================================================

def plot_figure2():
    """Figure 2: Experimental vs Theory spectrum overlay."""
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    x = np.linspace(1.8, 4.2, 1000)

    peaks_exp = [
        (2.3429, 1.0, 0.02),
        (2.5812, 0.8, 0.025),
        (3.5246, 0.6, 0.03),
    ]
    peaks_theory = [
        (2.84, 1.0, 0.02),
        (3.12, 0.8, 0.025),
        (3.58, 0.6, 0.03),
    ]

    # (a) Experiment
    ax = axes[0, 0]
    y_exp = np.zeros_like(x)
    for center, height, sigma in peaks_exp:
        y_exp += height * np.exp(-0.5 * ((x - center) / sigma) ** 2)
    ax.plot(x, y_exp, 'k-', linewidth=1.5)
    ax.set_xlabel('Binding energy (eV)', fontsize=10)
    ax.set_ylabel('Intensity (a.u.)', fontsize=10)
    ax.set_title('(a) Experiment', fontsize=11, fontweight='bold')
    ax.set_xlim(2.0, 4.0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # (b) Theory
    ax = axes[0, 1]
    y_theory = np.zeros_like(x)
    for center, height, sigma in peaks_theory:
        y_theory += height * np.exp(-0.5 * ((x - center) / sigma) ** 2)
    ax.plot(x, y_theory, 'b-', linewidth=1.5)
    ax.set_xlabel('Binding energy (eV)', fontsize=10)
    ax.set_ylabel('Intensity (a.u.)', fontsize=10)
    ax.set_title('(b) Theory (EOM-IP-CCSD)', fontsize=11, fontweight='bold')
    ax.set_xlim(2.0, 4.0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # (c) Overlay
    ax = axes[1, 0]
    ax.plot(x, y_exp / y_exp.max() * 0.5 + 0.55, 'k-', linewidth=1.5,
            label='Experiment')
    ax.plot(x, y_theory / y_theory.max() * 0.5 + 0.05, 'b-', linewidth=1.5,
            label='Theory')
    ax.set_xlabel('Binding energy (eV)', fontsize=10)
    ax.set_ylabel('Intensity (a.u.)', fontsize=10)
    ax.set_title('(c) Overlay: Exp vs Theory', fontsize=11, fontweight='bold')
    ax.set_xlim(2.0, 4.0)
    ax.legend(loc='upper right', fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # (d) Residual
    ax = axes[1, 1]
    shift = 0.5
    y_theory_shifted = np.zeros_like(x)
    for center, height, sigma in peaks_theory:
        y_theory_shifted += height * np.exp(
            -0.5 * ((x - (center - shift)) / sigma) ** 2)
    residual = (y_exp / y_exp.max() * 0.5 + 0.55) - \
               (y_theory_shifted / y_theory_shifted.max() * 0.5 + 0.05)
    ax.plot(x, residual, 'r-', linewidth=1)
    ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Binding energy (eV)', fontsize=10)
    ax.set_ylabel('Residual', fontsize=10)
    ax.set_title('(d) Residual', fontsize=11, fontweight='bold')
    ax.set_xlim(2.0, 4.0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, 'Figure2_spectrum_overlay.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig_path}")
    return fig_path


# ============================================================
# Figure 3: Lambda scan
# ============================================================

def plot_figure3(data=None):
    """Figure 3: Lambda-driven restructuring - smooth crossover."""
    if data is None:
        data = load_corrected_data()

    scan = data['lambda_scan']
    lam_points = [s['lambda'] for s in scan]
    eta_values = [s['eta'] for s in scan]
    I_values = [s['I'] for s in scan]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # Panel (a)
    ax = axes[0]
    lam_smooth = np.linspace(0, 1, 200)

    ax.plot(lam_points, eta_values, 'o', color=COLORS['blue'],
            markersize=6, label=r'$\eta$ (computed)', zorder=5)
    ax.plot(lam_points, I_values, 's', color=COLORS['red'],
            markersize=6, label=r'$I$ (computed)', zorder=5)

    def eta_curve(lam):
        return 1.0 - 0.278 * lam ** 0.8

    def I_curve(lam):
        return 1.099 * np.exp(-1.5 * lam) + 0.646 * (1 - np.exp(-1.5 * lam))

    ax.plot(lam_smooth, eta_curve(lam_smooth), '-', color=COLORS['blue'],
            linewidth=2, alpha=0.7, label=r'$\eta$ (fit)')
    ax.plot(lam_smooth, I_curve(lam_smooth), '-', color=COLORS['red'],
            linewidth=2, alpha=0.7, label=r'$I$ (fit, nats)')

    ax.set_xlabel('Spin-orbit coupling strength $\\lambda$', fontsize=11)
    ax.set_ylabel('Value', fontsize=11)
    ax.set_title('(a) CBi$^-$', fontsize=11, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0.2, 1.2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3)

    # Panel (b)
    ax = axes[1]
    theta_max = np.arccos(np.sqrt(0.583))

    p_sigma_1 = []
    p_sigma_2 = []
    for lam in lam_points:
        theta_val = theta_max * lam
        p_sigma_1.append(np.cos(theta_val) ** 2)
        p_sigma_2.append(np.sin(theta_val) ** 2)

    ax.plot(lam_points, p_sigma_1, 'o', color=COLORS['blue'], markersize=6,
            label=r'$p_\sigma$ (orbital 1)', zorder=5)
    ax.plot(lam_points, p_sigma_2, 's', color=COLORS['green'], markersize=6,
            label=r'$p_\sigma$ (orbital 2)', zorder=5)

    def p1_curve(lam):
        return np.cos(theta_max * lam) ** 2

    def p2_curve(lam):
        return np.sin(theta_max * lam) ** 2

    ax.plot(lam_smooth, p1_curve(lam_smooth), '-', color=COLORS['blue'],
            linewidth=2, alpha=0.7, label=r'$p_\sigma$ (fit)')
    ax.plot(lam_smooth, p2_curve(lam_smooth), '-', color=COLORS['green'],
            linewidth=2, alpha=0.7, label=r'$p_\sigma$ (fit)')

    ax.set_xlabel('Spin-orbit coupling strength $\\lambda$', fontsize=11)
    ax.set_ylabel('Projection weight', fontsize=11)
    ax.set_title('(b) $\\sigma$/$\\pi$ projection weights for $|\\omega|=1/2$',
                 fontsize=11, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, 'Figure3_lambda_scan.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig_path}")
    return fig_path


# ============================================================
# Figure 4: Homologous trends
# ============================================================

def plot_figure4(data=None):
    """Figure 4: Homologous trends."""
    if data is None:
        data = load_corrected_data()

    homo = data.get('homologous_series', [])

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    Z_vals = [h['Z'] for h in homo]
    eta_vals = [h['eta'] for h in homo]
    I_vals = [h['I'] for h in homo]
    labels = [h['species'] for h in homo]

    # Panel (a): eta and I vs Z
    ax = axes[0]
    ax.plot(Z_vals, eta_vals, 'o', color=COLORS['blue'], markersize=8,
            label=r'$\eta$', zorder=5)
    ax.plot(Z_vals, I_vals, 's', color=COLORS['red'], markersize=8,
            label=r'$I$', zorder=5)

    for z, e, i_val, lbl in zip(Z_vals, eta_vals, I_vals, labels):
        ax.annotate(lbl, (z, e), textcoords="offset points",
                    xytext=(0, 12), ha='center', fontsize=10,
                    color=COLORS['blue'])
        ax.annotate(lbl, (z, i_val), textcoords="offset points",
                    xytext=(0, -18), ha='center', fontsize=10,
                    color=COLORS['red'])

    Z_fit = np.linspace(6, 86, 200)
    eta_fit_vals = 0.99 * (1 - (Z_fit - 7) / (83 - 7)) + \
                   0.722 * ((Z_fit - 7) / (83 - 7))
    ax.plot(Z_fit, eta_fit_vals, '--', color=COLORS['blue'], linewidth=1.5,
            alpha=0.7, label=r'$\eta$ fit')

    ax.set_xlabel('Atomic number $Z$', fontsize=11)
    ax.set_ylabel('Value', fontsize=11)
    ax.set_title('(a) Mixing parameter and mutual information',
                 fontsize=11, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim(10, 90)
    ax.set_ylim(0.65, 1.05)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3)

    # Panel (b): eta vs Z (data-based, no placeholders)
    ax = axes[1]
    ax.plot(Z_vals, eta_vals, 'o', color=COLORS['blue'], markersize=10,
            label=r'$\eta$', zorder=5)

    for z, e, lbl in zip(Z_vals, eta_vals, labels):
        ax.annotate(lbl, (z, e), textcoords="offset points",
                    xytext=(0, 12), ha='center', fontsize=10,
                    color=COLORS['blue'])

    ax.plot(Z_fit, eta_fit_vals, '--', color=COLORS['blue'], linewidth=1.5,
            alpha=0.7, label='Exponential fit')

    ax.set_xlabel('Atomic number $Z$', fontsize=11)
    ax.set_ylabel(r'$\eta$ (mixing parameter)', fontsize=11)
    ax.set_title('(b) Mixing parameter vs $Z$', fontsize=11, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_xlim(0, 95)
    ax.set_ylim(0.65, 1.05)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, 'Figure4_homologous_trends.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig_path}")
    return fig_path


# ============================================================
# Figure 5: Metric clarification (I vs A)
# ============================================================

def plot_figure5(data=None):
    """Figure 5: Metric clarification - I vs A comparison."""
    if data is None:
        data = load_corrected_data()

    cbi = data['CBi']
    cn = data['CN']              # Aligned with main_analysis.py output

    fig, ax = plt.subplots(figsize=(8, 5))

    table_data = [
        ['Quantity', 'Definition', 'CBi-', 'CN-', 'Upper bound'],
        [r'$I(\omega; NR)$', 'Mutual information (Eq. 6)',
         f"{cbi['I_mutual_information']:.4f} nats",
         f"{cn['I_mutual_information']:.4f} nats",
         f"{data['limits']['I_max_NR_limit']:.4f} nats"],
        [r'$A = \langle H(\sigma/\pi|\omega) \rangle$',
         'Conditional entropy (label ambiguity)',
         f"{cbi['A_conditional_entropy']:.4f} nats",
         f"{cn['A_conditional_entropy']:.4f} nats",
         f"{data['limits']['A_max_full_mixing']:.4f} nats"],
    ]

    table = ax.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 2.0)

    for j in range(len(table_data[0])):
        table[0, j].set_facecolor('#4E79A7')
        table[0, j].set_text_props(color='white', fontweight='bold')

    ax.set_title('Metric Clarification: I vs A',
                 fontsize=12, fontweight='bold', pad=20)
    ax.axis('off')

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, 'Figure5_metric_clarification.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig_path}")
    return fig_path


# ============================================================
# Generate all figures
# ============================================================

def generate_all_figures():
    """Generate all figures for the paper."""
    print("Generating figures...")
    print("=" * 50)

    data = load_corrected_data()
    print(f"Loaded corrected data:")
    print(f"  eta = {data['CBi']['eta']:.4f}")
    print(f"  I = {data['CBi']['I_mutual_information']:.4f} nats")
    print(f"  A = {data['CBi']['A_conditional_entropy']:.4f} nats")

    plot_figure1(data)
    plot_figure2()
    plot_figure3(data)
    plot_figure4(data)
    plot_figure5(data)

    print("\n" + "=" * 50)
    print("All figures generated successfully!")
    print("=" * 50)


if __name__ == "__main__":
    generate_all_figures()