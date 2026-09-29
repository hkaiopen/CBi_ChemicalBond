"""
Analysis pipeline: self-consistent S-matrix from wavefunction data.

This module implements the full workflow for computing self-consistent
overlap matrices and information-dynamic metrics from DIRAC/CFOUR
wavefunction data.

Functions
---------
construct_symmetry_adapted_basis : Build sigma/pi symmetry-adapted NR orbitals
compute_overlap_matrix : Compute S_ij = <phi_i^SOC | phi_j^NR>
compute_eta : Mixing parameter with optional optimal pairing
compute_mutual_information : I(omega; NR) - mutual information
compute_conditional_entropy : A = <H(sigma/pi | omega)> - label ambiguity
check_unitarity : Full unitarity diagnostics
run_analysis : Complete pipeline from coefficient matrices to metrics

Usage
-----
import numpy as np
from analysis_pipeline import run_analysis

metrics = run_analysis(soc_coeffs, nr_coeffs, nr_labels, active_space=[0,1,2])
print(f"eta = {metrics['eta']:.4f}")
print(f"I = {metrics['I']:.4f} nats")
print(f"A = {metrics['A']:.4f} nats")
"""

import numpy as np
from scipy.optimize import linear_sum_assignment
from typing import Tuple, Dict, Optional, List


# ============================================================
# 1. Symmetry-adapted basis construction
# ============================================================

def construct_symmetry_adapted_basis(
    nr_coefficients: np.ndarray,
    ao_symmetry: Optional[List[str]] = None,
    active_space: Optional[List[int]] = None
) -> Tuple[np.ndarray, List[str]]:
    """
    Organize non-relativistic orbitals by symmetry label and construct
    the active subspace.

    For a diatomic molecule with C_infty_v symmetry:
    - sigma orbitals: m_l = 0
    - pi orbitals: m_l = +/-1
    """
    if active_space is None:
        active_space = list(range(nr_coefficients.shape[0]))

    active_coeffs = nr_coefficients[active_space]
    if ao_symmetry is None:
        labels = [f"orb_{i}" for i in active_space]
    else:
        labels = [ao_symmetry[i] for i in active_space]

    return active_coeffs, labels


# ============================================================
# 2. Overlap matrix computation
# ============================================================

def compute_overlap_matrix(
    soc_coefficients: np.ndarray,
    nr_coefficients: np.ndarray,
    ao_overlap: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Compute the full overlap matrix S_ij = <phi_i^SOC | phi_j^NR>.
    """
    if ao_overlap is None:
        return soc_coefficients @ nr_coefficients.T
    else:
        return soc_coefficients @ ao_overlap @ nr_coefficients.T


# ============================================================
# 3. S-matrix metrics
# ============================================================

def compute_eta(S, optimal_pairing=True):
    """
    Compute the mixing parameter eta.

    eta = (1/N) * max_pi sum_i |<phi_i^SOC | phi_{pi(i)}^NR>|^2

    The maximization is over all permutations pi of the NR orbitals,
    solved via the Hungarian algorithm. This definition is essential
    when orbital energy ordering is not aligned with physical pairing.

    When optimal_pairing=False, only the diagonal (pi = identity) is
    used, which may give physically meaningless results when orbital
    orderings differ between the two basis sets.
    """
    overlap_sq = np.abs(S) ** 2
    if optimal_pairing:
        row_ind, col_ind = linear_sum_assignment(-overlap_sq)
    else:
        row_ind = np.arange(S.shape[0])
        col_ind = np.arange(min(S.shape[0], S.shape[1]))
    paired_values = overlap_sq[row_ind, col_ind]
    N = len(paired_values)
    eta = paired_values.sum() / N
    return eta, row_ind, col_ind, paired_values


def compute_mutual_information(S):
    """
    Compute the mutual information I(omega; NR).

    Limiting behavior:
    - NR limit (S = identity): I = ln(N) (maximum)
    - Complete mixing (S uniform): I = 0 (minimum)
    """
    N = S.shape[1]
    P = np.abs(S) ** 2 / N
    Pi = P.sum(axis=1)
    Pj = P.sum(axis=0)
    I = 0.0
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            if P[i, j] > 1e-15:
                ratio = P[i, j] / (Pi[i] * Pj[j])
                I += P[i, j] * np.log(ratio)
    return I


def compute_conditional_entropy(S):
    """
    Compute the average conditional entropy A = <H(sigma/pi | omega)>.

    Limiting behavior:
    - NR limit: A = 0
    - Complete mixing: A = ln(2) (maximum)
    """
    H_list = []
    for i in range(S.shape[0]):
        row = np.abs(S[i, :]) ** 2
        total = row.sum()
        if total < 1e-15:
            H_list.append(0.0)
            continue
        p_sigma = row[0] / total
        p_pi = 1.0 - p_sigma
        H = 0.0
        if p_sigma > 1e-15:
            H -= p_sigma * np.log(p_sigma)
        if p_pi > 1e-15:
            H -= p_pi * np.log(p_pi)
        H_list.append(H)
    A = np.mean(H_list)
    return A, np.array(H_list)


# ============================================================
# 4. Diagnostics
# ============================================================

def check_unitarity(S, atol=1e-10):
    """Comprehensive unitarity check for the S-matrix."""
    St = S.conj().T
    S_St = S @ St
    St_S = St @ S
    row_sums = np.sum(np.abs(S) ** 2, axis=1)
    col_sums = np.sum(np.abs(S) ** 2, axis=0)
    max_dev_row = np.max(np.abs(row_sums - 1.0))
    max_dev_col = np.max(np.abs(col_sums - 1.0))
    is_unitary = (
        np.allclose(S_St, np.eye(S.shape[0]), atol=atol) and
        np.allclose(St_S, np.eye(S.shape[1]), atol=atol)
    )
    return {
        'S_St': S_St, 'St_S': St_S,
        'row_sums': row_sums, 'col_sums': col_sums,
        'max_deviation_row': max_dev_row,
        'max_deviation_col': max_dev_col,
        'is_unitary': bool(is_unitary),
    }


def compute_sigma_fractions(S):
    """Compute sigma fraction for each SOC orbital."""
    row = np.abs(S[:, 0]) ** 2
    total = np.sum(np.abs(S) ** 2, axis=1)
    return row / total


# ============================================================
# 5. Complete pipeline
# ============================================================

def run_analysis(
    soc_coefficients,
    nr_coefficients,
    nr_labels=None,
    active_space=None,
    ao_overlap=None,
    optimal_pairing=True,
):
    """
    Complete pipeline: from coefficient matrices to metrics.
    """
    S = compute_overlap_matrix(soc_coefficients, nr_coefficients, ao_overlap)
    if active_space is not None:
        S_active = S[:, active_space]
    else:
        S_active = S

    diag = check_unitarity(S_active)
    eta, row_ind, col_ind, paired_values = compute_eta(S_active, optimal_pairing)
    I = compute_mutual_information(S_active)
    A, H_list = compute_conditional_entropy(S_active)
    sigma_frac = compute_sigma_fractions(S_active)

    return {
        'S_matrix': S_active,
        'eta': eta,
        'optimal_pairing_used': optimal_pairing,
        'pairing': (row_ind.tolist(), col_ind.tolist()),
        'paired_overlaps': paired_values.tolist(),
        'I_mutual_information': I,
        'A_conditional_entropy': A,
        'H_per_orbital': H_list.tolist(),
        'sigma_fractions': sigma_frac.tolist(),
        'unitarity': diag,
        'nr_labels': nr_labels,
    }


# ============================================================
# 6. Analytical S-matrix builder (testing/validation)
# ============================================================

def build_analytical_S_matrix(theta):
    """
    Build the analytical 3x3 unitary S-matrix from the mixing angle theta.
    """
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [0.0, 0.0, 1.0],
        [c,   s,   0.0],
        [-s,  c,   0.0],
    ])


def theta_from_sigma_fraction(sigma_frac):
    """Invert theta from target sigma fraction."""
    if not 0 <= sigma_frac <= 1:
        raise ValueError("sigma_frac must be in [0, 1]")
    return np.arccos(np.sqrt(sigma_frac))


# ============================================================
# Self-test
# ============================================================

if __name__ == "__main__":
    print("Running self-test...")
    theta = theta_from_sigma_fraction(0.583)
    S = build_analytical_S_matrix(theta)

    eta, _, _, _ = compute_eta(S)
    I = compute_mutual_information(S)
    A, _ = compute_conditional_entropy(S)
    diag = check_unitarity(S)

    print(f"theta = {np.degrees(theta):.2f} deg")
    print(f"eta = {eta:.4f} (expected 0.722)")
    print(f"I = {I:.4f} nats (expected 0.646)")
    print(f"A = {A:.4f} nats (expected 0.453)")
    print(f"Unitary: {diag['is_unitary']}")
    print(f"Sigma fractions: {[f'{s*100:.1f}%' for s in compute_sigma_fractions(S)]}")

    passed = all([
        abs(eta - 0.722) < 0.001,
        abs(I - 0.646) < 0.001,
        abs(A - 0.453) < 0.001,
        diag['is_unitary'],
    ])
    print("Self-test passed." if passed else "Self-test FAILED!")