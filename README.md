# Information Dynamics of Relativistic Bond Reconstruction in CBi⁻

This repository contains the Python code and data for the manuscript:

**Relativistic Bond Reconstruction in Chemical Bonding: An Information Dynamics Perspective on the CBi⁻ Molecular Ion**

## Paper Link

https://doi.org/10.5281/zenodo.21734951

---
---
## Interactive Explainer

**[Open the interactive explainer](Update/interactive.html)** — a visual, jargon-light walkthrough of the paper, written in English for busy scientists and non-chemists.

- 30-second takeaway, plain-language experiment story, and a "real space vs virtual space" analogy
- Two live labs: drag the lambda slider to watch eta / I / A and orbital mixing evolve; drag the sigma1 slider to explore the single-parameter constraint — all numbers computed live in your browser from the paper's equations
- The five numerical validation experiments as expandable cards
- All five paper figures embedded

No build step, no server, no tracking — just open `Update/interactive.html` in any modern browser. (Tip: enable GitHub Pages on this repository to share it as a live link.)

---

---


## Key Conclusions

This work reinterprets the breakthrough Science 2026 experiment on CBi⁻ through the lens of Information Dynamics. By combining high-level relativistic quantum chemistry (Dirac-Coulomb CCSD(T)) with information-theoretic measures, we demonstrate that chemical bonding rules are not absolute but undergo physical restructuring under extreme conditions.

The main findings are:

1. **Quantification of Orbital Mixing**

   We quantitatively characterize the smearing of the classical sigma/pi boundary. The mixing parameter drops from eta ~ 0.99 (in non-relativistic CN⁻) to eta = 0.722 (in relativistic CBi⁻), while the orbital mutual information decreases from I ~ 1.099 nats (NR limit) to I = 0.646 nats. The label ambiguity (conditional entropy) A = 0.453 nats quantifies the loss of sigma/pi predictivity.

2. **Self-Consistent S-Matrix Framework**

   A single-parameter constraint model enforces unitarity and symmetry orthogonality: given sigma_1 = 58.3%, unitarity forces sigma_2 = 41.7% (complementary), and angular momentum projection orthogonality forces sigma_3 = 0.00% (strictly, by symmetry). All derived metrics (eta, I, A) are uniquely determined from this single input.

3. **Continuous Phase Transition in Bonding Rules**

   The restructuring of the virtual space (from the C-infinity-v group to the double group C-infinity-v*) occurs as a continuous crossover driven by spin-orbit coupling (SOC), with the steepest change occurring around lambda ~ 0.55.

4. **Predictive Power**

   The framework extrapolates the homologous series (C-N to C-Bi), predicting eta = 0.722 for CBi⁻ consistent with the experimental assignment.

---

## Repository Contents

### Root Level

| File | Description |
|------|-------------|
| README.md | This file |
| Figure_Generation_Script.py | Original figure generation script (legacy, uses hardcoded values) |
| Figure_P0_spectra_comparison.pdf | Experimental vs. Theoretical photoelectron spectra |
| Figure_P1_orbital_mixing.pdf | Visualization of sigma/pi orbital mixing |
| Figure_P2_phase_transition.pdf | Continuous phase transition of bonding rules vs. lambda |
| Figure_P3_homologous_trends.pdf | Periodic trends and extrapolated predictions |

### Update/ Directory (Self-Consistent Analysis Pipeline)

| File | Description |
|------|-------------|
| main_analysis.py | Main analysis script: implements the Information Dynamics Framework, computes eta, I, A from the S-matrix, performs lambda-scan and homologous series analysis |
| analysis_pipeline.py | Core pipeline module: symmetry-adapted basis construction, overlap matrix computation, S-matrix metrics, unitarity diagnostics |
| figure_generation.py | Updated figure generation using corrected data from JSON (generates 5 figures as PNG) |
| numerical_validation.py | 5 numerical experiments: pairing comparison, noise injection, subspace closure, CN⁻ null test, basis set incompleteness model |
| analysis_results.json | All computed values: S-matrix, eta, I, A, lambda-scan data, homologous series data, limits |
| S_matrix_CBi.csv | Full 3x3 unitary overlap matrix (rows: |omega|=3/2, |omega|=1/2(1), |omega|=1/2(2); cols: sigma, pi_1, pi_2) |
| figures/ | Directory for generated figure outputs |
| interactive.html | Interactive explainer webpage (English, for non-specialists): live lambda/sigma1 sliders, validation cards, embedded figures |

---

## How to Run

### Prerequisites

Ensure you have the required Python libraries installed:

pip install numpy scipy matplotlib

### Running the Self-Consistent Analysis (Recommended)

    cd Update/
    python main_analysis.py

This will:
- Load experimental data from Science 2026
- Build the self-consistent 3x3 unitary S-matrix
- Compute eta, I, A metrics
- Perform lambda-scan and homologous series analysis
- Save results to analysis_results.json and S_matrix_CBi.csv

### Generating Figures (Updated Pipeline)

    cd Update/
    python figure_generation.py

This generates 5 figures as PNG in the figures/ directory:
- Figure1_orbital_mixing.png - Orbital mixing visualization
- Figure2_spectrum_overlay.png - Experimental vs. Theory spectrum
- Figure3_lambda_scan.png - lambda-driven restructuring
- Figure4_homologous_trends.png - Homologous series trends
- Figure5_metric_clarification.png - I vs. A metric clarification

### Running Numerical Validation

    cd Update/
    python numerical_validation.py

This runs 5 experiments characterizing the robustness of the pipeline.

### Legacy Script (Root Level)

    python Figure_Generation_Script.py

Note: This legacy script uses hardcoded values from the original analysis. For corrected values, use the Update/figure_generation.py pipeline instead.

---

## Corrected Values Summary

| Quantity | CN⁻ (NR limit) | CBi⁻ (Relativistic) | Upper Bound |
|----------|---------------|---------------------|-------------|
| eta (mixing parameter) | 0.999 | 0.722 | 1.0 |
| I(omega; NR) (mutual information) | 1.092 | 0.646 nats | ln(3) ~ 1.099 nats |
| A = <H(sigma/pi|omega)> (label ambiguity) | 0.006 | 0.453 nats | ln(2) ~ 0.693 nats |

| Orbital | sigma character | pi character |
|---------|------------|-------------|
| |omega|=3/2 | 0.00% (symmetry forced) | 100.0% |
| |omega|=1/2 (1) | 58.3% | 41.7% |
| |omega|=1/2 (2) | 41.7% | 58.3% |

---

## Notes on Data Consistency

- The sigma fractions sum to exactly 100% (58.3% + 41.7% + 0.00% = 100.0%), satisfying unitarity.
- The |omega|=3/2 orbital has exactly zero sigma character, enforced by angular momentum projection orthogonality in the C-infinity-v* double group.
- The mutual information I and conditional entropy A are distinct quantities: I measures total label predictivity loss (upper bound ln 3), while A measures the average sigma/pi label ambiguity per SOC orbital (upper bound ln 2). The manuscript previously reported A under the label I; both are now reported explicitly.
- The CN⁻ conditional entropy A = 0.006 nats is consistent with the weak-SOC limit (theta = 2 degrees), not the strict NR limit (theta = 0 degrees) where A -> 0.

---

## Citation

Huang, K., Liu, H. & Huang, Z. (2026). Relativistic Bond Reconstruction in Chemical Bonding: An Information Dynamics Perspective on the CBi⁻ Molecular Ion. Zenodo. https://doi.org/10.5281/zenodo.21734951

---

## License

This code is provided for academic reproducibility purposes. Please contact the authors for permissions.
