# Information Dynamics of Relativistic Bond Reconstruction in CBi⁻

This repository contains the official Python code for generating all main figures presented in the manuscript:

**Relativistic Bond Reconstruction in the CBi⁻ Molecular Ion: An Information Dynamics Perspective**  

## 🔗 Paper Link
https://doi.org/10.5281/zenodo.21734952

---

## 📝 Key Conclusions

This work reinterprets the breakthrough *Science* 2026 experiment on CBi⁻ through the lens of Information Dynamics. By combining high-level relativistic quantum chemistry (Dirac-Coulomb CCSD(T)) with information-theoretic measures, we demonstrate that chemical bonding rules are not absolute but undergo physical restructuring under extreme conditions.

The main findings are:

1. **Quantification of Orbital Mixing**  
   We quantitatively characterize the "smearing" of the classical σ/π boundary. The mixing parameter drops from η = 0.99 (in non-relativistic CN⁻) to η = 0.73 (in relativistic CBi⁻), while the orbital mutual information increases sixfold (I = 0.47 nats), providing a rigorous measure of the loss of symmetry distinction.

2. **Perfect Experimental Reproduction**  
   The simulated photoelectron spectra (EOM-IP-CCSD) reproduce the experimental X, A, and B bands with an RMS deviation of only 0.035 eV, including the vibrational progression (FC factors 1.00 : 0.32 : 0.08).

3. **Continuous Phase Transition in Bonding Rules**  
   The "restructuring" of the virtual space (from the C∞v group to the double group C∞v*) occurs as a continuous phase transition driven by spin-orbit coupling (SOC), with a steep transition occurring around λ ≈ 0.55.

4. **Predictive Power**  
   The framework extrapolates the homologous series (C–N to C–Bi), successfully predicting the bonding parameters for as-yet unsynthesized species: CPb⁻ (η ≈ 0.729) and CAt⁻ (η ≈ 0.724).

---

## 📂 Repository Contents

- `generate_figures_v2.2.py` : The master script to generate all publication-ready figures.
- `Figure_P0_spectra_comparison.pdf` : Experimental vs. Theoretical photoelectron spectra.
- `Figure_P1_orbital_mixing.pdf` : Visualization of σ/π orbital mixing.
- `Figure_P2_phase_transition.pdf` : Continuous phase transition of bonding rules vs. λ.
- `Figure_P3_homologous_trends.pdf` : Periodic trends and extrapolated predictions.

---

## 🚀 How to Run

Ensure you have the required Python libraries installed:

```bash
pip install numpy scipy matplotlib
```

Run the figure generation script:

```bash
python generate_figures_v2.2.py
```

The script will automatically generate all PDF figures in the current working directory without any warnings.

---

## ⚖️ License

This code is provided for academic reproducibility purposes. Please contact the authors for permissions.
```
