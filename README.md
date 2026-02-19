# Eigenspace Confinement Barren Plateaus: Numerical Experiments

Numerical experiments accompanying:

> P. Sarker, "Eigenspace Confinement as a Structural Mechanism for Barren Plateaus," 2026 (submitted).

The code validates the Block-Preservation Theorem and the DESCENT mechanism numerically, reproduces all figures and tables in the paper, and provides a comparative evaluation of standard barren-plateau mitigation strategies.

---

## Overview

The *Eigenspace Confinement Barren Plateau* (ECBP) is a structural training obstruction in variational quantum algorithms that arises when the initial state and target state belong to distinct eigenspaces of the locality operator $L$. Within the centraliser $\mathcal{C}(L)$, the Block-Preservation Theorem (Theorem 6.4) guarantees that no gradient trajectory can transfer amplitude between eigenspaces; the cost landscape is therefore flat by construction rather than by statistical concentration. The DESCENT procedure resolves the obstruction in a single algebraic step by enlarging the ansatz algebra to include a cross-block generator.

All computations use exact linear algebra (NumPy/SciPy) with no shot noise, no stochastic sampling, and no external quantum-computing dependencies.

---

## Requirements

- Python ≥ 3.10
- NumPy ≥ 1.24
- SciPy ≥ 1.10

```bash
pip install -r requirements.txt
```

---

## Scripts

Each script is self-contained and may be executed directly. `proof_of_concept.py` defines shared utilities imported by the other scripts and should be present in the same directory.

### `proof_of_concept.py`

Validates the central claim of the paper. Constructs a confined ansatz, confirms the exact fidelity ceiling $\lceil = \lVert P_+ |\mathrm{target}\rangle \rVert^2$, and demonstrates that a single DESCENT step—adding one cross-block generator to the ansatz algebra—connects the initial and target states. Tested for $n = 2$–7 qubits.

```bash
python proof_of_concept.py
```

### `descent_scaling.py`

Scaling analysis of the DESCENT mechanism across Hilbert-space dimensions $d = 4$–$128$ ($n = 2$–7 qubits). Verifies that the analytic ceiling formula holds exactly at each scale and that gradient variance in the confined direction is identically zero, independently of $n$.

```bash
python descent_scaling.py
```

### `stress_tests.py`

Evaluates ten structured objections to the structural theory: expressivity-based arguments, symmetry alternatives, the ADAPT-VQE comparison, and others. Each experiment reports the outcome and, where applicable, the scope qualifier under which the theory's prediction holds.

```bash
python stress_tests.py
```

### `bridge_experiments.py`

Four experiments at the boundary between ECBP and Statistical Barren Plateaus (SBP): (i) signature-based distinguishability, (ii) analytic resolution of the 60/20/20 measurement discrepancy from the stress tests, (iii) a structural-initialisation bridge experiment in the SBP regime, and (iv) finite-depth approximate-block-structure analysis.

```bash
python bridge_experiments.py
```

### `mitigation_comparison.py`

Comparative evaluation of seven approaches to the GHZ preparation task ($n = 4$ qubits, cross-block target): standard VQA, identity-neighbourhood initialisation (Grant et al. 2021), local cost function (Cerezo et al. 2021), layer-wise training (Skolik et al. 2021), increased circuit depth, extended gradient steps, and DESCENT. All methods operate under identical exact-arithmetic conditions.

```bash
python mitigation_comparison.py
```

---

## Reproducibility

All results are deterministic. No random seeds need to be set manually; scripts that use random initialisations fix the seed internally. Output is printed to stdout.

---

## Citation

```bibtex
@article{sarker2026ecbp,
  author  = {Sarker, P.},
  title   = {Eigenspace Confinement as a Structural Mechanism for Barren Plateaus},
  year    = {2026},
  note    = {submitted}
}
```

---

## Licence

The code is released under the MIT Licence. See `LICENSE` for details.