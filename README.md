# NeuraLSP
**NeuraLSP: An Efficient and Rigorous Neural Left Singular Subspace Preconditioner for Conjugate Gradient Methods**


> **TL;DR:** NeuraLSP is a neural preconditioner for Conjugate Gradient (CG) methods that leverages a learned left singular subspace to accelerate convergence while preserving a rigorous solver framework. (See the paper for theory + guarantees.)

---

## What’s in this repo
This repository contains the reference implementation and experiment scripts for the NeuraLSP paper.

**Main entry points:**
- `main.py`: runs the trained models and outputs the main results present in the tables of the main paper. 
- `train_models.py`: trains NeuraLSP models used in the experiments
- `comparison_test.py`: trains the smaller models and compares the subspace loss and NLSS loss in terms of captured energy
- `rank_sweep.py`: contains code for running the rank sweeps for the main experiments
- `scalability_ablation.py`: scalability/ablation experiment script

Outputs and plots are written to `results/` (either generated or precomputed; see below).

---

## Quickstart (run a sanity check)
### 1) Setup
Please download the repository. 





