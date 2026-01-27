# NeuraLSP (Anonymous ICML Submission)

**Anonymous repository for double-blind review.**
This repo will be de-anonymized upon acceptance.

## Overview
NeuraLSP is a neural preconditioning approach for accelerating Conjugate Gradient (CG) by learning a left singular subspace surrogate used within a rigorous solver framework.

## Getting the code
- Download: use the “Download ZIP” button on the anonymous repository page and unzip locally.
- (Optional) Clone: `git clone <anonymous_repo_url>`

## Installation
We recommend creating a fresh environment.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip

## Dependencies
Install PyTorch then:
'''bash
pip install -r requirements.txt
bash'''

NOTE: torch-scatter may require a wheel matching your PyTorch/CUDA version. Install it only if the code imports it (need to comment the line out in requirements.txt)







