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
```

## Dependencies
Install PyTorch then:
```bash
pip install -r requirements.txt
```
NOTE: torch-scatter may require a wheel matching your PyTorch/CUDA version. Install it only if the code imports it (need to comment the line out in requirements.txt)

## Training 
To train the models for all PDEs run:
```bash
python train_models.py
```
This may take a while, but it only needs to be done once, as all models are saved via checkpoints after training for each PDE

## Perform Main Experiments
After the models are trained, you can perform the experiments done in the main body of the paper related to solve time by running main.py
```bash
python main.py
```
## Perform Captured-Energy Comparison Experiments 
To reproduce the results we presented for comparing captured energy of subspace loss vs. NLSS loss, please run the following code: 
```bash
python comparison_test.py
```

## Scalability Ablation
Finally, to run the scalability ablation, please run the following:
```bash
python scalability_ablation.py
```

## Running on Smaller Scale Problems 
The code is default to run on thr $N=64$ problem with $K=72$. This can be changed to whatever size problem you like; however, you must change the value of ```bash N``` and ```bash K_VECTORS``` in 









