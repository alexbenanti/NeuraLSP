# scripts/main.py
import numpy as np
import torch
import torch.optim as optim  # only needed if you want to restore optimizer; otherwise skip

from src.ckpt_utils import normalize_pde_types, get_ckpt_path, load_ckpt
from train_models import build_models
from rank_sweep import run_rank_sweep

# import your benchmark helpers from wherever you keep them
# e.g. if you move your big file to src/experiment_lib.py:
# from src.experiment_lib import build_models, run_rank_sweep, ...

# For now, assume you keep build_models + run_rank_sweep accessible.

CKPT_ROOT = "checkpoints"

N = 64
K_VECTORS = 72
RANKS = [2,4,6,8,10,12,14,20,24,26,28,30,32, 36,40,44,48,52,56,60,64,68,72]
RANK_MAX = max(RANKS)
TEST_SAMPLES = 100

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 0

PDE_TYPES_RAW = ["diffusion", "anisotropic", "screened_poisson"]
PDE_TYPES = normalize_pde_types(PDE_TYPES_RAW, dedupe=True)

np.random.seed(SEED)
torch.manual_seed(SEED)

if __name__ == "__main__":
    for pde_type in PDE_TYPES:
        print(f"\n==============================")
        print(f"BENCHMARK for PDE: {pde_type}")
        print(f"==============================")

        models = build_models()  # your same builder
        training_times = {}

        # Load all models for this PDE
        for name, cfg in models.items():
            ckpt_path = get_ckpt_path(
                root=CKPT_ROOT,
                model_name=name,
                pde_key=pde_type,
                N=N, K=K_VECTORS, R=RANK_MAX, seed=SEED,
            )
            meta = load_ckpt(ckpt_path, cfg["model"], optimizer=None, device=DEVICE)
            cfg["model"].eval()
            training_times[name] = float(meta.get("train_time_s", 0.0))
            print(f"  -> loaded {name} from {ckpt_path}")

        # Now run your sweep
        df = run_rank_sweep(
            models=models,
            training_times=training_times,
            ranks=RANKS,
            num_samples=TEST_SAMPLES,
            pde_type=pde_type,
        )

        df.to_csv(f"PERCENTILE_rank_sweep_{pde_type}_N{N}_K{K_VECTORS}_Rmax{RANK_MAX}.csv", index=False)
        print(f"Saved CSV for {pde_type}")