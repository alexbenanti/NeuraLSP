import time
import numpy as np
import scipy.sparse as sp
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import pyamg

from src.pdes import generate_pde_data, smooth_test_vectors
from src.model import (
    ProlongationMLP2,
    nested_lora_loss,
    subspace_loss,
    error_propagation_loss,
)
from src.multigrid import TwoGridPreconditioner, pcg_solve

# --- Optional GNN Import ---
try:
    from src.gnn_baseline import AMG_GNN
    GNN_AVAILABLE = True
except ImportError:
    GNN_AVAILABLE = False
    print("PyG / GNN Baseline not found. Skipping GNN experiments.")

# ==============================================================================
# GLOBAL CONFIGURATION
# ==============================================================================
N = 64
K_VECTORS = 72

# Train ONCE at this max rank, then test prefixes
RANKS = [2,4,6,8,10,12,14,20,24,26,28,30,32, 36,40,44,48,52,56,60,64,68,72]
RANK_MAX = max(RANKS)

TRAIN_EPOCHS = 1000
TEST_SAMPLES = 100
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pde_type = "diffusion"

SEED = 0
np.random.seed(SEED)
torch.manual_seed(SEED)

# If you want more stable timing on CPU:
# torch.set_num_threads(1)

# ==============================================================================
# FEATURES
# ==============================================================================
def get_features(A_csr, S_np):
    # Using only S as features (as in your current code)
    return torch.FloatTensor(S_np).to(DEVICE)

# ==============================================================================
# BASELINES
# ==============================================================================
def pyamg_inference(A, S):
    """
    PyAMG SA baseline. Returns PyAMG's prolongation P0 (dense float32).
    NOTE: PyAMG chooses coarse dimension automatically (not tied to RANKS).
    """
    B = np.ones((A.shape[0], 1), dtype=np.float64)
    ml = pyamg.smoothed_aggregation_solver(A, B=B, max_levels=2, max_coarse=10)
    P = ml.levels[0].P  # sparse

    if sp.issparse(P):
        P = P.toarray()
    P = np.asarray(P, dtype=np.float32)

    return P

def oracle_svd_Umax(S, rank_max):
    """
    Oracle coarse basis from top left singular vectors of S.
    U is already orthonormal.
    """
    U, _, _ = np.linalg.svd(S, full_matrices=False)
    return U[:, :rank_max].astype(np.float32)

# ==============================================================================
# TRAINING
# ==============================================================================
def train_model(model, optimizer, loss_fn, n_steps=50):
    t_start = time.perf_counter()
    model.train()
    loss_history = []

    print(f"  Training {model.__class__.__name__} for {n_steps} steps...")

    for step in range(n_steps):
        #A_csr, S_np = generate_pde_data(N, k_vectors=K_VECTORS, pde_type=pde_type)

        A_csr = generate_pde_data(N, pde_type=pde_type)

        S_np = smooth_test_vectors(A_csr, num_vectors = K_VECTORS)

        # Randomly permute columns of S so ordering doesn't leak
        perm = np.random.permutation(S_np.shape[1])
        S_np = S_np[:, perm]

        x = get_features(A_csr, S_np).unsqueeze(0)  # (1,n,K)
        S_target = torch.FloatTensor(S_np).unsqueeze(0).to(DEVICE)

        # Forward (handle GNN separately)
        if GNN_AVAILABLE and isinstance(model, torch.nn.Module) and "AMG_GNN" in str(type(model)):
            coo = A_csr.tocoo()
            edge_index = torch.tensor(
                np.vstack((coo.row, coo.col)), dtype=torch.long, device=DEVICE
            )
            edge_attr = torch.tensor(coo.data, dtype=torch.float, device=DEVICE).unsqueeze(1)

            x_flat = x.squeeze(0)  # (n,K)
            Q = model(x_flat, edge_index, edge_attr).unsqueeze(0)  # (1,n,r)

            
            if loss_fn.__name__ != "error_propagation_loss":
                Q, _ = torch.linalg.qr(Q)
        else:
            Q = model(x)  

        if loss_fn.__name__ == "error_propagation_loss":
            loss = loss_fn(Q, A_csr)
        else:
            loss = loss_fn(Q, S_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_history.append(float(loss.item()))

        if step == 0 or (step + 1) % 100 == 0:
            print(f"    Step {step+1}/{n_steps} | Loss: {float(loss.item()):.6f}")

    return loss_history, time.perf_counter() - t_start

# ==============================================================================
# INFERENCE HELPERS (compute P_max ONCE, then slice)
# ==============================================================================
def infer_Pmax_mlp(model, A, S):
    model.eval()
    with torch.no_grad():
        x = get_features(A, S).unsqueeze(0)      # (1,n,K)
        Q = model(x).squeeze(0).cpu().numpy()    # (n,RANK_MAX), already orthonormal for ProlongationMLP2
    return Q.astype(np.float32)

def infer_Pmax_gnn(model, A, S, do_qr=False):
    """
    If do_qr=True, we QR the output to make prefix-truncation meaningful.
    If do_qr=False, matches your earlier baseline behavior when using error_propagation_loss.
    """
    model.eval()
    with torch.no_grad():
        x = get_features(A, S)  # (n,K)
        coo = A.tocoo()
        edge_index = torch.tensor(
            np.vstack((coo.row, coo.col)), dtype=torch.long, device=DEVICE
        )
        edge_attr = torch.tensor(coo.data, dtype=torch.float, device=DEVICE).unsqueeze(1)

        Y = model(x, edge_index, edge_attr)  # (n,RANK_MAX)
        if do_qr:
            Y, _ = torch.linalg.qr(Y)
        return Y.cpu().numpy().astype(np.float32)

# ==============================================================================
# SOLVE ONE INSTANCE WITH GIVEN P
# ==============================================================================
def solve_with_P(A, b, P):
    t0 = time.perf_counter()
    M = TwoGridPreconditioner(A, P)
    t_setup = time.perf_counter() - t0

    x_sol, hist, t_hist_solve = pcg_solve(A, b, M, tol=1e-6)
    t_solve = t_hist_solve[-1]
    iters = len(hist) - 1
    return t_setup, t_solve, iters

# ==============================================================================
# RANK SWEEP BENCHMARK
# ==============================================================================
def run_rank_sweep(models, training_times, ranks, num_samples=100, pde_type=pde_type):
    """
    models: dict of base models (trained at RANK_MAX)
    training_times: dict base_name -> seconds
    ranks: list of prefix ranks to test
    """
    print("\n========================================================")
    print(f"STARTING RANK-SWEEP BENCHMARK ({num_samples} samples)")
    print(f"PDE Type: {pde_type}")
    print(f"Train rank: {RANK_MAX}  |  Test ranks: {ranks}")
    print("========================================================")

    # Build list of result-rows keys
    method_keys = []
    for base_name in models.keys():
        for r in ranks:
            method_keys.append(f"{base_name}_r={r}")
    for r in ranks:
        method_keys.append(f"Oracle_SVD_r={r}")
    method_keys.append("PyAMG_SA")  # auto rank

    stats = {k: {"infer": [], "setup": [], "solve": [], "total": [], "iter": []} for k in method_keys}

    pyamg_Pcols = []

    for i in range(num_samples):
        if i % 10 == 0:
            print(f"Processing sample {i}/{num_samples}...")

        A = generate_pde_data(N, pde_type=pde_type)

        
        b = np.random.randn(A.shape[0]).astype(np.float32)

        t0 = time.perf_counter()
        S = smooth_test_vectors(A, num_vectors = K_VECTORS)
        t_smooth_vectors = time.perf_counter() - t0

        # -----------------------
        # Oracle SVD (compute once)
        # -----------------------
        
        t0 = time.perf_counter()
        Umax = oracle_svd_Umax(S, RANK_MAX)
        t_svd = time.perf_counter() - t0 

        # -----------------------
        # PyAMG (compute once)
        # -----------------------
        t0 = time.perf_counter()
        P_pyamg = pyamg_inference(A, S)
        t_pyamg = time.perf_counter() - t0
        pyamg_Pcols.append(P_pyamg.shape[1])

        t_set, t_sol, iters = solve_with_P(A, b, P_pyamg)
        stats["PyAMG_SA"]["infer"].append(t_pyamg)
        stats["PyAMG_SA"]["setup"].append(t_set)
        stats["PyAMG_SA"]["solve"].append(t_sol)
        stats["PyAMG_SA"]["total"].append(t_pyamg + t_set + t_sol)
        stats["PyAMG_SA"]["iter"].append(iters)

        # -----------------------
        # Neural models (compute Pmax once per model)
        # -----------------------
        Pmax_cache = {}     # base_name -> (Pmax, t_inf)

        for base_name, cfg in models.items():
            model = cfg["model"]
            kind = cfg.get("kind", "mlp")

            t0 = time.perf_counter()
            if kind == "gnn":
                Pmax = infer_Pmax_gnn(model, A, S, do_qr=cfg.get("gnn_do_qr", False))
            else:
                Pmax = infer_Pmax_mlp(model, A, S)
            t_inf = time.perf_counter() - t0

            Pmax_cache[base_name] = (Pmax, t_inf)

        # -----------------------
        # Evaluate prefix ranks
        # -----------------------
        for r in ranks:
            # Oracle SVD prefix
            P = Umax[:, :r]
            t_set, t_sol, iters = solve_with_P(A, b, P)
            key = f"Oracle_SVD_r={r}"
            stats[key]["infer"].append(t_svd) 
            stats[key]["setup"].append(t_set)
            stats[key]["solve"].append(t_sol)
            stats[key]["total"].append(t_svd + t_set + t_sol + t_smooth_vectors)
            stats[key]["iter"].append(iters)

            # Each neural model prefix
            for base_name in models.keys():
                Pmax, t_inf = Pmax_cache[base_name]
                P = Pmax[:, :r]
                t_set, t_sol, iters = solve_with_P(A, b, P)

                key = f"{base_name}_r={r}"
                stats[key]["infer"].append(t_inf)             # NOTE: same t_inf for all r (Pmax computed once)
                stats[key]["setup"].append(t_set)
                stats[key]["solve"].append(t_sol)
                stats[key]["total"].append(t_inf + t_set + t_sol + t_smooth_vectors)
                stats[key]["iter"].append(iters)

    # -----------------------
    # Summarize
    # -----------------------
    rows = []
    for name, m in stats.items():
        if "_r=" in name:
            base, r_str = name.split("_r=")
            rank_val = int(r_str)
            train_time = training_times.get(base, 0.0)
        else:
            base = name
            rank_val = "auto" if name == "PyAMG_SA" else None
            train_time = training_times.get(base, 0.0)

        rows.append({
            "Method": base,
            "Rank": rank_val,
            "Train Time (s)": train_time,
            "Inference (ms)": 1000 * float(np.median(m["infer"])),
            "Setup (ms)": 1000 * float(np.median(m["setup"])),
            "Solve (ms)": 1000 * float(np.median(m["solve"])),
            "Total (ms)": 1000 * float(np.median(m["total"])),
            "25th Percentile (ms)": 1000 * float(np.percentile(m["total"], 25)),
            "75th Percentile (ms)": 1000 * float(np.percentile(m["total"], 75)),
            "Median Iterations": float(np.median(m["iter"])),
        })

    df = pd.DataFrame(rows)

    # Helpful debug: PyAMG coarse dimension
    print(f"\n[PyAMG] P0 columns: median={int(np.median(pyamg_Pcols))}, "
          f"mean={np.mean(pyamg_Pcols):.1f}, min={min(pyamg_Pcols)}, max={max(pyamg_Pcols)}")

    return df

# ==============================================================================
# MAIN
# ==============================================================================
if __name__ == "__main__":
    input_dim = K_VECTORS
    n_nodes = (N + 1) ** 2

    # -----------------------
    # Define models (ALL at RANK_MAX)
    # -----------------------
    models = {}

    mlp_nested = ProlongationMLP2(input_dim, 128, 256, RANK_MAX, n_nodes, RANK_MAX).to(DEVICE)
    models["MLP_Nested"] = {
        "model": mlp_nested,
        "loss": nested_lora_loss,
        "opt": optim.Adam(mlp_nested.parameters(), lr=LR),
        "kind": "mlp",
    }

    
    mlp_unnested = ProlongationMLP2(input_dim, 128, 256, RANK_MAX, n_nodes, RANK_MAX).to(DEVICE)
    models["MLP_Unnested"] = {
        "model": mlp_unnested,
        "loss": subspace_loss,
        "opt": optim.Adam(mlp_unnested.parameters(), lr=LR),
        "kind": "mlp",
    }

    if GNN_AVAILABLE:
        gnn = AMG_GNN(input_node_dim = input_dim, output_dim = RANK_MAX, hidden_dim =488, num_layers=5).to(DEVICE)
        models["GNN"] = {
            "model": gnn,
            "loss": error_propagation_loss,
            "opt": optim.Adam(gnn.parameters(), lr=1e-4),
            "kind": "gnn",
            "gnn_do_qr": False,
        }

# -----------------------
# Train ONCE
# -----------------------
    training_times = {}
    print(f"Starting Training on {DEVICE} (rank_max={RANK_MAX})...")
    for name, cfg in models.items():
        model_path = f"{name}_weights.pth"
        try:
            # Check if the model weights already exist
            cfg["model"].load_state_dict(torch.load(model_path))
            cfg["model"].eval()
            print(f"  -> {name} weights loaded from {model_path}")
        except FileNotFoundError:
            # Train the model if weights are not found
            _, t_train = train_model(cfg["model"], cfg["opt"], cfg["loss"], n_steps=TRAIN_EPOCHS)
            training_times[name] = t_train
            print(f"  -> {name} train time: {t_train:.2f}s")
            # Save the trained weights
            torch.save(cfg["model"].state_dict(), model_path)
            print(f"  -> {name} weights saved to {model_path}")

    # -----------------------
    # Rank sweep benchmark
    # -----------------------
    df = run_rank_sweep(
        models=models,
        training_times=training_times,
        ranks=RANKS,
        num_samples=TEST_SAMPLES,
        pde_type=pde_type,
    )

    
    # Put Rank-swept rows first, then auto baselines at bottom
    df_ranked = df.copy()
    df_ranked["Rank_sort"] = df_ranked["Rank"].apply(lambda x: 10**9 if x == "auto" else int(x))
    df_ranked = df_ranked.sort_values(["Rank_sort", "Method"]).drop(columns=["Rank_sort"])

    print("\n=== RANK SWEEP RESULTS ===")
    print(df_ranked.to_string(index=False))
    df_ranked.to_csv(f"PERCENTILE_rank_sweep_{pde_type}_N{N}_K{K_VECTORS}_Rmax{RANK_MAX}.csv", index=False)

    # Optional: quick plot of iterations vs rank for Nested vs Unnested vs Oracle
    try:
        df_plot = df_ranked[df_ranked["Rank"] != "auto"].copy()
        df_plot["Rank"] = df_plot["Rank"].astype(int)

        plt.figure(figsize=(9, 5))
        for method in ["MLP_Nested", "MLP_Unnested", "Oracle_SVD"]:
            sub = df_plot[df_plot["Method"] == method]
            if len(sub) > 0:
                plt.plot(sub["Rank"], sub["Median Iterations"], marker="o", label=method)
        plt.xlabel("Prefix rank r")
        plt.ylabel("Median PCG iterations")
        plt.title(f"Prefix-rank sweep | {pde_type} | N={N} | K={K_VECTORS}")
        plt.grid(True, ls="--", alpha=0.4)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"rank_sweep_iters_{pde_type}.png", dpi=150)
        print(f"Plot saved to rank_sweep_iters_{pde_type}.png")
    except Exception as e:
        print(f"[Plot skipped] {e}")
