"""
Chintalpudi FINAL — 10-chain parallel multichain run.

Runs 10 independent MCMC chains with the SAME inversion settings as
run_chintalpudi_FINAL.py (so their posteriors are directly comparable),
varying ONLY:
  - the initial depth field (a flat field at a randomly chosen value)
  - the random seed (so each chain rolls its own dice)

The 10 starting depths are stratified random draws across the full
depth prior [0, 5000] m — one draw per 500 m bin — so every major
"starting region" of the prior is covered (very shallow ... very deep).
This is the cleanest way to test whether the original FINAL run was
trapped in a local minimum: if all 10 chains converge to the same
posterior despite starting anywhere from ~440 m to ~4955 m, there is
no local-minima problem.

Parallel execution: uses Python multiprocessing with up to 8 worker
processes (matches the user's 8-core CPU). With 10 chains and 8 cores,
expect ~1.25× the wall-time of a single chain, i.e. roughly 5 hours
instead of the ~41 hours that running them sequentially would take.

Each chain saves to:
    results/exp_chintalpudi_FINAL_multichain/chain_NN.npz

Schema of each chain_NN.npz matches results_data.npz from the original
FINAL run (mean_depths, std_depths, ci_5, ci_95, posterior_samples_thinned,
all_misfits, etc.) plus chain_id, seed, and init_depth for traceability.

Usage (Windows, in the dissertation_research directory):
    python run_chintalpudi_FINAL_multichain.py
"""
import os
import sys
import time
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.mcmc_inversion import run_mcmc_3d, process_chain_3d

# ======================================================================
# CONFIG  (identical to run_chintalpudi_FINAL.py except where noted)
# ======================================================================
DATA_DIR = 'real_data/chintalpudi'
OUT_DIR  = 'results/exp_chintalpudi_FINAL_multichain'

# Grid + MCMC — identical to FINAL
NX, NY            = 20, 20
N_ITERATIONS      = 100_000
BURN_IN_FRAC      = 0.5
POSTERIOR_THIN    = 50
STEP_DEPTH        = 300.0
STRIDE            = 2

# Density (Rao parabolic, alpha fixed at literature value)
DRHO_0            = -550.0
ALPHA             = 2000.0

# Likelihood / prior
NOISE_STD         = 1.5
SMOOTHNESS_WEIGHT = 1e-5
DEPTH_MIN         = 0.0
DEPTH_MAX         = 5000.0
N_SUBLAYERS       = 10

# ----------------- The 10 chains: stratified across the prior -----------
# One random draw per 500 m stratum from [0, 5000]. These specific values
# were drawn once at script-creation time and frozen here for full
# reproducibility. Every major depth regime is represented exactly once.
INIT_DEPTHS = [
    440.3,   # 0-500 m       (very shallow)
    808.2,   # 500-1000 m    (very shallow)
    1079.1,  # 1000-1500 m   (shallow)
    1950.3,  # 1500-2000 m   (shallow)
    2371.4,  # 2000-2500 m   (mid)
    2685.4,  # 2500-3000 m   (mid)
    3177.4,  # 3000-3500 m   (deep)
    3622.6,  # 3500-4000 m   (deep)
    4026.8,  # 4000-4500 m   (very deep)
    4955.2,  # 4500-5000 m   (very deep)
]
# Each chain gets its own MCMC seed so proposal sequences are independent
SEEDS = [101, 202, 303, 404, 505, 606, 707, 808, 909, 1010]

# Parallel workers — 8-core CPU
N_WORKERS = 8


def parabolic_density(z):
    """Rao parabolic compaction law with fixed alpha."""
    z = np.asarray(z, dtype=float)
    return DRHO_0 * (ALPHA / (ALPHA + z)) ** 2


class _ChainPrefixStdout:
    """Wraps sys.stdout so every line is prefixed with the chain ID.
    Each worker process installs its own instance."""
    def __init__(self, real_stdout, chain_id):
        self._real = real_stdout
        self._prefix = f'[chain {chain_id:02d}] '
        self._at_line_start = True

    def write(self, s):
        if not s:
            return 0
        out_parts = []
        for ch in s:
            if self._at_line_start and ch != '\n':
                out_parts.append(self._prefix)
                self._at_line_start = False
            out_parts.append(ch)
            if ch == '\n':
                self._at_line_start = True
        return self._real.write(''.join(out_parts))

    def flush(self):
        self._real.flush()


# ======================================================================
# WORKER — runs ONE chain end-to-end and writes its npz
# ======================================================================
def run_one_chain(args):
    chain_id, init_depth, seed, payload = args
    # Redirect this worker's stdout so every print is auto-prefixed
    sys.stdout = _ChainPrefixStdout(sys.__stdout__, chain_id)
    obs_x         = payload['obs_x']
    obs_y         = payload['obs_y']
    gravity_obs   = payload['gravity_obs']
    block_x_edges = payload['block_x_edges']
    block_y_edges = payload['block_y_edges']

    initial_depths = np.full((NX, NY), float(init_depth))

    print(f"START  init={init_depth:.0f} m  seed={seed}  iter={N_ITERATIONS:,}",
          flush=True)

    t0 = time.time()
    result = run_mcmc_3d(
        obs_x=obs_x, obs_y=obs_y, gravity_obs=gravity_obs,
        block_x_edges=block_x_edges, block_y_edges=block_y_edges,
        density_func=parabolic_density,
        noise_std=NOISE_STD,
        n_iterations=N_ITERATIONS,
        step_size=STEP_DEPTH,
        depth_min=DEPTH_MIN, depth_max=DEPTH_MAX,
        smoothness_weight=SMOOTHNESS_WEIGHT,
        n_sublayers=N_SUBLAYERS,
        initial_depths=initial_depths, seed=seed,
        verbose=True,        # auto-prefixed with [chain NN] by stdout wrapper
    )
    elapsed_min = (time.time() - t0) / 60.0

    post = process_chain_3d(result, burn_in_frac=BURN_IN_FRAC, thin=1)
    samples_thinned = post['samples'][::POSTERIOR_THIN].astype(np.float32)

    out_path = os.path.join(OUT_DIR, f'chain_{chain_id:02d}.npz')
    np.savez_compressed(
        out_path,
        # ---- per-chain identity ----
        chain_id=chain_id,
        seed=seed,
        init_depth=float(init_depth),
        # ---- posterior summaries ----
        mean_depths=post['mean'],
        std_depths=post['std'],
        ci_5=post['ci_5'], ci_95=post['ci_95'],
        ci_2_5=post['ci_2_5'], ci_97_5=post['ci_97_5'],
        posterior_samples_thinned=samples_thinned,
        posterior_thin=POSTERIOR_THIN,
        burn_in_frac=BURN_IN_FRAC,
        # ---- diagnostics ----
        all_misfits=np.asarray(result['all_misfits']),
        acceptance_rate=result['acceptance_rate'],
        n_iterations=N_ITERATIONS,
        runtime_min=elapsed_min,
        # ---- shared config (handy for downstream analysis) ----
        block_x_edges=block_x_edges, block_y_edges=block_y_edges,
        obs_x=obs_x, obs_y=obs_y, obs_gravity=gravity_obs,
        drho_0=DRHO_0, alpha=ALPHA, density_law='rao_parabolic',
        noise_std=NOISE_STD, step_depth=STEP_DEPTH,
        smoothness_weight=SMOOTHNESS_WEIGHT, n_sublayers=N_SUBLAYERS,
        depth_min=DEPTH_MIN, depth_max=DEPTH_MAX,
        stride=STRIDE, grid_shape=np.asarray([NX, NY]),
        experiment='chintalpudi_FINAL_multichain',
    )
    return chain_id, init_depth, seed, elapsed_min, result['acceptance_rate'], out_path


# ======================================================================
# MAIN
# ======================================================================
def main():
    print("=" * 78)
    print("CHINTALPUDI FINAL — 10-CHAIN PARALLEL RUN")
    print(f"  Chains:     {len(INIT_DEPTHS)}")
    print(f"  Workers:    {N_WORKERS}  (your CPU)")
    print(f"  Iterations: {N_ITERATIONS:,} per chain")
    print(f"  Output:     {OUT_DIR}/")
    print("=" * 78)

    # ---- load data once, share with all workers ----
    xg = np.loadtxt(os.path.join(DATA_DIR, 'x_meshgrid.txt'))
    yg = np.loadtxt(os.path.join(DATA_DIR, 'y_meshgrid.txt'))
    gv = np.loadtxt(os.path.join(DATA_DIR, 'observed_gravity.txt'))

    obs_x       = xg[::STRIDE, ::STRIDE].flatten()
    obs_y       = yg[::STRIDE, ::STRIDE].flatten()
    gravity_obs = gv[::STRIDE, ::STRIDE].flatten()

    block_x_edges = np.linspace(obs_x.min(), obs_x.max(), NX + 1)
    block_y_edges = np.linspace(obs_y.min(), obs_y.max(), NY + 1)

    print(f"  Stations:   {len(obs_x)}   (stride={STRIDE})")
    print(f"  Grid:       {NX} x {NY} = {NX*NY} unknowns")
    print(f"  Density:    Rao parabolic  Δρ₀={DRHO_0:.0f}  α={ALPHA:.0f} m  (FIXED)")
    print()
    print("  Chain config:")
    print("    chain | init_depth (m) | seed")
    print("    ------+----------------+------")
    for k, (d, s) in enumerate(zip(INIT_DEPTHS, SEEDS)):
        print(f"     {k:4d} | {d:13.1f}  | {s:4d}")
    print()

    os.makedirs(OUT_DIR, exist_ok=True)

    payload = {
        'obs_x': obs_x, 'obs_y': obs_y, 'gravity_obs': gravity_obs,
        'block_x_edges': block_x_edges, 'block_y_edges': block_y_edges,
    }
    work = [(k, INIT_DEPTHS[k], SEEDS[k], payload) for k in range(len(INIT_DEPTHS))]

    print(f"Launching {len(work)} chains across {N_WORKERS} workers ...")
    t_start = time.time()
    with Pool(processes=N_WORKERS) as pool:
        for chain_id, init_d, seed, mins, acc, path in pool.imap_unordered(
                run_one_chain, work):
            print(f"  ✓ chain {chain_id:02d}  init={init_d:6.0f} m  seed={seed:>4d}  "
                  f"acceptance={acc*100:5.1f}%  runtime={mins:6.1f} min  -> {os.path.basename(path)}",
                  flush=True)
    total_min = (time.time() - t_start) / 60.0

    print()
    print("=" * 78)
    print(f"ALL 10 CHAINS DONE in {total_min:.1f} min wall-clock "
          f"({total_min/60:.2f} h)")
    print(f"Results saved to: {OUT_DIR}/chain_00.npz ... chain_09.npz")
    print("=" * 78)


if __name__ == '__main__':
    main()
