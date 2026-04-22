"""
Chintalpudi — JOINT depth + constant-Δρ MCMC (QUICK TEST)
==========================================================
Tests the new run_mcmc_3d_joint_drho function on Chintalpudi data:
  - Density model: Δρ(z) = constant  (no λ, no compaction)
  - Δρ jointly estimated with all 100 block depths
  - No borehole / no priors beyond uniform bounds

Quick config (so it finishes in ~5–10 min on a laptop):
  - 10×10 grid
  - 5,000 iterations
  - STRIDE=5 → 96 stations
  - prob_perturb_drho = 0.10

Goal: confirm that a constant Δρ jointly inverted with depths produces
a depocenter close to ONGC borehole truth (2935 m). If it works, scale
up to 20k iter for the production run.
"""
import sys, os, time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.mcmc_inversion import run_mcmc_3d_joint_drho, process_chain_3d_joint_drho

# ============================================================
DATA_DIR = 'real_data/chintalpudi'
OUT_DIR  = 'results/exp_chintalpudi_drho_quicktest'

NX, NY            = 10, 10
N_ITERATIONS      = 5_000
STRIDE            = 5                    # ~96 stations

# Constant Δρ joint inversion (NO lambda, NO depth-decay)
DRHO_INIT         = -200.0               # mid-prior start
DRHO_MIN          = -500.0               # geologically plausible bounds
DRHO_MAX          = -50.0
STEP_DRHO         = 8.0                  # ~3% of mid-prior
PROB_PERTURB_DRHO = 0.10

STEP_DEPTH        = 250.0
NOISE_STD         = 3.0
SMOOTHNESS_WEIGHT = 1e-6                 # very weak (don't flatten depocenter)
N_SUBLAYERS       = 5                    # constant density => fewer sublayers needed
DEPTH_MIN, DEPTH_MAX = 0.0, 5500.0
BURN_IN_FRAC      = 0.5

ONGC_DEPTH = 2935.0


def main():
    print("=" * 72)
    print(f"CHINTALPUDI QUICK TEST — depth + constant Δρ joint MCMC")
    print(f"  {NX}×{NY} grid | {N_ITERATIONS:,} iter | STRIDE={STRIDE}")
    print("=" * 72)

    # Load
    xg = np.loadtxt(os.path.join(DATA_DIR, 'x_meshgrid.txt'))
    yg = np.loadtxt(os.path.join(DATA_DIR, 'y_meshgrid.txt'))
    gv = np.loadtxt(os.path.join(DATA_DIR, 'observed_gravity.txt'))
    bd = np.loadtxt(os.path.join(DATA_DIR, 'basement_depth.txt'))

    obs_x = xg[::STRIDE, ::STRIDE].flatten()
    obs_y = yg[::STRIDE, ::STRIDE].flatten()
    gravity_obs = gv[::STRIDE, ::STRIDE].flatten()
    print(f"  Stations: {len(obs_x)}  | Unknowns: {NX*NY} depths + 1 Δρ")
    print(f"  Data/param: {len(obs_x)/(NX*NY+1):.2f}")
    print(f"  Gravity range: {gravity_obs.min():.1f} to {gravity_obs.max():.1f} mGal")

    block_x_edges = np.linspace(obs_x.min(), obs_x.max(), NX + 1)
    block_y_edges = np.linspace(obs_y.min(), obs_y.max(), NY + 1)

    # Rebin Chak 2007 reference for benchmark
    ref_x = np.loadtxt(os.path.join(DATA_DIR, 'x_coords.txt'))
    ref_y = np.loadtxt(os.path.join(DATA_DIR, 'y_coords.txt'))
    bd_cells = 0.25 * (bd[:-1, :-1] + bd[1:, :-1] + bd[:-1, 1:] + bd[1:, 1:])
    ref_xc = 0.5 * (ref_x[:-1] + ref_x[1:])
    ref_yc = 0.5 * (ref_y[:-1] + ref_y[1:])
    ix_of = np.clip(np.digitize(ref_xc, block_x_edges) - 1, 0, NX - 1)
    iy_of = np.clip(np.digitize(ref_yc, block_y_edges) - 1, 0, NY - 1)
    ref_blocks = np.zeros((NX, NY))
    counts = np.zeros((NX, NY), dtype=int)
    for jj, iy in enumerate(iy_of):
        for ii, ix in enumerate(ix_of):
            ref_blocks[ix, iy] += bd_cells[jj, ii]
            counts[ix, iy] += 1
    ref_blocks = np.where(counts > 0, ref_blocks / np.maximum(counts, 1), np.nan)
    if np.isnan(ref_blocks).any():
        from scipy.ndimage import distance_transform_edt
        idx = distance_transform_edt(np.isnan(ref_blocks),
                                     return_distances=False, return_indices=True)
        ref_blocks = ref_blocks[tuple(idx)]
    ib, jb = [int(k) for k in np.unravel_index(np.argmax(ref_blocks), ref_blocks.shape)]

    # MCMC
    initial_depths = np.full((NX, NY), 1500.0)
    print(f"\nMCMC: Δρ joint, prior [{DRHO_MIN}, {DRHO_MAX}], init {DRHO_INIT}")
    print(f"      σ={NOISE_STD} mGal, smooth={SMOOTHNESS_WEIGHT:g}, "
          f"prob_Δρ={PROB_PERTURB_DRHO}")
    t0 = time.time()
    result = run_mcmc_3d_joint_drho(
        obs_x=obs_x, obs_y=obs_y, gravity_obs=gravity_obs,
        block_x_edges=block_x_edges, block_y_edges=block_y_edges,
        noise_std=NOISE_STD,
        n_iterations=N_ITERATIONS,
        step_depth=STEP_DEPTH, step_drho=STEP_DRHO,
        depth_min=DEPTH_MIN, depth_max=DEPTH_MAX,
        drho_min=DRHO_MIN, drho_max=DRHO_MAX, drho_init=DRHO_INIT,
        prob_perturb_drho=PROB_PERTURB_DRHO,
        smoothness_weight=SMOOTHNESS_WEIGHT,
        n_sublayers=N_SUBLAYERS,
        initial_depths=initial_depths,
        seed=42, verbose=True,
    )
    elapsed = time.time() - t0
    print(f"\nRuntime: {elapsed/60:.1f} min")

    post = process_chain_3d_joint_drho(result, burn_in_frac=BURN_IN_FRAC)
    mean_d, std_d = post['mean'], post['std']
    ci_lo, ci_hi   = post['ci_5'],  post['ci_95']
    drho_mean = post['drho_mean']
    drho_std  = post['drho_std']
    drho_ci   = (post['drho_ci_5'], post['drho_ci_95'])

    rms_ref  = float(np.sqrt(np.mean((mean_d - ref_blocks)**2)))
    bias_ref = float(np.mean(mean_d - ref_blocks))
    cov90 = float(np.mean((ref_blocks >= ci_lo) & (ref_blocks <= ci_hi)))

    bore_recov = float(mean_d[ib, jb])
    bore_std   = float(std_d[ib, jb])
    err_pct    = 100.0 * (bore_recov - ONGC_DEPTH) / ONGC_DEPTH
    in_ci      = ci_lo[ib, jb] <= ONGC_DEPTH <= ci_hi[ib, jb]

    print(f"\n=== RESULTS ===")
    print(f"  Δρ posterior:   {drho_mean:.1f} ± {drho_std:.1f} kg/m³  "
          f"(90% CI [{drho_ci[0]:.1f}, {drho_ci[1]:.1f}])")
    print(f"  Recovered:      {mean_d.min():.0f}–{mean_d.max():.0f} m  "
          f"(Chak benchmark {ref_blocks.min():.0f}–{ref_blocks.max():.0f})")
    print(f"  RMS vs Chak:    {rms_ref:.0f} m  | Bias: {bias_ref:+.0f} m")
    print(f"  90% coverage:   {cov90*100:.0f}%")
    print(f"  Mean σ:         {std_d.mean():.0f} m")
    print(f"\n=== ONGC borehole (real ground truth) ===")
    print(f"  Depocenter block ({ib},{jb})")
    print(f"  Recovered: {bore_recov:.0f} ± {bore_std:.0f} m  vs ONGC {ONGC_DEPTH:.0f} m")
    print(f"  Error: {err_pct:+.1f}%  | ONGC in 90% CI: "
          f"{'YES' if in_ci else 'NO'}  ({ci_lo[ib,jb]:.0f}–{ci_hi[ib,jb]:.0f} m)")

    # Save
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez_compressed(
        os.path.join(OUT_DIR, 'results_data.npz'),
        mean_depths=mean_d, std_depths=std_d,
        ci_5=ci_lo, ci_95=ci_hi,
        drho_mean=drho_mean, drho_std=drho_std,
        drho_ci_5=drho_ci[0], drho_ci_95=drho_ci[1],
        drho_chain=np.asarray(result['all_drhos']),
        all_misfits=np.asarray(result['all_misfits']),
        ref_blocks=ref_blocks,
        block_x_edges=block_x_edges, block_y_edges=block_y_edges,
        obs_x=obs_x, obs_y=obs_y, obs_gravity=gravity_obs,
        borehole_depth=ONGC_DEPTH, borehole_block=np.asarray([ib, jb]),
        rms_ref=rms_ref, bias_ref=bias_ref, coverage_90=cov90,
        runtime_min=elapsed/60, n_iterations=N_ITERATIONS,
        acceptance_rate=result['acceptance_rate'],
        depth_acceptance=result['depth_acceptance_rate'],
        drho_acceptance=result['drho_acceptance_rate'],
        experiment='chintalpudi_drho_quicktest',
    )

    # Quick 4-panel diagnostic plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    ax = axes[0, 0]
    im = ax.pcolormesh(block_x_edges/1000, block_y_edges/1000, ref_blocks.T,
                       cmap='viridis_r', shading='flat')
    plt.colorbar(im, ax=ax, label='Depth (m)')
    ax.scatter([0.5*(block_x_edges[ib]+block_x_edges[ib+1])/1000],
               [0.5*(block_y_edges[jb]+block_y_edges[jb+1])/1000],
               marker='*', s=300, c='gold', edgecolor='k', label='ONGC')
    ax.set_title(f'Chak 2007 reference  (max {ref_blocks.max():.0f} m)',
                 fontweight='bold')
    ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)'); ax.legend()
    ax.set_aspect('equal')

    ax = axes[0, 1]
    im = ax.pcolormesh(block_x_edges/1000, block_y_edges/1000, mean_d.T,
                       cmap='viridis_r', shading='flat')
    plt.colorbar(im, ax=ax, label='Depth (m)')
    ax.scatter([0.5*(block_x_edges[ib]+block_x_edges[ib+1])/1000],
               [0.5*(block_y_edges[jb]+block_y_edges[jb+1])/1000],
               marker='*', s=300, c='gold', edgecolor='k')
    ax.set_title(f'MCMC mean (constant Δρ joint)  '
                 f'depocenter {bore_recov:.0f} m', fontweight='bold')
    ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)'); ax.set_aspect('equal')

    ax = axes[1, 0]
    ax.scatter(ref_blocks.ravel(), mean_d.ravel(), s=40, alpha=0.6,
               c=std_d.ravel(), cmap='hot_r', edgecolor='k', linewidth=0.2)
    lim = max(ref_blocks.max(), mean_d.max()) * 1.1
    ax.plot([0, lim], [0, lim], 'k--', label='1:1')
    ax.scatter([ONGC_DEPTH], [bore_recov], marker='*', s=300, c='gold',
               edgecolor='k', zorder=10, label=f'ONGC ({ONGC_DEPTH}m)')
    ax.set_xlabel('Chak 2007 depth (m)'); ax.set_ylabel('MCMC mean (m)')
    ax.set_title(f'Validation  RMS={rms_ref:.0f}m, bias={bias_ref:+.0f}m, '
                 f'cov={cov90*100:.0f}%', fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3); ax.set_aspect('equal')
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)

    ax = axes[1, 1]
    ax.plot(np.asarray(result['all_drhos']), lw=0.5, color='C0')
    ax.axvline(N_ITERATIONS // 2, color='red', ls='--', label='burn-in')
    ax.axhline(drho_mean, color='k', ls=':', label=f'mean = {drho_mean:.1f}')
    ax.set_xlabel('iteration'); ax.set_ylabel('Δρ (kg/m³)')
    ax.set_title(f'Δρ trace  (post-burn {drho_mean:.1f} ± {drho_std:.1f} kg/m³)',
                 fontweight='bold')
    ax.legend(); ax.grid(alpha=0.3)

    fig.suptitle('Chintalpudi quick test — joint depth + constant Δρ MCMC',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, 'quicktest_summary.png')
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nPlot: {out_png}")
    print(f"Data: {os.path.join(OUT_DIR, 'results_data.npz')}")


if __name__ == '__main__':
    main()
