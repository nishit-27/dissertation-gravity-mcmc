# Lab-Computer Setup (Conda)

Run the Chintalpudi v5 Bayesian MCMC inversion end-to-end on a fresh machine.

## One-time setup

```bash
# 1. Clone the repo
git clone https://github.com/nishit-27/dissertation-gravity-mcmc.git
cd dissertation-gravity-mcmc

# 2. Create a conda environment
conda create -n gravity-mcmc python=3.11 -y

# 3. Activate it
conda activate gravity-mcmc

# 4. Install dependencies from conda-forge
conda install -c conda-forge numpy scipy matplotlib -y

# 5. Sanity check
python -c "import numpy, scipy, matplotlib; print(numpy.__version__, scipy.__version__, matplotlib.__version__)"
```

## Run the Chintalpudi v5 inversion

**One command does MCMC + all 8 plots:**

```bash
# Quick inversion + plots (~10–15 min)
python run_chintalpudi_v5_50k.py

# Publication-quality inversion + plots (~25–30 min)
python run_chintalpudi_v5_100k.py
```

To re-generate plots from a saved run without rerunning MCMC:

```bash
python generate_chintalpudi_v5_plots.py results/exp_chintalpudi_v5_50k
```

Each produces an 8-figure plot suite in the results directory:

| # | File | Content |
|---|---|---|
| 01 | `01_depth_comparison.png`        | 2D: ours vs Chakravarthi 2007 vs difference |
| 02 | `02_depth_3d_surface.png`        | 3D perspective, side-by-side |
| 03 | `03_uncertainty_map.png`         | 2D posterior std |
| 04 | `04_uncertainty_3d_surface.png`  | 3D depth colored by std |
| 05 | `05_cross_sections.png`          | E–W & N–S with 90% CI bands |
| 06 | `06_gravity_fit.png`             | Observed, predicted, residual (mGal) |
| 07 | `07_accuracy.png`                | Scatter + error histogram |
| 08 | `08_mcmc_diagnostics.png`        | Misfit trace + depocenter posterior |

Gold ★ on spatial plots marks the **ONGC borehole (2935 m)** — the only measured ground truth.

`results/.../results_data.npz` saves everything (posterior stats, thinned samples, inputs) so plots can be regenerated without rerunning MCMC.

## Subsequent sessions

```bash
cd dissertation-gravity-mcmc
conda activate gravity-mcmc
git pull                 # grab latest changes from GitHub
python run_chintalpudi_v5_50k.py
```

## Faster dependency resolution (mamba)

```bash
conda install -n base -c conda-forge mamba -y
mamba create -n gravity-mcmc -c conda-forge python=3.11 numpy scipy matplotlib -y
conda activate gravity-mcmc
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `CommandNotFoundError: conda` | Load conda module (lab-specific): `module load anaconda` or `module load miniconda` |
| `conda activate` fails | Run `conda init bash` once, then restart the shell |
| `ModuleNotFoundError: src` | You're not at the repo root — `cd dissertation-gravity-mcmc` first |
| Headless server, no display | Already handled — `matplotlib.use('Agg')` is set in all scripts |
| `zsh: no matches found: requirements*` | Use bash, or quote the glob: `"requirements*"` |

## One-liner (eager mode)

```bash
git clone https://github.com/nishit-27/dissertation-gravity-mcmc.git && \
cd dissertation-gravity-mcmc && \
conda create -n gravity-mcmc -c conda-forge python=3.11 numpy scipy matplotlib -y && \
conda run -n gravity-mcmc python run_chintalpudi_v5_50k.py
```

## Minimum requirements

- Python 3.9+ (3.11 recommended)
- NumPy ≥ 1.24, SciPy ≥ 1.10, Matplotlib ≥ 3.6
- ~100 MB disk for code + data
- ~10–30 min runtime per inversion (single core)
