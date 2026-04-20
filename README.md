# Gravity Data Inversion for Basement Depth with Uncertainty Quantification

**M.Tech Dissertation — Department of Earth Sciences, IIT Roorkee**

Bayesian MCMC inversion of gravity data to estimate basement depth beneath
sedimentary basins, with full posterior uncertainty quantification using
rectangular prism (block) models and depth-dependent density.

> **Setting this up on a lab computer?** See [`SETUP.md`](SETUP.md) for the conda-based install and run instructions.

---

## Folder Structure

```
dissertation_research/
│
├── README.md                  ← You are here
│
├── src/                       ← Source code (all written from scratch)
│   ├── forward_model.py       │  Nagy (2000) rectangular prism gravity formula
│   ├── mcmc_inversion.py      │  Bayesian MCMC (Metropolis-Hastings) inversion
│   ├── synthetic.py           │  Synthetic basin model generator
│   ├── utils.py               │  Density-depth functions (exponential, etc.)
│   └── visualization.py       │  Plotting functions (TODO)
│
├── notebooks/                 ← Jupyter notebooks for analysis
│   ├── 01_forward_model_test.ipynb   (TODO)
│   ├── 02_synthetic_test.ipynb       (TODO)
│   └── 03_analysis.ipynb            (TODO)
│
├── results/                   ← Output figures, data, saved chains
│
├── docs/                      ← Documentation & notes
│   ├── progress_log.md        │  Running log of all work done
│   ├── literature_review.md   │  Formatted literature review
│   ├── literature_review_notes.md  │  Summary notes for 10 papers
│   ├── paper_knowledge_base.md     │  Deep extraction from papers
│   ├── bayesian_mcmc_explained.md  │  MCMC concept explained simply
│   ├── methodology_plan.md    │  Original methodology (superseded)
│   └── workflow.md            │  Workflow notes
│
├── papers/                    ← Research PDFs (11 papers)
│   ├── 01_Florio_2020_Review_ITRESC_Method.pdf
│   ├── 02_Alvarado_2022_3D_Satellite_Gravity_Basement.pdf
│   ├── 03_Pallero_2018_PSO_Uncertainty_Inverse_Problems.pdf
│   ├── 04_Soudeh_2020_IPSO_Basement_Depth.pdf
│   ├── 05_Alqahtani_2022_PSO_Geothermal_Basement.pdf
│   ├── 06_Athens_2022_Stochastic_Inversion_Structural_Uncertainty.pdf
│   ├── 07_Elghrabawy_2025_CNN_LSTM_Basement.pdf
│   ├── 08_Kamto_2023_Nonlinear_Gravity_Inversion_Borehole.pdf
│   ├── 09_Rossi_2017_Bayesian_Gravity_Inversion_Monte_Carlo.pdf
│   ├── 10_Pallero_2024_Posterior_Analysis_PSO_Gravity.pdf
│   ├── GravMCMC_Field_2026.pdf
│   └── extracted_text/        │  Text extractions from some papers
│
└── plans/                     ← Methodology plans (PDFs)
    ├── methodology_plan.pdf
    ├── mcmc_plan.pdf
    ├── short_plan.pdf
    └── scripts/               │  Python scripts that generated the PDFs
```

---

## Method Overview

1. **Forward Model:** Gravity effect of rectangular prisms (Nagy et al., 2000)
2. **Density:** Depth-dependent exponential compaction Δρ(z) = Δρ₀·e^(-λz)
3. **Inversion:** Metropolis-Hastings MCMC sampling of basement depths
4. **Output:** Posterior distribution → mean, std, credible intervals per block
5. **Validation:** Synthetic data with known true basement

---

## Quick Start

```python
import sys
sys.path.insert(0, '/Users/nishit/Desktop/IITR/dissertation_research')

from src.synthetic import create_synthetic_basin_2d, generate_synthetic_gravity
from src.utils import make_density_func
from src.mcmc_inversion import run_mcmc, process_chain

# 1. Create synthetic basin
model = create_synthetic_basin_2d(n_blocks=30)

# 2. Generate gravity data
density_func = make_density_func('exponential', drho_0=-500.0, lam=0.0003)
data = generate_synthetic_gravity(model, density_func, noise_std=0.3)

# 3. Run MCMC inversion
result = run_mcmc(
    obs_x=data['obs_x'],
    gravity_obs=data['gravity_obs'],
    block_x_edges=model['block_x_edges'],
    block_y_width=model['block_y_width'],
    density_func=density_func,
    noise_std=data['noise_std'],
    n_iterations=50000,
    step_size=100.0,
)

# 4. Get posterior statistics
posterior = process_chain(result, burn_in_frac=0.5)
print(f"Mean depths: {posterior['mean']}")
print(f"Uncertainty: {posterior['std']}")
```

---

## Dependencies

- Python 3.x
- NumPy
- SciPy
- Matplotlib
