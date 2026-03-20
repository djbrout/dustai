# dustai: Fixing the fsigma8 bias from P23 non-Gaussianity

## The Problem

Carreres et al. (2025, arXiv:2505.13290) showed that the realistic P23 dust model for SN Ia intrinsic scatter biases the growth rate measurement fsigma8 by -20% to -26% when using the standard Gaussian velocity likelihood. This is a systematic larger than the ~14% statistical error of LSST-era surveys.

## Our Solution

Replace the Gaussian likelihood with a **multivariate Student-t distribution**. One line changes in the log-likelihood:

```
# Gaussian:
nll = 0.5 * log|C| + 0.5 * v^T C^{-1} v

# Student-t:
nll = 0.5 * log|C| + (nu+N)/2 * log(1 + v^T C^{-1} v / nu)
```

The degrees-of-freedom parameter nu is estimated from the data kurtosis: `nu = 4 + 6/kurtosis`.

## Key Results

| Method | fsigma8 bias | Source |
|--------|:---:|---|
| Gaussian, simple fit (C25) | **-20%** | Carreres+2025, 8×6600 SNe |
| Gaussian + BBC (C25) | **-26%** | Carreres+2025, 8×6600 SNe |
| **Student-t (this work)** | **-2%** | Analytical mocks, 8×1500 SNe |
| **Student-t, Uchuu N-body** | **+8%** | 8×6600 Uchuu mocks |
| **Student-t, real SNANA** | **-2%** | 1500 SALT3-fitted P23 SNe |
| Gaussian, real SNANA | **-98%** | Same SNANA data (boundary hit) |

## Why It Works

The P23 dust model creates skewed, heavy-tailed Hubble residuals. The fsigma8 likelihood uses chi2 = v^T C^{-1} v, which **squares** residuals — erasing the sign (skewness) but amplifying the amplitude (kurtosis). The top 10% of outlier SNe contribute 66% of the Gaussian chi2 but only 59% of the Student-t. This rebalancing de-leverages the dust-contaminated outliers and recovers fsigma8.

BBC doesn't help because it fixes the **mean** bias per bin (which M_0 already absorbs) but not the **shape** (kurtosis). BBC actually makes P23 worse (-20% → -26% in C25) because z-dependent corrections can remove velocity signal.

## Repository Structure

```
dustai/
├── likelihood.py          # THE KEY FILE: Student-t likelihood for fsigma8
├── bbc.py                 # BBC implementation (pass-through for PV analysis)
├── simulate.py            # Analytical mock generator (READ ONLY)
├── evaluate.py            # Scoring harness for analytical mocks (READ ONLY)
├── evaluate_uchuu.py      # Scoring harness for Uchuu N-body mocks
├── uchuu_mock.py          # Uchuu-based realistic mock generator
├── run_snana_pipeline.py  # Full SNANA → BBC → fsigma8 pipeline
├── bbc_proper.py          # Proper BBC with biascor simulation
├── make_uchuu_plots.py    # Generate comparison plots
├── paper.tex              # Draft paper (MNRAS format)
├── program.md             # Original experiment specification
├── results.tsv            # Score tracking across iterations
├── snana_sims/            # SNANA simulation configs and outputs
│   ├── sim_SP23_lowz.input    # SNANA input for low-z P23 sim
│   ├── fit_SALT3_lowz.nml     # SALT3 fitting config
│   └── FIT_DUSTAI_LOWZ.FITRES.TEXT  # Fitted light curves
└── uchuu/                 # Uchuu halo catalog (7.8 GB, not in git)
    └── halolist_z0p00_m200c1e12_*.h5  # 100 HDF5 files
```

## How to Reproduce

### 1. Setup

```bash
git clone <this repo>
cd dustai
pip install numpy scipy iminuit h5py matplotlib
```

### 2. Run on analytical mocks (fast, ~12s)

```bash
python3 evaluate.py
```

Expected output: `final_score: ~1.88`, `bias: +0.029`

### 3. Download Uchuu halo catalog (7.8 GB)

```bash
mkdir -p uchuu
for i in $(seq 0 99); do
  curl -sL -o uchuu/halolist_z0p00_m200c1e12_${i}.h5 \
    "https://skun.iaa.es/SUsimulations/UchuuDR1/Uchuu/RockstarExtendedM200c1e12/halodir_050/halolist_z0p00_m200c1e12_${i}.h5"
done
```

### 4. Run on Uchuu N-body mocks (~50 min for 8×6600 SNe)

```bash
python3 evaluate_uchuu.py
```

Expected output: `final_score: ~0.84`, `bias: +0.079`

### 5. Run on real SNANA P23 data

Requires SNANA installation. See `snana_sims/` for configs.

```bash
# Generate low-z P23 light curves
source /path/to/snana/setup_env.sh
snlc_sim.exe snana_sims/sim_SP23_lowz.input

# Fit with SALT3
snlc_fit.exe snana_sims/fit_SALT3_lowz.nml

# Run the pipeline
python3 run_snana_pipeline.py
```

### 6. Compile the paper

```bash
pdflatex paper.tex
```

## The Physics in Detail

### Why the Gaussian likelihood fails

The P23 model decomposes SN color as: `c = c_int + E_dust`

- `c_int ~ N(-0.07, 0.05)` — Gaussian intrinsic color
- `E_dust ~ Exp(tau)` — exponential dust reddening

The Tripp formula uses a single beta for both:
`mu = mB - M0 + alpha*x1 - beta*c - Delta_M`

But the true model has two color terms:
`mu_true ∝ beta_int*c_int + (R_V+1)*E_dust`

with beta_int ≈ 2.07 and (R_V+1) ≈ 2.7-4.3. The mismatch creates:
- **Skewness** ~1: positive tail from the exponential dust
- **Kurtosis** ~5-22: heavy tails from extreme dust values

### Why skewness doesn't matter (and kurtosis does)

The fsigma8 likelihood uses chi2 = v^T C^{-1} v, a sum of **squared** terms. Squaring erases the sign of residuals — skewness (positive vs negative outliers) has no effect on chi2. Only the **amplitude** of outliers matters, which is the kurtosis.

The Tripp M_0 parameter absorbs the global mean shift from skewness. We verified this experimentally: a forward model that eliminates skewness (0.8 → 0.04) does NOT improve fsigma8 (-2.2% → -2.8%).

### Why the Student-t fixes it

The Student-t replaces `chi2/2` with `(nu+N)/2 * log(1 + chi2/nu)`. This logarithmic compression means:
- A 1σ residual contributes ~1 (same as Gaussian)
- A 3σ residual contributes ~ln(9) ≈ 2.2 instead of 9
- A 5σ residual contributes ~ln(25) ≈ 3.2 instead of 25

The dust-contaminated outliers lose their leverage, and the fsigma8 signal is recovered from the well-behaved majority of SNe.

### Why BBC makes P23 worse

BBC corrects the **mean** Hubble residual per (z, c, x1) bin. But:
1. M_0 already absorbs the global mean — BBC is fixing something already fixed
2. The within-bin distribution remains non-Gaussian (kurtosis persists)
3. BBC corrections depend on z, and tau(z) in P23 creates color-z correlations that leak velocity signal into the correction

Carreres+2025 confirmed: BBC worsens P23 bias from -20% to -26%.

## Cosmological Parameters

Planck 2015 (matching Uchuu simulation):
- h = 0.6774
- Omega_m = 0.3089
- Omega_b = 0.0486
- sigma_8 = 0.8159
- n_s = 0.9667
- fsigma8 = Omega_m^0.55 * sigma_8 = 0.4276

## References

- Carreres et al. 2025, arXiv:2505.13290 (the problem)
- Brout & Scolnic 2021 (BS21 dust model)
- Popovic et al. 2023 (P23 dust parameters)
- Ishiyama et al. 2021 (Uchuu simulation)
- Bel et al. 2019 (non-linear velocity corrections)
- Howlett et al. 2017 (velocity covariance formalism)
