# dustai — Fixing the fsigma8 Bias from P23 Non-Gaussianity

You are an autonomous research agent. Your task is to develop a
likelihood function and/or BBC bias correction method that recovers
unbiased fsigma8 measurements from Type Ia supernovae when the
intrinsic scatter follows the realistic P23 dust model.

## The Problem

Carreres+2025 (arXiv:2505.13290, copy in this directory) showed that:

1. The P23 dust model (Brout & Scolnic 2021, parameterized by
   Popovic+2023) is the most realistic SN Ia intrinsic scatter model
2. It produces **non-Gaussian Hubble diagram residuals** (skewness
   ~0.3–0.5, excess kurtosis ~1.6)
3. The standard Gaussian likelihood for fsigma8 gives a **~20–26%
   bias** on fsigma8 as a consequence
4. BBC partially corrects color-dependent biases but does NOT fix
   the non-Gaussianity

Your goal: **modify the likelihood and/or BBC to recover unbiased
fsigma8 with calibrated uncertainties.**

## What You Must Do

The current baseline uses a Gaussian likelihood with no BBC
corrections. It scores ~5.3 with ~60% bias on fsigma8.

You need to do one or both of:

1. **Implement BBC bias corrections** in `bbc.py` — the current
   `apply_corrections_to_mock()` is a pass-through that applies
   quality cuts but no actual bias corrections. Implement real BBC
   (bin in z, c, x1, host mass; compute mean bias per bin; subtract).
   This should reduce the non-Gaussianity somewhat and improve the
   Tripp fit residuals.

2. **Create a new likelihood** in `likelihood.py` that accounts for
   the predicted P23 non-Gaussianity. The current Gaussian likelihood
   (Eq. 41 of the paper) assumes velocities are multivariate Gaussian.
   They are not under P23 — the Hubble residuals have skewness ~1.3
   and excess kurtosis ~7.6 (before BBC). You need a likelihood that
   handles this.

The P23 non-Gaussianity has a known, predictable form: it comes from
the exponential dust distribution E_dust ~ Exp(τ) convolved with the
Gaussian intrinsic color. The resulting color distribution has a
positive tail, which after Tripp standardization produces positively
skewed Hubble residuals. Your likelihood or BBC should use this
knowledge.

**Key insight**: the non-Gaussianity is NOT random — it is predicted
by the P23 dust model parameters (R_V, E_dust distributions). A good
solution will use the P23 model to predict the shape of the residual
distribution and account for it, rather than just using a generic
heavy-tailed distribution.

## Goal

Minimise `final_score` (lower is better) from `evaluate.py`.

The score penalizes:
- **Bias** (weight 5.0): |mean(fsigma8) / fsigma8_fid - 1|
- **Coverage** (weight 2.0): |coverage_68 - 0.68| — are 68% CIs
  calibrated?
- **Width** (weight 1.0): mean interval width / fsigma8_fid — don't
  just make huge error bars

---

## Setup

1. **Agree on a run tag** (e.g. `mar18`). Branch: `dustai/<tag>`
2. **Create the branch**: `git checkout -b dustai/<tag>`
3. **Read the in-scope files**:
   - This file (`program.md`)
   - `likelihood.py` — fsigma8 likelihood (**editable**)
   - `bbc.py` — BBC bias corrections (**editable**)
   - `evaluate.py` — scoring harness (read-only)
   - `simulate.py` — mock data generator (read-only)
   - `2505.13290v2.pdf` — the paper (read-only)
4. **Initialise results.tsv** with the header row.
5. **Confirm and go.**

---

## Physics Background

### Peculiar velocities and fsigma8

Observed redshift = cosmological redshift + peculiar velocity:
  1 + z_obs = (1 + z_cos)(1 + z_p)

Peculiar velocities trace the matter density field. The growth rate
fsigma8 = f(z) * sigma8(z) sets the amplitude of the velocity
power spectrum. By measuring SN Ia distances and comparing to
redshifts, we can infer peculiar velocities and constrain fsigma8.

### The Gaussian likelihood (current, biased)

  L = (2π)^(-n/2) |C|^(-1/2) exp(-1/2 v^T C^{-1} v)

where v = estimated velocities, C = C^vv(fsigma8) + C^obs + σ_v² I

This assumes v is multivariate Gaussian. When P23 scatter makes the
Hubble residuals (and therefore velocities) non-Gaussian, this
likelihood is misspecified → biased fsigma8.

### Why P23 produces non-Gaussianity

The BS21/P23 model decomposes SN color into:
- Intrinsic: c_int ~ N(-0.07, 0.05), β ~ N(2.07, 0.22)
- Dust: (R_V + 1) * E_dust, where E_dust ~ Exp(τ)

The exponential dust distribution creates a **positive tail** in
the color distribution. When the standard Tripp formula fits a
single β to this mixture, the residuals are skewed. BBC partially
corrects the mean bias in color bins but doesn't fix the shape.

### Key equations (from the paper)

Velocity covariance (Eq. 35):
  C^vv_ij = H0²/(2π²) * (fsigma8/fsigma8_fid)² * ∫ P_θθ(k) D_u² W_ij dk

Damping function (Eq. 39):
  D_u(k) = sin(k*σ_u) / (k*σ_u)

Hubble residual → velocity (Eq. 29-30):
  v = J(z) * Δμ

The P23 dust model parameters:
  β_int ~ N(2.07, 0.22)
  c_int ~ N(-0.07, 0.05)
  R_V|M_host: N(1.66, 0.95) if log(M) > 10, N(3.25, 0.93) otherwise
  E_dust|M_host,z ~ Exp(τ) with τ from Eq. 10 of the paper
  Observed color: c = c_int + E_dust
  Magnitude: includes (R_V + 1)*E_dust + β_int*c_int terms

Tripp formula (Eq. 3):
  M*_sim = M_b,fid - α*x1 + β*c + Δ_M(M_host, γ)

BS21 modified Tripp (Eq. 6):
  M*_sim = M_b,fid - α*x1 + β_sim*c_sim + (R_V + 1)*E_dust

Mass step (Eq. 5):
  Δ_M = -γ/2 if log(M) > 10, +γ/2 if log(M) < 10
  γ = 0.05 mag

---

## Files you CAN edit

### `likelihood.py`
The fsigma8 likelihood function. Currently implements the Gaussian
likelihood (Eq. 41). Key function: `fit_fsigma8(...)`.

### `bbc.py`
The BBC bias correction code. Currently implements simplified BBC.
Key function: `apply_corrections(...)`.

## Files you CANNOT edit

- `evaluate.py` — the scoring harness
- `simulate.py` — the mock data generator

---

## Running an experiment

```bash
python3 evaluate.py > run.log 2>&1
```

Extract key metrics:
```bash
grep "^final_score:\|^bias:\|^coverage_68:" run.log
```

---

## Logging results

Tab-separated `results.tsv`:

```
commit	final_score	bias	coverage_68	status	description
```

---

## The experiment loop

LOOP FOREVER:

1. Look at git state.
2. Modify `likelihood.py` and/or `bbc.py`.
3. `git commit`
4. Run: `python3 evaluate.py > run.log 2>&1`
5. Read results: `grep "^final_score:\|^bias:\|^coverage_68:" run.log`
6. If crashed: `tail -n 50 run.log`, fix if trivial.
7. Record in `results.tsv`.
8. If `final_score` improved → keep.
9. If equal or worse → `git reset` back.

**NEVER STOP.** The human may be asleep. You are autonomous.

---

## Research Directions

### Approach 1: Fix the likelihood

The Gaussian likelihood is wrong because velocities are non-Gaussian
under P23. Ideas:

- **Skew-normal**: Replace N(0,C) with a skew-normal. Fit skewness
  as a nuisance parameter.
- **Mixture model**: Model the velocity distribution as a mixture
  of Gaussians (low-dust + high-dust populations).
- **Student-t**: Heavy-tailed distribution, robust to
  non-Gaussianity. ν parameter absorbs excess kurtosis.
- **Color split**: Fit fsigma8 separately for blue (c < 0) and
  red (c > 0) SNe, then combine. Blue SNe have less dust → more
  Gaussian.
- **Forward model**: Explicitly model P23 scatter in the likelihood.
  Marginalize over dust parameters per-SN.

### Approach 2: Fix BBC

BBC doesn't fully Gaussianize P23 residuals. Ideas:

- **Higher-dimensional grid**: More parameters (x1, M_host).
- **Asymmetric corrections**: Separate for c > 0 and c < 0.
- **Color-dependent σ_int**: σ_int(c) instead of constant.
- **Iterative BBC**: Run BBC, examine residuals, re-run.
- **Population-aware BBC**: Separate corrections for high/low dust.

### Approach 3: Combine both

Fix BBC to reduce non-Gaussianity, then use a robust likelihood
for the remainder.

### What the paper found (Table 4)

| Model | True vel. | Simple fit | BBC fit |
|-------|-----------|-----------|---------|
| S_COH | 0.990 | 1.000 ± 14% | 0.990 ± 14% |
| S_G10 | 0.984 | 1.038 ± 13% | 1.026 ± 13% |
| S_C11 | 0.991 | 1.052 ± 14% | 1.038 ± 13% |
| S_P23 | 0.988 | **0.808 ± 15%** | **0.743 ± 11%** |

The S_P23 row is what we need to fix. The true velocity fit gives
unbiased results (~0.99), proving the velocity field itself is fine.
The bias comes entirely from the non-Gaussian Hubble residuals.

### SNANA validation

Full SNANA simulation configs for reproducing Figure 6 exactly are
in `snana_sims/`. Use these for validation but NOT for the iteration
loop (too slow). The fast Python mock generator (`simulate.py`) is
used for the agent's iteration loop.

### Available packages

numpy, scipy, iminuit. For MCMC, implement simple Metropolis-Hastings
in pure numpy if needed.

### Reference code

- flip library: https://github.com/corentinravoux/flip
- SNANA: https://github.com/RickKessler/SNANA
- Pippin: https://github.com/dessn/Pippin
