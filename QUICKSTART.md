# dustai - Autoresearch Quickstart

## What This Is
An autonomous AI research agent that fixes the fsigma8 bias caused
by non-Gaussian Hubble diagram residuals from the P23 dust model.

## The Problem
The Gaussian likelihood for fsigma8 gives ~20-26% bias when the
intrinsic scatter model is P23 (dust-based). See `2505.13290v2.pdf`.

## How to Kick It Off

### Step 1: Open a terminal in this folder
```bash
cd /Volumes/External24TB/dustai
```

### Step 2: Set up SNANA environment (if needed for validation)
```bash
source /Volumes/External24TB/snana_install/setup_env.sh
```

### Step 3: Launch Claude Code
```bash
claude
```

### Step 4: Say the magic words
```
Have a look at program.md and let's kick off a new experiment.
```

## What the Agent Edits
- `likelihood.py` — the fsigma8 likelihood function
- `bbc.py` — the BBC bias correction code

## Key Files
- `evaluate.py` — scoring harness (read-only)
- `simulate.py` — mock data generator (read-only)
- `program.md` — full instructions
- `2505.13290v2.pdf` — the paper

## Check Progress
```bash
cat results.tsv
python3 plot_progress.py
```

## Known Setup Issues
The velocity covariance calibration between simulate.py and
likelihood.py may need tuning. The agent should verify that
the Gaussian likelihood baseline reproduces the expected ~20%
bias before attempting fixes. If the baseline doesn't show
the bias, the mock generator or likelihood covariance needs
debugging first.
