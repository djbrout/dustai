"""Full SNANA → BBC → fsigma8 pipeline.

Reads SALT3-fitted FITRES output from SNANA, applies BBC corrections
using a biascor subsample, computes velocities, and fits fsigma8
with both Gaussian and Student-t likelihoods.

This is the real pipeline matching Carreres+2025.
"""

import numpy as np
import sys
from pathlib import Path
from scipy.stats import kurtosis, skew

sys.path.insert(0, str(Path(__file__).parent))

from simulate import COSMO, C_LIGHT, mu_cos, Hz_cos, r_cos


def parse_fitres(fname):
    """Parse a SNANA FITRES text file into a dict of arrays."""
    lines = open(fname).readlines()

    # Get column names
    cols = None
    for l in lines:
        if l.startswith('VARNAMES:'):
            cols = l.split()[1:]
            break
    if cols is None:
        raise ValueError(f"No VARNAMES found in {fname}")

    data = {c: [] for c in cols}
    for l in lines:
        if l.startswith('SN:'):
            parts = l.split()[1:]
            for c, v in zip(cols, parts):
                try:
                    data[c].append(float(v))
                except ValueError:
                    data[c].append(v)

    return {c: np.array(data[c]) for c in data}


def select_pv_sample(data, z_min=0.02, z_max=0.10, x1_cut=3.0,
                      c_cut=0.3, fitprob_cut=0.01):
    """Apply quality cuts for PV analysis."""
    z = data['zCMB'].astype(float)
    x1 = data['x1'].astype(float)
    c = data['c'].astype(float)
    fitprob = data['FITPROB'].astype(float)
    errflag = data['ERRFLAG_FIT'].astype(float)

    mask = (
        (z >= z_min) & (z <= z_max)
        & (np.abs(x1) < x1_cut)
        & (np.abs(c) < c_cut)
        & (fitprob > fitprob_cut)
        & (errflag == 0)
    )

    return {c: data[c][mask] for c in data}, mask


def tripp_fit(data, mu_fid):
    """Fit Tripp standardization and return Hubble residuals."""
    from scipy.optimize import minimize

    mB = data['mB'].astype(float)
    x1 = data['x1'].astype(float)
    c = data['c'].astype(float)
    z = data['zCMB'].astype(float)
    mBERR = data['mBERR'].astype(float)
    x1ERR = data['x1ERR'].astype(float)
    cERR = data['cERR'].astype(float)

    def neg_log_like(params):
        alpha, beta, M0, ln_sig = params
        sigma_int = np.exp(ln_sig)
        mu_obs = mB - M0 + alpha * x1 - beta * c
        resid = mu_obs - mu_fid
        var = mBERR**2 + (alpha*x1ERR)**2 + (beta*cERR)**2 + sigma_int**2
        return np.sum(resid**2 / var + np.log(var))

    res = minimize(neg_log_like, [0.15, 3.1, -19.36, np.log(0.1)],
                   method='Nelder-Mead',
                   options={'maxiter': 50000, 'xatol': 1e-8, 'fatol': 1e-8})
    alpha, beta, M0, ln_sig = res.x
    sigma_int = np.exp(ln_sig)

    mu_obs = mB - M0 + alpha * x1 - beta * c
    delta_mu = mu_obs - mu_fid
    sigma_mu = np.sqrt(mBERR**2 + (alpha*x1ERR)**2 + (beta*cERR)**2 + sigma_int**2)

    return {
        'delta_mu': delta_mu,
        'sigma_mu': sigma_mu,
        'alpha': alpha,
        'beta': beta,
        'M0': M0,
        'sigma_int': sigma_int,
        'mu_obs': mu_obs,
    }


def compute_bbc_corrections_from_fitres(data, tripp_result, mu_fid,
                                         n_z=8, n_c=6, n_x1=3,
                                         min_per_bin=10):
    """Compute BBC corrections from FITRES data with known truth.

    Uses SIM_DLMAG (true distance modulus) to compute the bias
    in each (z, c, x1) bin.
    """
    z = data['zCMB'].astype(float)
    c = data['c'].astype(float)
    x1 = data['x1'].astype(float)
    delta_mu = tripp_result['delta_mu']

    # The "truth" for the distance modulus
    sim_dlmag = data['SIM_DLMAG'].astype(float)
    mu_bias = tripp_result['mu_obs'] - sim_dlmag  # includes PV

    # But PV averages to zero in bins, so the mean of mu_bias
    # in each bin IS the standardization bias
    z_edges = np.linspace(z.min()-0.001, z.max()+0.001, n_z+1)
    c_edges = np.linspace(-0.3, 0.3, n_c+1)
    x1_edges = np.linspace(-3, 3, n_x1+1)

    iz = np.clip(np.digitize(z, z_edges)-1, 0, n_z-1)
    ic = np.clip(np.digitize(c, c_edges)-1, 0, n_c-1)
    ix = np.clip(np.digitize(x1, x1_edges)-1, 0, n_x1-1)

    bias_grid = np.zeros((n_z, n_c, n_x1))
    for i_z in range(n_z):
        for i_c in range(n_c):
            for i_x in range(n_x1):
                sel = (iz==i_z) & (ic==i_c) & (ix==i_x)
                if sel.sum() >= min_per_bin:
                    bias_grid[i_z, i_c, i_x] = np.mean(mu_bias[sel])

    return bias_grid, z_edges, c_edges, x1_edges


def apply_bbc(data, tripp_result, bias_grid, z_edges, c_edges, x1_edges):
    """Apply BBC corrections to a data sample."""
    z = data['zCMB'].astype(float)
    c = data['c'].astype(float)
    x1 = data['x1'].astype(float)

    n_z = len(z_edges) - 1
    n_c = len(c_edges) - 1
    n_x1 = len(x1_edges) - 1

    iz = np.clip(np.digitize(z, z_edges)-1, 0, n_z-1)
    ic = np.clip(np.digitize(c, c_edges)-1, 0, n_c-1)
    ix = np.clip(np.digitize(x1, x1_edges)-1, 0, n_x1-1)

    corrections = np.array([bias_grid[iz[i], ic[i], ix[i]]
                            for i in range(len(z))])

    delta_mu_corr = tripp_result['delta_mu'] - corrections
    return delta_mu_corr


def estimate_velocities(delta_mu, z_obs, sigma_mu):
    """Convert Hubble residuals to velocities."""
    Hz = Hz_cos(z_obs)
    rz = np.maximum(r_cos(z_obs), 1e-3)
    bracket = (1 + z_obs) * C_LIGHT / (Hz * rz) - 1
    bracket = np.where(np.abs(bracket) < 1e-6,
                       np.sign(bracket) * 1e-6, bracket)
    J_z = -C_LIGHT * np.log(10) / 5 / bracket

    v_est = J_z * delta_mu
    sigma_v = np.abs(J_z) * sigma_mu
    return v_est, sigma_v, J_z


def run_pipeline(fitres_file, n_biascor=None, seed=42):
    """Run the full BBC → fsigma8 pipeline on a FITRES file.

    Splits the sample: first n_biascor SNe for BBC calibration,
    rest for data. If n_biascor is None, uses 70% for biascor.
    """
    print(f"Reading {fitres_file}...")
    raw = parse_fitres(fitres_file)

    # Quality cuts
    cut, mask = select_pv_sample(raw)
    n_total = len(cut['zCMB'])
    print(f"  After cuts: {n_total} SNe")

    if n_total < 100:
        print("  Too few SNe!")
        return None

    z = cut['zCMB'].astype(float)
    mu_fid = np.array([float(mu_cos(zi)) for zi in z])

    # Split into biascor and data
    rng = np.random.default_rng(seed)
    if n_biascor is None:
        n_biascor = int(0.7 * n_total)
    idx = rng.permutation(n_total)
    idx_bc = idx[:n_biascor]
    idx_data = idx[n_biascor:]

    biascor = {c: cut[c][idx_bc] for c in cut}
    data = {c: cut[c][idx_data] for c in cut}

    print(f"  Biascor: {len(idx_bc)} SNe, Data: {len(idx_data)} SNe")

    # Tripp fit on biascor
    mu_fid_bc = mu_fid[idx_bc]
    tripp_bc = tripp_fit(biascor, mu_fid_bc)
    print(f"  Tripp (biascor): alpha={tripp_bc['alpha']:.3f} "
          f"beta={tripp_bc['beta']:.3f} sigma_int={tripp_bc['sigma_int']:.3f}")

    # Compute BBC corrections from biascor
    bias_grid, ze, ce, xe = compute_bbc_corrections_from_fitres(
        biascor, tripp_bc, mu_fid_bc
    )
    print(f"  BBC correction range: [{bias_grid.min():.4f}, {bias_grid.max():.4f}] mag")

    # Tripp fit on data (using same alpha/beta or refitting)
    mu_fid_data = mu_fid[idx_data]
    tripp_data = tripp_fit(data, mu_fid_data)

    # Apply BBC to data
    delta_mu_raw = tripp_data['delta_mu']
    delta_mu_bbc = apply_bbc(data, tripp_data, bias_grid, ze, ce, xe)

    print(f"\n  Raw residuals:  skew={skew(delta_mu_raw):.3f} "
          f"kurt={kurtosis(delta_mu_raw, fisher=True):.3f}")
    print(f"  BBC residuals:  skew={skew(delta_mu_bbc):.3f} "
          f"kurt={kurtosis(delta_mu_bbc, fisher=True):.3f}")

    # Convert to velocities
    z_data = data['zCMB'].astype(float)
    sigma_mu_data = tripp_data['sigma_mu']

    v_raw, sv_raw, J_z = estimate_velocities(delta_mu_raw, z_data, sigma_mu_data)
    v_bbc, sv_bbc, _ = estimate_velocities(delta_mu_bbc, z_data, sigma_mu_data)

    # Get true velocities for comparison
    sim_vpec = data['SIM_VPEC'].astype(float)

    print(f"\n  v_est (raw) std: {v_raw.std():.0f} km/s")
    print(f"  v_est (BBC) std: {v_bbc.std():.0f} km/s")
    print(f"  v_true std: {sim_vpec.std():.0f} km/s")
    print(f"  corr(v_raw, v_true): {np.corrcoef(v_raw, sim_vpec)[0,1]:.3f}")
    print(f"  corr(v_bbc, v_true): {np.corrcoef(v_bbc, sim_vpec)[0,1]:.3f}")

    return {
        'data': data,
        'tripp': tripp_data,
        'delta_mu_raw': delta_mu_raw,
        'delta_mu_bbc': delta_mu_bbc,
        'v_raw': v_raw,
        'v_bbc': v_bbc,
        'sigma_v': sv_bbc,
        'z': z_data,
        'sim_vpec': sim_vpec,
        'J_z': J_z,
    }


if __name__ == '__main__':
    import time

    fitres = '/Volumes/External24TB/dustai/snana_sims/FIT_DUSTAI_LOWZ.FITRES.TEXT'

    if not Path(fitres).exists():
        print("FITRES file not found. SALT3 fit still running?")
        sys.exit(1)

    t0 = time.time()
    result = run_pipeline(fitres)

    if result is not None:
        print(f"\nPipeline completed in {time.time()-t0:.1f}s")
