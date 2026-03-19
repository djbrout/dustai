"""BBC (BEAMS with Bias Corrections) implementation. THE AGENT EDITS THIS FILE.

Implements bias corrections for the Hubble diagram following
Kessler & Scolnic (2017) and the BBC-7D / BBC-BS20 approaches.

The initial implementation uses a simplified 2D binning in (z, c) for
speed.  The agent can extend to the full 7D or BS20 formulations.
"""

import numpy as np
from scipy import interpolate

try:
    from iminuit import Minuit
    HAS_IMINUIT = True
except ImportError:
    HAS_IMINUIT = False


# ---------------------------------------------------------------------------
# Bias correction computation
# ---------------------------------------------------------------------------

def _bin_edges(values, n_bins, pad=0.01):
    """Create bin edges spanning the data range with padding."""
    lo = np.min(values) - pad * np.ptp(values)
    hi = np.max(values) + pad * np.ptp(values)
    return np.linspace(lo, hi, n_bins + 1)


def _digitize_clipped(values, edges):
    """Digitize values into bins, clipping to valid range."""
    idx = np.digitize(values, edges) - 1
    return np.clip(idx, 0, len(edges) - 2)


def compute_bias_corrections(sim_data, method='BBC7D',
                             n_z_bins=20, n_c_bins=10,
                             n_x1_bins=5, n_alpha_bins=1,
                             n_beta_bins=1, n_host_bins=2,
                             min_per_bin=5):
    """Compute bias corrections from a bias-correction simulation.

    Parameters
    ----------
    sim_data : dict
        Simulated SN data with keys:
        - 'z'       : ndarray, redshifts
        - 'mB_fit'  : ndarray, fitted peak magnitude
        - 'mB_sim'  : ndarray, simulated (true) peak magnitude
        - 'x1_fit'  : ndarray, fitted stretch
        - 'x1_sim'  : ndarray, simulated stretch
        - 'c_fit'   : ndarray, fitted colour
        - 'c_sim'   : ndarray, simulated colour
        - 'mu_obs'  : ndarray, observed distance modulus (optional, for BS20)
        - 'mu_fid'  : ndarray, fiducial distance modulus (optional, for BS20)
        - 'alpha'   : ndarray or float (optional, for 7D)
        - 'beta'    : ndarray or float (optional, for 7D)
        - 'host_mass': ndarray (optional, for host-mass step)
    method : str
        'BBC7D' or 'BBCBS20'.
    n_z_bins, n_c_bins, n_x1_bins : int
        Number of bins in each dimension.
    n_alpha_bins, n_beta_bins : int
        Number of bins for alpha, beta (BBC7D only).
    n_host_bins : int
        Number of bins for host-mass step (typically 2: low/high mass).
    min_per_bin : int
        Minimum number of simulated events per bin.

    Returns
    -------
    bias_functions : dict
        Contains interpolation functions and metadata.
        For BBC7D: 'delta_mB', 'delta_x1', 'delta_c' functions of (z, c)
        For BBCBS20: 'delta_mu' function of (z, c)
        Plus 'method', 'bin_edges', and 'bin_counts'.
    """
    z = np.asarray(sim_data['z'])
    c_fit = np.asarray(sim_data['c_fit'])

    z_edges = _bin_edges(z, n_z_bins)
    c_edges = _bin_edges(c_fit, n_c_bins)

    iz = _digitize_clipped(z, z_edges)
    ic = _digitize_clipped(c_fit, c_edges)

    z_centres = 0.5 * (z_edges[:-1] + z_edges[1:])
    c_centres = 0.5 * (c_edges[:-1] + c_edges[1:])

    if method == 'BBC7D':
        return _compute_bbc7d(sim_data, iz, ic, z_centres, c_centres,
                              z_edges, c_edges, n_z_bins, n_c_bins,
                              min_per_bin)
    elif method == 'BBCBS20':
        return _compute_bbcbs20(sim_data, iz, ic, z_centres, c_centres,
                                z_edges, c_edges, n_z_bins, n_c_bins,
                                min_per_bin)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'BBC7D' or 'BBCBS20'.")


def _compute_bbc7d(sim_data, iz, ic, z_centres, c_centres,
                   z_edges, c_edges, n_z_bins, n_c_bins, min_per_bin):
    """BBC-7D: compute per-parameter biases delta_mB, delta_x1, delta_c.

    In each (z, c) cell:  delta_p = <p_fit - p_sim>  for p in {mB, x1, c}.
    """
    mB_fit = np.asarray(sim_data['mB_fit'])
    mB_sim = np.asarray(sim_data['mB_sim'])
    x1_fit = np.asarray(sim_data['x1_fit'])
    x1_sim = np.asarray(sim_data['x1_sim'])
    c_fit = np.asarray(sim_data['c_fit'])
    c_sim = np.asarray(sim_data['c_sim'])

    delta_mB_grid = np.zeros((n_z_bins, n_c_bins))
    delta_x1_grid = np.zeros((n_z_bins, n_c_bins))
    delta_c_grid = np.zeros((n_z_bins, n_c_bins))
    counts_grid = np.zeros((n_z_bins, n_c_bins), dtype=int)

    for i_z in range(n_z_bins):
        for i_c in range(n_c_bins):
            mask = (iz == i_z) & (ic == i_c)
            n_in_bin = np.sum(mask)
            counts_grid[i_z, i_c] = n_in_bin

            if n_in_bin >= min_per_bin:
                delta_mB_grid[i_z, i_c] = np.mean(mB_fit[mask] - mB_sim[mask])
                delta_x1_grid[i_z, i_c] = np.mean(x1_fit[mask] - x1_sim[mask])
                delta_c_grid[i_z, i_c] = np.mean(c_fit[mask] - c_sim[mask])
            else:
                # Use NaN for interpolation to fill later
                delta_mB_grid[i_z, i_c] = np.nan
                delta_x1_grid[i_z, i_c] = np.nan
                delta_c_grid[i_z, i_c] = np.nan

    # Fill NaN cells via nearest-neighbour interpolation
    delta_mB_grid = _fill_nan_nearest(delta_mB_grid)
    delta_x1_grid = _fill_nan_nearest(delta_x1_grid)
    delta_c_grid = _fill_nan_nearest(delta_c_grid)

    # Build 2D interpolators (z, c) -> delta_p
    interp_mB = interpolate.RegularGridInterpolator(
        (z_centres, c_centres), delta_mB_grid,
        method='linear', bounds_error=False, fill_value=None
    )
    interp_x1 = interpolate.RegularGridInterpolator(
        (z_centres, c_centres), delta_x1_grid,
        method='linear', bounds_error=False, fill_value=None
    )
    interp_c = interpolate.RegularGridInterpolator(
        (z_centres, c_centres), delta_c_grid,
        method='linear', bounds_error=False, fill_value=None
    )

    return {
        'method': 'BBC7D',
        'delta_mB': interp_mB,
        'delta_x1': interp_x1,
        'delta_c': interp_c,
        'bin_edges': {'z': z_edges, 'c': c_edges},
        'bin_centres': {'z': z_centres, 'c': c_centres},
        'bin_counts': counts_grid,
        'grids': {
            'delta_mB': delta_mB_grid,
            'delta_x1': delta_x1_grid,
            'delta_c': delta_c_grid,
        },
    }


def _compute_bbcbs20(sim_data, iz, ic, z_centres, c_centres,
                     z_edges, c_edges, n_z_bins, n_c_bins, min_per_bin):
    """BBC-BS20: compute the direct distance-modulus bias delta_mu.

    In each (z, c) cell:  delta_mu = <mu_obs - mu_fid>.
    """
    mu_obs = np.asarray(sim_data['mu_obs'])
    mu_fid = np.asarray(sim_data['mu_fid'])

    delta_mu_grid = np.zeros((n_z_bins, n_c_bins))
    counts_grid = np.zeros((n_z_bins, n_c_bins), dtype=int)

    for i_z in range(n_z_bins):
        for i_c in range(n_c_bins):
            mask = (iz == i_z) & (ic == i_c)
            n_in_bin = np.sum(mask)
            counts_grid[i_z, i_c] = n_in_bin

            if n_in_bin >= min_per_bin:
                delta_mu_grid[i_z, i_c] = np.mean(mu_obs[mask] - mu_fid[mask])
            else:
                delta_mu_grid[i_z, i_c] = np.nan

    delta_mu_grid = _fill_nan_nearest(delta_mu_grid)

    interp_mu = interpolate.RegularGridInterpolator(
        (z_centres, c_centres), delta_mu_grid,
        method='linear', bounds_error=False, fill_value=None
    )

    return {
        'method': 'BBCBS20',
        'delta_mu': interp_mu,
        'bin_edges': {'z': z_edges, 'c': c_edges},
        'bin_centres': {'z': z_centres, 'c': c_centres},
        'bin_counts': counts_grid,
        'grids': {
            'delta_mu': delta_mu_grid,
        },
    }


def _fill_nan_nearest(grid):
    """Fill NaN entries in a 2D grid using nearest valid value."""
    mask = np.isnan(grid)
    if not np.any(mask):
        return grid

    if np.all(mask):
        # No valid data at all -- fill with zeros
        return np.zeros_like(grid)

    from scipy.ndimage import distance_transform_edt

    filled = grid.copy()
    # Indices of nearest valid pixel
    _, nearest_idx = distance_transform_edt(mask, return_distances=True,
                                            return_indices=True)
    filled[mask] = grid[nearest_idx[0][mask], nearest_idx[1][mask]]
    return filled


# ---------------------------------------------------------------------------
# Apply corrections
# ---------------------------------------------------------------------------

def apply_corrections(data, bias_functions, method=None):
    """Apply bias corrections to observed SN data.

    Parameters
    ----------
    data : dict
        Observed SN data with keys:
        - 'z'      : ndarray, redshifts
        - 'mB'     : ndarray, peak magnitude (for BBC7D)
        - 'x1'     : ndarray, stretch (for BBC7D)
        - 'c'      : ndarray, colour (for BBC7D)
        - 'mu_obs' : ndarray, observed distance modulus (for BS20)
        - 'mB_err' : ndarray, uncertainty on mB (optional)
        - 'x1_err' : ndarray, uncertainty on x1 (optional)
        - 'c_err'  : ndarray, uncertainty on c (optional)
        - 'mu_err' : ndarray, uncertainty on mu_obs (optional)
        - 'alpha'  : float, stretch coefficient
        - 'beta'   : float, colour coefficient
        - 'M0'     : float, absolute magnitude
        - 'Delta_M': ndarray, host-mass correction per SN (optional)
    bias_functions : dict
        Output of compute_bias_corrections().
    method : str or None
        Override the method. If None, uses bias_functions['method'].

    Returns
    -------
    result : dict
        - 'mu_corrected': ndarray, bias-corrected distance moduli
        - 'mu_err'      : ndarray, uncertainties on corrected mu
    """
    if method is None:
        method = bias_functions['method']

    z = np.asarray(data['z'])
    c = np.asarray(data['c'])

    # Points for interpolation
    points = np.column_stack([z, c])

    if method == 'BBC7D':
        mB = np.asarray(data['mB'])
        x1 = np.asarray(data['x1'])
        alpha = data.get('alpha', 0.15)
        beta = data.get('beta', 3.1)
        M0 = data.get('M0', -19.36)
        Delta_M = np.asarray(data.get('Delta_M', np.zeros(len(z))))

        # Look up bias corrections
        d_mB = bias_functions['delta_mB'](points)
        d_x1 = bias_functions['delta_x1'](points)
        d_c = bias_functions['delta_c'](points)

        # Tripp formula with bias corrections (Eq. in Kessler & Scolnic 2017):
        #   mu = (mB - d_mB) - M0 + alpha*(x1 - d_x1) - beta*(c - d_c) - Delta_M
        mu_corrected = (
            (mB - d_mB) - M0
            + alpha * (x1 - d_x1)
            - beta * (c - d_c)
            - Delta_M
        )

        # Propagate uncertainties (simplified: ignore bias correction errors)
        mB_err = np.asarray(data.get('mB_err', np.zeros(len(z))))
        x1_err = np.asarray(data.get('x1_err', np.zeros(len(z))))
        c_err = np.asarray(data.get('c_err', np.zeros(len(z))))
        mu_err = np.sqrt(
            mB_err ** 2
            + (alpha * x1_err) ** 2
            + (beta * c_err) ** 2
        )

    elif method == 'BBCBS20':
        mu_obs = np.asarray(data['mu_obs'])

        d_mu = bias_functions['delta_mu'](points)
        mu_corrected = mu_obs - d_mu

        mu_err = np.asarray(data.get('mu_err', np.zeros(len(z))))

    else:
        raise ValueError(f"Unknown method: {method}")

    return {
        'mu_corrected': mu_corrected,
        'mu_err': mu_err,
    }


# ---------------------------------------------------------------------------
# Nuisance parameter fitting
# ---------------------------------------------------------------------------

def _hubble_residuals(data, alpha, beta, gamma, M0, sigma_int,
                      bias_functions, mu_theory):
    """Compute Hubble residuals and chi2 for given nuisance parameters.

    Parameters
    ----------
    data : dict
        SN data (see apply_corrections).
    alpha, beta, gamma, M0, sigma_int : float
        Nuisance parameters.
    bias_functions : dict
        Bias corrections.
    mu_theory : ndarray
        Theoretical distance moduli at each SN redshift.

    Returns
    -------
    residuals : ndarray
    chi2 : float
    """
    z = np.asarray(data['z'])
    c = np.asarray(data['c'])
    mB = np.asarray(data['mB'])
    x1 = np.asarray(data['x1'])

    host_mass = np.asarray(data.get('host_mass', np.full(len(z), 10.5)))
    host_step = np.where(host_mass > 10.0, gamma / 2.0, -gamma / 2.0)

    points = np.column_stack([z, c])

    d_mB = bias_functions['delta_mB'](points)
    d_x1 = bias_functions['delta_x1'](points)
    d_c = bias_functions['delta_c'](points)

    mu_obs = (
        (mB - d_mB) - M0
        + alpha * (x1 - d_x1)
        - beta * (c - d_c)
        - host_step
    )

    residuals = mu_obs - mu_theory

    mB_err = np.asarray(data.get('mB_err', np.zeros(len(z))))
    x1_err = np.asarray(data.get('x1_err', np.zeros(len(z))))
    c_err = np.asarray(data.get('c_err', np.zeros(len(z))))
    cov_x1_c = np.asarray(data.get('cov_x1_c', np.zeros(len(z))))
    cov_mB_x1 = np.asarray(data.get('cov_mB_x1', np.zeros(len(z))))
    cov_mB_c = np.asarray(data.get('cov_mB_c', np.zeros(len(z))))

    # Diagonal uncertainty on mu
    var_mu = (
        mB_err ** 2
        + (alpha * x1_err) ** 2
        + (beta * c_err) ** 2
        + 2.0 * alpha * cov_mB_x1
        - 2.0 * beta * cov_mB_c
        - 2.0 * alpha * beta * cov_x1_c
        + sigma_int ** 2
    )

    chi2 = np.sum(residuals ** 2 / var_mu) + np.sum(np.log(var_mu))

    return residuals, chi2


def fit_nuisance_params(data, bias_functions, mu_theory,
                        method='BBC7D', verbose=False):
    """Fit nuisance parameters alpha, beta, gamma, M0, sigma_int.

    Minimises chi^2 of the Hubble residuals using the Tripp formula
    with bias corrections.

    Parameters
    ----------
    data : dict
        Observed SN data (see apply_corrections for required keys).
    bias_functions : dict
        Output of compute_bias_corrections().
    mu_theory : ndarray
        Theoretical distance moduli at each SN redshift.
    method : str
        'BBC7D' or 'BBCBS20'. Currently only BBC7D is supported for
        nuisance fitting.
    verbose : bool
        Print fitting output.

    Returns
    -------
    result : dict
        - 'alpha'      : best-fit stretch coefficient
        - 'beta'       : best-fit colour coefficient
        - 'gamma'      : best-fit host-mass step (mag)
        - 'M0'         : best-fit absolute magnitude
        - 'sigma_int'  : best-fit intrinsic dispersion (mag)
        - 'chi2'       : minimum chi2
        - 'ndof'       : degrees of freedom
        - 'residuals'  : Hubble residuals at best fit
        - 'mu_corrected': corrected distance moduli
        - 'converged'  : bool
    """
    if method != 'BBC7D':
        raise NotImplementedError(
            "Nuisance fitting is currently implemented only for BBC7D. "
            "For BBCBS20, nuisance parameters are absorbed into the "
            "bias correction."
        )

    if HAS_IMINUIT:
        return _fit_nuisance_iminuit(data, bias_functions, mu_theory, verbose)
    else:
        return _fit_nuisance_scipy(data, bias_functions, mu_theory, verbose)


def _fit_nuisance_iminuit(data, bias_functions, mu_theory, verbose):
    """Fit nuisance parameters using iminuit."""

    def cost(alpha, beta, gamma, M0, sigma_int):
        _, chi2 = _hubble_residuals(
            data, alpha, beta, gamma, M0, sigma_int,
            bias_functions, mu_theory
        )
        return chi2

    m = Minuit(cost,
               alpha=0.15, beta=3.1, gamma=0.06, M0=-19.36, sigma_int=0.10)
    m.limits['alpha'] = (-0.5, 1.0)
    m.limits['beta'] = (0.5, 8.0)
    m.limits['gamma'] = (-0.5, 0.5)
    m.limits['M0'] = (-25.0, -15.0)
    m.limits['sigma_int'] = (0.001, 1.0)
    m.errordef = Minuit.LEAST_SQUARES  # chi2 minimisation

    m.print_level = 2 if verbose else 0
    m.migrad()
    m.hesse()

    alpha_fit = m.values['alpha']
    beta_fit = m.values['beta']
    gamma_fit = m.values['gamma']
    M0_fit = m.values['M0']
    sigma_int_fit = m.values['sigma_int']

    residuals, chi2 = _hubble_residuals(
        data, alpha_fit, beta_fit, gamma_fit, M0_fit, sigma_int_fit,
        bias_functions, mu_theory
    )

    N = len(data['z'])
    ndof = N - 5  # 5 fitted parameters

    # Compute corrected distance moduli at best-fit nuisance params
    data_copy = dict(data)
    data_copy['alpha'] = alpha_fit
    data_copy['beta'] = beta_fit
    data_copy['M0'] = M0_fit

    host_mass = np.asarray(data.get('host_mass', np.full(N, 10.5)))
    data_copy['Delta_M'] = np.where(
        host_mass > 10.0, gamma_fit / 2.0, -gamma_fit / 2.0
    )

    corrected = apply_corrections(data_copy, bias_functions, method='BBC7D')

    return {
        'alpha': alpha_fit,
        'beta': beta_fit,
        'gamma': gamma_fit,
        'M0': M0_fit,
        'sigma_int': sigma_int_fit,
        'chi2': chi2,
        'ndof': ndof,
        'residuals': residuals,
        'mu_corrected': corrected['mu_corrected'],
        'converged': m.valid,
        'errors': {
            'alpha': m.errors['alpha'],
            'beta': m.errors['beta'],
            'gamma': m.errors['gamma'],
            'M0': m.errors['M0'],
            'sigma_int': m.errors['sigma_int'],
        },
    }


def _fit_nuisance_scipy(data, bias_functions, mu_theory, verbose):
    """Fallback nuisance fitting using scipy.optimize.minimize."""
    from scipy.optimize import minimize

    def cost(params):
        alpha, beta, gamma, M0, sigma_int = params
        _, chi2 = _hubble_residuals(
            data, alpha, beta, gamma, M0, sigma_int,
            bias_functions, mu_theory
        )
        return chi2

    x0 = [0.15, 3.1, 0.06, -19.36, 0.10]
    bounds = [(-0.5, 1.0), (0.5, 8.0), (-0.5, 0.5), (-25.0, -15.0),
              (0.001, 1.0)]

    res = minimize(cost, x0, method='L-BFGS-B', bounds=bounds,
                   options={'disp': verbose, 'maxiter': 500})

    alpha_fit, beta_fit, gamma_fit, M0_fit, sigma_int_fit = res.x

    residuals, chi2 = _hubble_residuals(
        data, alpha_fit, beta_fit, gamma_fit, M0_fit, sigma_int_fit,
        bias_functions, mu_theory
    )

    N = len(data['z'])
    ndof = N - 5

    data_copy = dict(data)
    data_copy['alpha'] = alpha_fit
    data_copy['beta'] = beta_fit
    data_copy['M0'] = M0_fit

    host_mass = np.asarray(data.get('host_mass', np.full(N, 10.5)))
    data_copy['Delta_M'] = np.where(
        host_mass > 10.0, gamma_fit / 2.0, -gamma_fit / 2.0
    )

    corrected = apply_corrections(data_copy, bias_functions, method='BBC7D')

    # Estimate uncertainties from Hessian
    try:
        hess_inv = np.asarray(res.hess_inv.todense() if hasattr(
            res.hess_inv, 'todense') else res.hess_inv)
        errors = {
            'alpha': np.sqrt(max(hess_inv[0, 0], 0)),
            'beta': np.sqrt(max(hess_inv[1, 1], 0)),
            'gamma': np.sqrt(max(hess_inv[2, 2], 0)),
            'M0': np.sqrt(max(hess_inv[3, 3], 0)),
            'sigma_int': np.sqrt(max(hess_inv[4, 4], 0)),
        }
    except Exception:
        errors = {k: np.nan for k in ['alpha', 'beta', 'gamma', 'M0',
                                       'sigma_int']}

    return {
        'alpha': alpha_fit,
        'beta': beta_fit,
        'gamma': gamma_fit,
        'M0': M0_fit,
        'sigma_int': sigma_int_fit,
        'chi2': chi2,
        'ndof': ndof,
        'residuals': residuals,
        'mu_corrected': corrected['mu_corrected'],
        'converged': res.success,
        'errors': errors,
    }


# ---------------------------------------------------------------------------
# Top-level wrapper for evaluate.py
# ---------------------------------------------------------------------------

def apply_corrections_to_mock(mock):
    """Apply BBC corrections to a full mock dict from simulate.generate_mock().

    This is the entry point called by evaluate.py. It extracts the
    relevant fields from the mock, applies corrections (currently a
    simplified pass-through that uses the Tripp-fit velocities), and
    returns a dict with corrected velocities and uncertainties.

    The agent should improve this function to better handle P23
    non-Gaussianity — e.g. by implementing full BBC-7D/BS20, by
    using color-dependent corrections, or by modifying the Tripp
    fit to account for the dust mixture.

    Returns
    -------
    dict with keys:
        velocities : ndarray, corrected estimated velocities
        sigma_v    : ndarray, velocity uncertainties
        n_sn       : int, number of SNe after cuts
        delta_mu   : ndarray, Hubble residuals
        colors     : ndarray, SALT colors (for diagnostics)
    """
    tripp = mock["tripp_fit"]
    vel = mock["vel_est"]
    obs = mock["observables"]

    x1 = obs["x1"]
    c = obs["c"]
    sigma_mu = tripp["sigma_mu"]

    # Must match evaluate.py mask exactly
    mask = (
        np.abs(x1) < 3.0
    ) & (
        np.abs(c) < 0.3
    ) & (
        np.isfinite(vel["v_est"])
    ) & (
        np.isfinite(sigma_mu)
    ) & (
        sigma_mu > 0
    )

    return {
        "velocities": vel["v_est"][mask],
        "sigma_v": vel["sigma_v"][mask],
        "n_sn": int(np.sum(mask)),
        "delta_mu": tripp["delta_mu"][mask],
        "colors": c[mask],
        "x1": x1[mask],
        "z": obs["z_obs"][mask],
        "host_mass": obs["log_mass"][mask],
    }
