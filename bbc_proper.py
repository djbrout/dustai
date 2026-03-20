"""Proper BBC implementation with bias correction simulation.

The BBC (BEAMS with Bias Corrections) framework:
1. Generate a large "biascor" simulation with the same scatter model
   and survey properties, including selection effects.
2. In (z, c, x1) bins, compute the mean difference between fitted
   and true light-curve parameters.
3. Apply these corrections to the data.

This is how real SN Ia cosmology is done — BBC corrects the mean
bias from selection effects and fitting biases. The Student-t
likelihood then handles the residual non-Gaussianity that BBC
cannot fix.
"""

import numpy as np
from scipy import interpolate


def generate_biascor_sim(n_biascor=50000, scatter_model='P23',
                          z_range=(0.02, 0.1), seed=None):
    """Generate a large bias correction simulation.

    This is a Monte Carlo of SNe with known truth, used to compute
    the expected bias in each (z, c, x1) bin.

    Parameters
    ----------
    n_biascor : int
        Number of SNe in the biascor simulation (typically 10-50x
        the data sample).
    scatter_model : str
        Must match the data scatter model.
    z_range : tuple
        Redshift range.
    seed : int or None

    Returns
    -------
    dict with true and fitted parameters for all biascor SNe.
    """
    from simulate import generate_mock

    mock = generate_mock(
        n_sn=n_biascor,
        scatter_model=scatter_model,
        z_range=z_range,
        seed=seed,
    )

    obs = mock['observables']
    tripp = mock['tripp_fit']

    return {
        # True values
        'm_b_true': obs['m_b_true'],
        'x1_true': obs['x1_true'],
        'c_true': obs['c_true'],
        'mu_cos': obs['mu_cos'],
        # Fitted values (true + noise, as seen by the analysis)
        'm_b_fit': obs['m_b'],
        'x1_fit': obs['x1'],
        'c_fit': obs['c'],
        # Observational properties
        'z_obs': obs['z_obs'],
        'z_cos': obs['z_cos'],
        'log_mass': obs['log_mass'],
        'sigma_mB': obs['sigma_mB'],
        'sigma_x1': obs['sigma_x1'],
        'sigma_c': obs['sigma_c'],
        # Tripp fit from the biascor sample
        'alpha': tripp['alpha'],
        'beta': tripp['beta'],
        'M_0': tripp['M_0'],
        'gamma': tripp['gamma'],
        'sigma_int': tripp['sigma_int'],
        'delta_mu': tripp['delta_mu'],
        'mu_obs': tripp['mu_obs'],
        'mu_cos_obs': tripp['mu_cos_obs'],
    }


def compute_bbc_corrections(biascor, n_z_bins=10, n_c_bins=8,
                             n_x1_bins=4, min_per_bin=20):
    """Compute BBC corrections from the biascor simulation.

    In each (z, c, x1) bin, compute:
      delta_mB = mean(mB_fit - mB_true)
      delta_x1 = mean(x1_fit - x1_true)
      delta_c  = mean(c_fit - c_true)
      delta_mu = mean(mu_tripp_fit - mu_cos)

    The delta_mu is the key correction: it captures the total bias
    in the distance modulus from selection, fitting, and the
    scatter model's non-Gaussian effects.

    Parameters
    ----------
    biascor : dict
        Output of generate_biascor_sim().
    n_z_bins, n_c_bins, n_x1_bins : int
        Binning.
    min_per_bin : int
        Minimum biascor SNe per bin.

    Returns
    -------
    dict with correction functions and metadata.
    """
    z = biascor['z_obs']
    c = biascor['c_fit']
    x1 = biascor['x1_fit']

    # Use the biascor Tripp parameters to compute mu_obs for biascor SNe
    alpha = biascor['alpha']
    beta = biascor['beta']
    M_0 = biascor['M_0']
    gamma = biascor['gamma']
    log_mass = biascor['log_mass']

    mass_step = np.where(log_mass > 10.0, -gamma / 2.0, gamma / 2.0)
    mu_tripp = (biascor['m_b_fit'] - M_0
                + alpha * biascor['x1_fit']
                - beta * biascor['c_fit']
                - mass_step)

    # True distance modulus at cosmological redshift
    mu_true = biascor['mu_cos']

    # Total distance modulus bias (includes PV contribution, but
    # averaged over the biascor sample the PV contribution averages
    # to zero since PV is uncorrelated with SN properties)
    delta_mu_biascor = mu_tripp - biascor['mu_cos_obs']

    # Bin edges
    z_edges = np.linspace(z.min() - 0.001, z.max() + 0.001, n_z_bins + 1)
    c_edges = np.linspace(-0.3, 0.3, n_c_bins + 1)
    x1_edges = np.linspace(-3.0, 3.0, n_x1_bins + 1)

    iz = np.clip(np.digitize(z, z_edges) - 1, 0, n_z_bins - 1)
    ic = np.clip(np.digitize(c, c_edges) - 1, 0, n_c_bins - 1)
    ix = np.clip(np.digitize(x1, x1_edges) - 1, 0, n_x1_bins - 1)

    # Compute mean bias in each 3D bin
    delta_mu_grid = np.full((n_z_bins, n_c_bins, n_x1_bins), np.nan)
    counts_grid = np.zeros((n_z_bins, n_c_bins, n_x1_bins), dtype=int)

    for i_z in range(n_z_bins):
        for i_c in range(n_c_bins):
            for i_x in range(n_x1_bins):
                mask = (iz == i_z) & (ic == i_c) & (ix == i_x)
                n = np.sum(mask)
                counts_grid[i_z, i_c, i_x] = n
                if n >= min_per_bin:
                    delta_mu_grid[i_z, i_c, i_x] = np.mean(
                        delta_mu_biascor[mask]
                    )

    # Fill NaN bins with nearest valid value
    from scipy.ndimage import distance_transform_edt
    nan_mask = np.isnan(delta_mu_grid)
    if nan_mask.any() and not nan_mask.all():
        _, nearest = distance_transform_edt(nan_mask,
                                             return_distances=True,
                                             return_indices=True)
        delta_mu_grid[nan_mask] = delta_mu_grid[
            nearest[0][nan_mask],
            nearest[1][nan_mask],
            nearest[2][nan_mask]
        ]
    elif nan_mask.all():
        delta_mu_grid[:] = 0.0

    # Build interpolator
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    c_centers = 0.5 * (c_edges[:-1] + c_edges[1:])
    x1_centers = 0.5 * (x1_edges[:-1] + x1_edges[1:])

    interp = interpolate.RegularGridInterpolator(
        (z_centers, c_centers, x1_centers), delta_mu_grid,
        method='linear', bounds_error=False, fill_value=None
    )

    return {
        'interp_delta_mu': interp,
        'grid': delta_mu_grid,
        'counts': counts_grid,
        'edges': {'z': z_edges, 'c': c_edges, 'x1': x1_edges},
        'tripp_params': {
            'alpha': alpha, 'beta': beta,
            'M_0': M_0, 'gamma': gamma,
        },
    }


def apply_bbc_to_mock(mock, bbc_corrections):
    """Apply BBC corrections to a data mock.

    Recomputes the Tripp distance moduli using the biascor Tripp
    parameters, subtracts the binned bias corrections, and converts
    to velocities.

    Parameters
    ----------
    mock : dict
        Output of generate_mock() or generate_uchuu_mock().
    bbc_corrections : dict
        Output of compute_bbc_corrections().

    Returns
    -------
    dict with corrected velocities and metadata.
    """
    obs = mock['observables']
    tripp = mock['tripp_fit']
    vel = mock['vel_est']

    x1 = obs['x1']
    c = obs['c']
    z_obs = obs['z_obs']
    log_mass = obs['log_mass']
    sigma_mu = tripp['sigma_mu']

    # Quality cuts (must match evaluate.py)
    mask = (
        (np.abs(x1) < 3.0)
        & (np.abs(c) < 0.3)
        & np.isfinite(vel['v_est'])
        & np.isfinite(sigma_mu)
        & (sigma_mu > 0)
    )

    z_m = z_obs[mask]
    c_m = c[mask]
    x1_m = x1[mask]
    delta_mu_m = tripp['delta_mu'][mask]
    sigma_mu_m = sigma_mu[mask]
    J_z_m = vel['J_z'][mask]

    # Look up BBC correction for each SN
    points = np.column_stack([z_m, c_m, x1_m])
    delta_mu_bbc = bbc_corrections['interp_delta_mu'](points)

    # Corrected Hubble residuals: subtract the expected bias
    delta_mu_corr = delta_mu_m - delta_mu_bbc

    # Convert to velocities
    v_corr = J_z_m * delta_mu_corr
    sigma_v_m = np.abs(J_z_m) * sigma_mu_m

    return {
        'velocities': v_corr,
        'sigma_v': sigma_v_m,
        'n_sn': int(np.sum(mask)),
        'delta_mu': delta_mu_corr,
        'colors': c_m,
        'x1': x1_m,
        'z': z_m,
        'host_mass': log_mass[mask],
    }


if __name__ == '__main__':
    import time

    print('Generating biascor simulation (50k SNe, P23)...')
    t0 = time.time()
    biascor = generate_biascor_sim(n_biascor=50000, scatter_model='P23',
                                    seed=99999)
    t1 = time.time()
    print(f'  Done in {t1-t0:.1f}s')

    print('Computing BBC corrections...')
    corrections = compute_bbc_corrections(biascor)
    t2 = time.time()
    print(f'  Done in {t2-t1:.1f}s')

    # Check the correction grid
    grid = corrections['grid']
    print(f'  Grid shape: {grid.shape}')
    print(f'  Mean correction: {np.nanmean(grid):.4f} mag')
    print(f'  Correction range: [{np.nanmin(grid):.4f}, {np.nanmax(grid):.4f}] mag')
    print(f'  Tripp params: alpha={corrections["tripp_params"]["alpha"]:.3f}, '
          f'beta={corrections["tripp_params"]["beta"]:.3f}')

    # Test on a data mock
    from simulate import generate_mock, COSMO
    print('\nGenerating test data mock (1500 SNe, P23)...')
    test_mock = generate_mock(n_sn=1500, scatter_model='P23', seed=42)

    print('Applying BBC corrections...')
    corrected = apply_bbc_to_mock(test_mock, corrections)
    print(f'  N_SN after cuts: {corrected["n_sn"]}')
    print(f'  delta_mu std (raw):       {test_mock["tripp_fit"]["delta_mu"].std():.4f}')
    print(f'  delta_mu std (corrected): {corrected["delta_mu"].std():.4f}')

    from scipy.stats import kurtosis, skew
    raw = test_mock['tripp_fit']['delta_mu']
    raw = raw[np.isfinite(raw)]
    print(f'  Skewness (raw):       {skew(raw):.3f}')
    print(f'  Skewness (corrected): {skew(corrected["delta_mu"]):.3f}')
    print(f'  Kurtosis (raw):       {kurtosis(raw, fisher=True):.3f}')
    print(f'  Kurtosis (corrected): {kurtosis(corrected["delta_mu"], fisher=True):.3f}')
