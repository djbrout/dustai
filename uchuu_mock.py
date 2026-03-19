"""Build realistic SN Ia peculiar velocity survey mocks from Uchuu halos.

Uses the Uchuu N-body simulation halo catalog (Ishiyama+2021) for
realistic correlated peculiar velocities, replacing the analytical
covariance-based mock generator.

The pipeline:
1. Load Uchuu halos at z=0 (Rockstar catalog, M200c > 10^12 Msun/h)
2. Place an observer at a random location in the 2 Gpc/h box
3. Select halos within the survey volume (z_min < z < z_max)
4. Assign SN Ia host galaxy properties (stellar mass from halo mass)
5. Generate SN Ia observables with P23 scatter on top of Uchuu velocities
6. Run Tripp fit and velocity estimation (reusing simulate.py machinery)
"""

import numpy as np
from pathlib import Path
import h5py

# Reuse cosmology and SN machinery from simulate.py
from simulate import (
    COSMO, C_LIGHT, H0_km_s_Mpc, M_B_FID,
    mu_cos, r_cos, Hz_cos, E_z,
    simulate_sn_observables, fit_tripp, estimate_velocities,
    _observed_redshift, _mass_step, _measurement_noise,
)

UCHUU_DIR = Path(__file__).parent / "uchuu"
BOX_SIZE = 2000.0  # Mpc/h


def load_uchuu_halos(data_dir=None, max_files=100):
    """Load all Uchuu halo catalog files and concatenate.

    Returns dict with keys: x, y, z (Mpc/h), vx, vy, vz (km/s),
    Mvir (Msun/h), upid (parent ID, -1 for hosts).
    """
    if data_dir is None:
        data_dir = UCHUU_DIR

    data_dir = Path(data_dir)
    files = sorted(data_dir.glob("halolist_z0p00_m200c1e12_*.h5"))
    if not files:
        raise FileNotFoundError(f"No Uchuu halo files found in {data_dir}")

    files = files[:max_files]

    arrays = {k: [] for k in ['x', 'y', 'z', 'vx', 'vy', 'vz', 'Mvir', 'upid']}

    for fpath in files:
        try:
            with h5py.File(fpath, 'r') as f:
                for k in arrays:
                    arrays[k].append(f[k][:])
        except OSError:
            # Skip truncated/corrupt files (still downloading)
            continue

    return {k: np.concatenate(v) for k, v in arrays.items()}


def select_survey_volume(halos, observer_pos, z_range=(0.02, 0.1)):
    """Select halos within the survey redshift range from an observer.

    Parameters
    ----------
    halos : dict
        Output of load_uchuu_halos().
    observer_pos : array-like, shape (3,)
        Observer position in Mpc/h (comoving).
    z_range : tuple
        (z_min, z_max) for the survey.

    Returns
    -------
    dict with keys:
        ra, dec (deg), z_cos, r_com (Mpc/h), v_los (km/s),
        log_mass (log10 stellar mass), mvir (Msun/h)
    """
    ox, oy, oz = observer_pos

    # Periodic box: wrap coordinates
    dx = halos['x'] - ox
    dy = halos['y'] - oy
    dz = halos['z'] - oz
    dx = dx - BOX_SIZE * np.round(dx / BOX_SIZE)
    dy = dy - BOX_SIZE * np.round(dy / BOX_SIZE)
    dz = dz - BOX_SIZE * np.round(dz / BOX_SIZE)

    # Comoving distance from observer
    r_com = np.sqrt(dx**2 + dy**2 + dz**2)

    # Convert comoving distance to cosmological redshift
    # Build lookup table: z -> r_com
    z_tab = np.linspace(0.001, 0.3, 2000)
    r_tab = r_cos(z_tab) * COSMO['h']  # Mpc -> Mpc/h
    from scipy.interpolate import interp1d
    z_from_r = interp1d(r_tab, z_tab, kind='cubic', fill_value='extrapolate')

    z_cos = z_from_r(r_com)

    # Select halos in redshift range and host halos only
    r_min = r_tab[np.searchsorted(z_tab, z_range[0])]
    r_max = r_tab[np.searchsorted(z_tab, z_range[1])]
    mask = (
        (r_com > r_min) & (r_com < r_max)
        & (halos['upid'] == -1)  # host halos only
        & (r_com > 1.0)  # avoid division by zero
    )

    # RA, Dec from Cartesian offsets
    ra = np.degrees(np.arctan2(dy, dx)) % 360.0
    dec = np.degrees(np.arcsin(np.clip(dz / np.maximum(r_com, 1e-10), -1, 1)))

    # Line-of-sight unit vector
    rhat_x = dx / np.maximum(r_com, 1e-10)
    rhat_y = dy / np.maximum(r_com, 1e-10)
    rhat_z = dz / np.maximum(r_com, 1e-10)

    # Project peculiar velocity onto line of sight
    v_los = (halos['vx'] * rhat_x + halos['vy'] * rhat_y
             + halos['vz'] * rhat_z)

    # Halo mass -> stellar mass (abundance matching approximation)
    # Using Behroozi+2013 SMHM relation
    mvir = halos['Mvir'][mask]
    log_mh = np.log10(mvir)
    # Simplified Behroozi+2013: log(M*/Msun) = f(log(Mh/Msun))
    # At z~0: log M* ~ 10.5 for log Mh ~ 12, rising slowly
    log_mstar = 10.5 + 0.5 * (log_mh - 12.0) - 0.1 * (log_mh - 12.0)**2
    log_mstar = np.clip(log_mstar, 8.0, 12.5)

    return {
        'ra': ra[mask],
        'dec': dec[mask],
        'z_cos': z_cos[mask],
        'r_com': r_com[mask],
        'v_los': v_los[mask],
        'log_mass': log_mstar,
        'mvir': mvir,
        'n_total': int(np.sum(mask)),
    }


def generate_uchuu_mock(n_sn=1500, scatter_model='P23', z_range=(0.02, 0.1),
                         seed=None, halos=None):
    """Generate a complete SN Ia PV survey mock using Uchuu velocities.

    Parameters
    ----------
    n_sn : int
        Number of SNe to select.
    scatter_model : str
        Intrinsic scatter model: 'COH', 'G10', 'C11', 'P23'.
    z_range : tuple
        (z_min, z_max) redshift range.
    seed : int or None
        Random seed.
    halos : dict or None
        Pre-loaded halo catalog (to avoid re-reading files).

    Returns
    -------
    dict : same format as simulate.generate_mock()
    """
    rng = np.random.default_rng(seed)

    # Load halos if not provided
    if halos is None:
        halos = load_uchuu_halos()

    # Place observer at random position in box
    observer_pos = rng.uniform(0, BOX_SIZE, 3)

    # Select halos in survey volume
    survey = select_survey_volume(halos, observer_pos, z_range=z_range)

    if survey['n_total'] < n_sn:
        raise ValueError(
            f"Only {survey['n_total']} halos in survey volume, need {n_sn}. "
            f"Try a wider z_range or use the full Uchuu catalog."
        )

    # Randomly select n_sn halos (weighted by volumetric rate)
    # Rate propto (1+z)^1.7, weighted by survey volume element
    z_cos = survey['z_cos']
    weights = (1.0 + z_cos)**1.7
    weights /= weights.sum()
    idx = rng.choice(len(z_cos), size=n_sn, replace=False, p=weights)

    # Build positions dict matching simulate.py format
    positions = {
        'ra': survey['ra'][idx],
        'dec': survey['dec'][idx],
        'z_cos': survey['z_cos'][idx],
        'log_mass': survey['log_mass'][idx],
    }

    # True peculiar velocities from Uchuu
    v_pec = survey['v_los'][idx]

    # Generate SN observables using simulate.py machinery
    seed_obs = int(rng.integers(0, 2**31))
    observables = simulate_sn_observables(positions, v_pec,
                                           scatter_model=scatter_model,
                                           seed=seed_obs)

    # Tripp fit
    tripp_fit = fit_tripp(observables)

    # Velocity estimates
    vel_est = estimate_velocities(tripp_fit['delta_mu'],
                                   observables['z_obs'],
                                   tripp_fit['sigma_mu'])

    config = {
        'n_sn': n_sn,
        'scatter_model': scatter_model,
        'fsigma8_fid': COSMO['fsigma8'],
        'z_range': z_range,
        'seed': seed,
        'cosmology': COSMO.copy(),
        'source': 'uchuu',
        'observer_pos': observer_pos.tolist(),
    }

    return {
        'positions': positions,
        'v_pec': v_pec,
        'observables': observables,
        'tripp_fit': tripp_fit,
        'vel_est': vel_est,
        'config': config,
    }


if __name__ == '__main__':
    import time

    print("Loading Uchuu halos...")
    t0 = time.time()
    halos = load_uchuu_halos()
    t1 = time.time()
    print(f"  Loaded {len(halos['x'])} halos in {t1-t0:.1f}s")
    print(f"  Host halos: {np.sum(halos['upid']==-1)}")

    print("\nGenerating mock...")
    t2 = time.time()
    mock = generate_uchuu_mock(n_sn=1500, scatter_model='P23', seed=42,
                                halos=halos)
    t3 = time.time()
    print(f"  Generated in {t3-t2:.1f}s")

    pos = mock['positions']
    v = mock['v_pec']
    tripp = mock['tripp_fit']
    vel = mock['vel_est']

    print(f"\n  z range: [{pos['z_cos'].min():.4f}, {pos['z_cos'].max():.4f}]")
    print(f"  v_pec std: {v.std():.1f} km/s")
    print(f"  Tripp beta: {tripp['beta']:.3f}")
    print(f"  delta_mu std: {tripp['delta_mu'].std():.4f}")
    print(f"  Correlation(v_true, v_est): {np.corrcoef(v, vel['v_est'])[0,1]:.3f}")
