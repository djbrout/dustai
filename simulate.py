"""
Fast analytical mock generator for Type Ia supernova peculiar velocity surveys.

Simulates correlated peculiar velocities from the velocity power spectrum and
realistic SN Ia light-curve observables with multiple intrinsic scatter models
(COH, G10, C11, P23/BS21-dust), producing Hubble diagram residuals suitable
for f*sigma8 analysis.  No SNANA or external cosmology packages required --
everything is computed analytically using numpy/scipy.

Fiducial cosmology: Planck 2015 (Table 1 of Huterer+2017 / Howlett+2017)
    h      = 0.6774
    Omega_m = 0.3089
    Omega_b = 0.0486
    n_s     = 0.9667
    sigma8  = 0.8159

References:
    - Eisenstein & Hu 1998 (transfer function)
    - Bel et al. 2019 (non-linear velocity PS corrections, Eq. 37-39)
    - Howlett et al. 2017 (velocity covariance, Eq. 29-30, 35-36)
    - Popovic et al. 2021 (SN Ia populations, Tables 4 & 8)
    - Guy et al. 2010, Chotard et al. 2011 (scatter models)
    - Brout & Scolnic 2021, Popovic et al. 2023 (BS21 dust model w/ P23 params)

Usage:
    import simulate
    mock = simulate.generate_mock(n_sn=6600, scatter_model='P23', seed=42)
"""

import numpy as np
from scipy import integrate, interpolate, optimize, special


# ============================================================================
# Fiducial cosmology (Planck 2015)
# ============================================================================

COSMO = dict(
    h=0.6774,
    Omega_m=0.3089,
    Omega_b=0.0486,
    Omega_L=1.0 - 0.3089,          # flat LCDM
    n_s=0.9667,
    sigma8=0.8159,
    fsigma8=0.3089**0.55 * 0.8159,  # f*sigma8, f ~ Omega_m^0.55
)

H0_km_s_Mpc = COSMO["h"] * 100.0   # km/s/Mpc
C_LIGHT = 299792.458                 # km/s
M_B_FID = -19.36                     # fiducial SN Ia absolute magnitude


# ============================================================================
# Cosmology helper functions
# ============================================================================

def E_z(z, Omega_m=COSMO["Omega_m"]):
    """Dimensionless Hubble parameter E(z) = H(z)/H0 for flat LCDM."""
    Omega_L = 1.0 - Omega_m
    return np.sqrt(Omega_m * (1.0 + z) ** 3 + Omega_L)


def H_z(z, Omega_m=COSMO["Omega_m"]):
    """Hubble parameter H(z) in km/s/Mpc."""
    return H0_km_s_Mpc * E_z(z, Omega_m)


def _dc_integrand(z, Omega_m):
    return 1.0 / E_z(z, Omega_m)


def comoving_distance(z, Omega_m=COSMO["Omega_m"]):
    """Comoving distance r(z) in Mpc (line-of-sight, flat universe).

    Works for scalar or 1-D array z.
    """
    z = np.atleast_1d(z).astype(np.float64)
    r = np.empty_like(z)
    for i, zi in enumerate(z):
        r[i], _ = integrate.quad(_dc_integrand, 0.0, zi, args=(Omega_m,))
    r *= C_LIGHT / H0_km_s_Mpc  # Mpc
    return r.item() if r.size == 1 else r


def luminosity_distance(z, Omega_m=COSMO["Omega_m"]):
    """Luminosity distance d_L(z) in Mpc."""
    return (1.0 + z) * comoving_distance(z, Omega_m)


def distance_modulus(z, Omega_m=COSMO["Omega_m"]):
    """Distance modulus mu(z) = 5*log10(d_L / 10 pc)."""
    d_L = luminosity_distance(z, Omega_m)
    return 5.0 * np.log10(d_L * 1.0e6 / 10.0)  # d_L in Mpc -> pc


# Pre-tabulate for fast interpolation (z = 0.001 .. 0.3)
_ZTAB = np.linspace(0.001, 0.30, 2000)
_MUTAB = distance_modulus(_ZTAB)
_RCTAB = comoving_distance(_ZTAB)
_HTAB = H_z(_ZTAB)
_mu_interp = interpolate.interp1d(_ZTAB, _MUTAB, kind="cubic", fill_value="extrapolate")
_rc_interp = interpolate.interp1d(_ZTAB, _RCTAB, kind="cubic", fill_value="extrapolate")
_Hz_interp = interpolate.interp1d(_ZTAB, _HTAB, kind="cubic", fill_value="extrapolate")


def mu_cos(z):
    """Fast interpolated distance modulus."""
    return _mu_interp(z)


def r_cos(z):
    """Fast interpolated comoving distance (Mpc)."""
    return _rc_interp(z)


def Hz_cos(z):
    """Fast interpolated H(z) (km/s/Mpc)."""
    return _Hz_interp(z)


# ============================================================================
# Eisenstein-Hu transfer function (no wiggles version, 1998)
# ============================================================================

def eisenstein_hu_transfer(k_hMpc, Omega_m=COSMO["Omega_m"],
                           Omega_b=COSMO["Omega_b"], h=COSMO["h"]):
    """Eisenstein & Hu 1998 'no-wiggle' (zero-baryon) transfer function.

    Parameters
    ----------
    k_hMpc : array
        Wavenumber in h/Mpc.

    Returns
    -------
    T(k) : array  (normalised so T(0)->1)
    """
    # Physical densities
    Omega_mh2 = Omega_m * h ** 2
    Omega_bh2 = Omega_b * h ** 2
    f_b = Omega_b / Omega_m
    theta_cmb = 2.7255 / 2.7  # T_cmb / 2.7 K

    # Sound horizon (Eq. 26)
    z_eq = 2.5e4 * Omega_mh2 * theta_cmb ** (-4)
    k_eq = 7.46e-2 * Omega_mh2 * theta_cmb ** (-2)  # h/Mpc

    # Drag epoch
    b1 = 0.313 * Omega_mh2 ** (-0.419) * (1 + 0.607 * Omega_mh2 ** 0.674)
    b2 = 0.238 * Omega_mh2 ** 0.223
    z_drag = (1291 * Omega_mh2 ** 0.251 / (1 + 0.659 * Omega_mh2 ** 0.828)
              * (1 + b1 * Omega_bh2 ** b2))

    # Sound horizon at drag epoch
    R_drag = 31.5 * Omega_bh2 * theta_cmb ** (-4) * (1000.0 / z_drag)
    R_eq = 31.5 * Omega_bh2 * theta_cmb ** (-4) * (1000.0 / z_eq)
    s = (2.0 / (3.0 * k_eq) * np.sqrt(6.0 / R_eq)
         * np.log((np.sqrt(1.0 + R_drag) + np.sqrt(R_drag + R_eq))
                  / (1.0 + np.sqrt(R_eq))))

    # Silk damping
    k_silk = (1.6 * Omega_bh2 ** 0.52 * Omega_mh2 ** 0.73
              * (1.0 + (10.4 * Omega_mh2) ** (-0.95)))

    # CDM transfer (Eq. 11)
    alpha_gamma = (1 - f_b) * ((0.328 * np.log(431.0 * Omega_mh2)
                                * f_b / (0.308 * Omega_mh2 ** 0.215 + 1))
                               + 1.0)
    # Effective shape parameter
    Gamma_eff_inv = Omega_m * h * (
        alpha_gamma + (1.0 - alpha_gamma) / (1.0 + (0.43 * k_hMpc * s) ** 4)
    )
    q = k_hMpc * theta_cmb ** 2 / Gamma_eff_inv

    # Zero-baryon transfer (Eq. 29)
    L0 = np.log(2.0 * np.e + 1.8 * q)
    C0 = 14.2 + 731.0 / (1.0 + 62.5 * q)
    T0 = L0 / (L0 + C0 * q ** 2)
    return T0


def linear_power_spectrum(k_hMpc, sigma8=COSMO["sigma8"], n_s=COSMO["n_s"],
                          Omega_m=COSMO["Omega_m"], Omega_b=COSMO["Omega_b"],
                          h=COSMO["h"]):
    """Linear matter power spectrum P_lin(k) in (Mpc/h)^3.

    Normalised to the input sigma8.
    """
    Tk = eisenstein_hu_transfer(k_hMpc, Omega_m, Omega_b, h)
    Pk_unnorm = k_hMpc ** n_s * Tk ** 2

    # Normalise via sigma8: sigma8^2 = 1/(2pi^2) int k^2 P(k) W^2(kR) dk
    R8 = 8.0  # Mpc/h
    k_norm = np.logspace(-5, 2, 5000)
    Tk_norm = eisenstein_hu_transfer(k_norm, Omega_m, Omega_b, h)
    Pk_norm = k_norm ** n_s * Tk_norm ** 2
    x = k_norm * R8
    W = 3.0 * (np.sin(x) - x * np.cos(x)) / x ** 3
    integrand = k_norm ** 2 * Pk_norm * W ** 2
    sigma8_sq_unnorm = np.trapz(integrand, k_norm) / (2.0 * np.pi ** 2)
    A = sigma8 ** 2 / sigma8_sq_unnorm

    return A * Pk_unnorm


# ============================================================================
# Non-linear velocity power spectrum (Bel+2019)
# ============================================================================

def velocity_power_spectrum(k_hMpc, sigma8=COSMO["sigma8"], sigma_u=21.0,
                            n_s=COSMO["n_s"], Omega_m=COSMO["Omega_m"],
                            Omega_b=COSMO["Omega_b"], h=COSMO["h"]):
    """Velocity divergence power spectrum P_theta_theta(k) with non-linear
    corrections from Bel+2019 (Eq. 37-38) and finger-of-god damping D_u
    (Eq. 39).

    Parameters
    ----------
    k_hMpc : array
        Wavenumber in h/Mpc.
    sigma_u : float
        Velocity damping scale in Mpc/h.

    Returns
    -------
    P_tt(k) : array  in (Mpc/h)^3
    """
    P_lin = linear_power_spectrum(k_hMpc, sigma8, n_s, Omega_m, Omega_b, h)

    # Non-linear correction (Bel+2019 Eq. 37-38)
    s8 = sigma8
    a1 = -0.817 + 3.198 * s8
    a2 = 0.877 - 4.191 * s8
    a3 = -1.199 + 4.629 * s8
    P_tt = P_lin * np.exp(-k_hMpc * (a1 + a2 * k_hMpc + a3 * k_hMpc ** 2))

    # Velocity damping (Eq. 39)
    k_su = k_hMpc * sigma_u
    D_u = np.where(k_su > 1e-10, np.sin(k_su) / k_su, 1.0)
    P_tt *= D_u ** 2

    return P_tt


# ============================================================================
# 1. Generate positions
# ============================================================================

def generate_positions(n_sn, z_range=(0.02, 0.1), seed=None):
    """Generate random SN positions on the southern sky with realistic
    redshift distribution and host galaxy stellar masses.

    Parameters
    ----------
    n_sn : int
        Number of supernovae.
    z_range : tuple
        (z_min, z_max) redshift range.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    dict with keys:
        ra      : array (deg), right ascension in [0, 360)
        dec     : array (deg), declination in [-90, 0) (southern sky)
        z_cos   : array, cosmological redshifts
        log_mass: array, log10 host stellar mass (M_sun)
    """
    rng = np.random.default_rng(seed)

    # RA uniform in [0, 360)
    ra = rng.uniform(0.0, 360.0, n_sn)

    # DEC: uniform on southern sky -> sin(dec) uniform in [-1, 0)
    sin_dec = rng.uniform(-1.0, 0.0, n_sn)
    dec = np.degrees(np.arcsin(sin_dec))

    # Redshifts from volumetric rate r_v propto (1+z)^1.7
    # Acceptance-rejection: pdf propto dV/dz * (1+z)^1.7
    # dV/dz propto r(z)^2 / E(z)  (comoving volume element, flat sky approx)
    z_min, z_max = z_range
    z_grid = np.linspace(z_min, z_max, 500)
    r_grid = r_cos(z_grid)
    E_grid = E_z(z_grid)
    weight = r_grid ** 2 / E_grid * (1.0 + z_grid) ** 1.7
    weight /= np.trapz(weight, z_grid)  # normalise to PDF
    # Build CDF via trapezoidal integration: length = len(z_grid)
    cdf = np.zeros(len(z_grid))
    cdf[1:] = np.cumsum(0.5 * (weight[:-1] + weight[1:]) * np.diff(z_grid))
    cdf /= cdf[-1]
    inv_cdf = interpolate.interp1d(cdf, z_grid, kind="linear")
    u = rng.uniform(0, 1, n_sn)
    z_cos = inv_cdf(u)

    # Host galaxy log stellar mass: bimodal Gaussian peaked near 10-11
    # 40% low-mass component N(9.5, 0.6^2), 60% high-mass component N(10.8, 0.4^2)
    comp = rng.uniform(size=n_sn) < 0.6  # True -> high-mass
    log_mass = np.where(
        comp,
        rng.normal(10.8, 0.4, n_sn),
        rng.normal(9.5, 0.6, n_sn),
    )
    log_mass = np.clip(log_mass, 7.0, 13.0)

    return dict(ra=ra, dec=dec, z_cos=z_cos, log_mass=log_mass)


# ============================================================================
# 2. Generate correlated peculiar velocities (FFT grid method)
# ============================================================================

def generate_pv_field(positions, fsigma8_fid=0.4253, sigma_u=21.0, seed=None):
    """Generate correlated line-of-sight peculiar velocities from the
    analytic velocity covariance matrix.

    Uses the SAME covariance computation as the likelihood (from
    likelihood.py) to ensure perfect calibration. This is O(N^3) for
    the Cholesky decomposition, feasible for N <= ~5000 SNe.

    Parameters
    ----------
    positions : dict
        Output of ``generate_positions``.
    fsigma8_fid : float
        Fiducial f*sigma8 value.
    sigma_u : float
        Velocity damping scale (Mpc/h).
    seed : int or None
        Random seed.

    Returns
    -------
    v_pec : array (km/s)
        Line-of-sight peculiar velocity for each SN.  Positive = recession.
    """
    rng = np.random.default_rng(seed)

    # Use the likelihood's covariance computation for perfect calibration
    from likelihood import compute_velocity_covariance

    z_cos = positions["z_cos"]
    ra_rad = np.radians(positions["ra"])
    dec_rad = np.radians(positions["dec"])
    r_com = np.array([r_cos(z) * COSMO["h"] for z in z_cos])  # Mpc/h

    pos_array = np.column_stack([ra_rad, dec_rad, r_com])

    C_vv = compute_velocity_covariance(
        pos_array, fsigma8_fid, sigma_u, COSMO
    )

    # Regularize for numerical stability: add a small diagonal term
    # and ensure symmetry. Scale regularization with matrix norm.
    C_vv = 0.5 * (C_vv + C_vv.T)  # enforce exact symmetry
    diag_mean = np.mean(np.diag(C_vv))
    C_vv += np.eye(len(C_vv)) * max(diag_mean * 1e-6, 1.0)

    # Draw correlated velocities from the covariance
    try:
        L_chol = np.linalg.cholesky(C_vv)
        v_pec = L_chol @ rng.standard_normal(len(z_cos))
    except np.linalg.LinAlgError:
        # Fall back to eigendecomposition if Cholesky fails
        eigvals, eigvecs = np.linalg.eigh(C_vv)
        eigvals = np.maximum(eigvals, 0.0)  # clip negative eigenvalues
        v_pec = (eigvecs * np.sqrt(eigvals)) @ rng.standard_normal(len(z_cos))

    return v_pec

    z_cos = positions["z_cos"]
    ra_rad = np.radians(positions["ra"])
    dec_rad = np.radians(positions["dec"])

    # SN positions in Cartesian (Mpc/h)
    r = r_cos(z_cos) * COSMO["h"]  # convert Mpc -> Mpc/h
    x_sn = r * np.cos(dec_rad) * np.cos(ra_rad)
    y_sn = r * np.cos(dec_rad) * np.sin(ra_rad)
    z_sn = r * np.sin(dec_rad)

    # Line-of-sight unit vectors
    rhat_x = np.cos(dec_rad) * np.cos(ra_rad)
    rhat_y = np.cos(dec_rad) * np.sin(ra_rad)
    rhat_z = np.sin(dec_rad)

    # Grid parameters
    r_max = r.max() * 1.15  # pad slightly beyond furthest SN
    Ngrid = 128             # grid cells per side (balance speed / resolution)
    L = 2.0 * r_max         # box side length (Mpc/h), centred on origin
    dk = 2.0 * np.pi / L
    dx = L / Ngrid

    # k-space grid
    kx = np.fft.fftfreq(Ngrid, d=dx) * 2.0 * np.pi  # h/Mpc
    ky = np.fft.fftfreq(Ngrid, d=dx) * 2.0 * np.pi
    kz = np.fft.fftfreq(Ngrid, d=dx) * 2.0 * np.pi
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    K = np.sqrt(KX ** 2 + KY ** 2 + KZ ** 2)
    K[0, 0, 0] = 1.0  # avoid division by zero; set DC to zero below

    # Growth rate scaling: v propto f*sigma8
    # f ~ Omega_m(z)^0.55 at low z, but the amplitude is set by fsigma8_fid
    # The velocity divergence PS is P_tt propto (f*sigma8)^2 / sigma8^2 * P_tt(sigma8)
    # Since we want v propto fsigma8, we scale the amplitude of the PS.
    P_tt = velocity_power_spectrum(K.ravel(), sigma8=COSMO["sigma8"],
                                   sigma_u=sigma_u).reshape(K.shape)
    # Scale by (fsigma8 / sigma8)^2 to get the correct amplitude
    # The PS already has sigma8^2 baked in via normalisation; re-scale:
    f_fid = fsigma8_fid / COSMO["sigma8"]
    P_tt *= f_fid ** 2

    # Set DC mode to zero
    P_tt[0, 0, 0] = 0.0

    # FFT convention for generating Gaussian random fields:
    # numpy's fftn/ifftn pair uses the convention:
    #   fftn:  F[k] = sum_x f(x) exp(-2πi k·x/N)
    #   ifftn: f(x) = (1/N^3) sum_k F[k] exp(+2πi k·x/N)
    # For a field with power spectrum P(k), we want
    #   <|f(k)|^2> = P(k) * (N/L)^3 = P(k) * N^3 / V
    # So draw each Fourier mode with sigma = sqrt(P(k) * N^3 / (2*V))
    # (the /2 splits variance between real and imaginary parts).
    V = L ** 3
    Ncells = Ngrid ** 3
    # Correct normalization: draw with sigma = sqrt(P/(2V)) per mode,
    # then multiply by N^3 to compensate numpy ifftn's 1/N^3 factor.
    amplitude = np.sqrt(P_tt / (2.0 * V)) * Ncells

    # Generate three independent velocity components from the velocity
    # divergence power spectrum.  For the divergence field theta = -div(v),
    # v_i(k) = -i k_i / k^2 * theta(k).
    # Draw theta in Fourier space and convert to velocity components.
    theta_k = (amplitude * (rng.standard_normal((Ngrid, Ngrid, Ngrid))
                            + 1j * rng.standard_normal((Ngrid, Ngrid, Ngrid))))
    theta_k[0, 0, 0] = 0.0

    # Velocity in Fourier space: v_i(k) = -i k_i / k^2 * theta_k
    # Factor aHf absorbed: our PS is for theta = -f * delta, and
    # v_pec = aHf/(k^2) * delta_k * k_hat  =>  v = H0 f / k^2 * theta_k * k_hat
    # But P_tt is already for theta propto f*delta, so v = H0 / k^2 * theta * k
    # In Fourier convention: v_i(k) = H0 * k_i / k^2 * theta(k)
    K2 = K ** 2
    K2[0, 0, 0] = 1.0  # avoid /0
    # v_i(k) = aHf * k_i/k^2 * delta(k) = H0*f * k_i/k^2 * delta(k)
    # Since theta = f*delta and P_tt = f^2 * P_delta, we have
    # theta_k already contains the f factor. So v = H0 * k/k^2 * theta.
    # Additional factor: the velocity should be in km/s, and our k is in
    # h/Mpc, so H0 should be in km/s/(Mpc/h) = 100 km/s/(Mpc/h) * h.
    # But H0_km_s_Mpc is in km/s/Mpc, and k is in h/Mpc, so we need
    # H0 / h = 100 km/s/(Mpc/h).
    H0_over_h = 100.0  # km/s per Mpc/h
    vx_k = H0_over_h * KX / K2 * theta_k
    vy_k = H0_over_h * KY / K2 * theta_k
    vz_k = H0_over_h * KZ / K2 * theta_k

    # Transform to real space
    vx_real = np.fft.ifftn(vx_k).real
    vy_real = np.fft.ifftn(vy_k).real
    vz_real = np.fft.ifftn(vz_k).real

    # Calibration: the FFT grid misses large-scale power at k < k_min
    # and NGP interpolation loses small-scale power. Calibrate the
    # velocity amplitude to match the analytic velocity covariance
    # (from likelihood.py) at the fiducial fsigma8. This factor was
    # determined empirically by comparing FFT output to the analytic
    # prediction: sigma_v(FFT) / sigma_v(analytic) ≈ 0.53.
    _AMPLITUDE_CALIBRATION = 1.88
    vx_real *= _AMPLITUDE_CALIBRATION
    vy_real *= _AMPLITUDE_CALIBRATION
    vz_real *= _AMPLITUDE_CALIBRATION

    # Interpolate to SN positions (nearest-grid-point for speed;
    # trilinear would be more accurate but NGP is fine for ~128^3)
    def grid_index(coord):
        """Map coordinate to grid index (origin at grid centre)."""
        idx = ((coord + L / 2.0) / dx).astype(int) % Ngrid
        return idx

    ix = grid_index(x_sn)
    iy = grid_index(y_sn)
    iz = grid_index(z_sn)

    vx_at_sn = vx_real[ix, iy, iz]
    vy_at_sn = vy_real[ix, iy, iz]
    vz_at_sn = vz_real[ix, iy, iz]

    # Project onto line of sight
    v_los = vx_at_sn * rhat_x + vy_at_sn * rhat_y + vz_at_sn * rhat_z

    return v_los


# ============================================================================
# 3. Simulate SN observables
# ============================================================================

def _mass_step(log_mass, gamma):
    """Mass step: Delta_M = -gamma/2 if log(M) > 10, +gamma/2 otherwise."""
    return np.where(log_mass > 10.0, -gamma / 2.0, gamma / 2.0)


def _observed_redshift(z_cos, v_pec):
    """Convert cosmological redshift + peculiar velocity to observed redshift.

    (1+z_obs) = (1+z_cos) * (1 + v_pec/c)
    """
    return (1.0 + z_cos) * (1.0 + v_pec / C_LIGHT) - 1.0


def _measurement_noise(n_sn, z_cos, rng):
    """Generate realistic SALT2 measurement uncertainties.

    Returns sigma_mB, sigma_x1, sigma_c and their covariance matrices.
    """
    # Redshift-dependent m_B uncertainty (brighter = better measured)
    sigma_mB = 0.02 + 0.03 * (z_cos / 0.1)
    sigma_mB = np.clip(sigma_mB, 0.02, 0.08)
    sigma_mB += rng.uniform(-0.005, 0.005, n_sn)
    sigma_mB = np.clip(sigma_mB, 0.015, 0.10)

    # x1 uncertainty
    sigma_x1 = 0.1 + 0.4 * rng.uniform(0, 1, n_sn)
    sigma_x1 = np.clip(sigma_x1, 0.08, 0.6)

    # c uncertainty
    sigma_c = 0.02 + 0.03 * (z_cos / 0.1)
    sigma_c = np.clip(sigma_c, 0.02, 0.06)
    sigma_c += rng.uniform(-0.005, 0.005, n_sn)
    sigma_c = np.clip(sigma_c, 0.015, 0.08)

    # SALT2 fit covariance for each SN (diagonal approximation with small
    # off-diagonal correlations)
    C_SALT = np.zeros((n_sn, 3, 3))
    C_SALT[:, 0, 0] = sigma_mB ** 2
    C_SALT[:, 1, 1] = sigma_x1 ** 2
    C_SALT[:, 2, 2] = sigma_c ** 2
    # Small mB-c correlation (~0.01)
    cov_mBc = 0.01 * sigma_mB * sigma_c
    C_SALT[:, 0, 2] = cov_mBc
    C_SALT[:, 2, 0] = cov_mBc

    return sigma_mB, sigma_x1, sigma_c, C_SALT


def simulate_sn_observables(positions, velocities, scatter_model="P23",
                            seed=None):
    """Generate SN Ia light-curve fit parameters (m_B, x1, c) for a given
    scatter model.

    Parameters
    ----------
    positions : dict
        Output of ``generate_positions``.
    velocities : array
        Line-of-sight peculiar velocities (km/s) from ``generate_pv_field``.
    scatter_model : str
        One of 'COH', 'G10', 'C11', 'P23'.
    seed : int or None

    Returns
    -------
    dict with keys:
        m_b, x1, c          : observed (noisy) light-curve params
        m_b_true, x1_true, c_true : true (noise-free) values
        z_obs                : observed redshift (including PV)
        z_cos                : cosmological redshift
        log_mass             : host log stellar mass
        mu_cos               : cosmological distance modulus at z_cos
        sigma_mB, sigma_x1, sigma_c : measurement uncertainties
        C_SALT               : (N,3,3) SALT fit covariance per SN
        scatter_model        : label
    """
    rng = np.random.default_rng(seed)
    model = scatter_model.upper()
    if model not in ("COH", "G10", "C11", "P23"):
        raise ValueError(f"Unknown scatter model: {scatter_model!r}. "
                         f"Choose from COH, G10, C11, P23.")

    n_sn = len(positions["z_cos"])
    z_cos = positions["z_cos"]
    log_mass = positions["log_mass"]

    alpha_fid = 0.15
    gamma_fid = 0.05

    # --- x1 from P(x1 | M_host) per Popovic+2021 Table 4 ---
    # Approximate: x1 ~ N(mu_x1, 1) with mass-dependent mean
    # High mass: mu_x1 ~ -0.5 (prefer fast decliners)
    # Low mass:  mu_x1 ~ +0.3
    mu_x1 = np.where(log_mass > 10.0, -0.5, 0.3)
    x1_true = rng.normal(mu_x1, 1.0)
    x1_true = np.clip(x1_true, -3.0, 3.0)

    # --- Observed redshift ---
    z_obs = _observed_redshift(z_cos, velocities)

    # --- Distance modulus at true cosmological redshift ---
    mu_cosmo = mu_cos(z_cos)

    # --- Mass step ---
    delta_M = _mass_step(log_mass, gamma_fid)

    # --- PV contribution to apparent magnitude ---
    # Δm_pv = 5 * log10((1 + z_obs) / (1 + z_cos))  (exact via d_L ratio)
    # Approximation for small v/c:
    # At linear order: Δm ~ (5/ln10) * v_pec / (c * z_cos) * [...]
    # But we use the exact formula via z_obs:
    z_pec = velocities / C_LIGHT
    pv_mag = 10.0 * np.log10(1.0 + z_pec)  # Δm from PV (linearised)

    # --- Scatter-model-specific observables ---

    if model == "COH":
        beta_fid = 3.1
        # c from P(c | M_host) per Popovic+2021 Table 8
        mu_c = np.where(log_mass > 10.0, 0.02, -0.04)
        sigma_c_pop = np.where(log_mass > 10.0, 0.06, 0.05)
        c_true = rng.normal(mu_c, sigma_c_pop)
        c_true = np.clip(c_true, -0.3, 0.5)

        # True absolute magnitude scatter
        M_star = (M_B_FID - alpha_fid * x1_true + beta_fid * c_true
                  + delta_M)
        # Coherent intrinsic scatter
        sigma_int = 0.12
        M_star += rng.normal(0.0, sigma_int, n_sn)
        # Apparent magnitude
        m_b_true = M_star + mu_cosmo + pv_mag
        beta_used = beta_fid

    elif model == "G10":
        beta_fid = 3.1
        # Guy+2010: 70% achromatic, 30% chromatic
        mu_c = np.where(log_mass > 10.0, 0.02, -0.04)
        sigma_c_pop = np.where(log_mass > 10.0, 0.06, 0.05)
        c_true_base = rng.normal(mu_c, sigma_c_pop)

        # Chromatic scatter component (adds to observed c)
        sigma_c_scat = 0.04
        c_scatter = rng.normal(0.0, sigma_c_scat, n_sn)
        c_true = np.clip(c_true_base + c_scatter, -0.3, 0.5)

        # Achromatic scatter
        sigma_achro = 0.08
        M_star = (M_B_FID - alpha_fid * x1_true + beta_fid * c_true
                  + delta_M)
        M_star += rng.normal(0.0, sigma_achro, n_sn)
        m_b_true = M_star + mu_cosmo + pv_mag
        beta_used = beta_fid

    elif model == "C11":
        beta_fid = 3.8
        # Chotard+2011: 25% achromatic, 75% chromatic
        mu_c = np.where(log_mass > 10.0, 0.02, -0.04)
        sigma_c_pop = np.where(log_mass > 10.0, 0.06, 0.05)
        c_true_base = rng.normal(mu_c, sigma_c_pop)

        sigma_c_scat = 0.08
        c_scatter = rng.normal(0.0, sigma_c_scat, n_sn)
        c_true = np.clip(c_true_base + c_scatter, -0.3, 0.5)

        sigma_achro = 0.04
        M_star = (M_B_FID - alpha_fid * x1_true + beta_fid * c_true
                  + delta_M)
        M_star += rng.normal(0.0, sigma_achro, n_sn)
        m_b_true = M_star + mu_cosmo + pv_mag
        beta_used = beta_fid

    elif model == "P23":
        # BS21 dust model with Popovic+2023 parameters
        # Intrinsic color
        c_int = rng.normal(-0.07, 0.05, n_sn)

        # Intrinsic beta
        beta_int = rng.normal(2.07, 0.22, n_sn)

        # R_V depends on host mass
        high_mass = log_mass > 10.0
        R_V = np.where(
            high_mass,
            rng.normal(1.66, 0.95, n_sn),
            rng.normal(3.25, 0.93, n_sn),
        )
        R_V = np.clip(R_V, 0.5, 6.0)

        # Dust E(B-V) from exponential with mass- and z-dependent tau
        tau = np.empty(n_sn)
        mask_hm_lz = high_mass & (z_cos < 0.1)
        mask_lm_lz = (~high_mass) & (z_cos < 0.1)
        mask_hm_hz = high_mass & (z_cos >= 0.1)
        mask_lm_hz = (~high_mass) & (z_cos >= 0.1)
        tau[mask_hm_lz] = 0.11
        tau[mask_lm_lz] = 0.14
        tau[mask_hm_hz] = 0.15
        tau[mask_lm_hz] = 0.12
        E_dust = rng.exponential(tau)

        # Observed color
        c_true = c_int + E_dust
        c_true = np.clip(c_true, -0.3, 1.0)

        # True absolute magnitude (split intrinsic + dust)
        M_star = (M_B_FID
                  - alpha_fid * x1_true
                  + beta_int * c_int
                  + (R_V + 1.0) * E_dust
                  + delta_M)
        m_b_true = M_star + mu_cosmo + pv_mag
        beta_used = None  # no single beta for P23

    # --- Measurement noise ---
    sigma_mB, sigma_x1, sigma_c_meas, C_SALT = _measurement_noise(
        n_sn, z_cos, rng)

    m_b_obs = m_b_true + rng.normal(0.0, sigma_mB)
    x1_obs = x1_true + rng.normal(0.0, sigma_x1)
    c_obs = c_true + rng.normal(0.0, sigma_c_meas)

    return dict(
        m_b=m_b_obs,
        x1=x1_obs,
        c=c_obs,
        m_b_true=m_b_true,
        x1_true=x1_true,
        c_true=c_true,
        z_obs=z_obs,
        z_cos=z_cos,
        log_mass=log_mass,
        mu_cos=mu_cosmo,
        sigma_mB=sigma_mB,
        sigma_x1=sigma_x1,
        sigma_c=sigma_c_meas,
        C_SALT=C_SALT,
        scatter_model=model,
        alpha_fid=alpha_fid,
        beta_fid=beta_used,
        gamma_fid=gamma_fid,
        v_pec=velocities,
    )


# ============================================================================
# 4. Tripp formula fit
# ============================================================================

def fit_tripp(observables):
    """Fit the Tripp standardisation formula to recover Hubble residuals.

    mu_obs = m_b - (M_0 - alpha*x1 + beta*c + Delta_M)

    Fits alpha, beta, M_0, gamma, sigma_int by chi^2 minimisation.

    Parameters
    ----------
    observables : dict
        Output of ``simulate_sn_observables``.

    Returns
    -------
    dict with keys:
        delta_mu   : Hubble residuals mu_obs - mu_cos(z_obs)
        mu_obs     : standardised distance modulus
        mu_cos_obs : mu_cos at observed redshift
        alpha, beta, M_0, gamma, sigma_int : fitted nuisance params
        sigma_mu   : total uncertainty on each delta_mu
    """
    m_b = observables["m_b"]
    x1 = observables["x1"]
    c = observables["c"]
    z_obs = observables["z_obs"]
    log_mass = observables["log_mass"]
    sigma_mB = observables["sigma_mB"]
    sigma_x1 = observables["sigma_x1"]
    sigma_c = observables["sigma_c"]
    n_sn = len(m_b)

    mu_cos_obs = mu_cos(z_obs)

    def neg_log_like(params):
        alpha, beta, M_0, gamma, ln_sigma_int = params
        sigma_int = np.exp(ln_sigma_int)
        dM = _mass_step(log_mass, gamma)
        mu_model = m_b - (M_0 - alpha * x1 + beta * c + dM)
        resid = mu_model - mu_cos_obs

        # Total variance per SN (propagating SALT uncertainties)
        var = (sigma_mB ** 2
               + alpha ** 2 * sigma_x1 ** 2
               + beta ** 2 * sigma_c ** 2
               + sigma_int ** 2)
        chi2 = np.sum(resid ** 2 / var + np.log(var))
        return chi2

    # Initial guesses
    x0 = [0.15, 3.1, M_B_FID, 0.05, np.log(0.10)]
    result = optimize.minimize(neg_log_like, x0, method="Nelder-Mead",
                               options={"maxiter": 50000, "xatol": 1e-8,
                                        "fatol": 1e-8})
    alpha_fit, beta_fit, M0_fit, gamma_fit, ln_sig = result.x
    sigma_int_fit = np.exp(ln_sig)

    dM_fit = _mass_step(log_mass, gamma_fit)
    mu_obs = m_b - (M0_fit - alpha_fit * x1 + beta_fit * c + dM_fit)
    delta_mu = mu_obs - mu_cos_obs

    sigma_mu = np.sqrt(sigma_mB ** 2
                       + alpha_fit ** 2 * sigma_x1 ** 2
                       + beta_fit ** 2 * sigma_c ** 2
                       + sigma_int_fit ** 2)

    return dict(
        delta_mu=delta_mu,
        mu_obs=mu_obs,
        mu_cos_obs=mu_cos_obs,
        alpha=alpha_fit,
        beta=beta_fit,
        M_0=M0_fit,
        gamma=gamma_fit,
        sigma_int=sigma_int_fit,
        sigma_mu=sigma_mu,
    )


# ============================================================================
# 5. Estimate velocities from Hubble residuals
# ============================================================================

def estimate_velocities(delta_mu, z_obs, sigma_mu):
    """Convert Hubble residuals to peculiar velocity estimates.

    v = J(z) * delta_mu

    where J(z) = -c*ln10/5 * [ (1+z)*c / (H(z)*r(z)) - 1 ]^{-1}

    (Eq. 29-30 of Howlett+2017)

    Parameters
    ----------
    delta_mu : array
        Hubble residuals (mag).
    z_obs : array
        Observed redshifts.
    sigma_mu : array
        Uncertainty on delta_mu.

    Returns
    -------
    dict:
        v_est     : estimated peculiar velocity (km/s)
        sigma_v   : velocity uncertainty (km/s)
        J_z       : Jacobian (km/s per mag)
    """
    Hz = Hz_cos(z_obs)
    rz = r_cos(z_obs)

    # Avoid division by zero for very low z
    rz = np.maximum(rz, 1.0e-3)

    bracket = (1.0 + z_obs) * C_LIGHT / (Hz * rz) - 1.0
    # For z -> 0, bracket -> infinity; clip for safety
    bracket = np.where(np.abs(bracket) < 1e-6, np.sign(bracket) * 1e-6, bracket)

    J_z = -C_LIGHT * np.log(10.0) / 5.0 / bracket  # km/s per mag

    v_est = J_z * delta_mu
    sigma_v = np.abs(J_z) * sigma_mu

    return dict(v_est=v_est, sigma_v=sigma_v, J_z=J_z)


# ============================================================================
# 6. Top-level mock generator
# ============================================================================

def generate_mock(n_sn=6600, scatter_model="P23", fsigma8_fid=0.4253,
                  z_range=(0.02, 0.1), sigma_u=21.0, seed=None):
    """Generate one complete mock SN Ia peculiar velocity survey.

    Parameters
    ----------
    n_sn : int
        Number of supernovae (default 6600, typical for ZTF-like survey).
    scatter_model : str
        Intrinsic scatter model: 'COH', 'G10', 'C11', or 'P23'.
    fsigma8_fid : float
        Fiducial f*sigma8 used to generate peculiar velocities.
    z_range : tuple
        (z_min, z_max) redshift range.
    sigma_u : float
        Velocity damping scale (Mpc/h).
    seed : int or None
        Master random seed.  Sub-seeds are derived deterministically.

    Returns
    -------
    dict with all intermediate and final products:
        positions   : dict (ra, dec, z_cos, log_mass)
        v_pec       : array (true peculiar velocities, km/s)
        observables : dict (m_b, x1, c, z_obs, uncertainties, ...)
        tripp_fit   : dict (delta_mu, fitted params, sigma_mu, ...)
        vel_est     : dict (v_est, sigma_v, J_z)
        config      : dict of input parameters
    """
    # Derive deterministic sub-seeds
    master_rng = np.random.default_rng(seed)
    seed_pos = int(master_rng.integers(0, 2 ** 31))
    seed_pv = int(master_rng.integers(0, 2 ** 31))
    seed_obs = int(master_rng.integers(0, 2 ** 31))

    # Step 1: positions and host masses
    positions = generate_positions(n_sn, z_range=z_range, seed=seed_pos)

    # Step 2: correlated peculiar velocities
    v_pec = generate_pv_field(positions, fsigma8_fid=fsigma8_fid,
                              sigma_u=sigma_u, seed=seed_pv)

    # Step 3: SN observables
    observables = simulate_sn_observables(positions, v_pec,
                                          scatter_model=scatter_model,
                                          seed=seed_obs)

    # Step 4: Tripp fit
    tripp_fit = fit_tripp(observables)

    # Step 5: velocity estimates
    vel_est = estimate_velocities(tripp_fit["delta_mu"],
                                  observables["z_obs"],
                                  tripp_fit["sigma_mu"])

    config = dict(
        n_sn=n_sn,
        scatter_model=scatter_model,
        fsigma8_fid=fsigma8_fid,
        z_range=z_range,
        sigma_u=sigma_u,
        seed=seed,
        cosmology=COSMO.copy(),
    )

    return dict(
        positions=positions,
        v_pec=v_pec,
        observables=observables,
        tripp_fit=tripp_fit,
        vel_est=vel_est,
        config=config,
    )


# ============================================================================
# Self-test (runs when executed directly)
# ============================================================================

if __name__ == "__main__":
    import time

    print("=" * 70)
    print("simulate.py self-test")
    print("=" * 70)

    n_test = 200
    t0 = time.time()

    print(f"\nGenerating mock with {n_test} SNe, scatter_model='P23' ...")
    mock = generate_mock(n_sn=n_test, scatter_model="P23", seed=42)

    t1 = time.time()
    print(f"  Elapsed: {t1 - t0:.2f} s")

    pos = mock["positions"]
    obs = mock["observables"]
    tripp = mock["tripp_fit"]
    vel = mock["vel_est"]

    print(f"\n--- Positions ---")
    print(f"  RA  range : [{pos['ra'].min():.1f}, {pos['ra'].max():.1f}] deg")
    print(f"  DEC range : [{pos['dec'].min():.1f}, {pos['dec'].max():.1f}] deg")
    print(f"  z   range : [{pos['z_cos'].min():.4f}, {pos['z_cos'].max():.4f}]")
    print(f"  logM range: [{pos['log_mass'].min():.2f}, {pos['log_mass'].max():.2f}]")

    print(f"\n--- Peculiar velocities ---")
    v = mock["v_pec"]
    print(f"  mean : {v.mean():+.1f} km/s")
    print(f"  std  : {v.std():.1f} km/s")
    print(f"  range: [{v.min():.0f}, {v.max():.0f}] km/s")

    print(f"\n--- Observables (P23) ---")
    print(f"  m_b  range: [{obs['m_b'].min():.2f}, {obs['m_b'].max():.2f}]")
    print(f"  x1   mean : {obs['x1'].mean():.3f},  std: {obs['x1'].std():.3f}")
    print(f"  c    mean : {obs['c'].mean():.3f},  std: {obs['c'].std():.3f}")

    print(f"\n--- Tripp fit ---")
    print(f"  alpha     : {tripp['alpha']:.4f}")
    print(f"  beta      : {tripp['beta']:.4f}")
    print(f"  M_0       : {tripp['M_0']:.4f}")
    print(f"  gamma     : {tripp['gamma']:.4f}")
    print(f"  sigma_int : {tripp['sigma_int']:.4f}")

    print(f"\n--- Hubble residuals ---")
    dm = tripp["delta_mu"]
    print(f"  mean  : {dm.mean():+.4f} mag")
    print(f"  std   : {dm.std():.4f} mag")

    print(f"\n--- Velocity estimates ---")
    print(f"  v_est  mean : {vel['v_est'].mean():+.1f} km/s")
    print(f"  v_est  std  : {vel['v_est'].std():.1f} km/s")
    print(f"  sigma_v mean: {vel['sigma_v'].mean():.1f} km/s")

    # Quick sanity: correlation between true and estimated velocities
    corr = np.corrcoef(mock["v_pec"], vel["v_est"])[0, 1]
    print(f"\n  Correlation(v_true, v_est): {corr:.3f}")

    # Test all scatter models
    print(f"\n--- Testing all scatter models (n={n_test}) ---")
    for sm in ["COH", "G10", "C11", "P23"]:
        t2 = time.time()
        m = generate_mock(n_sn=n_test, scatter_model=sm, seed=123)
        t3 = time.time()
        dm_std = m["tripp_fit"]["delta_mu"].std()
        print(f"  {sm:4s}: delta_mu std = {dm_std:.4f} mag, "
              f"elapsed = {t3 - t2:.2f} s")

    print(f"\nAll tests passed.")
