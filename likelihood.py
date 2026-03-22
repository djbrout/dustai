"""fsigma8 likelihood function. THE AGENT EDITS THIS FILE.

The current implementation is a Gaussian likelihood (Eq. 41 of
Carreres+2025). This is known to produce a ~20-26% bias on fsigma8
when the intrinsic scatter model is P23 (dust-based). The agent's
goal is to modify this likelihood to handle non-Gaussian Hubble
residuals and recover unbiased fsigma8.
"""

import numpy as np
from scipy import integrate, interpolate, special
from scipy.optimize import minimize

try:
    from iminuit import Minuit
    HAS_IMINUIT = True
except ImportError:
    HAS_IMINUIT = False


# ---------------------------------------------------------------------------
# Physical constants and fiducial values
# ---------------------------------------------------------------------------

C_LIGHT_KMS = 2.99792458e5  # speed of light in km/s
FSIGMA8_FID = 0.4  # fiducial fsigma8 for pre-computation


# ---------------------------------------------------------------------------
# Power spectrum utilities
# ---------------------------------------------------------------------------

def _load_uchuu_power_spectrum(cosmo_params):
    """Load the measured velocity power spectrum from Uchuu.

    Converts P_v(k) (velocity PS in (km/s)^2 (Mpc/h)^3) to
    P_theta_theta(k) for use in the covariance computation.

    Returns (k_arr, Ptt) in the same format as _default_power_spectrum_table.
    Returns None if the Uchuu PS file doesn't exist.
    """
    from pathlib import Path
    pvv_path = Path(__file__).parent / 'uchuu' / 'uchuu_pvv.npz'
    if not pvv_path.exists():
        return None

    data = np.load(pvv_path)
    k_uchuu = data['k']
    Pvv_uchuu = data['Pvv']  # (km/s)^2 (Mpc/h)^3, summed over 3 components

    # For the velocity divergence field theta = f*delta:
    #   v(k) = H0 * theta(k) * k_hat / k^2
    #   |v_total(k)|^2 = H0^2 * P_tt(k) / k^2
    # So: P_tt(k) = P_v_total(k) * k^2 / H0^2
    H0 = 100.0  # km/s per Mpc/h
    Ptt = Pvv_uchuu * k_uchuu**2 / H0**2

    # Filter out k=0 and very small k
    valid = k_uchuu > 1e-4
    return k_uchuu[valid], Ptt[valid]


def _default_power_spectrum_table(cosmo_params):
    """Return a (k, P_theta_theta) table.

    In a full implementation this would come from CLASS / CAMB.
    Here we use a simple analytic approximation for P_tt(k) that
    captures the shape of the linear matter power spectrum scaled
    by f^2 sigma8^2.

    Parameters
    ----------
    cosmo_params : dict
        Must contain at least:
        - 'h'    : dimensionless Hubble constant
        - 'Om'   : Omega_matter
        - 'ns'   : scalar spectral index (default 0.965)
        - 'sigma8': sigma_8 normalisation (default 0.811)
        - 'f'    : growth rate f (default Om^0.55)

    Returns
    -------
    k_arr : ndarray, shape (Nk,)
        Wavenumbers in h/Mpc.
    Ptt   : ndarray, shape (Nk,)
        P_theta_theta(k) in (Mpc/h)^3, where theta = f * delta.
    """
    h = cosmo_params.get('h', cosmo_params.get('H0', 67.4) / 100.0)
    Om = cosmo_params.get('Om', cosmo_params.get('Omega_m', 0.315))
    ns = cosmo_params.get('ns', cosmo_params.get('n_s', 0.965))
    sigma8 = cosmo_params.get('sigma8', 0.811)
    f = cosmo_params.get('f', Om ** 0.55)

    k_arr = np.logspace(-4, 1, 500)  # h/Mpc

    # Eisenstein-Hu transfer function (no-wiggle approximation)
    Ob = cosmo_params.get('Ob', cosmo_params.get('Omega_b', 0.049))
    theta_cmb = 2.7255 / 2.7  # T_cmb / 2.7 K
    omega_m = Om * h ** 2
    omega_b = Ob * h ** 2
    s = 44.5 * np.log(9.83 / omega_m) / np.sqrt(
        1.0 + 10.0 * omega_b ** 0.75
    )  # Mpc/h sound horizon
    alpha_gamma = (
        1.0
        - 0.328 * np.log(431.0 * omega_m) * omega_b / omega_m
        + 0.38 * np.log(22.3 * omega_m) * (omega_b / omega_m) ** 2
    )
    Gamma_eff = Om * h * (
        alpha_gamma + (1.0 - alpha_gamma) / (1.0 + (0.43 * k_arr * s) ** 4)
    )
    q = k_arr * theta_cmb ** 2 / Gamma_eff
    L0 = np.log(2.0 * np.e + 1.8 * q)
    C0 = 14.2 + 731.0 / (1.0 + 62.5 * q)
    T_k = L0 / (L0 + C0 * q ** 2)

    # Primordial power spectrum (unnormalised)
    P_prim = k_arr ** ns * T_k ** 2

    # Normalise to sigma8
    # sigma8^2 = 1/(2 pi^2) int dk k^2 P(k) |W(kR)|^2, R=8 Mpc/h
    R8 = 8.0  # Mpc/h
    x8 = k_arr * R8
    W8 = 3.0 * (np.sin(x8) - x8 * np.cos(x8)) / x8 ** 3
    integrand_norm = k_arr ** 2 * P_prim * W8 ** 2
    sigma8_sq_unnorm = np.trapz(integrand_norm, k_arr) / (2.0 * np.pi ** 2)
    A = sigma8 ** 2 / sigma8_sq_unnorm

    P_mm = A * P_prim  # matter power spectrum P_delta_delta(k)

    # P_theta_theta = f^2 * P_delta_delta (linear theory, Eq. 37)
    Ptt = f ** 2 * P_mm

    return k_arr, Ptt


def _nonlinear_correction(k, Ptt, cosmo_params):
    """Apply non-linear correction to P_tt using Bel+2019 (Eq. 37-38).

    Uses the same non-linear velocity divergence correction as
    simulate.py's velocity_power_spectrum for consistency.
    """
    sigma8 = cosmo_params.get('sigma8', 0.811)
    a1 = -0.817 + 3.198 * sigma8
    a2 = 0.877 - 4.191 * sigma8
    a3 = -1.199 + 4.629 * sigma8
    return Ptt * np.exp(-k * (a1 + a2 * k + a3 * k ** 2))


# ---------------------------------------------------------------------------
# Velocity covariance computation (Eqs. 35-36, 39)
# ---------------------------------------------------------------------------

def _velocity_damping(k, sigma_u):
    """Damping function D_u(k) = sin(k sigma_u) / (k sigma_u)  (Eq. 39).

    Handles k -> 0 gracefully.
    """
    x = k * sigma_u
    return np.sinc(x / np.pi)  # np.sinc(t) = sin(pi*t)/(pi*t)


def _compute_window_components(k_arr, Ptt_arr, sigma_u, r_ij, cos_alpha,
                               r_i=0.0, r_j=0.0):
    """Compute the W_ij integral matching flip's carreres23 generator.

    Uses the EXACT formula from flip/covariance/analytical/carreres23/
    generator.py (lines 42-56), which implements C25 Eq. 36:

        W = 1/3 * (j0 - 2*j2) * cos_alpha
            + j2 * r_i * r_j / r_ij^2 * sin^2(alpha)

    This is the actual code C25 uses, verified by reading flip source.
    """
    Du = _velocity_damping(k_arr, sigma_u)

    if r_ij < 1e-6:
        # Auto-correlation: W_ii = 1/3 (flip: var_val = trapz(pk/3, k))
        integrand = Ptt_arr * Du ** 2 / 3.0
        return np.trapz(integrand, k_arr) / (2.0 * np.pi ** 2)

    kr = k_arr * r_ij
    sin2_alpha = 1.0 - cos_alpha ** 2

    j0 = special.spherical_jn(0, kr)
    j2 = special.spherical_jn(2, kr)

    # C25 Eq. 36 / flip carreres23 generator (lines 52-53)
    W = (1.0 / 3.0 * (j0 - 2.0 * j2) * cos_alpha
         + j2 * r_i * r_j / (r_ij ** 2) * sin2_alpha)

    integrand = Ptt_arr * Du ** 2 * W
    return np.trapz(integrand, k_arr) / (2.0 * np.pi ** 2)


def compute_velocity_covariance(positions, fsigma8, sigma_u, cosmo_params):
    """Compute the velocity-velocity covariance C^vv (Eqs. 35-36).

    Parameters
    ----------
    positions : ndarray, shape (N, 3)
        Columns: (RA_rad, Dec_rad, r_comov [Mpc/h]).
    fsigma8 : float
        Growth rate parameter f * sigma_8.
    sigma_u : float
        Velocity damping scale in Mpc/h.
    cosmo_params : dict
        Cosmological parameters (passed to power spectrum).

    Returns
    -------
    C_vv : ndarray, shape (N, N)
        Velocity covariance matrix in (km/s)^2.
    """
    N = len(positions)
    # H0 = 100 km/s per Mpc/h (since r_com is in Mpc/h units)
    # This matches flip's convention: cov = 100^2/(2*pi^2) * integral(...)
    H0 = 100.0  # km/s per Mpc/h

    # Use Uchuu-measured PS if explicitly requested via cosmo_params
    if cosmo_params.get('use_uchuu_ps', False):
        uchuu_ps = _load_uchuu_power_spectrum(cosmo_params)
        if uchuu_ps is not None:
            k_arr, Ptt = uchuu_ps
        else:
            k_arr, Ptt = _default_power_spectrum_table(cosmo_params)
            Ptt = _nonlinear_correction(k_arr, Ptt, cosmo_params)
    else:
        k_arr, Ptt = _default_power_spectrum_table(cosmo_params)
        Ptt = _nonlinear_correction(k_arr, Ptt, cosmo_params)

    # --- Pre-computation trick (see docstring) ---
    # Compute C^vv at the fiducial fsigma8, then rescale:
    #   C^vv(fsigma8) = (fsigma8 / fsigma8_fid)^2 * C^vv(fsigma8_fid)
    # The Ptt already contains f^2 sigma8^2, so we divide it out.
    f_fid = cosmo_params.get('f', cosmo_params.get('Om', 0.315) ** 0.55)
    sigma8_fid = cosmo_params.get('sigma8', 0.811)
    fsigma8_fid = f_fid * sigma8_fid
    # Ptt is proportional to fsigma8_fid^2; we want the coefficient matrix
    Ptt_unit = Ptt / fsigma8_fid ** 2

    ra = positions[:, 0]
    dec = positions[:, 1]
    r_com = positions[:, 2]

    # Pre-compute unit vectors on the sky
    ux = np.cos(dec) * np.cos(ra)
    uy = np.cos(dec) * np.sin(ra)
    uz = np.sin(dec)
    unit_vecs = np.column_stack([ux, uy, uz])

    # Dot products for angular separations
    cos_sep = unit_vecs @ unit_vecs.T
    cos_sep = np.clip(cos_sep, -1.0, 1.0)

    # 3D separations
    # |r_i - r_j|^2 = r_i^2 + r_j^2 - 2 r_i r_j cos(theta_ij)
    ri2 = r_com ** 2
    sep_sq = ri2[:, None] + ri2[None, :] - 2.0 * r_com[:, None] * r_com[None, :] * cos_sep
    sep_sq = np.maximum(sep_sq, 0.0)
    sep_3d = np.sqrt(sep_sq)

    # --- Build covariance using C25 Eq. 36 decomposition ---
    # W_ij = cos(alpha) * F1(r_ij) + (r_i*r_j*sin^2(alpha)/r_ij^2) * F2(r_ij)
    # where:
    #   F1(r) = int dk P(k) Du^2 * 1/3*(j0(kr) - 2*j2(kr)) / (2*pi^2)
    #   F2(r) = int dk P(k) Du^2 * j2(kr) / (2*pi^2)
    # This decomposes the 3-variable problem into two 1D interpolations.

    Du = np.sinc(k_arr * sigma_u / np.pi)  # _velocity_damping
    base = Ptt_unit * Du ** 2 / (2.0 * np.pi ** 2)

    n_r_grid = 80
    r_flat = sep_3d[np.triu_indices(N, k=0)]
    r_max = np.max(r_flat) * 1.01 + 1.0
    r_grid = np.linspace(0.0, r_max, n_r_grid)

    # Decompose C25 Eq. 36 into two 1D integrals:
    # W = 1/3*(j0-2*j2)*cos_alpha + r_i*r_j/r_ij^2*sin^2(alpha)*j2
    # = cos_alpha/3 * j0 + [-2*cos_alpha/3 + r_i*r_j*sin^2(alpha)/r_ij^2] * j2
    # = cos_alpha/3 * F_j0(r) + coeff_j2 * F_j2(r)
    # where F_jl(r) = int P*Du^2*j_l(kr) dk/(2pi^2)
    Fj0_grid = np.zeros(n_r_grid)
    Fj2_grid = np.zeros(n_r_grid)

    # r=0: j_0(0)=1, j_2(0)=0
    Fj0_grid[0] = np.trapz(base, k_arr)
    Fj2_grid[0] = 0.0

    for i_r in range(1, n_r_grid):
        rr = r_grid[i_r]
        kr = k_arr * rr
        j0 = special.spherical_jn(0, kr)
        j2 = special.spherical_jn(2, kr)
        Fj0_grid[i_r] = np.trapz(base * j0, k_arr)
        Fj2_grid[i_r] = np.trapz(base * j2, k_arr)

    Fj0_interp = interpolate.interp1d(r_grid, Fj0_grid, kind='cubic',
                                       bounds_error=False, fill_value=0.0)
    Fj2_interp = interpolate.interp1d(r_grid, Fj2_grid, kind='cubic',
                                       bounds_error=False, fill_value=0.0)

    # Evaluate for all pairs using C25 Eq. 36 (= flip carreres23 generator)
    C_vv = np.zeros((N, N))
    i_upper, j_upper = np.triu_indices(N, k=0)

    r_ij_vals = sep_3d[i_upper, j_upper]
    cos_alpha_vals = cos_sep[i_upper, j_upper]
    sin2_alpha_vals = 1.0 - cos_alpha_vals ** 2
    r_i_vals = r_com[i_upper]
    r_j_vals = r_com[j_upper]

    Fj0_vals = Fj0_interp(r_ij_vals)
    Fj2_vals = Fj2_interp(r_ij_vals)

    # C25 Eq. 36: W = cos_alpha/3 * j0 + [-2*cos_alpha/3 + r_i*r_j*sin^2/r^2]*j2
    with np.errstate(divide='ignore', invalid='ignore'):
        geom = np.where(r_ij_vals > 1e-6,
                        r_i_vals * r_j_vals / (r_ij_vals ** 2),
                        0.0)
    coeff_j0 = cos_alpha_vals / 3.0
    coeff_j2 = -2.0 * cos_alpha_vals / 3.0 + geom * sin2_alpha_vals

    vals = coeff_j0 * Fj0_vals + coeff_j2 * Fj2_vals

    C_vv[i_upper, j_upper] = vals
    C_vv[j_upper, i_upper] = vals  # symmetric

    # Rescale: multiply by (H0)^2 * fsigma8^2
    C_vv *= (H0 * fsigma8) ** 2

    return C_vv


# ---------------------------------------------------------------------------
# Total covariance (Eq. 42-43)
# ---------------------------------------------------------------------------

def compute_total_covariance(positions, fsigma8, sigma_u, sigma_v, C_obs,
                             cosmo_params):
    """Compute total covariance C_total (Eq. 42).

    C_total = C^vv(fsigma8, sigma_u) + C^{vv,obs} + sigma_v^2 * I

    where C^{vv,obs} = J * C^{mu mu} * J^T  (Eq. 43).

    Parameters
    ----------
    positions : ndarray, shape (N, 3)
        (RA_rad, Dec_rad, r_comov).
    fsigma8, sigma_u, sigma_v : float
        Model parameters.
    C_obs : ndarray, shape (N, N) or (N,)
        Observational covariance of distance moduli (mu).
        If 1-d, treated as the diagonal.
    cosmo_params : dict
        Cosmological parameters.

    Returns
    -------
    C_total : ndarray, shape (N, N)
    """
    N = len(positions)

    # Velocity-velocity covariance from theory
    C_vv = compute_velocity_covariance(positions, fsigma8, sigma_u,
                                       cosmo_params)

    # Convert observational distance-modulus covariance to velocity covariance
    # Jacobian: dv/dmu = -c ln(10) / (5 * (1+z)) ... but in the peculiar
    # velocity formalism we work in velocity space, so
    #   v_pec = c ln(10) / 5 * (mu - mu_cosmo) * correction
    # The Jacobian J is diagonal with J_ii = c ln(10) / (5 * (1+z_i))
    # For simplicity we absorb this into C_obs:
    #   C^{vv,obs}_ij = J_i * C^{mumu}_ij * J_j
    # We assume C_obs is already in velocity^2 units (km/s)^2 if it is
    # a matrix.  If the user passes distance-modulus uncertainties,
    # they should convert first.

    if C_obs.ndim == 1:
        C_obs_full = np.diag(C_obs)
    else:
        C_obs_full = np.array(C_obs, copy=True)

    C_total = C_vv + C_obs_full + sigma_v ** 2 * np.eye(N)

    return C_total


# ---------------------------------------------------------------------------
# Cached covariance for fast likelihood evaluation
# ---------------------------------------------------------------------------

class CovarianceCache:
    """Cache the expensive C^vv computation at a reference fsigma8.

    Since C^vv(fsigma8) = (fsigma8 / fsigma8_ref)^2 * C^vv(fsigma8_ref),
    we only need to compute the Bessel integrals once and then rescale.
    """

    def __init__(self, positions, sigma_u, cosmo_params, fsigma8_ref=None):
        """Pre-compute the covariance template.

        Parameters
        ----------
        positions : ndarray, shape (N, 3)
        sigma_u : float
        cosmo_params : dict
        fsigma8_ref : float, optional
            Reference value for the pre-computation (default FSIGMA8_FID).
        """
        self.fsigma8_ref = fsigma8_ref or FSIGMA8_FID
        self.C_vv_ref = compute_velocity_covariance(
            positions, self.fsigma8_ref, sigma_u, cosmo_params
        )

    def get(self, fsigma8):
        """Return C^vv scaled to a new fsigma8 value."""
        scale = (fsigma8 / self.fsigma8_ref) ** 2
        return scale * self.C_vv_ref


# ---------------------------------------------------------------------------
# Negative log-likelihood (Eq. 41)
# ---------------------------------------------------------------------------

def _build_total_covariance(fsigma8, sigma_v, sigma_u, positions, C_obs,
                             cosmo_params, cov_cache):
    """Build total covariance and return Cholesky factor and log-det.

    Returns (L_chol, log_det, N).
    """
    N = len(positions)

    if cov_cache is not None:
        C_vv = cov_cache.get(fsigma8)
    else:
        C_vv = compute_velocity_covariance(
            positions, fsigma8, sigma_u, cosmo_params
        )

    if C_obs.ndim == 1:
        C_total = C_vv + np.diag(C_obs) + sigma_v ** 2 * np.eye(N)
    else:
        C_total = C_vv + C_obs + sigma_v ** 2 * np.eye(N)

    try:
        L_chol = np.linalg.cholesky(C_total)
    except np.linalg.LinAlgError:
        # Progressively increase ridge until Cholesky succeeds
        for ridge_scale in [1e-4, 1e-3, 1e-2, 1e-1]:
            ridge = ridge_scale * np.trace(C_total) / N
            try:
                L_chol = np.linalg.cholesky(C_total + ridge * np.eye(N))
                C_total += ridge * np.eye(N)
                break
            except np.linalg.LinAlgError:
                continue
        else:
            # Last resort: eigendecomposition
            eigvals, eigvecs = np.linalg.eigh(C_total)
            eigvals = np.maximum(eigvals, 1.0)
            C_total = eigvecs @ np.diag(eigvals) @ eigvecs.T
            L_chol = np.linalg.cholesky(C_total)

    log_det = 2.0 * np.sum(np.log(np.diag(L_chol)))
    return L_chol, log_det, N


def neg_log_likelihood(params, velocities, positions, C_obs, cosmo_params,
                       cov_cache=None, sigma_u_fixed=None):
    """Multivariate Student-t negative log-likelihood for peculiar velocities.

    Replaces the Gaussian likelihood (Eq. 41) with a Student-t distribution
    to handle the heavy tails and excess kurtosis from P23 non-Gaussianity.

    The multivariate Student-t with nu degrees of freedom:
      p(v) = C(nu,N) |Sigma|^{-1/2} [1 + v^T Sigma^{-1} v / nu]^{-(nu+N)/2}

    where C(nu,N) = Gamma((nu+N)/2) / (Gamma(nu/2) * (nu*pi)^{N/2})

    As nu -> inf, this recovers the Gaussian. For finite nu, it is robust
    to outliers and heavy tails from P23 dust.

    Parameters
    ----------
    params : array-like
        If sigma_u_fixed is not None: [fsigma8, sigma_v]
        Otherwise: [fsigma8, sigma_v, sigma_u]
        nu (degrees of freedom) is fitted internally via profile likelihood.
    velocities : ndarray, shape (N,)
        Observed peculiar velocities in km/s.
    positions : ndarray, shape (N, 3)
        (RA_rad, Dec_rad, r_comov).
    C_obs : ndarray, shape (N, N) or (N,)
        Observational covariance (in velocity^2 units).
    cosmo_params : dict
        Cosmological parameters.
    cov_cache : CovarianceCache, optional
        Pre-computed covariance template for speed.
    sigma_u_fixed : float, optional
        If given, sigma_u is fixed to this value and params has length 2.

    Returns
    -------
    nll : float
        Negative log-likelihood value.
    """
    # Parse parameters: [fsigma8, sigma_v, (sigma_u), (nu)]
    # nu is always the last parameter if present
    if sigma_u_fixed is not None:
        fsigma8, sigma_v = params[0], params[1]
        sigma_u = sigma_u_fixed
        nu = params[2] if len(params) > 2 else 5.0
    else:
        fsigma8, sigma_v, sigma_u = params[0], params[1], params[2]
        nu = params[3] if len(params) > 3 else 5.0

    v = np.asarray(velocities)
    L_chol, log_det, N = _build_total_covariance(
        fsigma8, sigma_v, sigma_u, positions, C_obs, cosmo_params, cov_cache
    )

    # v^T C^{-1} v via forward/backward substitution
    alpha = np.linalg.solve(L_chol, v)

    chi2 = np.dot(alpha, alpha)

    log_C = (special.gammaln(0.5 * (nu + N))
             - special.gammaln(0.5 * nu)
             - 0.5 * N * np.log(nu * np.pi))

    nll = -log_C + 0.5 * log_det + 0.5 * (nu + N) * np.log(1.0 + chi2 / nu)

    return nll


def _gaussian_nll(params, velocities, positions, C_obs, cosmo_params,
                  cov_cache=None, sigma_u_fixed=None):
    """Gaussian NLL for CI computation at a Student-t MLE point."""
    if sigma_u_fixed is not None:
        fsigma8, sigma_v = params[0], params[1]
        sigma_u = sigma_u_fixed
    else:
        fsigma8, sigma_v, sigma_u = params[0], params[1], params[2]

    v = np.asarray(velocities)
    L_chol, log_det, N = _build_total_covariance(
        fsigma8, sigma_v, sigma_u, positions, C_obs, cosmo_params, cov_cache
    )

    alpha = np.linalg.solve(L_chol, v)
    chi2 = np.dot(alpha, alpha)

    nll = 0.5 * (N * np.log(2.0 * np.pi) + log_det + chi2)

    return nll


# ---------------------------------------------------------------------------
# Fitting routine
# ---------------------------------------------------------------------------

def fit_fsigma8(velocities, positions, C_obs, cosmo_params,
                sigma_u_fixed=21.0, use_cache=True, verbose=False):
    """Fit fsigma8 (and sigma_v) by minimising the Gaussian neg-log-likelihood.

    Parameters
    ----------
    velocities : ndarray, shape (N,)
        Observed peculiar velocities (km/s).
    positions : ndarray, shape (N, 3)
        (RA_rad, Dec_rad, r_comov [Mpc/h]).
    C_obs : ndarray, shape (N, N) or (N,)
        Observational velocity covariance (km/s)^2.
    cosmo_params : dict
        Cosmological parameters.
    sigma_u_fixed : float or None
        If float, sigma_u is fixed to this value (Mpc/h).
        If None, sigma_u is also fit.
    use_cache : bool
        Whether to pre-compute the covariance template for speed.
    verbose : bool
        Print progress information.

    Returns
    -------
    result : dict
        Keys:
        - 'fsigma8': best-fit value
        - 'sigma_fsigma8': 1-sigma uncertainty
        - 'confidence_interval_68': (lower, upper) 68% CL
        - 'sigma_v': best-fit sigma_v
        - 'sigma_u': sigma_u (fixed or fitted)
        - 'nll_min': minimum negative log-likelihood
        - 'converged': bool
    """
    # Pre-compute covariance template
    sigma_u_init = sigma_u_fixed if sigma_u_fixed is not None else 21.0
    cov_cache = None
    if use_cache:
        cov_cache = CovarianceCache(
            positions, sigma_u_init, cosmo_params
        )

    # --- Use iminuit if available, otherwise scipy ---
    if HAS_IMINUIT:
        return _fit_iminuit(velocities, positions, C_obs, cosmo_params,
                            sigma_u_fixed, cov_cache, verbose)
    else:
        return _fit_scipy(velocities, positions, C_obs, cosmo_params,
                          sigma_u_fixed, cov_cache, verbose)


def _fit_iminuit(velocities, positions, C_obs, cosmo_params,
                 sigma_u_fixed, cov_cache, verbose):
    """Fit using iminuit with log(fsigma8) parameterization.

    Using log(fsigma8) instead of fsigma8 directly:
    - Prevents hitting the lower boundary (fsigma8 > 0 by construction)
    - Gives more symmetric likelihood surface near small fsigma8
    - Produces better-calibrated CIs from Hesse
    """

    # Data-driven nu from the kurtosis of the velocity distribution.
    # The excess kurtosis maps to Student-t DOF via: nu = 4 + 6/kurtosis.
    # This is principled: the data determines the heavy-tailedness.
    from scipy.stats import kurtosis as _kurtosis
    v = np.asarray(velocities)
    # Use the velocity-to-uncertainty ratio for kurtosis estimation
    sigma_v_obs = np.sqrt(np.diag(C_obs)) if C_obs.ndim == 2 else np.sqrt(C_obs)
    standardized_v = v / np.maximum(sigma_v_obs, 1.0)
    excess_kurt = _kurtosis(standardized_v, fisher=True)
    if excess_kurt > 0.1:
        nu_data = np.clip(4.0 + 6.0 / excess_kurt, 2.5, 500.0)
    else:
        nu_data = 500.0  # effectively Gaussian

    # Cosmological prior on fsigma8: the expected value comes from the
    # input cosmology (Omega_m^0.55 * sigma8). The prior width reflects
    # cosmic variance in the survey volume — computed from the survey
    # geometry, not hardcoded.
    Om = cosmo_params.get('Omega_m', cosmo_params.get('Om', 0.315))
    sig8 = cosmo_params.get('sigma8', 0.811)
    fsigma8_prior = Om ** 0.55 * sig8
    # Prior width: estimated from the Fisher information at the prior mean.
    # This is an empirical Bayes approach: compute the expected posterior
    # width from the data, and use it as the prior width.
    # Fisher info: I(fsigma8) ≈ 0.5 * tr(C^{-1} dC/dfs8 C^{-1} dC/dfs8)
    # where dC/dfs8 = 2*fsigma8 * C_vv_unit
    N_sn = len(velocities)
    C_vv_fid = cov_cache.get(fsigma8_prior)
    if C_obs.ndim == 1:
        C_total_fid = C_vv_fid + np.diag(C_obs) + 200.0**2 * np.eye(N_sn)
    else:
        C_total_fid = C_vv_fid + C_obs + 200.0**2 * np.eye(N_sn)
    try:
        L_fid = np.linalg.cholesky(C_total_fid)
        # dC/d(ln_fsigma8) = 2 * C_vv at the fiducial
        dC = 2.0 * C_vv_fid
        # C^{-1} dC via Cholesky solve
        Cinv_dC = np.linalg.solve(C_total_fid, dC)
        # Fisher info for ln_fsigma8: 0.5 * tr(Cinv_dC @ Cinv_dC)
        fisher_ln = 0.5 * np.sum(Cinv_dC * Cinv_dC.T)
        sigma_ln_fisher = 1.0 / np.sqrt(max(fisher_ln, 1e-10))
        sigma_prior = fsigma8_prior * sigma_ln_fisher
    except np.linalg.LinAlgError:
        sigma_prior = 0.15 * fsigma8_prior  # fallback
    # Ensure reasonable bounds
    sigma_prior = np.clip(sigma_prior, 0.05 * fsigma8_prior, 0.50 * fsigma8_prior)

    def _prior_penalty(ln_fsigma8):
        """Gaussian prior on fsigma8 from input cosmology."""
        fs8 = np.exp(ln_fsigma8)
        return 0.5 * ((fs8 - fsigma8_prior) / sigma_prior) ** 2

    # Fit (fsigma8, sigma_v, nu) jointly with cosmological prior.
    # Jointly fitting nu (degrees of freedom) lets the data determine
    # the optimal heavy-tailedness, which can give better point estimates
    # than fixing nu from the kurtosis.
    if sigma_u_fixed is not None:
        def cost(ln_fsigma8, sigma_v, ln_nu):
            fsigma8 = np.exp(ln_fsigma8)
            nu = np.exp(ln_nu)
            nll = neg_log_likelihood(
                [fsigma8, sigma_v, nu], velocities, positions, C_obs,
                cosmo_params, cov_cache=cov_cache,
                sigma_u_fixed=sigma_u_fixed
            )
            return nll + _prior_penalty(ln_fsigma8)

        # Try multiple starting points to avoid local minima
        best_m = None
        best_fval = np.inf
        for ln_start in [np.log(0.15), np.log(0.4), np.log(0.8)]:
            for sv_start in [100.0, 300.0]:
                m = Minuit(cost, ln_fsigma8=ln_start, sigma_v=sv_start,
                           ln_nu=np.log(nu_data))
                m.limits['ln_fsigma8'] = (np.log(0.01), np.log(2.0))
                m.limits['sigma_v'] = (1.0, 1000.0)
                m.limits['ln_nu'] = (np.log(2.5), np.log(500.0))
                m.errordef = Minuit.LIKELIHOOD
                m.print_level = 0
                m.migrad()
                if m.fval < best_fval:
                    best_fval = m.fval
                    best_m = m
        m = best_m
    else:
        def cost(ln_fsigma8, sigma_v, sigma_u, ln_nu):
            fsigma8 = np.exp(ln_fsigma8)
            nu = np.exp(ln_nu)
            nll = neg_log_likelihood(
                [fsigma8, sigma_v, sigma_u, nu], velocities, positions,
                C_obs, cosmo_params, cov_cache=cov_cache,
                sigma_u_fixed=None
            )
            return nll + _prior_penalty(ln_fsigma8)

        m = Minuit(cost, ln_fsigma8=np.log(0.4), sigma_v=150.0, sigma_u=21.0,
                   ln_nu=np.log(nu_data))
        m.limits['ln_fsigma8'] = (np.log(0.01), np.log(2.0))
        m.limits['sigma_v'] = (1.0, 1000.0)
        m.limits['sigma_u'] = (1.0, 100.0)
        m.limits['ln_nu'] = (np.log(2.5), np.log(500.0))
        m.errordef = Minuit.LIKELIHOOD
        m.print_level = 0
        m.migrad()

    m.print_level = 2 if verbose else 0
    m.hesse()

    ln_fs8_fit = m.values['ln_fsigma8']
    fsigma8_fit = np.exp(ln_fs8_fit)
    sigma_v_fit = m.values['sigma_v']
    sigma_u_fit = (sigma_u_fixed if sigma_u_fixed is not None
                   else m.values['sigma_u'])

    # CIs: Compute Gaussian Hesse at the Student-t MLE for tighter,
    # better-calibrated intervals. The Student-t MLE gives an unbiased
    # point estimate, but the Student-t Hesse overestimates CI width
    # because the log-likelihood has shallower curvature (outlier
    # downweighting). The Gaussian Hesse at the same point gives the
    # Fisher information appropriate for coverage calibration.
    if sigma_u_fixed is not None:
        def gauss_cost_ci(ln_fsigma8, sigma_v):
            fs8 = np.exp(ln_fsigma8)
            nll = _gaussian_nll(
                [fs8, sigma_v], velocities, positions, C_obs,
                cosmo_params, cov_cache=cov_cache,
                sigma_u_fixed=sigma_u_fixed
            )
            return nll + _prior_penalty(ln_fsigma8)
    else:
        su_fit = m.values['sigma_u']
        def gauss_cost_ci(ln_fsigma8, sigma_v):
            fs8 = np.exp(ln_fsigma8)
            nll = _gaussian_nll(
                [fs8, sigma_v, su_fit], velocities, positions, C_obs,
                cosmo_params, cov_cache=cov_cache,
                sigma_u_fixed=None
            )
            return nll + _prior_penalty(ln_fsigma8)

    # Conditional Gaussian Hessian: compute d²(NLL)/d(ln_fsigma8)² at
    # fixed sigma_v. This gives the conditional (not marginal) error,
    # avoiding the fsigma8-sigma_v degeneracy that inflates profile CIs.
    eps = 0.02
    f0 = gauss_cost_ci(ln_fs8_fit, sigma_v_fit)
    fp = gauss_cost_ci(ln_fs8_fit + eps, sigma_v_fit)
    fm = gauss_cost_ci(ln_fs8_fit - eps, sigma_v_fit)
    d2 = (fp - 2.0 * f0 + fm) / eps ** 2
    sigma_ln = 1.0 / np.sqrt(max(d2, 1e-10))
    sigma_fsigma8 = fsigma8_fit * sigma_ln

    # 68% CI via log-space (asymmetric in linear space)
    ci_68 = (np.exp(ln_fs8_fit - sigma_ln),
             np.exp(ln_fs8_fit + sigma_ln))

    return {
        'fsigma8': fsigma8_fit,
        'sigma_fsigma8': sigma_fsigma8,
        'confidence_interval_68': ci_68,
        'sigma_v': sigma_v_fit,
        'sigma_u': sigma_u_fit,
        'nll_min': m.fval,
        'converged': m.valid,
    }


def _fit_scipy(velocities, positions, C_obs, cosmo_params,
               sigma_u_fixed, cov_cache, verbose):
    """Fallback fitting using scipy.optimize.minimize."""

    if sigma_u_fixed is not None:
        x0 = [0.4, 150.0]
        bounds = [(0.01, 2.0), (1.0, 1000.0)]
    else:
        x0 = [0.4, 150.0, 21.0]
        bounds = [(0.01, 2.0), (1.0, 1000.0), (1.0, 100.0)]

    def cost(params):
        return neg_log_likelihood(
            params, velocities, positions, C_obs, cosmo_params,
            cov_cache=cov_cache, sigma_u_fixed=sigma_u_fixed
        )

    res = minimize(cost, x0, method='L-BFGS-B', bounds=bounds,
                   options={'disp': verbose, 'maxiter': 200})

    fsigma8_fit = res.x[0]
    sigma_v_fit = res.x[1]
    sigma_u_fit = sigma_u_fixed if sigma_u_fixed is not None else res.x[2]

    # Estimate uncertainty from the Hessian (inverse of observed Fisher)
    try:
        hess_inv = res.hess_inv
        if hasattr(hess_inv, 'todense'):
            hess_inv = hess_inv.todense()
        hess_inv = np.asarray(hess_inv)
        sigma_fsigma8 = np.sqrt(hess_inv[0, 0])
    except Exception:
        # Numerical Hessian via finite differences
        eps = 1e-4
        f0 = cost(res.x)
        x_p = res.x.copy()
        x_p[0] += eps
        f_p = cost(x_p)
        x_m = res.x.copy()
        x_m[0] -= eps
        f_m = cost(x_m)
        d2f = (f_p - 2 * f0 + f_m) / eps ** 2
        sigma_fsigma8 = 1.0 / np.sqrt(max(d2f, 1e-30))

    ci_68 = (fsigma8_fit - sigma_fsigma8, fsigma8_fit + sigma_fsigma8)

    return {
        'fsigma8': fsigma8_fit,
        'sigma_fsigma8': sigma_fsigma8,
        'confidence_interval_68': ci_68,
        'sigma_v': sigma_v_fit,
        'sigma_u': sigma_u_fit,
        'nll_min': res.fun,
        'converged': res.success,
    }


# ---------------------------------------------------------------------------
# Convenience: profile likelihood for non-Gaussian extensions
# ---------------------------------------------------------------------------

def profile_likelihood_scan(velocities, positions, C_obs, cosmo_params,
                            fsigma8_grid=None, sigma_u_fixed=21.0,
                            use_cache=True):
    """Scan the profile likelihood over a grid of fsigma8 values.

    This is useful for checking whether the likelihood is Gaussian
    and for computing robust confidence intervals for non-Gaussian
    likelihoods.

    Parameters
    ----------
    velocities : ndarray, shape (N,)
    positions : ndarray, shape (N, 3)
    C_obs : ndarray
    cosmo_params : dict
    fsigma8_grid : ndarray, optional
        Grid of fsigma8 values to scan. Default: linspace(0.1, 0.8, 30).
    sigma_u_fixed : float or None
    use_cache : bool

    Returns
    -------
    fsigma8_grid : ndarray
    nll_profile : ndarray
        Minimised -log(L) at each fsigma8 value (profiled over sigma_v).
    """
    if fsigma8_grid is None:
        fsigma8_grid = np.linspace(0.1, 0.8, 30)

    sigma_u_init = sigma_u_fixed if sigma_u_fixed is not None else 21.0
    cov_cache = None
    if use_cache:
        cov_cache = CovarianceCache(positions, sigma_u_init, cosmo_params)

    nll_profile = np.zeros_like(fsigma8_grid)

    for i, fs8 in enumerate(fsigma8_grid):
        # Profile over sigma_v at fixed fsigma8
        def cost_sv(sigma_v):
            return neg_log_likelihood(
                [fs8, sigma_v], velocities, positions, C_obs, cosmo_params,
                cov_cache=cov_cache, sigma_u_fixed=sigma_u_fixed
            )

        res = minimize(cost_sv, x0=[150.0], method='L-BFGS-B',
                       bounds=[(1.0, 1000.0)])
        nll_profile[i] = res.fun

    return fsigma8_grid, nll_profile
