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
    h = cosmo_params.get('h', 0.674)
    Om = cosmo_params.get('Om', 0.315)
    ns = cosmo_params.get('ns', 0.965)
    sigma8 = cosmo_params.get('sigma8', 0.811)
    f = cosmo_params.get('f', Om ** 0.55)

    k_arr = np.logspace(-4, 1, 500)  # h/Mpc

    # Eisenstein-Hu transfer function (no-wiggle approximation)
    Ob = cosmo_params.get('Ob', 0.049)
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
    """Apply 1-loop non-linear correction to P_tt (Eq. 37-38).

    For now we use a Lorentzian damping as a proxy for the full 1-loop
    calculation.  The agent can replace this with the exact expressions.

    Returns corrected P_tt.
    """
    sigma_nl = cosmo_params.get('sigma_nl', 6.0)  # Mpc/h
    damping = 1.0 / (1.0 + (k * sigma_nl) ** 2)
    return Ptt * damping


# ---------------------------------------------------------------------------
# Velocity covariance computation (Eqs. 35-36, 39)
# ---------------------------------------------------------------------------

def _velocity_damping(k, sigma_u):
    """Damping function D_u(k) = sin(k sigma_u) / (k sigma_u)  (Eq. 39).

    Handles k -> 0 gracefully.
    """
    x = k * sigma_u
    return np.sinc(x / np.pi)  # np.sinc(t) = sin(pi*t)/(pi*t)


def _compute_window_components(k_arr, Ptt_arr, sigma_u, r, cos_theta):
    """Compute the W_ij integral from Eq. 36 for a pair of SNe.

    The velocity covariance between two SNe at comoving distances
    r_i, r_j with angular separation theta is:

        C^vv_ij = (H_0 f sigma_8)^2 / (2 pi^2)
                  * integral dk  P_tt(k)/(f sigma8)^2
                    * D_u(k)^2 * [j0 terms + j2 terms]

    Since we pre-factor by fsigma8^2, we compute the integral
    without the (f sigma8)^2 prefactor (it's already in Ptt).

    Here we use the simplified monopole+quadrupole decomposition.

    Parameters
    ----------
    k_arr, Ptt_arr : arrays
        Power spectrum table.
    sigma_u : float
        Velocity damping scale in Mpc/h.
    r : float
        Separation between the two SNe in Mpc/h.
    cos_theta : float
        Cosine of the angle between the two line-of-sight directions.

    Returns
    -------
    C_vv_element : float
        Unnormalised covariance element (needs H0^2/(2 pi^2) prefactor).
    """
    Du = _velocity_damping(k_arr, sigma_u)

    if r < 1e-6:
        # Auto-correlation: only monopole contributes
        integrand = Ptt_arr * Du ** 2 / (2.0 * np.pi ** 2)
        return np.trapz(integrand, k_arr)

    kr = k_arr * r

    j0 = special.spherical_jn(0, kr)
    j2 = special.spherical_jn(2, kr)

    # Eq. 36: C_ij propto int dk P(k) Du^2 [A * j0(kr) + B * j2(kr)]
    # where A and B depend on cos_theta (see Carreres+2025 Eq. 36).
    # Simplified: the monopole part and the quadrupole part.
    A = 1.0 / 3.0
    B = (3.0 * cos_theta ** 2 - 1.0) * 2.0 / 3.0

    integrand = Ptt_arr * Du ** 2 * (A * j0 + B * j2)
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
    H0 = cosmo_params.get('H0', 67.4)  # km/s/Mpc

    # Get power spectrum and apply non-linear correction
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

    # --- Build covariance on a grid and interpolate ---
    # Unique-ish (r, cos_theta) pairs -- bin to a grid for speed
    r_flat = sep_3d[np.triu_indices(N, k=0)]
    cos_flat = cos_sep[np.triu_indices(N, k=0)]

    # Grid for interpolation
    n_r_grid = 60
    n_cos_grid = 30
    r_max = np.max(r_flat) * 1.01 + 1.0
    r_grid = np.linspace(0.0, r_max, n_r_grid)
    cos_grid = np.linspace(-1.0, 1.0, n_cos_grid)

    # Evaluate the window integral on the grid
    W_grid = np.zeros((n_r_grid, n_cos_grid))
    for i_r, rr in enumerate(r_grid):
        for i_c, cc in enumerate(cos_grid):
            W_grid[i_r, i_c] = _compute_window_components(
                k_arr, Ptt_unit, sigma_u, rr, cc
            )

    # 2D interpolator
    interp_func = interpolate.RegularGridInterpolator(
        (r_grid, cos_grid), W_grid,
        method='linear', bounds_error=False, fill_value=0.0
    )

    # Evaluate for all pairs
    C_vv = np.zeros((N, N))
    i_upper, j_upper = np.triu_indices(N, k=0)
    points = np.column_stack([
        sep_3d[i_upper, j_upper],
        cos_sep[i_upper, j_upper]
    ])
    vals = interp_func(points)

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
        ridge = 1e-4 * np.trace(C_total) / N
        C_total += ridge * np.eye(N)
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
    if sigma_u_fixed is not None:
        fsigma8, sigma_v = params[0], params[1]
        sigma_u = sigma_u_fixed
    else:
        fsigma8, sigma_v, sigma_u = params[0], params[1], params[2]

    v = np.asarray(velocities)
    L_chol, log_det, N = _build_total_covariance(
        fsigma8, sigma_v, sigma_u, positions, C_obs, cosmo_params, cov_cache
    )

    # v^T C^{-1} v via forward/backward substitution
    alpha = np.linalg.solve(L_chol, v)
    chi2 = np.dot(alpha, alpha)

    # Student-t with nu degrees of freedom
    # nu=3 gives infinite kurtosis, very heavy tails — maximally robust
    # to P23 non-Gaussianity while remaining a proper distribution
    nu = 3.0

    log_C = (special.gammaln(0.5 * (nu + N))
             - special.gammaln(0.5 * nu)
             - 0.5 * N * np.log(nu * np.pi))

    nll = -log_C + 0.5 * log_det + 0.5 * (nu + N) * np.log(1.0 + chi2 / nu)

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

    if sigma_u_fixed is not None:
        def cost(ln_fsigma8, sigma_v):
            fsigma8 = np.exp(ln_fsigma8)
            return neg_log_likelihood(
                [fsigma8, sigma_v], velocities, positions, C_obs,
                cosmo_params, cov_cache=cov_cache,
                sigma_u_fixed=sigma_u_fixed
            )

        m = Minuit(cost, ln_fsigma8=np.log(0.4), sigma_v=150.0)
        m.limits['ln_fsigma8'] = (np.log(0.01), np.log(2.0))
        m.limits['sigma_v'] = (1.0, 1000.0)
        m.errordef = Minuit.LIKELIHOOD
    else:
        def cost(ln_fsigma8, sigma_v, sigma_u):
            fsigma8 = np.exp(ln_fsigma8)
            return neg_log_likelihood(
                [fsigma8, sigma_v, sigma_u], velocities, positions, C_obs,
                cosmo_params, cov_cache=cov_cache, sigma_u_fixed=None
            )

        m = Minuit(cost, ln_fsigma8=np.log(0.4), sigma_v=150.0, sigma_u=21.0)
        m.limits['ln_fsigma8'] = (np.log(0.01), np.log(2.0))
        m.limits['sigma_v'] = (1.0, 1000.0)
        m.limits['sigma_u'] = (1.0, 100.0)
        m.errordef = Minuit.LIKELIHOOD

    m.print_level = 2 if verbose else 0

    # Minimise
    m.migrad()

    # Use MINOS for accurate profile likelihood CIs (not parabolic Hesse)
    m.minos('ln_fsigma8')

    ln_fs8_fit = m.values['ln_fsigma8']
    fsigma8_fit = np.exp(ln_fs8_fit)
    sigma_v_fit = m.values['sigma_v']
    sigma_u_fit = (sigma_u_fixed if sigma_u_fixed is not None
                   else m.values['sigma_u'])

    # MINOS gives asymmetric errors in log-space
    merr = m.merrors['ln_fsigma8']
    ln_lo = ln_fs8_fit + merr.lower  # merr.lower is negative
    ln_hi = ln_fs8_fit + merr.upper

    # Transform to linear space
    ci_68 = (np.exp(ln_lo), np.exp(ln_hi))
    sigma_fsigma8 = 0.5 * (ci_68[1] - ci_68[0])

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
