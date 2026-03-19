"""Evaluation harness using Uchuu N-body simulation for realistic mocks.

Same scoring as evaluate.py but replaces the analytical mock generator
with Uchuu-based mocks that have real N-body peculiar velocities.

Usage:
    python3 evaluate_uchuu.py > run_uchuu.log 2>&1
"""

from __future__ import annotations

import time
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from simulate import COSMO as FIDUCIAL_COSMO
from uchuu_mock import load_uchuu_halos, generate_uchuu_mock
from bbc import apply_corrections_to_mock
from likelihood import fit_fsigma8

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_SN = 1500
N_MOCKS = 16          # more mocks for better statistics
SCATTER_MODEL = "P23"
FSIGMA8_FID = FIDUCIAL_COSMO["fsigma8"]
SIGMA_U_FIXED = 21.0
# Use Uchuu-measured velocity PS for the covariance model
UCHUU_COSMO = {**FIDUCIAL_COSMO, 'use_uchuu_ps': True}
Z_RANGE = (0.02, 0.1)


# Pre-load halos once (expensive, ~10s)
print("Loading Uchuu halos...")
t_load = time.time()
UCHUU_HALOS = load_uchuu_halos()
print(f"  Loaded {len(UCHUU_HALOS['x']):,} halos in {time.time()-t_load:.1f}s")
print()


def run_one_mock(seed: int) -> dict:
    """Run one mock: simulate -> BBC -> fit fsigma8."""
    # 1. Generate mock from Uchuu
    mock = generate_uchuu_mock(
        n_sn=N_SN,
        scatter_model=SCATTER_MODEL,
        z_range=Z_RANGE,
        seed=seed,
        halos=UCHUU_HALOS,
    )

    pos = mock["positions"]
    obs = mock["observables"]
    tripp = mock["tripp_fit"]
    vel = mock["vel_est"]

    # 2. Apply BBC corrections
    corrected = apply_corrections_to_mock(mock)

    velocities = corrected["velocities"]
    sigma_v_obs = corrected["sigma_v"]
    n_sn = corrected["n_sn"]
    z_used = corrected["z"]

    C_obs = np.diag(sigma_v_obs**2)

    r_com = np.array([_comoving_distance(z) for z in z_used])

    all_z = obs["z_obs"]
    all_ra = pos["ra"][:len(all_z)]
    all_dec = pos["dec"][:len(all_z)]

    x1_all = obs["x1"]
    c_all = obs["c"]
    sigma_mu_all = tripp["sigma_mu"]
    vel_all = vel["v_est"]
    bbc_mask = (
        (np.abs(x1_all) < 3.0)
        & (np.abs(c_all) < 0.3)
        & np.isfinite(vel_all)
        & np.isfinite(sigma_mu_all)
        & (sigma_mu_all > 0)
    )

    ra_rad = np.deg2rad(all_ra[bbc_mask])
    dec_rad = np.deg2rad(all_dec[bbc_mask])
    positions = np.column_stack([ra_rad, dec_rad, r_com])

    # 3. Fit fsigma8
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = fit_fsigma8(
            velocities=velocities,
            positions=positions,
            C_obs=C_obs,
            cosmo_params=FIDUCIAL_COSMO,
            sigma_u_fixed=SIGMA_U_FIXED,
        )

    ci = result.get("confidence_interval_68", (float("nan"), float("nan")))
    return {
        "fsigma8": result["fsigma8"],
        "fsigma8_err": result.get("sigma_fsigma8", float("nan")),
        "ci68_lo": ci[0],
        "ci68_hi": ci[1],
        "sigma_v": result.get("sigma_v", float("nan")),
        "n_sn": n_sn,
        "seed": seed,
    }


def _comoving_distance(z: float) -> float:
    """Comoving distance in Mpc/h."""
    from scipy.integrate import quad
    Om = FIDUCIAL_COSMO["Omega_m"]
    c_km = 299792.458
    def integrand(zp):
        return 1.0 / np.sqrt(Om * (1 + zp)**3 + (1 - Om))
    val, _ = quad(integrand, 0, z)
    return c_km / 100.0 * val


def compute_score(results: list[dict]) -> dict:
    """Same scoring as evaluate.py."""
    fs8_values = np.array([r["fsigma8"] for r in results])
    ci_lo = np.array([r["ci68_lo"] for r in results])
    ci_hi = np.array([r["ci68_hi"] for r in results])

    mean_fs8 = float(np.mean(fs8_values))
    bias = mean_fs8 / FSIGMA8_FID - 1.0

    covered = np.sum((ci_lo <= FSIGMA8_FID) & (FSIGMA8_FID <= ci_hi))
    coverage_68 = float(covered) / len(results)

    widths = ci_hi - ci_lo
    mean_width = float(np.mean(widths)) / FSIGMA8_FID

    w_bias = 5.0
    w_coverage = 2.0
    w_width = 1.0

    final_score = (
        w_bias * abs(bias)
        + w_coverage * abs(coverage_68 - 0.68)
        + w_width * mean_width
    )

    return {
        "final_score": float(final_score),
        "bias": float(bias),
        "abs_bias": float(abs(bias)),
        "coverage_68": float(coverage_68),
        "mean_width_rel": float(mean_width),
        "mean_fsigma8": float(mean_fs8),
        "std_fsigma8": float(np.std(fs8_values)),
        "fsigma8_fid": float(FSIGMA8_FID),
        "n_mocks": len(results),
    }


def main():
    t0 = time.time()

    print(f"Uchuu evaluation: {N_MOCKS} mocks, {N_SN} SNe each, model={SCATTER_MODEL}")
    print(f"Fiducial fsigma8 = {FSIGMA8_FID:.4f}")
    print()

    results = []
    for i in range(N_MOCKS):
        seed = 2000 + i * 137
        t1 = time.time()
        try:
            r = run_one_mock(seed)
            results.append(r)
            dt = time.time() - t1
            print(
                f"  Mock {i+1}/{N_MOCKS}: fsigma8={r['fsigma8']:.4f} "
                f"+/- {r['fsigma8_err']:.4f}  "
                f"[{r['ci68_lo']:.4f}, {r['ci68_hi']:.4f}]  "
                f"({dt:.1f}s)"
            )
        except Exception as exc:
            import traceback
            print(f"  Mock {i+1}/{N_MOCKS}: FAILED - {exc}")
            traceback.print_exc()

    if len(results) < 2:
        print("\nToo few successful mocks for scoring.")
        print("---")
        print("final_score:      999.000000")
        return

    sc = compute_score(results)
    t_end = time.time()

    print()
    print("---")
    print(f"final_score:      {sc['final_score']:.6f}")
    print(f"bias:             {sc['bias']:.6f}")
    print(f"abs_bias:         {sc['abs_bias']:.6f}")
    print(f"coverage_68:      {sc['coverage_68']:.4f}")
    print(f"mean_width_rel:   {sc['mean_width_rel']:.4f}")
    print(f"mean_fsigma8:     {sc['mean_fsigma8']:.6f}")
    print(f"std_fsigma8:      {sc['std_fsigma8']:.6f}")
    print(f"fsigma8_fid:      {sc['fsigma8_fid']:.6f}")
    print(f"n_mocks:          {sc['n_mocks']}")
    print(f"total_seconds:    {t_end - t0:.1f}")

    print()
    print("Per-mock results:")
    for r in results:
        inside = "Y" if r["ci68_lo"] <= FSIGMA8_FID <= r["ci68_hi"] else "N"
        print(
            f"  seed={r['seed']:5d}  fsigma8={r['fsigma8']:.4f} "
            f"[{r['ci68_lo']:.4f}, {r['ci68_hi']:.4f}]  "
            f"covers_fid={inside}"
        )


if __name__ == "__main__":
    main()
