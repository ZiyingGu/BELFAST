# import necessary packages
import os
# import copy
import time
from astropy import units
import astropy.constants as ct
import numpy as np
# from scipy.stats import uniform
from scipy.stats import norm
# from statsmodels.tsa.stattools import acf
import emcee
# import arviz as az
import corner
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d
from pa.lib import star as st
import torch
import multiprocess as mp
import platform
from datetime import datetime
from scipy.interpolate import LinearNDInterpolator
from lib.target_utils import cal_obs_mag, cal_parsec, add_extinction
# from lib.isochrone_utils import make_model

import warnings
import math


from pathlib import Path
# import os
# import numpy as np
# import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages



warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in scalar subtract",
    category=RuntimeWarning,
)

warnings.filterwarnings(
    "ignore",
    message=r"invalid value encountered in subtract",
    category=RuntimeWarning,
    module=r".*mcmc_utils",
)

warnings.filterwarnings(
    "ignore",
    message=r"divide by zero encountered in log",
    category=RuntimeWarning,
    module=r".*mcmc_utils",
)
target = None

# priors

# import numpy as np
#
# def _log_prior_lagged_maxwell(v, alpha, ell, v_max=500.0):
#     """
#     Generic log-prior for lagged Maxwellian:
#         v = ell + Maxwell(alpha)
#
#     Parameters
#     ----------
#     v : float or array
#         Equatorial rotational velocity (km/s)
#     alpha : float
#         Maxwellian scale parameter
#     ell : float
#         Lag (km/s)
#     v_max : float
#         Upper cutoff (km/s)
#     """
#     v = np.asarray(v, dtype=float)
#
#     out = np.full_like(v, -np.inf, dtype=float)
#     ok = (v >= ell) & (v <= v_max)
#
#     x = v[ok] - ell
#     out[ok] = (
#         0.5 * np.log(2.0 / np.pi)
#         + 2.0 * np.log(x)
#         - 3.0 * np.log(alpha)
#         - x**2 / (2.0 * alpha**2)
#     )
#
#     return out if out.ndim else float(out)
#
# # ---------- individual mass-bin priors ----------
#
# def log_prior_Vrot_1d60_to_2d00_Msun(v, v_max=500.0):
#     return _log_prior_lagged_maxwell(
#         v, alpha=114.0 / np.sqrt(2.0), ell=46.0, v_max=v_max
#     )
#
# def log_prior_Vrot_1d70_to_2d10_Msun(v, v_max=500.0):
#     return _log_prior_lagged_maxwell(
#         v, alpha=124.0 / np.sqrt(2.0), ell=46.0, v_max=v_max
#     )
#
# def log_prior_Vrot_1d80_to_2d20_Msun(v, v_max=500.0):
#     return _log_prior_lagged_maxwell(
#         v, alpha=128.0 / np.sqrt(2.0), ell=48.0, v_max=v_max
#     )
#
# def log_prior_Vrot_1d90_to_2d30_Msun(v, v_max=500.0):
#     return _log_prior_lagged_maxwell(
#         v, alpha=133.0 / np.sqrt(2.0), ell=44.0, v_max=v_max
#     )
#
# def log_prior_Vrot_2d00_to_2d40_Msun(v, v_max=500.0):
#     return _log_prior_lagged_maxwell(
#         v, alpha=140.0 / np.sqrt(2.0), ell=45.0, v_max=v_max
#     )
#
# def log_prior_Vrot_2d10_to_2d50_Msun(v, v_max=500.0):
#     return _log_prior_lagged_maxwell(
#         v, alpha=154.0 / np.sqrt(2.0), ell=27.0, v_max=v_max
#     )
#
# def log_prior_Vrot_2d20_to_2d60_Msun(v, v_max=500.0):
#     return _log_prior_lagged_maxwell(
#         v, alpha=164.0 / np.sqrt(2.0), ell=18.0, v_max=v_max
#     )
#
# def log_prior_Vrot_2d30_to_2d70_Msun(v, v_max=500.0):
#     return _log_prior_lagged_maxwell(
#         v, alpha=179.0 / np.sqrt(2.0), ell=12.0, v_max=v_max
#     )
#
# def log_prior_Vrot_2d35_to_2d85_Msun(v, v_max=500.0):
#     return _log_prior_lagged_maxwell(
#         v, alpha=210.0 / np.sqrt(2.0), ell=0.0, v_max=v_max
#     )
#
# def log_prior_Vrot_2d45_to_2d95_Msun(v, v_max=500.0):
#     return _log_prior_lagged_maxwell(
#         v, alpha=185.0 / np.sqrt(2.0), ell=14.0, v_max=v_max
#     )
#
# def log_prior_Vrot_2d55_to_3d05_Msun(v, v_max=500.0):
#     return _log_prior_lagged_maxwell(
#         v, alpha=191.0 / np.sqrt(2.0), ell=20.0, v_max=v_max
#     )
#
# def log_prior_Vrot_2d65_to_3d15_Msun(v, v_max=500.0):
#     return _log_prior_lagged_maxwell(
#         v, alpha=183.0 / np.sqrt(2.0), ell=24.0, v_max=v_max
#     )
#
# def log_prior_Vrot_2d75_to_3d45_Msun(v, v_max=500.0):
#     return _log_prior_lagged_maxwell(
#         v, alpha=201.0 / np.sqrt(2.0), ell=18.0, v_max=v_max
#     )
#
# log_prior_Vrot_dict = {
#     "1.60-2.00 Msun": log_prior_Vrot_1d60_to_2d00_Msun,
#     "1.70-2.10 Msun": log_prior_Vrot_1d70_to_2d10_Msun,
#     "1.80-2.20 Msun": log_prior_Vrot_1d80_to_2d20_Msun,
#     "1.90-2.30 Msun": log_prior_Vrot_1d90_to_2d30_Msun,
#     "2.00-2.40 Msun": log_prior_Vrot_2d00_to_2d40_Msun,
#     "2.10-2.50 Msun": log_prior_Vrot_2d10_to_2d50_Msun,
#     "2.20-2.60 Msun": log_prior_Vrot_2d20_to_2d60_Msun,
#     "2.30-2.70 Msun": log_prior_Vrot_2d30_to_2d70_Msun,
#     "2.35-2.85 Msun": log_prior_Vrot_2d35_to_2d85_Msun,
#     "2.45-2.95 Msun": log_prior_Vrot_2d45_to_2d95_Msun,
#     "2.55-3.05 Msun": log_prior_Vrot_2d55_to_3d05_Msun,
#     "2.65-3.15 Msun": log_prior_Vrot_2d65_to_3d15_Msun,
#     "2.75-3.45 Msun": log_prior_Vrot_2d75_to_3d45_Msun,
# }
#
#
# def parse_mass_range(key):
#     """
#     key: e.g. '1.60-2.00 Msun'
#     returns: (m_min, m_max)
#     """
#     rng = key.replace("Msun", "").strip()
#     m_min, m_max = rng.split("-")
#     return float(m_min), float(m_max)
#
#
# def select_Vrot_prior_by_rough_mass(
#     rough_mass,
#     log_prior_Vrot_dict,
# ):
#     """
#     Select the Vrot prior whose mass-bin center is closest to rough_mass,
#     but ONLY if rough_mass lies within the bin.
#     If rough_mass is not covered by any bin, return (None, None).
#     """
#
#     best_key = None
#     best_dist = np.inf
#
#     for key in log_prior_Vrot_dict:
#         m_min, m_max = parse_mass_range(key)
#
#         # 必须先在区间内
#         if not (m_min <= rough_mass <= m_max):
#             continue
#
#         m_center = 0.5 * (m_min + m_max)
#         dist = abs(rough_mass - m_center)
#
#         if dist < best_dist:
#             best_dist = dist
#             best_key = key
#
#     if best_key is None:
#         return None, None
#
#     return best_key, log_prior_Vrot_dict[best_key]


import numpy as np
from scipy.special import erf


def _logsumexp_two(a, b):
    """
    Stable log(exp(a) + exp(b))
    """
    a = np.asarray(a)
    b = np.asarray(b)
    m = np.maximum(a, b)
    return m + np.log(np.exp(a - m) + np.exp(b - m))


def _maxwell_cdf(x, alpha):
    """
    CDF of the standard Maxwell distribution with scale alpha.

    Parameters
    ----------
    x : float or array
        Velocity, must be >= 0 for physical support.
    alpha : float
        Maxwellian scale parameter
    """
    x = np.asarray(x, dtype=float)
    z = x / (np.sqrt(2.0) * alpha)

    cdf = (
        erf(z)
        - np.sqrt(2.0 / np.pi) * (x / alpha) * np.exp(-x**2 / (2.0 * alpha**2))
    )

    # Numerical safety
    cdf = np.clip(cdf, 0.0, 1.0)
    return cdf


def _log_prior_maxwell_truncated(v, alpha, v_max=500.0):
    """
    Log-pdf of a Maxwellian truncated to [0, v_max], properly normalized.

    Parameters
    ----------
    v : float or array
        Equatorial rotational velocity (km/s)
    alpha : float
        Maxwellian scale parameter
    v_max : float
        Upper cutoff (km/s)
    """
    v = np.asarray(v, dtype=float)

    out = np.full_like(v, -np.inf, dtype=float)
    ok = (v >= 0.0) & (v <= v_max)

    if not np.any(ok):
        return out if out.ndim else float(out)

    x = v[ok]

    # Untruncated log-pdf
    log_pdf = (
        0.5 * np.log(2.0 / np.pi)
        + 2.0 * np.log(x)
        - 3.0 * np.log(alpha)
        - x**2 / (2.0 * alpha**2)
    )

    # Normalization constant on [0, v_max]
    Z = _maxwell_cdf(v_max, alpha)

    if Z <= 0.0:
        raise ValueError(f"Normalization constant is non-positive for alpha={alpha}, v_max={v_max}")

    out[ok] = log_pdf - np.log(Z)
    return out if out.ndim else float(out)


def _log_prior_lagged_maxwell_truncated(v, alpha, ell, v_max=500.0):
    """
    Log-pdf of a lagged Maxwellian:
        v = ell + Maxwell(alpha)
    truncated to [0, v_max], properly normalized.

    Since the physical support is v >= ell, the effective support is [ell, v_max].

    Parameters
    ----------
    v : float or array
        Equatorial rotational velocity (km/s)
    alpha : float
        Maxwellian scale parameter
    ell : float
        Lag (km/s)
    v_max : float
        Upper cutoff (km/s)
    """
    v = np.asarray(v, dtype=float)

    out = np.full_like(v, -np.inf, dtype=float)
    ok = (v >= ell) & (v <= v_max)

    if not np.any(ok):
        return out if out.ndim else float(out)

    x = v[ok] - ell

    # Untruncated log-pdf in shifted variable x = v - ell
    log_pdf = (
        0.5 * np.log(2.0 / np.pi)
        + 2.0 * np.log(x)
        - 3.0 * np.log(alpha)
        - x**2 / (2.0 * alpha**2)
    )

    # Normalization over v in [ell, v_max]
    # equivalent to x in [0, v_max - ell]
    upper = v_max - ell
    if upper <= 0.0:
        raise ValueError(f"Need v_max > ell, but got v_max={v_max}, ell={ell}")

    Z = _maxwell_cdf(upper, alpha)

    if Z <= 0.0:
        raise ValueError(
            f"Normalization constant is non-positive for alpha={alpha}, ell={ell}, v_max={v_max}"
        )

    out[ok] = log_pdf - np.log(Z)
    return out if out.ndim else float(out)


def _log_prior_mixture_maxwell_lagged_truncated(
    v,
    slow_frac,
    slow_mu,
    fast_frac,
    fast_sqrt2alpha,
    fast_ell,
    v_max=500.0,
):
    """
    Properly normalized truncated mixture prior:
        p(v) = slow_frac * p_slow(v) + fast_frac * p_fast(v)

    where
    - p_slow is a Maxwellian normalized on [0, v_max]
    - p_fast is a lagged Maxwellian normalized on [fast_ell, v_max]

    Parameters
    ----------
    v : float or array
        Equatorial rotational velocity (km/s)
    slow_frac : float
        Fraction of slow rotators, e.g. 0.08
    slow_mu : float or None
        Mode of the slow-rotator Maxwellian (km/s).
        For a Maxwellian, mu = sqrt(2) * alpha.
    fast_frac : float
        Fraction of fast rotators, e.g. 0.92
    fast_sqrt2alpha : float
        Table value sqrt(2)*alpha for fast rotators
    fast_ell : float
        Lag parameter for fast rotators
    v_max : float
        Upper cutoff (km/s)
    """
    v = np.asarray(v, dtype=float)

    frac_sum = slow_frac + fast_frac
    if not np.isclose(frac_sum, 1.0, atol=1e-12):
        raise ValueError(f"slow_frac + fast_frac must equal 1. Got {frac_sum}")

    log_terms = []

    if slow_frac > 0.0:
        if slow_mu is None:
            raise ValueError("slow_mu cannot be None when slow_frac > 0")
        alpha_slow = slow_mu / np.sqrt(2.0)
        logp_slow = np.log(slow_frac) + _log_prior_maxwell_truncated(
            v, alpha_slow, v_max=v_max
        )
        log_terms.append(logp_slow)

    if fast_frac > 0.0:
        alpha_fast = fast_sqrt2alpha / np.sqrt(2.0)
        logp_fast = np.log(fast_frac) + _log_prior_lagged_maxwell_truncated(
            v, alpha_fast, fast_ell, v_max=v_max
        )
        log_terms.append(logp_fast)

    if len(log_terms) == 0:
        out = np.full_like(v, -np.inf, dtype=float)
        return out if out.ndim else float(out)

    if len(log_terms) == 1:
        out = log_terms[0]
        return out if out.ndim else float(out)

    out = _logsumexp_two(log_terms[0], log_terms[1])
    return out if out.ndim else float(out)


# -------------------------------------------------------------------
# Parameters from Zorec & Royer table
# slow_frac and fast_frac are true mixture weights
# slow_mu is the slow-rotator Maxwellian mode mu_s (km/s)
# fast_sqrt2alpha is the table value sqrt(2)*alpha for fast rotators
# -------------------------------------------------------------------
_VROT_PRIOR_PARAMS = {
    "1.60-2.00 Msun": dict(slow_frac=0.00, slow_mu=None, fast_frac=1.00, fast_sqrt2alpha=114.0, fast_ell=46.0),
    "1.70-2.10 Msun": dict(slow_frac=0.00, slow_mu=None, fast_frac=1.00, fast_sqrt2alpha=124.0, fast_ell=46.0),
    "1.80-2.20 Msun": dict(slow_frac=0.00, slow_mu=None, fast_frac=1.00, fast_sqrt2alpha=128.0, fast_ell=48.0),
    "1.90-2.30 Msun": dict(slow_frac=0.00, slow_mu=None, fast_frac=1.00, fast_sqrt2alpha=133.0, fast_ell=44.0),
    "2.00-2.40 Msun": dict(slow_frac=0.00, slow_mu=None, fast_frac=1.00, fast_sqrt2alpha=140.0, fast_ell=45.0),
    "2.10-2.50 Msun": dict(slow_frac=0.03, slow_mu=25.0, fast_frac=0.97, fast_sqrt2alpha=154.0, fast_ell=27.0),
    "2.20-2.60 Msun": dict(slow_frac=0.08, slow_mu=37.0, fast_frac=0.92, fast_sqrt2alpha=164.0, fast_ell=18.0),
    "2.30-2.70 Msun": dict(slow_frac=0.12, slow_mu=44.0, fast_frac=0.88, fast_sqrt2alpha=179.0, fast_ell=12.0),
    "2.35-2.85 Msun": dict(slow_frac=0.12, slow_mu=47.0, fast_frac=0.88, fast_sqrt2alpha=210.0, fast_ell=0.0),
    "2.45-2.95 Msun": dict(slow_frac=0.18, slow_mu=52.0, fast_frac=0.82, fast_sqrt2alpha=185.0, fast_ell=14.0),
    "2.55-3.05 Msun": dict(slow_frac=0.20, slow_mu=61.0, fast_frac=0.80, fast_sqrt2alpha=191.0, fast_ell=20.0),
    "2.65-3.15 Msun": dict(slow_frac=0.15, slow_mu=55.0, fast_frac=0.85, fast_sqrt2alpha=183.0, fast_ell=24.0),
    "2.65-3.35 Msun": dict(slow_frac=0.14, slow_mu=52.0, fast_frac=0.86, fast_sqrt2alpha=187.0, fast_ell=29.0),
    "2.75-3.45 Msun": dict(slow_frac=0.13, slow_mu=47.0, fast_frac=0.87, fast_sqrt2alpha=201.0, fast_ell=18.0),
    "2.85-3.55 Msun": dict(slow_frac=0.10, slow_mu=39.0, fast_frac=0.90, fast_sqrt2alpha=202.0, fast_ell=20.0),
    "2.95-3.65 Msun": dict(slow_frac=0.06, slow_mu=32.0, fast_frac=0.94, fast_sqrt2alpha=224.0, fast_ell=0.0),
    "3.05-3.75 Msun": dict(slow_frac=0.05, slow_mu=31.0, fast_frac=0.95, fast_sqrt2alpha=229.0, fast_ell=0.0),
    "3.15-3.85 Msun": dict(slow_frac=0.08, slow_mu=32.0, fast_frac=0.92, fast_sqrt2alpha=224.0, fast_ell=15.0),
}


def _make_log_prior_Vrot(params):
    def _f(v, v_max=500.0):
        return _log_prior_mixture_maxwell_lagged_truncated(
            v,
            slow_frac=params["slow_frac"],
            slow_mu=params["slow_mu"],
            fast_frac=params["fast_frac"],
            fast_sqrt2alpha=params["fast_sqrt2alpha"],
            fast_ell=params["fast_ell"],
            v_max=v_max,
        )
    return _f


log_prior_Vrot_dict = {
    key: _make_log_prior_Vrot(params)
    for key, params in _VROT_PRIOR_PARAMS.items()
}


def parse_mass_range(key):
    """
    key: e.g. '1.60-2.00 Msun'
    returns: (m_min, m_max)
    """
    rng = key.replace("Msun", "").strip()
    m_min, m_max = rng.split("-")
    return float(m_min), float(m_max)


def select_Vrot_prior_by_rough_mass(
    rough_mass,
    log_prior_Vrot_dict,
):
    """
    Select the Vrot prior whose mass-bin center is closest to rough_mass,
    but ONLY if rough_mass lies within the bin.
    If rough_mass is not covered by any bin, return (None, None).
    """
    best_key = None
    best_dist = np.inf

    for key in log_prior_Vrot_dict:
        m_min, m_max = parse_mass_range(key)

        if not (m_min <= rough_mass <= m_max):
            continue

        m_center = 0.5 * (m_min + m_max)
        dist = abs(rough_mass - m_center)

        if dist < best_dist:
            best_dist = dist
            best_key = key

    if best_key is None:
        return None, None

    return best_key, log_prior_Vrot_dict[best_key]



def log_prior(sampling_variables, verbose=False):
    """
    Compute the logarithm of the prior probability for a given set of sampling variables.

    Args:
        sampling_variables (list or np.ndarray): Current parameter values being sampled.

    Variables inside:
        Mass (float): Stellar mass (Msun).
        logAge (float): Logarithm of stellar age (yr).
        MH (float): Metallicity [M/H].
        omega (float): Rotation parameter (fraction of critical angular velocity).
        Inclination (float): Inclination angle (deg).
        parallax (float): Stellar parallax (mas).
        extinction (float): Extinction parameter.
        Mini_prior, logAge_prior, MH_prior, omega_ini_prior, Inclination_prior, parallax_prior, extinction_prior:
            Individual log-priors for each parameter, based on default or user-defined priors.

    Returns:
        float: Logarithm of the total prior probability (sum of individual priors).
               Returns -np.inf if any variable is outside allowed limits.
    """

    if isinstance(sampling_variables, list):
        sampling_variables = np.array(sampling_variables)

    # if 'Mass' in target.sampling_variables_list:
    #     Mass_index = target.sampling_variables_list.index('Mass')
    #     Mass = sampling_variables[Mass_index]
    #     if target.mass_prior == 'default':
    #         Mass_prior = -2.35 * np.log( np.maximum(1e-8, Mass) )
    #     else:
    #         Mass_prior = target.mass_prior(Mass)
    # else:
    #     Mass_prior = 0

    if 'Mini' in target.sampling_variables_list:
        Mini_index = target.sampling_variables_list.index('Mini')
        Mini = sampling_variables[Mini_index]
        if target.mini_prior == 'default':
            # Mini_prior = -2.35 * np.log( np.maximum(1e-8, Mini) )
            Mini_prior = -2.3 * np.log(np.maximum(1e-8, Mini))
        else:
            Mini_prior = target.mini_prior(Mini)
    else:
        Mini_prior = 0
    if verbose:
        print(f'Mini_prior: {Mini_prior}')

    if 'logAge' in target.sampling_variables_list:
        logAge_index = target.sampling_variables_list.index('logAge')
        logAge = sampling_variables[logAge_index]
        if target.logAge_prior == 'default':
            # convert age_prior to logAge_prior
            logAge_upper = target.sampling_upper_limit[logAge_index]
            logAge_lower = target.sampling_lower_limit[logAge_index]
            C = np.log(np.log(10)) - np.log(10**logAge_upper - 10**logAge_lower)
            logAge_prior = C + np.log(10) * logAge
        else:
            logAge_prior = target.logAge_prior(logAge)
    else:
        logAge_prior = 0
    if verbose:
        print(f'logAge_prior: {logAge_prior}')

    if 'MH' in target.sampling_variables_list:
        MH_index = target.sampling_variables_list.index('MH')
        MH = sampling_variables[MH_index]
        if target.MH_prior == 'default':
            MH_prior = norm.logpdf(MH, loc=-0.0427, scale=0.2461) # non-informative prior from the GALAH survey (Buder et al. 2019)
            # MH_prior = norm.logpdf(MH, loc=0.072, scale=0.21) # narrower; young solar type star in the solar neighbourhood; data from Buder et al. 2019
        else:
            MH_prior = target.MH_prior(MH)
    else:
        MH_prior = 0
    if verbose:
        print(f'MH_prior: {MH_prior}')

    if 'omega_ini' in target.sampling_variables_list:
        omega_ini_index = target.sampling_variables_list.index('omega_ini')
        omega_ini = sampling_variables[omega_ini_index]
        if target.omega_ini_prior == 'default':
            omega_ini_prior = 0
        else:
            omega_ini_prior = target.omega_ini_prior(omega_ini)
    else:
        omega_ini_prior = 0
    if verbose:
        print(f'omega_ini_prior: {omega_ini_prior}')

    if 'Inclination' in target.sampling_variables_list:
        Inclination_index = target.sampling_variables_list.index('Inclination')
        Inclination = sampling_variables[Inclination_index]
        if target.inclination_prior == 'default':
            Inclination_prior = np.log(np.maximum(1e-8, np.sin(Inclination)))
        else:
            Inclination_prior = target.inclination_prior(Inclination)
    else:
        Inclination_prior = 0
    if verbose:
        print(f'Inclination_prior: {Inclination_prior}')

    if 'parallax' in target.sampling_variables_list:
        parallax_index = target.sampling_variables_list.index('parallax')
        parallax = sampling_variables[parallax_index]
        if target.parallax_prior == 'default':
            parallax_prior = -4 * np.log( np.maximum(1e-8, parallax) )
        else:
            parallax_prior = target.parallax_prior(parallax)
    else:
        parallax_prior = 0
    if verbose:
        print(f'parallax_prior: {parallax_prior}')

    if 'extinction' in target.sampling_variables_list:
        extinction_index = target.sampling_variables_list.index('extinction')
        extinction = sampling_variables[extinction_index]
        if target.extinction_prior == 'default':
            extinction_prior = 0
            # extinction_prior = norm.logpdf(extinction, loc=target.extinction, scale=target.extinction_err)
        else:
            extinction_prior = target.extinction_prior(extinction)
    else:
        extinction_prior = 0
    if verbose:
        print(f'extinction_prior: {extinction_prior}')

    # if target.cluster == 'Hyades':
    #     MH_prior = norm.logpdf(MH, loc=0.12, scale=0.04) # Hyades (Brandt et al. 2015)
    # elif target.cluster == 'Praesepe':
    #     MH_prior = norm.logpdf(MH, loc=0.12, scale=0.04) # Praesepe (Brandt et al. 2015)
    # elif target.cluster == 'Pleiades':
    #     MH_prior = norm.logpdf(MH, loc=0.03, scale=0.05)   # Pleiades
    #
    # if target.name == 'Kappa Andromedae':
    #     Inclination_prior = norm.logpdf(Inclination, loc=30, scale=10)
    #     MH_prior = norm.logpdf(MH, loc=-0.0, scale=0.14)

    if np.all((sampling_variables > target.sampling_lower_limit) & (sampling_variables < target.sampling_upper_limit)):
        return Mini_prior + logAge_prior + MH_prior + omega_ini_prior + Inclination_prior + parallax_prior + extinction_prior
    else:
        return -np.inf




# def log_likelihood_fewer_INTERP(sampling_variables):
#     """
#     Compute the log-likelihood for the INETRP method.
#
#         Args:
#         sampling_variables (list or np.ndarray): Current parameter values being sampled.
#
#     Key variables:
#         Mass, logAge, MH, omega, Inclination, parallax, extinction (floats): Stellar parameters.
#         sigma_adjustment_parameter (int): Determines whether observational errors are rescaled.
#         Model_interp: Interpolated model used to compute stellar magnitudes and radii.
#
#     Returns:
#         tuple: (lnL, blobs)
#             lnL (float): Log-likelihood value.
#             blobs (list): List of diagnostic outputs including magnitudes, vsini, parallax, chi-squared, vtaneq, Req.
#     """
#
#     if isinstance(sampling_variables, list):
#         sampling_variables = np.array(sampling_variables)
#
#     if 'Mass' in target.sampling_variables_list:
#         Mass_index = target.sampling_variables_list.index('Mass')
#         Mass = sampling_variables[Mass_index]
#     elif target.known_Mass is not None:
#         Mass = target.known_Mass
#
#     if 'logAge' in target.sampling_variables_list:
#         logAge_index = target.sampling_variables_list.index('logAge')
#         logAge = sampling_variables[logAge_index]
#     elif target.known_logAge is not None:
#         logAge = target.known_logAge
#
#     if 'MH' in target.sampling_variables_list:
#         MH_index = target.sampling_variables_list.index('MH')
#         MH = sampling_variables[MH_index]
#     elif target.known_MH is not None:
#         MH = target.known_MH
#
#     if 'omega' in target.sampling_variables_list:
#         omega_index = target.sampling_variables_list.index('omega')
#         omega = sampling_variables[omega_index]
#     elif target.known_omega is not None:
#         omega = target.known_omega
#
#     if 'Inclination' in target.sampling_variables_list:
#         Inclination_index = target.sampling_variables_list.index('Inclination')
#         Inclination = sampling_variables[Inclination_index]
#     elif target.known_Inclination is not None:
#         Inclination = target.known_Inclination
#
#     if 'parallax' in target.sampling_variables_list:
#         parallax_index = target.sampling_variables_list.index('parallax')
#         parallax = sampling_variables[parallax_index]
#     elif target.known_parallax is not None:
#         parallax = target.known_parallax
#
#     if 'extinction' in target.sampling_variables_list:
#         extinction_index = target.sampling_variables_list.index('extinction')
#         extinction = sampling_variables[extinction_index]
#         extinction_err = 0
#     else:
#         extinction = target.extinction
#         extinction_err = target.extinction_err
#
#     if target.sigma_adjustment_parameter == 1:
#         sigma = sampling_variables[target.sampling_variables_list.index('sigma')]
#         obs_err_mags_sigma = np.ones(len(target.bands_list)) * sigma
#         obs_err_mags_new = np.sqrt(target.obs_err_mags**2 + obs_err_mags_sigma**2)
#     elif target.sigma_adjustment_parameter == 3:
#         sigma_gaia = sampling_variables[target.sampling_variables_list.index('sigma_gaia')]
#         sigma_tycho2 = sampling_variables[target.sampling_variables_list.index('sigma_tycho2')]
#         sigma_2mass = sampling_variables[target.sampling_variables_list.index('sigma_2mass')]
#         obs_err_mags_sigma = np.array([1, 1, 1, 0, 0, 0, 0, 0]) * sigma_gaia + \
#                              np.array([0, 0, 0, 1, 1, 0, 0, 0]) * sigma_tycho2 + \
#                              np.array([0, 0, 0, 0, 0, 1, 1, 1]) * sigma_2mass
#         obs_err_mags_new = np.sqrt(target.obs_err_mags**2 + obs_err_mags_sigma**2)
#     elif target.sigma_adjustment_parameter == 0:
#         obs_err_mags_new = target.obs_err_mags
#     else:
#         raise ValueError('sigma_adjustment_parameter must be any of 0, 1, or 3.')
#
#     if target.fixed_parameter_indices:
#         for idx in sorted(target.fixed_parameter_indices):
#             sampling_variables = np.insert(sampling_variables, idx, target.true_values[idx])
#
#     # logL, Req, Rpol = target.Model_interp(sampling_variables[:len(target.interp_variables_list)])[0]
#
#     y_interpolated = target.Model_interp(sampling_variables[:len(target.interp_variables_list)]).flatten()
#
#     if np.isnan(y_interpolated).any():
#         return -np.inf, np.ones(len(target.bands_list) + 9)*np.nan
#
#     Rpol = y_interpolated[-1]
#     Req = y_interpolated[-2]
#     logL = y_interpolated[-3]
#     abs_mags_interpolated = y_interpolated[:-3]
#
#     obs_mags_model = cal_obs_mag(abs_mags_interpolated, parallax)
#     if target.extinction_parameter == True:
#         obs_mags_model_extincted, obs_mags_model_err_extincted = add_extinction(obs_mags_model, 0, target.bands_list, extinction, extinction_err, target.extinction_type, print_flag=False)
#     else:
#         obs_mags_model_extincted, obs_mags_model_err_extincted = obs_mags_model, 0
#     final_mags = obs_mags_model_extincted
#     final_err_mags = np.sqrt(obs_err_mags_new**2 + obs_mags_model_err_extincted**2)
#
#    omega_angular = omega * np.sqrt(ct.GM_sun * Mass / (ct.R_sun * 1.5 * Rpol) ** 3)
#    vtaneq = (omega_angular * Req * ct.R_sun).to('km/s').value
#    vsini = vtaneq * np.sin(Inclination)
#    # v_keplerian = np.sqrt(ct.GM_sun * Mass / (ct.R_sun * Req)).to('km/s').value
#    v_critical = np.sqrt(ct.GM_sun * Mass / (ct.R_sun * 1.5 * Rpol)).to('km/s').value
#
#     if np.isnan(target.vsini_obs) or np.isnan(target.vsini_obs_err):
#         chi_vsini = 0
#     else:
#         chi_vsini = ((vsini - target.vsini_obs) / target.vsini_obs_err) ** 2
#
#     chi_squared = np.sum(((final_mags - target.obs_mags)/final_err_mags)**2, axis=1) \
#                   + ((parallax - target.parallax_obs)/target.parallax_obs_err)**2 + chi_vsini
#
#     chi_squared = chi_squared.item()
#
#     if np.isnan(chi_squared).any():
#         chi_squared = np.nan_to_num(chi_squared, nan=np.inf)
#
#     lnL0 = -0.5 * chi_squared
#
#     if target.sigma_adjustment_parameter == 1 or target.sigma_adjustment_parameter == 3:
#         lnL = lnL0 - np.sum(np.log(final_err_mags))
#     elif target.sigma_adjustment_parameter == 0:
#         lnL = lnL0
#
#     blobs = final_mags.tolist()
#     blobs.append(vsini.item())
#     blobs.append(parallax)
#     blobs.append(chi_squared)
#     blobs.append(vtaneq.item())
#     blobs.append(Req.item())
#     blobs.append(Rpol.item())
#     blobs.append(logL.item())
#     blobs.append(lnL0)
#     blobs.append(lnL)
#     unified_blobs = [float(np.squeeze(x)) for x in blobs]
#
#     if target.omega_prior == 'default':
#         if target.Vrot_prior is not None:
#             Vrot_prior = target.Vrot_prior(vtaneq) + np.log(v_critical)
#             lnL += Vrot_prior
#
#     return lnL, unified_blobs



def log_likelihood_INTERP(sampling_variables, verbose=False):
    """
    Compute the log-likelihood for the INETRP method.

        Args:
        sampling_variables (list or np.ndarray): Current parameter values being sampled.

    Key variables:
        Mini, logAge, MH, omega, Inclination, parallax, extinction (floats): Stellar parameters.
        sigma_adjustment_parameter (int): Determines whether observational errors are rescaled.
        Model_interp: Interpolated model used to compute stellar magnitudes and radii.

    Returns:
        tuple: (lnL, blobs)
            lnL (float): Log-likelihood value.
            blobs (list): List of diagnostic outputs including magnitudes, vsini, parallax, chi-squared, vtaneq, Req.
    """

    if isinstance(sampling_variables, list):
        sampling_variables = np.array(sampling_variables)

    if 'Mini' in target.sampling_variables_list:
        Mini_index = target.sampling_variables_list.index('Mini')
        Mini = sampling_variables[Mini_index]
    elif target.known_Mini is not None:
        Mini = target.known_Mini

    if 'logAge' in target.sampling_variables_list:
        logAge_index = target.sampling_variables_list.index('logAge')
        logAge = sampling_variables[logAge_index]
    elif target.known_logAge is not None:
        logAge = target.known_logAge

    if 'MH' in target.sampling_variables_list:
        MH_index = target.sampling_variables_list.index('MH')
        MH = sampling_variables[MH_index]
    elif target.known_MH is not None:
        MH = target.known_MH

    if 'omega_ini' in target.sampling_variables_list:
        omega_ini_index = target.sampling_variables_list.index('omega_ini')
        omega_ini = sampling_variables[omega_ini_index]
    elif target.known_omega_ini is not None:
        omega_ini = target.known_omega_ini

    if 'Inclination' in target.sampling_variables_list:
        Inclination_index = target.sampling_variables_list.index('Inclination')
        Inclination = sampling_variables[Inclination_index]
    elif target.known_Inclination is not None:
        Inclination = target.known_Inclination

    if 'parallax' in target.sampling_variables_list:
        parallax_index = target.sampling_variables_list.index('parallax')
        parallax = sampling_variables[parallax_index]
    elif target.known_parallax is not None:
        parallax = target.known_parallax

    if 'extinction' in target.sampling_variables_list:
        extinction_index = target.sampling_variables_list.index('extinction')
        extinction = sampling_variables[extinction_index]
        extinction_err = 0
    else:
        extinction = target.extinction
        extinction_err = target.extinction_err

    if target.sigma_adjustment_parameter == 1:
        sigma = sampling_variables[target.sampling_variables_list.index('sigma')]
        obs_err_mags_sigma = np.ones(len(target.bands_list)) * sigma
        obs_err_mags_new = np.sqrt(target.obs_err_mags**2 + obs_err_mags_sigma**2)
    elif target.sigma_adjustment_parameter == 3:
        sigma_gaia = sampling_variables[target.sampling_variables_list.index('sigma_gaia')]
        sigma_tycho2 = sampling_variables[target.sampling_variables_list.index('sigma_tycho2')]
        sigma_2mass = sampling_variables[target.sampling_variables_list.index('sigma_2mass')]
        obs_err_mags_sigma = np.array([1, 1, 1, 0, 0, 0, 0, 0]) * sigma_gaia + \
                             np.array([0, 0, 0, 1, 1, 0, 0, 0]) * sigma_tycho2 + \
                             np.array([0, 0, 0, 0, 0, 1, 1, 1]) * sigma_2mass
        obs_err_mags_new = np.sqrt(target.obs_err_mags**2 + obs_err_mags_sigma**2)
    elif target.sigma_adjustment_parameter == 0:
        obs_err_mags_new = target.obs_err_mags
    else:
        raise ValueError('sigma_adjustment_parameter must be any of 0, 1, or 3.')

    if target.fixed_parameter_indices:
        for idx in sorted(target.fixed_parameter_indices):
            sampling_variables = np.insert(sampling_variables, idx, target.true_values[idx])

    # logL, Req, Rpol = target.Model_interp(sampling_variables[:len(target.interp_variables_list)])[0]

    y_interpolated = target.Model_interp(sampling_variables[:len(target.interp_variables_list)]).flatten()


    if np.isnan(y_interpolated).any():
        if verbose:
            print('Failed in interpolation.')
        return -np.inf, np.ones(len(target.bands_list) + 11)*np.nan

    Mass, omega, Req, Rpol = y_interpolated[:4]
    abs_mags_interpolated = y_interpolated[4:]


    obs_mags_model = cal_obs_mag(abs_mags_interpolated, parallax)
    if target.extinction_parameter == True:
        obs_mags_model_extincted, obs_mags_model_err_extincted = add_extinction(obs_mags_model, 0, target.bands_list, extinction, extinction_err, target.extinction_type, print_flag=False)
    else:
        obs_mags_model_extincted, obs_mags_model_err_extincted = obs_mags_model, 0
    final_mags = obs_mags_model_extincted

    final_err_mags = np.sqrt(obs_err_mags_new**2 + obs_mags_model_err_extincted**2)

    omega_angular = omega * np.sqrt(ct.GM_sun*Mass/(ct.R_sun * 1.5 * Rpol)**3)
    vtaneq = (omega_angular*Req*ct.R_sun).to('km/s').value
    vsini = vtaneq * np.sin(Inclination)
    # v_keplerian = np.sqrt(ct.GM_sun * Mass / (ct.R_sun * Req)).to('km/s').value
    v_critical = np.sqrt(ct.GM_sun * Mass / (ct.R_sun * 1.5 * Rpol)).to('km/s').value

    chi_photo = np.sum(((final_mags - target.obs_mags) / final_err_mags) ** 2)

    chi_parallax = ((parallax - target.parallax_obs)/target.parallax_obs_err)**2

    if np.isnan(target.vsini_obs) or np.isnan(target.vsini_obs_err):
        chi_vsini = 0
    else:
        chi_vsini = ((vsini - target.vsini_obs) / target.vsini_obs_err) ** 2

    if any(x is not None for x in target.true_values) and not all(x is not None for x in target.true_values):
        if target.known_omega_ini is not None and target.known_Inclination is not None:
            if np.isclose(target.known_omega_ini * np.sin(target.known_Inclination), 0):
                chi_vsini = 0

    if verbose:
        print(f'photometric Chi_square: {chi_photo}')
        print(f'parallax Chi_square: {chi_parallax}')
        print(f'vsini Chi_square: {chi_vsini}')

    chi_squared = chi_photo + chi_parallax + chi_vsini

    chi_squared = chi_squared.item()

    if np.isnan(chi_squared).any():
        chi_squared = np.nan_to_num(chi_squared, nan=np.inf)

    lnL0 = -0.5 * chi_squared

    if target.sigma_adjustment_parameter == 1 or target.sigma_adjustment_parameter == 3:
        lnL = lnL0 - np.sum(np.log(final_err_mags))
    elif target.sigma_adjustment_parameter == 0:
        lnL = lnL0

    blobs = final_mags.tolist()
    blobs.append(vsini)
    blobs.append(parallax)
    blobs.append(chi_squared)
    blobs.append(vtaneq)
    blobs.append(Mass)
    blobs.append(omega)
    blobs.append(Req)
    blobs.append(Rpol)
    blobs.append(lnL0)
    blobs.append(lnL)
    u = target.interp_u([Mini, logAge, MH, omega_ini, Inclination])
    blobs.append(u)
    unified_blobs = [float(np.squeeze(x)) for x in blobs]

    if target.known_omega_ini == 0:
        pass
    elif target.omega_ini_prior == 'default':
        if target.Vrot_prior is not None:
            Vrot_prior = target.Vrot_prior(vtaneq) + np.log(v_critical)
            if np.isnan(Vrot_prior):
                if verbose:
                    print('Failed in Vrot_prior calculation.')
                return -np.inf, np.ones(len(target.bands_list) + 11)*np.nan
            lnL += Vrot_prior

    return lnL, np.array(unified_blobs)


def log_likelihood_PARS(sampling_variables, verbose=False):
    """
    Compute the log-likelihood for the PARS models.

        Args:
        sampling_variables (list or np.ndarray): Current parameter values being sampled.

    Key variables:
        logL, Req, Rpol: Interpolated stellar properties.
        made_limb_darkening_dict (dict): Limb darkening coefficients for various metallicities.
        PARS_MH_offset (float): Metallicity offset for PARS.
        PARS_nz (int): Number of z grid points used in PARS modeling.

    Returns:
        tuple: (lnL, blobs)
            lnL (float): Log-likelihood value.
            blobs (list): List of diagnostics including magnitudes, vsini, parallax, chi-squared, vtaneq, Req, Rpol, logL.
    """

    if isinstance(sampling_variables, list):
        sampling_variables = np.array(sampling_variables)

    if 'Mini' in target.sampling_variables_list:
        Mini_index = target.sampling_variables_list.index('Mini')
        Mini = sampling_variables[Mini_index]
    elif target.known_Mini is not None:
        Mini = target.known_Mini

    if 'logAge' in target.sampling_variables_list:
        logAge_index = target.sampling_variables_list.index('logAge')
        logAge = sampling_variables[logAge_index]
    elif target.known_logAge is not None:
        logAge = target.known_logAge

    if 'MH' in target.sampling_variables_list:
        MH_index = target.sampling_variables_list.index('MH')
        MH = sampling_variables[MH_index]
    elif target.known_MH is not None:
        MH = target.known_MH

    if 'omega_ini' in target.sampling_variables_list:
        omega_ini_index = target.sampling_variables_list.index('omega_ini')
        omega_ini = sampling_variables[omega_ini_index]
    elif target.known_omega_ini is not None:
        omega_ini = target.known_omega_ini

    if 'Inclination' in target.sampling_variables_list:
        Inclination_index = target.sampling_variables_list.index('Inclination')
        Inclination = sampling_variables[Inclination_index]
    elif target.known_Inclination is not None:
        Inclination = target.known_Inclination

    if 'parallax' in target.sampling_variables_list:
        parallax_index = target.sampling_variables_list.index('parallax')
        parallax = sampling_variables[parallax_index]
    elif target.known_parallax is not None:
        parallax = target.known_parallax

    if 'extinction' in target.sampling_variables_list:
        extinction_index = target.sampling_variables_list.index('extinction')
        extinction = sampling_variables[extinction_index]
        extinction_err = 0
    else:
        extinction = target.extinction
        extinction_err = target.extinction_err

    if target.sigma_adjustment_parameter == 1:
        sigma = sampling_variables[target.sampling_variables_list.index('sigma')]
        obs_err_mags_sigma = np.ones(len(target.bands_list)) * sigma
        obs_err_mags_new = np.sqrt(target.obs_err_mags**2 + obs_err_mags_sigma**2)
    elif target.sigma_adjustment_parameter == 3:
        sigma_gaia = sampling_variables[target.sampling_variables_list.index('sigma_gaia')]
        sigma_tycho2 = sampling_variables[target.sampling_variables_list.index('sigma_tycho2')]
        sigma_2mass = sampling_variables[target.sampling_variables_list.index('sigma_2mass')]
        obs_err_mags_sigma = np.array([1, 1, 1, 0, 0, 0, 0, 0]) * sigma_gaia + \
                             np.array([0, 0, 0, 1, 1, 0, 0, 0]) * sigma_tycho2 + \
                             np.array([0, 0, 0, 0, 0, 1, 1, 1]) * sigma_2mass
        obs_err_mags_new = np.sqrt(target.obs_err_mags**2 + obs_err_mags_sigma**2)
    elif target.sigma_adjustment_parameter == 0:
        obs_err_mags_new = target.obs_err_mags
    else:
        raise ValueError('sigma_adjustment_parameter must be any of 0, 1, or 3.')


    if target.fixed_parameter_indices:
        for idx in sorted(target.fixed_parameter_indices):
            sampling_variables = np.insert(sampling_variables, idx, target.true_values[idx])

    # logL, Req, Rpol = target.Model_interp(sampling_variables[:len(target.interp_variables_list)])[0]
    if target.iso_name.lower() == 'PARSEC'.lower():
        Mass, omega, logL, Req, Rpol = target.Model_interp(sampling_variables[:len(target.interp_variables_list)]).flatten()
        omega_angular = omega * np.sqrt(ct.GM_sun*Mass/(ct.R_sun * 1.5 * Rpol)**3)
        omega_keplerian = np.sqrt(ct.GM_sun * Mass / (ct.R_sun * Req)**3)
        omega_k = omega_angular / omega_keplerian
        omega_k = omega_k.value
        v_critical = np.sqrt(ct.GM_sun * Mass / (ct.R_sun * 1.5 * Rpol)).to('km/s').value
    elif target.iso_name.lower() == 'MIST'.lower():
        Mass, omega, logL, Req = target.Model_interp(sampling_variables[:len(target.interp_variables_list)]).flatten()
        Rpol = Req
        omega_angular = omega * np.sqrt(ct.GM_sun*Mass/(ct.R_sun * Req)**3)
        omega_k = omega
        v_critical = np.sqrt(ct.GM_sun*Mass/(ct.R_sun * Req)).to('km/s').value

    vtaneq = (omega_angular * Req * ct.R_sun).to('km/s').value
    vsini = vtaneq * np.sin(Inclination)
    distance_pc = cal_parsec(parallax)
    distance_cm = distance_pc * units.parsec.to('cm')

    output_rearramge_index = [list(target.made_limb_darkening_dict.values())[0].bands.index(band) for band in target.bands_list]

    MH_PARS = MH + target.PARS_MH_offset # default: MH_offset=-0.110

    if MH_PARS in list(target.made_limb_darkening_dict.keys()):
        try:
            star = st.Star(omega_k, 10**logL, Mass, Req, distance_cm, target.PARS_nz, target.made_limb_darkening_dict[MH_PARS])
            original_mags = star.integrate(Inclination)
            PARS_mags_list = [original_mags[index] for index in output_rearramge_index]
            obs_mags_model = np.array(PARS_mags_list)
        except:
            if verbose:
                print('Failed in PARS calculation.')
            return -np.inf, np.ones(len(target.bands_list) + 12)*np.nan
    else:
        sorted_values = sorted(target.made_limb_darkening_dict.keys())
        lower_value = max((v for v in sorted_values if v < MH_PARS), default=None)
        upper_value = min((v for v in sorted_values if v > MH_PARS), default=None)
        closest_dict = {v: target.made_limb_darkening_dict[v] for v in [lower_value, upper_value] if v is not None}
        # print(closest_dict.keys())
        PARS_dict = {}
        for key in closest_dict.keys():
            limb_darkening_file = closest_dict[key]
            try:
                star = st.Star(omega_k, 10**logL, Mass, Req, distance_cm, target.PARS_nz, limb_darkening_file)
                original_mags = star.integrate(Inclination)
                PARS_mags_list = [original_mags[index] for index in output_rearramge_index]
                model_obs_mags_array = np.array(PARS_mags_list)
                PARS_dict[key] = model_obs_mags_array
            except:
                if verbose:
                    print('Failed in PARS calculation.')
                return -np.inf, np.ones(len(target.bands_list) + 12)*np.nan

        interp_MH = interp1d(np.array(list(PARS_dict.keys())), np.array(list(PARS_dict.values())).T)
        obs_mags_model = interp_MH(MH_PARS)
    # print(f'obs_mags_model={obs_mags_model}')

    if target.extinction_parameter == True:
        obs_mags_model_extincted, obs_mags_model_err_extincted = add_extinction(obs_mags_model, 0, target.bands_list, extinction, extinction_err, target.extinction_type, print_flag=False)
    else:
        obs_mags_model_extincted, obs_mags_model_err_extincted = obs_mags_model, 0

    final_mags = obs_mags_model_extincted
    final_err_mags = np.sqrt(obs_err_mags_new**2 + obs_mags_model_err_extincted**2)

    chi_photo = np.sum(((final_mags - target.obs_mags) / final_err_mags) ** 2)

    chi_parallax = ((parallax - target.parallax_obs) / target.parallax_obs_err) ** 2

    if np.isnan(target.vsini_obs) or np.isnan(target.vsini_obs_err):
        chi_vsini = 0
    else:
        chi_vsini = ((vsini - target.vsini_obs) / target.vsini_obs_err) ** 2

    if any(x is not None for x in target.true_values) and not all(x is not None for x in target.true_values):
        if target.known_omega_ini is not None and target.known_Inclination is not None:
            if np.isclose(target.known_omega_ini * np.sin(target.known_Inclination), 0):
                chi_vsini = 0

    if verbose:
        print(f'photometric Chi_square: {chi_photo}')
        print(f'parallax Chi_square: {chi_parallax}')
        print(f'vsini Chi_square: {chi_vsini}')

    chi_squared = chi_photo + chi_parallax + chi_vsini

    chi_squared = chi_squared.item()

    if np.isnan(chi_squared).any():
        chi_squared = np.nan_to_num(chi_squared, nan=np.inf)

    lnL0 = -0.5 * chi_squared

    if target.sigma_adjustment_parameter == 1 or target.sigma_adjustment_parameter == 3:
        lnL = lnL0 - np.sum(np.log(final_err_mags))
    elif target.sigma_adjustment_parameter == 0:
        lnL = lnL0

    blobs = final_mags.tolist()
    blobs.append(vsini)
    blobs.append(parallax)
    blobs.append(chi_squared)
    blobs.append(vtaneq)
    blobs.append(Mass)
    blobs.append(omega)
    blobs.append(Req)
    blobs.append(Rpol)
    blobs.append(logL)
    blobs.append(lnL0)
    blobs.append(lnL)
    u = target.interp_u([Mini, logAge, MH, omega_ini, Inclination])
    blobs.append(u)
    unified_blobs = [float(np.squeeze(x)) for x in blobs]

    if target.Vrot_prior is not None:
        Vrot_prior = target.Vrot_prior(vtaneq) + np.log(v_critical)
        if np.isnan(Vrot_prior):
            if verbose:
                print('Failed in Vrot_prior calculation.')
            return -np.inf, np.ones(len(target.bands_list) + 12)*np.nan
        else:
            lnL += Vrot_prior

    return lnL, np.array(unified_blobs)



# def log_likelihood_NN(sampling_variables):
#     """
#     Compute the log-likelihoodfor the NN method.
#
#     Args:
#         sampling_variables (list or np.ndarray): Current parameter values being sampled.
#
#     Key variables:
#         NN_model (torch model): Pre-trained neural network for stellar magnitudes.
#         NN_scalers (dict): Dictionary of scalers for input/output normalization.
#         logL, Req, Rpol: Interpolated stellar properties.
#
#     Returns:
#         tuple: (lnL, blobs)
#             lnL (float): Log-likelihood value.
#             blobs (list): Diagnostics including magnitudes, vsini, parallax, chi-squared, vtaneq, Req, Rpol, logL.
#     """
#
#
#     if isinstance(sampling_variables, list):
#         sampling_variables = np.array(sampling_variables)
#
#     if 'Mass' in target.sampling_variables_list:
#         Mass_index = target.sampling_variables_list.index('Mass')
#         Mass = sampling_variables[Mass_index]
#     elif target.known_Mass is not None:
#         Mass = target.known_Mass
#
#     if 'logAge' in target.sampling_variables_list:
#         logAge_index = target.sampling_variables_list.index('logAge')
#         logAge = sampling_variables[logAge_index]
#     elif target.known_logAge is not None:
#         logAge = target.known_logAge
#
#     if 'MH' in target.sampling_variables_list:
#         MH_index = target.sampling_variables_list.index('MH')
#         MH = sampling_variables[MH_index]
#     elif target.known_MH is not None:
#         MH = target.known_MH
#
#     if 'omega' in target.sampling_variables_list:
#         omega_index = target.sampling_variables_list.index('omega')
#         omega = sampling_variables[omega_index]
#     elif target.known_omega is not None:
#         omega = target.known_omega
#
#     if 'Inclination' in target.sampling_variables_list:
#         Inclination_index = target.sampling_variables_list.index('Inclination')
#         Inclination = sampling_variables[Inclination_index]
#     elif target.known_Inclination is not None:
#         Inclination = target.known_Inclination
#
#     if 'parallax' in target.sampling_variables_list:
#         parallax_index = target.sampling_variables_list.index('parallax')
#         parallax = sampling_variables[parallax_index]
#     elif target.known_parallax is not None:
#         parallax = target.known_parallax
#
#     if 'extinction' in target.sampling_variables_list:
#         extinction_index = target.sampling_variables_list.index('extinction')
#         extinction = sampling_variables[extinction_index]
#         extinction_err = 0
#     else:
#         extinction = target.extinction
#         extinction_err = target.extinction_err
#
#     if target.sigma_adjustment_parameter == 1:
#         sigma = sampling_variables[target.sampling_variables_list.index('sigma')]
#         obs_err_mags_sigma = np.ones(len(target.bands_list)) * sigma
#         obs_err_mags_new = np.sqrt(target.obs_err_mags**2 + obs_err_mags_sigma**2)
#     elif target.sigma_adjustment_parameter == 3:
#         sigma_gaia = sampling_variables[target.sampling_variables_list.index('sigma_gaia')]
#         sigma_tycho2 = sampling_variables[target.sampling_variables_list.index('sigma_tycho2')]
#         sigma_2mass = sampling_variables[target.sampling_variables_list.index('sigma_2mass')]
#         obs_err_mags_sigma = np.array([1, 1, 1, 0, 0, 0, 0, 0]) * sigma_gaia + \
#                              np.array([0, 0, 0, 1, 1, 0, 0, 0]) * sigma_tycho2 + \
#                              np.array([0, 0, 0, 0, 0, 1, 1, 1]) * sigma_2mass
#         obs_err_mags_new = np.sqrt(target.obs_err_mags**2 + obs_err_mags_sigma**2)
#     elif target.sigma_adjustment_parameter == 0:
#         obs_err_mags_new = target.obs_err_mags
#     else:
#         raise ValueError('sigma_adjustment_parameter must be any of 0, 1, or 3.')
#
#
#     if target.fixed_parameter_indices:
#         for idx in sorted(target.fixed_parameter_indices):
#             sampling_variables = np.insert(sampling_variables, idx, target.true_values[idx])
#
#     logL, Req, Rpol = target.Model_interp(sampling_variables[:len(target.interp_variables_list)])[0]
#
#     ############################## NN ##############################
#     target_x = np.array([[Mass, MH, Req, logL, omega, Inclination]])
#
#     mass_scaler = target.NN_scalers['mass']
#     mass = mass_scaler.transform(target_x[:, [0]])
#
#     mh_scaler = target.NN_scalers['mh']
#     mh = mh_scaler.transform(target_x[:, [1]])
#
#     req_scaler = target.NN_scalers['log1p_req']
#     req = req_scaler.transform(np.log1p(target_x[:, [2]]))  # Attention on log1p
#
#     logl_scaler = target.NN_scalers['logl']
#     logl = logl_scaler.transform(target_x[:, [3]])
#
#     omega = target_x[:, [4]]  # no changes for omega
#
#     incl_scaler = target.NN_scalers['incl']
#     incl = incl_scaler.transform(target_x[:, [5]])
#
#     y_scaler = target.NN_scalers['y']
#
#     # combine them
#     x_scaled = np.concatenate([mass, mh, req, logl, omega, incl], axis=1)
#
#     # force CPU here
#     x_tensor = torch.tensor(x_scaled, dtype=torch.float32, device="cpu")
#
#     # also force model to CPU
#     target.NN_model.to("cpu")
#     target.NN_model.eval()
#
#     with torch.no_grad():
#         y_pred_scaled = target.NN_model(x_tensor).detach().cpu().numpy()
#         y_pred = y_scaler.inverse_transform(y_pred_scaled)
#     ############################## NN ##############################
#
#     if np.isnan(y_pred).any():
#         return -np.inf, np.ones(len(target.bands_list) + 12)*np.nan
#
#     obs_mags_model = cal_obs_mag(y_pred.flatten(), parallax)
#
#     if target.extinction_parameter == True:
#         obs_mags_model_extincted, obs_mags_model_err_extincted = add_extinction(obs_mags_model, 0, target.bands_list, extinction, extinction_err, target.extinction_type, print_flag=False)
#     else:
#         obs_mags_model_extincted, obs_mags_model_err_extincted = obs_mags_model, 0
#     final_mags = obs_mags_model_extincted
#     final_err_mags = np.sqrt(obs_err_mags_new ** 2 + obs_mags_model_err_extincted ** 2)
#
#     omega_angular = omega * np.sqrt(ct.GM_sun*Mass/(ct.R_sun * 1.5 * Rpol)**3)
#     vtaneq = (omega_angular*Req*ct.R_sun).to('km/s').value
#     vsini = vtaneq * np.sin(Inclination)
#     # v_keplerian = np.sqrt(ct.GM_sun * Mass / (ct.R_sun * Req)).to('km/s').value
#     v_critical = np.sqrt(ct.GM_sun * Mass / (ct.R_sun * 1.5 * Rpol)).to('km/s').value
#
#     # print('NN final_err_mags = ', final_err_mags)
#
#     if np.isnan(target.vsini_obs) or np.isnan(target.vsini_obs_err):
#         chi_vsini = 0
#     else:
#         chi_vsini = ((vsini - target.vsini_obs) / target.vsini_obs_err) ** 2
#
#     chi_squared = np.sum(((final_mags - target.obs_mags) / final_err_mags) ** 2, axis=0) \
#                   + ((parallax - target.parallax_obs) / target.parallax_obs_err) ** 2 + chi_vsini
#
#     chi_squared = chi_squared.item()
#
#     if np.isnan(chi_squared).any():
#         chi_squared = np.nan_to_num(chi_squared, nan=np.inf)
#
#     lnL0 = -0.5 * chi_squared
#
#     if target.sigma_adjustment_parameter == 1 or target.sigma_adjustment_parameter == 3:
#         lnL = lnL0 - np.sum(np.log(final_err_mags))
#     elif target.sigma_adjustment_parameter == 0:
#         lnL = lnL0
#
#     blobs = final_mags.tolist()
#     blobs.append(vsini)
#     blobs.append(parallax)
#     blobs.append(chi_squared)
#     blobs.append(vtaneq)
#     blobs.append(Req)
#     blobs.append(Rpol)
#     blobs.append(logL)
#     blobs.append(lnL0)
#     blobs.append(lnL)
#     unified_blobs = [float(np.squeeze(x)) for x in blobs]
#
#     if target.known_omega == 0:
#         pass
#     elif target.omega_prior == 'default':
#         if target.Vrot_prior is not None:
#             Vrot_prior = target.Vrot_prior(vtaneq) + np.log(v_critical)
#             lnL += Vrot_prior
#
#     return lnL, unified_blobs


log_prior_function = log_prior

def log_posterior(sampling_variables):

    """
    Compute the posterior log-posterior for a given set of sampling variables.

    Args:
        sampling_variables (list or np.ndarray): Current parameter values being sampled.

    Process:
        - Evaluates the prior probability (log_prior).
        - If prior is invalid (-inf), returns immediately.
        - Otherwise, evaluates the likelihood (log_likelihood_xxx) depending on METHOD.
        - Sums prior and likelihood to obtain posterior probability.

    Returns:
        tuple: (log_prob, blobs)
            log_prob (float): Posterior log-posterior.
            blobs (list): Diagnostics from likelihood evaluation.
    """

    log_pr = log_prior_function(sampling_variables)

    if not np.isfinite(log_pr):
        if target.METHOD == "DIRECT":
            blobs = np.ones(len(target.bands_list) + 11)*np.nan
        elif target.METHOD == "PARS":
            blobs = np.ones(len(target.bands_list) + 12)*np.nan
        return -np.inf, blobs

    if target.METHOD in ['DIRECT']:
        log_lkh, blobs = log_likelihood_INTERP(sampling_variables)
    elif target.METHOD == "PARS":
        log_lkh, blobs = log_likelihood_PARS(sampling_variables)
    # elif target.METHOD == "NN":
    #     log_lkh, blobs = log_likelihood_NN(sampling_variables)

    log_prob = log_pr + log_lkh

    return log_prob, blobs



def running_in_jupyter():
    try:
        from IPython import get_ipython
        if 'IPKernelApp' in get_ipython().config:
            return True
    except Exception:
        pass



def runnning_in_pool(
    initial_steps, max_hours, emcee_moves, pool, dtype, use_tqdm,
    min_length_ratio_list, max_tau_relative_difference_list
):
    """
    Adaptive MCMC sampling with convergence monitoring:
      1) Tau stability: |tau_0.5 - tau_0.3| / max(tau_0.5, tau_0.3) < 15% (2 consecutive)
      3) min(ESS) > 1000
      Stops when either convergence or time limit (max_hours) reached.
    """
    sampler = emcee.EnsembleSampler(
        target.nwalkers,
        target.ndim,
        log_posterior,
        moves=emcee_moves,
        pool=pool,
        blobs_dtype=dtype,
    )

    # ----- Initial sampling -----
    start_time_total = time.time()
    if not use_tqdm:
        print(f"Starting Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("🔄 Starting initial sampling...")
    sampler.run_mcmc(target.initial_states, initial_steps, progress=use_tqdm)
    if not use_tqdm:
        elapsed = int(time.time() - start_time_total)
        print(f"Sampling time: {elapsed//3600:02d}:{(elapsed%3600)//60:02d}:{elapsed%60:02d}")

    # ===== Main adaptive loop =====
    stable_tau_counter = 0
    max_runtime = max_hours * 3600  # convert hours to seconds

    prev_state = None

    while True:
        total_samples = sampler.iteration
        elapsed_total = time.time() - start_time_total

        if elapsed_total > 2 * 3600:
            min_length_ratio = min_length_ratio_list[2]
            max_tau_relative_difference = max_tau_relative_difference_list[2]
            if prev_state != "2h":
                print(f"⏰ Runtime > 2h → Relaxing criteria 💤: "
                      f"N/τ > {min_length_ratio}, τ_diff < {max_tau_relative_difference:.0%}")
                prev_state = "2h"

        elif elapsed_total > 1 * 3600:
            min_length_ratio = min_length_ratio_list[1]
            max_tau_relative_difference = max_tau_relative_difference_list[1]
            if prev_state != "1h":
                print(f"⌛ Runtime > 1h → Slightly relaxed criteria 😌: "
                      f"N/τ > {min_length_ratio}, τ_diff < {max_tau_relative_difference:.0%}")
                prev_state = "1h"

        else:
            min_length_ratio = min_length_ratio_list[0]
            max_tau_relative_difference = max_tau_relative_difference_list[0]
            if prev_state != "standard":
                print(f"🚀 Standard criteria in use 💪: "
                      f"N/τ > {min_length_ratio}, τ_diff < {max_tau_relative_difference:.0%}")
                prev_state = "standard"

        if elapsed_total > max_runtime:
            raise TimeoutError(f"⏰ Sampling exceeded {max_hours} hours limit — stopping execution.")

        chain = sampler.get_chain()  # shape: (steps, walkers, dims)

        try:
            # --- 1️⃣ Compute tau on the last 50% and 30% of samples ---
            S = chain.shape[0]
            start_50 = int(S * 0.5)
            start_30 = int(S * 0.7)

            tau_05 = emcee.autocorr.integrated_time(chain[start_50:], tol=0)
            tau_03 = emcee.autocorr.integrated_time(chain[start_30:], tol=0)

            # --- robust tau_late computation ---
            taus = [np.nanmax(t) for t in [tau_05, tau_03] if np.any(np.isfinite(t))]
            tau_late = np.max(taus) if len(taus) > 0 else 1000  # fallback in case of NaNs

            # --- relative difference, safe handling ---
            tau05_max = np.nanmax(tau_05) if np.any(np.isfinite(tau_05)) else np.nan
            tau03_max = np.nanmax(tau_03) if np.any(np.isfinite(tau_03)) else np.nan

            if np.isfinite(tau05_max) and np.isfinite(tau03_max):
                rel_diff = abs(tau05_max - tau03_max) / max(tau05_max, tau03_max)
            else:
                rel_diff = np.nan

            # --- 2️⃣ ESS using ArviZ ---
            n_dim = sampler.get_chain().shape[-1]
            var_names = target.sampling_variables_list

            burnin = int(10 * tau_late)
            length_ratio = (total_samples - burnin) / tau_late

            # idata = az.from_emcee(sampler)

            # --- Display checkpoint summary ---
            hours = int(elapsed_total // 3600)
            minutes = int((elapsed_total % 3600) // 60)
            print(f"\n🔍 Checkpoint @ {total_samples} samples (elapsed {hours}h {minutes}m)")
            print(f"  τ₀․₅={np.max(tau_05):.1f}, τ₀․₃={np.max(tau_03):.1f}, rel_diff={rel_diff*100:.1f}%")
            print(f"  Length ratio (after burn=10τ): {length_ratio:.2f} vs requirement {min_length_ratio}")

            # --- 3️⃣ Convergence criteria ---
            stable_tau = rel_diff < max_tau_relative_difference
            length_ok = length_ratio >= min_length_ratio

            if stable_tau and length_ok:
                stable_tau_counter += 1
            else:
                stable_tau_counter = 0

            if (stable_tau_counter >= 2) and length_ok:
                print(f"✅ Converged: τ stable (2×) rel_diff < {max_tau_relative_difference}, N/τ > {min_length_ratio}")
                break

        except emcee.autocorr.AutocorrError:
            print("⚠️ ACT estimation failed, chain too short. Continuing...")

        # --- Continue sampling ---
        if not use_tqdm:
            start_time = time.time()
        check_interval = int(10 * tau_late)
        check_interval = max(2000, min(check_interval, 20000))
        if np.isnan(check_interval):
            check_interval = 10000
        print(f"⏱ Adaptive check_interval = {check_interval} (10×τ_late={10 * tau_late:.0f})")
        sampler.run_mcmc(None, check_interval, progress=use_tqdm)
        if not use_tqdm:
            elapsed = int(time.time() - start_time)
            print(f"Sampling time: {elapsed//3600:02d}:{(elapsed%3600)//60:02d}:{elapsed%60:02d}")

    return sampler



def run_test_sampling(
    nsteps_test=None,
    nsteps_evaluate=None,
    use_tqdm=True,
    show_fig=True,
    n_processes=8,
    emcee_moves=None,
    emcee_StretchMove_a_value=2.5,
    emcee_StretchMove_fraction=0.5,
    emcee_DEMove_gamma0_value=0.5,
    emcee_DEMove_fraction=0.3,
    emcee_KDEMove_fraction=0.2,
    corner_age_type="linear",
    corner_inc_type="degree",
    quantiles=None,
    hist_bin_factor=3,
    dig_num=2,
    save_path=".",
    save_samples=False,
    save_64=False,
):
    """
    Run a shorter MCMC sampling for testing and visualization.

    Special logic:
      - If target.omega_ini_prior == 'default', we do a short run first to estimate a rough mass,
        select a default Vrot prior, then run again for the final test samples.

    Returns
    -------
    samples_test_emcee : np.ndarray
        Flattened posterior samples after discard.
    target : object
        Updated target (may include chosen default Vrot prior).
    """

    # -------------------------
    # Defaults
    # -------------------------
    if nsteps_test is None:
        nsteps_test = 3000
    if nsteps_evaluate is None:
        nsteps_evaluate = nsteps_test
    if quantiles is None:
        quantiles = [0.16, 0.5, 0.84]

    if save_samples or not show_fig:
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    if save_path[-1] == '/':
        save_path = save_path[:-1]

    # -------------------------
    # Helpers
    # -------------------------
    def _build_emcee_moves():
        """Construct emcee moves list (or None) with strict parameter consistency."""
        if emcee_moves is not None:
            return emcee_moves

        params = [
            emcee_StretchMove_a_value,
            emcee_StretchMove_fraction,
            emcee_DEMove_gamma0_value,
            emcee_DEMove_fraction,
            emcee_KDEMove_fraction,
        ]

        if all(x is None for x in params):
            return None

        if not all(x is not None for x in params):
            raise ValueError("If using moves, all move parameters must be set (or all None).")

        return [
            (emcee.moves.StretchMove(a=emcee_StretchMove_a_value), emcee_StretchMove_fraction),
            (emcee.moves.DEMove(gamma0=emcee_DEMove_gamma0_value), emcee_DEMove_fraction),
            (emcee.moves.KDEMove(), emcee_KDEMove_fraction),
        ]

    def _should_disable_mp_pool():
        """
        Keep your original special-case:
        - METHOD == 'NN'
        - Linux
        - running in Jupyter
        """
        return (
            target.METHOD == "NN"
            and platform.system() == "Linux"
            and running_in_jupyter()
        )

    def _run_once(progress_flag: bool, nsteps_test: float):
        """
        Run one emcee sampling and return (sampler, flat_samples).
        """
        # if target.METHOD == "DIRECT":
        #     dtype = [("blob", float, (18,))]
        # elif target.METHOD == "PARS":
        #     dtype = [("blob", float, (19,))]

        dtype = [("blob", np.ndarray)]

        moves_local = _build_emcee_moves()

        if _should_disable_mp_pool():
            sampler = emcee.EnsembleSampler(
                target.nwalkers,
                target.ndim,
                log_posterior,
                moves=moves_local,
                blobs_dtype=dtype,
            )
            sampler.run_mcmc(target.initial_states, nsteps_test, progress=progress_flag)
        else:
            with mp.Pool(processes=n_processes) as pool:
                sampler = emcee.EnsembleSampler(
                    target.nwalkers,
                    target.ndim,
                    log_posterior,
                    moves=moves_local,
                    pool=pool,
                    blobs_dtype=dtype,
                )
                sampler.run_mcmc(target.initial_states, nsteps_test, progress=progress_flag)

        discard = int(sampler.iteration / 10)
        flat = sampler.get_chain(discard=discard, thin=1, flat=True)
        return sampler, flat

    def _estimate_rough_mass(samples_flat: np.ndarray):
        """Estimate rough mass summary from marginal posterior."""
        mass_index = target.sampling_variables_list.index("Mini")
        mass_samples = samples_flat[:, mass_index]

        qs = np.percentile(mass_samples, [100 * q for q in quantiles])
        return {
            "median": qs[1],
            "q_low": qs[0],
            "q_high": qs[2],
            "plus": qs[2] - qs[1],
            "minus": qs[1] - qs[0],
            "samples": mass_samples,
        }

    def _make_corner_inputs(samples_flat: np.ndarray):
        """Prepare samples/labels for corner plotting with age transform."""
        if corner_age_type not in ("log", "linear"):
            raise ValueError('"corner_age_type" must be "log" or "linear".')

        corner_samples = samples_flat.copy()
        corner_labels = target.sampling_variables_list.copy()

        if corner_age_type == "linear":
            age_index = target.sampling_variables_list.index("logAge")
            corner_samples[:, age_index] = 10 ** (corner_samples[:, age_index] - 6)
            corner_labels[age_index] = "Age"

        if target.extinction_parameter:
            ext_idx = target.sampling_variables_list.index("extinction")
            corner_labels[ext_idx] = target.extinction_type

        return corner_samples, corner_labels

    def _build_name_prefix():
        """Build filename prefix for outputs."""
        base = (
            f"{target.name.replace(' ', '_')}"
            f"{target.cluster_name}_{target.METHOD}"
            f"_sigma{target.sigma_adjustment_parameter}_"
        )
        if target.extinction_parameter == 0:
            return base + "extinction0"
        return base + f"{target.extinction_type}"

    # -------------------------
    # Run sampling (1 or 2-stage)
    # -------------------------
    name_pre = _build_name_prefix()

    if target.omega_ini_prior == "default":
        if target.Vrot_prior is None:
            # Stage 1: no progress bar (your original behavior), just to estimate mass
            print("Estimating rough mass...")
            # sampler_test, samples_test_emcee = _run_once(progress_flag=False, nsteps_test=nsteps_evaluate)
            sampler_test, samples_test_emcee = _run_once(progress_flag=use_tqdm, nsteps_test=nsteps_evaluate)

            rough_mass = _estimate_rough_mass(samples_test_emcee)
            key, Vrot_prior = select_Vrot_prior_by_rough_mass(
                rough_mass["median"], log_prior_Vrot_dict
            )
            target.Vrot_prior = Vrot_prior

            if key is None:
                print("====> No default Vrot prior available.")
            else:
                print("====> Using default Vrot prior:", key)

        # Stage 2: actual test run
        print(f"Running a test with {nsteps_test} samples...")
        sampler_test, samples_test_emcee = _run_once(progress_flag=use_tqdm, nsteps_test=nsteps_test)

    else:
        print(f"Running a test with {nsteps_test} samples...")
        sampler_test, samples_test_emcee = _run_once(progress_flag=use_tqdm, nsteps_test=nsteps_test)

    # -------------------------
    # Corner plot
    # -------------------------
    # corner_samples, corner_labels = _make_corner_inputs(samples_test_emcee)

    samples_flat = samples_test_emcee

    # -------------------------
    # Prepare corner samples/labels (age transform)
    # -------------------------
    title_map = {
        'Mini': r"$M_{\mathrm{ini}}\ (M_\odot)$",
        'MH': r"$[\mathrm{M/H}]$",
        'omega_ini': r"$\omega_{\mathrm{ini}}$",
        'parallax': r"$\varpi\ (\mathrm{mas})$",
        'sigma': r"$\sigma_{\mathrm{adjust}}\ (\mathrm{mag})$",
        'sigma_gaia': r"$\sigma_{\mathrm{adjust},\ Gaia}\ (\mathrm{mag})$",
        'sigma_tycho2': r"$\sigma_{\mathrm{adjust,\ Tycho-2}}\ (\mathrm{mag})$",
        'sigma_2mass': r"$\sigma_{\mathrm{adjust,\ 2MASS}}\ (\mathrm{mag})$",
    }

    extinction_map = {
        'A_V': r"$A_V\ (\mathrm{mag})$",
        'EB_V': r"$E(B-V)\ (\mathrm{mag})$",
        'EBP_RP': r"$E(G_{\mathrm{BP}} - G_{\mathrm{RP}})\ (\mathrm{mag})$",
    }

    corner_sampling_variables_list = [
        r"$\log(\tau/\mathrm{yr})$" if v == 'logAge' and corner_age_type == 'log'
        else r"$\tau\ (\mathrm{Myr})$" if v == 'logAge' and corner_age_type == 'linear'
        else r"$i\ (\mathrm{rad})$" if v == 'Inclination' and corner_inc_type == 'radian'
        else r"$i\ (^\circ)$" if v == 'Inclination' and corner_inc_type == 'degree'
        else extinction_map[target.extinction_type] if v == 'extinction'
        else title_map[v]
        for v in target.sampling_variables_list
    ]

    corner_samples_flat = samples_flat.copy()
    if corner_age_type == "log" and corner_inc_type == "radian":
        pass
    elif corner_age_type == "linear" and corner_inc_type == "radian":
        age_index = target.sampling_variables_list.index("logAge")
        corner_samples_flat[:, age_index] = 10 ** (corner_samples_flat[:, age_index] - 6)
    elif corner_age_type == "log" and corner_inc_type == "degree":
        if 'Inclination' in target.sampling_variables_list:
            inc_index = target.sampling_variables_list.index("Inclination")
            corner_samples_flat[:, inc_index] = np.rad2deg(corner_samples_flat[:, inc_index])
    elif corner_age_type == "linear" and corner_inc_type == "degree":
        age_index = target.sampling_variables_list.index("logAge")
        if 'Inclination' in target.sampling_variables_list:
            inc_index = target.sampling_variables_list.index("Inclination")
            corner_samples_flat[:, inc_index] = np.rad2deg(corner_samples_flat[:, inc_index])
        corner_samples_flat[:, age_index] = 10 ** (corner_samples_flat[:, age_index] - 6)

    else:
        raise ValueError('"corner_age_type" must be either "log" or "linear"; "corner_inc_type" must be either "radian" or "degree".')

    fig = corner.corner(
        corner_samples_flat,
        labels=corner_sampling_variables_list,
        quantiles=quantiles,
        show_titles=True,
        hist_bin_factor=hist_bin_factor,
        title_fmt=f".{dig_num}f",
    )

    if not show_fig:
        fig.savefig(
            f"{save_path}/{name_pre}_steps{nsteps_test}.pdf",
            bbox_inches="tight",
        )
        plt.close(fig)

    # -------------------------
    # Diagnostics
    # -------------------------
    acceptance_fraction_arr = sampler_test.acceptance_fraction
    print("acceptance ratio:")
    for i, val in enumerate(acceptance_fraction_arr):
        if i % 9 == 0:
            print("\t", end="")
        print(f"{val:.2f}", end=" ")
        if (i + 1) % 9 == 0:
            print()
    if len(acceptance_fraction_arr) % 9 != 0:
        print()

    print(f"nsteps/60: {nsteps_test / 60:.2f}")
    print(
        "autocorrelation time:",
        np.ceil(sampler_test.get_autocorr_time(tol=0)).astype(int),
    )

    # -------------------------
    # Save samples
    # -------------------------

    if save_samples:
        if  save_64:
            samples = sampler_test.get_chain()
            log_prob = sampler_test.get_log_prob()
            blobs_full = sampler_test.get_blobs()["blob"]
        else:
            samples = sampler_test.get_chain().astype(np.float32)
            log_prob = sampler_test.get_log_prob().astype(np.float32)
            blobs_full = sampler_test.get_blobs()["blob"].astype(np.float32)
        np.savez_compressed(
            f"{save_path}/posterior_{name_pre}_steps{nsteps_test}.npz",
            samples=samples,
            log_prob=log_prob,
            blobs=blobs_full if blobs_full is not None else np.array(None, dtype=object),
            param_names=np.array(target.sampling_variables_list),
        )

    return sampler_test, target





def run_sampling(initial_steps=None,
                 use_tqdm=True,
                 max_hours=4,
                 n_processes=8,
                 emcee_moves=None,
                 emcee_StretchMove_a_value=None,
                 emcee_StretchMove_fraction=None,
                 emcee_DEMove_gamma0_value=None,
                 emcee_DEMove_fraction=None,
                 emcee_KDEMove_fraction=None,
                 min_length_ratio=None,
                 max_tau_relative_difference=None,
                 ):
    """
    Run full MCMC sampling until convergence or until reaching max samples.

    Args:
        initial_steps (int, optional): Interval (number of steps) between convergence checks.
            Defaults depend on METHOD (fewer-INTERP: 10000, PARS: 30000, NN: 60000).
        max_samples (int, optional): Maximum number of samples to run before stopping.
            Default is 100 × check_interval.
        convergence_ratio (float): Required ratio of total samples / autocorrelation time for convergence.
        n_processes (int): Number of parallel processes for multiprocessing.
        emcee_a_value (float): StretchMove 'a' parameter.
        emcee_a_fraction (float): Fraction of StretchMove in proposals.
        emcee_gamma0_value (float): DEMove gamma0 parameter.
        emcee_gamma0_fraction (float): Fraction of DEMove in proposals.

    Returns:
        emcee.EnsembleSampler: The sampler object containing the full chains.
    """

    start_time_final = time.time()
    if min_length_ratio is not None:
        if not isinstance(min_length_ratio, (list, tuple, np.ndarray)) or not len(min_length_ratio) == 3:
            raise ValueError(f"❌ `min_length_ratio` must be a list/tuple/array of length 3.")
    if max_tau_relative_difference is not None:
        if not isinstance(max_tau_relative_difference, (list, tuple, np.ndarray)) or not len(max_tau_relative_difference) == 3:
            raise ValueError(f"❌ `max_tau_relative_difference` must be a list/tuple/array of length 3.")

    if min_length_ratio is None:
        min_length_ratio = [70, 60, 50]
    if max_tau_relative_difference is None:
        max_tau_relative_difference = [0.20, 0.25, 0.30]

    if initial_steps is None:
        if target.METHOD == 'fewer-INTERP':
            initial_steps = 6000
        elif  target.METHOD == 'PARS':
            initial_steps = 12000
        elif target.METHOD == 'NN':
            initial_steps = 24000

    if min_length_ratio[2] < 50:
        print('We do not encourage you to use a "N/ACT < 50"!')

    dtype = [("blob", np.ndarray)]

    if all(x is not None for x in [emcee_moves]):
        emcee_moves = emcee_moves
    else:
        if all(x is None for x in [
            emcee_StretchMove_a_value,
            emcee_StretchMove_fraction,
            emcee_DEMove_gamma0_value,
            emcee_DEMove_fraction,
            emcee_KDEMove_fraction]):
            emcee_moves = None

        elif all(x is not None for x in [
            emcee_StretchMove_a_value,
            emcee_StretchMove_fraction,
            emcee_DEMove_gamma0_value,
            emcee_DEMove_fraction,
            emcee_KDEMove_fraction]):
            emcee_moves = [
                (emcee.moves.StretchMove(a=emcee_StretchMove_a_value), emcee_StretchMove_fraction),
                (emcee.moves.DEMove(gamma0=emcee_DEMove_gamma0_value), emcee_DEMove_fraction),
                (emcee.moves.KDEMove(), emcee_KDEMove_fraction)
            ]

        else:
            raise ValueError(
                'If using "moves", you need to set all 5 parameters: \n'
                'emcee_StretchMove_a_value, emcee_StretchMove_fraction,\n'
                'emcee_DEMove_gamma0_value, emcee_DEMove_fraction,\n'
                'emcee_KDEMove_fraction')

    if target.METHOD == 'NN' and platform.system() == "Linux" and running_in_jupyter():
        pool = None
        sampler = runnning_in_pool(initial_steps, max_hours, emcee_moves, pool, dtype, use_tqdm,
                                   min_length_ratio, max_tau_relative_difference)
    else:
        with mp.Pool(processes=n_processes) as pool:
            sampler = runnning_in_pool(initial_steps, max_hours, emcee_moves, pool, dtype, use_tqdm,
                                       min_length_ratio, max_tau_relative_difference)

    end_time_final = time.time()
    elapsed_final = int(end_time_final - start_time_final)
    hours_final = elapsed_final // 3600
    minutes_final = (elapsed_final % 3600) // 60
    seconds_final = elapsed_final % 60

    print(f"Total sampling time: {hours_final:02d}:{minutes_final:02d}:{seconds_final:02d}")

    return sampler





def save_chains_plot(
    sampler,
    show_fig=True,
    burnin_ACT=10,
    group_size=3,
    plot_samples="all",
    folder_suffix="",
    hist_bin_factor=3,
    dig_num=2,
    quantiles=None,
    corner_age_type="linear",
    corner_inc_type="degree",
    save_samples=True,
    save_64=False,
    plot_recovered_corner=True,
    plot_blobs_hist=True,
    blobs_hist_bins=80,
    blobs_hist_ncols=4,
):
    """
    Save diagnostic plots for MCMC chains.

    This function saves:
        1. Original corner plot for sampled parameters.
        2. Recovered-parameter corner plot:
            Mini      -> blob Mass
            omega_ini -> blob omega
        3. Trace plots for sampled parameters and blobs.
        4. Histogram summary of all blobs.

    Expected blob shape:
        (n_steps, n_walkers, n_blob)
    """

    import os
    import math
    import warnings
    from pathlib import Path

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import corner
    import emcee

    # -------------------------
    # Defaults
    # -------------------------
    if quantiles is None:
        quantiles = [0.16, 0.5, 0.84]

    if plot_samples != "all" and not isinstance(plot_samples, int):
        raise ValueError('"plot_samples" must be an integer or a string: "all".')

    # -------------------------
    # Helper functions
    # -------------------------
    def running_in_jupyter():
        try:
            from IPython import get_ipython
            shell = get_ipython().__class__.__name__
            return shell == "ZMQInteractiveShell"
        except Exception:
            return False

    def _normalize_blobs(blobs_in):
        """
        Convert emcee blobs to ndarray with shape:
            (n_steps, n_walkers, n_blob)
        """
        if blobs_in is None:
            raise ValueError("sampler.get_blobs() returned None.")

        arr = np.asarray(blobs_in)

        if arr.dtype.names is not None:
            if "blob" in arr.dtype.names:
                arr = arr["blob"]
            else:
                raise ValueError(f"Unexpected blob dtype names: {arr.dtype.names}")

        if arr.dtype == object:
            if arr.ndim == 2:
                arr = np.stack(
                    [
                        np.stack([arr[i, j] for j in range(arr.shape[1])], axis=0)
                        for i in range(arr.shape[0])
                    ],
                    axis=0,
                )
            elif arr.ndim == 1:
                arr = np.stack(arr, axis=0)
            else:
                arr = np.array(arr.tolist())

        arr = np.asarray(arr, dtype=np.float64)
        arr = np.squeeze(arr)

        if arr.ndim != 3:
            raise ValueError(
                f"Expected blobs_array shape (n_steps, n_walkers, n_blob), "
                f"but got {arr.shape}."
            )

        return arr

    def _latex_summary(x, digits=3):
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]

        if len(x) == 0:
            return None, None, None, None

        q16, q50, q84 = np.percentile(x, [16, 50, 84])
        err_minus = q50 - q16
        err_plus = q84 - q50

        label = rf"${q50:.{digits}f}^{{+{err_plus:.{digits}f}}}_{{-{err_minus:.{digits}f}}}$"
        return q16, q50, q84, label

    def _plot_blobs_hist(
            blobs_flat,
            blob_names,
            out_path,
            bins=80,
            ncols=4,
            digits=3,
            color="#6BAED6",
            shade_color="grey",
    ):
        hist_data = np.asarray(blobs_flat, dtype=np.float64)

        if hist_data.ndim == 1:
            hist_data = hist_data[:, None]

        hist_names = list(blob_names)

        if hist_data.shape[1] != len(hist_names):
            raise ValueError(
                f"blobs_flat has {hist_data.shape[1]} columns, "
                f"but blob_names has {len(hist_names)} entries."
            )

        n_hist = hist_data.shape[1]
        nrows = math.ceil(n_hist / ncols)

        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(4.2 * ncols, 3.2 * nrows),
            squeeze=False,
        )
        axes = axes.ravel()

        for i in range(n_hist):
            if hist_names[i] == "u":
                ax = axes[i]

                x = hist_data[:, i]
                x = x[np.isfinite(x)]

                if len(x) == 0:
                    ax.set_title(hist_names[i])
                    ax.text(
                        0.5,
                        0.5,
                        "No finite samples",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    continue

                q16, q50, q84, label = _latex_summary(x, digits=digits)

                label = (
                    rf"$1\sigma: {q50:.{digits}f}"
                    rf"^{{+{q84-q50:.{digits}f}}}"
                    rf"_{{-{q50-q16:.{digits}f}}}$"
                )

                ax.hist(
                    x,
                    bins=bins,
                    density=True,
                    alpha=0.65,
                    color=color,
                    edgecolor="none",
                )

                ax.axvspan(
                    q16,
                    q84,
                    color=shade_color,
                    alpha=0.25,
                    zorder=3,
                    label=label,
                )

                ax.axvline(
                    q50,
                    color="black",
                    linestyle="--",
                    linewidth=1.5,
                )

                q2_low, q2_med, q2_high = np.percentile(x,[2.28, 50.0, 97.72])

                label_2sigma = (
                    rf"$2\sigma: {q2_med:.{digits}f}"
                    rf"^{{+{q2_high-q2_med:.{digits}f}}}"
                    rf"_{{-{q2_med-q2_low:.{digits}f}}}$"
                )

                ax.axvspan(
                    q2_low,
                    q2_high,
                    color=shade_color,
                    alpha=0.12,
                    zorder=2,
                    label=label_2sigma,
                )

                ax.axvline(
                    q2_low,
                    color="black",
                    linestyle=":",
                    linewidth=1.2,
                    # label=label_2sigma,
                )

                ax.axvline(
                    q2_high,
                    color="black",
                    linestyle=":",
                    linewidth=1.2,
                )

            else:
                ax = axes[i]

                x = hist_data[:, i]
                x = x[np.isfinite(x)]

                if len(x) == 0:
                    ax.set_title(hist_names[i])
                    ax.text(
                        0.5,
                        0.5,
                        "No finite samples",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    continue

                q16, q50, q84, label = _latex_summary(x, digits=digits)

                ax.hist(
                    x,
                    bins=bins,
                    density=True,
                    alpha=0.65,
                    color=color,
                    edgecolor="none",
                )

                ax.axvspan(
                    q16,
                    q84,
                    color=shade_color,
                    alpha=0.25,
                    zorder=3,
                    label=label,
                )

                ax.axvline(
                    q50,
                    color="black",
                    linestyle="--",
                    linewidth=1.5,
                )

            ax.set_title(hist_names[i])
            ax.set_xlabel(hist_names[i])
            ax.set_ylabel("Density")
            ax.legend(frameon=False, fontsize=9)

        for j in range(n_hist, len(axes)):
            fig.delaxes(axes[j])

        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
        plt.close(fig)

    def _make_recovered_corner_samples(
        corner_samples_flat,
        corner_labels,
        sampling_names,
        blobs_flat,
        blob_names,
        corner_true_values=None,
        true_values_flag=False,
    ):
        """
        Replace:
            Mini      -> Mass blob
            omega_ini -> omega blob
        """
        recovered_samples = corner_samples_flat.copy()
        recovered_labels = list(corner_labels)

        recovered_truths = None
        if true_values_flag and corner_true_values is not None:
            recovered_truths = list(corner_true_values).copy()

        name_to_idx = {name: i for i, name in enumerate(sampling_names)}
        blob_to_idx = {name: i for i, name in enumerate(blob_names)}

        if "Mini" in name_to_idx and "Mass" in blob_to_idx:
            idx = name_to_idx["Mini"]
            recovered_samples[:, idx] = blobs_flat[:, blob_to_idx["Mass"]]
            recovered_labels[idx] = r"$M\ (M_\odot)$"

            # Usually true_values has Mini, not current Mass.
            # Keep the original truth only if user explicitly wants it.
            if recovered_truths is not None:
                recovered_truths[idx] = None

        if "omega_ini" in name_to_idx and "omega" in blob_to_idx:
            idx = name_to_idx["omega_ini"]
            recovered_samples[:, idx] = blobs_flat[:, blob_to_idx["omega"]]
            recovered_labels[idx] = r"$\omega$"

            if recovered_truths is not None:
                recovered_truths[idx] = None

        if recovered_truths is not None:
            # corner does not accept None inside truths cleanly in all versions.
            # If any truth is None, disable truths for this recovered corner.
            if any(v is None for v in recovered_truths):
                recovered_truths = None

        return recovered_samples, recovered_labels, recovered_truths

    # -------------------------
    # Observed values
    # -------------------------
    obs_values = target.obs_mags.tolist()
    obs_values.append(target.vsini_obs)
    obs_values.append(target.parallax_obs)

    obs_errors = target.obs_err_mags.tolist()
    obs_errors.append(target.vsini_obs_err)
    obs_errors.append(target.parallax_obs_err)

    # -------------------------
    # True values
    # -------------------------
    if isinstance(target.true_values, list):
        true_values = target.true_values.copy()
    elif isinstance(target.true_values, np.ndarray):
        true_values = target.true_values.tolist().copy()
    else:
        true_values = target.true_values

    corner_true_values = true_values.copy() if isinstance(true_values, list) else true_values

    true_values_flag = False
    if isinstance(true_values, list):
        if any(v is None for v in true_values):
            true_values_flag = False
        elif all(isinstance(v, (int, float, np.integer, np.floating)) for v in true_values):
            true_values_flag = True

            if target.sigma_adjustment_parameter == 1:
                true_values += [0]
                corner_true_values += [0]
            elif target.sigma_adjustment_parameter == 3:
                true_values += [0, 0, 0]
                corner_true_values += [0, 0, 0]

            if target.extinction_parameter is True:
                true_values += [target.extinction]
                corner_true_values += [target.extinction]
        else:
            raise ValueError("'true_values' must be either None or numbers.")

    # -------------------------
    # Compute burn-in
    # -------------------------
    chain = sampler.get_chain()
    S = chain.shape[0]
    start_50 = int(S * 0.5)
    start_30 = int(S * 0.7)

    try:
        tau_05 = emcee.autocorr.integrated_time(chain[start_50:], tol=0)
        tau_03 = emcee.autocorr.integrated_time(chain[start_30:], tol=0)

        tau05_max = np.nanmax(tau_05) if np.ndim(tau_05) > 0 else float(tau_05)
        tau03_max = np.nanmax(tau_03) if np.ndim(tau_03) > 0 else float(tau_03)
        tau_late = np.nanmax([tau05_max, tau03_max])

        n_discard = int(burnin_ACT * tau_late)
        print(
            f"📏 τ₀․₅={tau05_max:.1f}, τ₀․₃={tau03_max:.1f}, "
            f"τ_late={tau_late:.1f} → burn-in={n_discard}"
        )

        autocorr_time = np.maximum(tau_05, tau_03)
        autocorr_flag = True

    except Exception as e:
        print(f"⚠️ Failed to compute τ adaptively ({e}), fallback to get_autocorr_time().")
        tau_fallback = np.nanmax(sampler.get_autocorr_time())
        n_discard = int(burnin_ACT * tau_fallback)
        autocorr_flag = False
        autocorr_time = None

    extinction = int(target.extinction_parameter)

    # -------------------------
    # Create results folder
    # -------------------------
    results_base = os.path.join(os.getcwd(), "results")
    os.makedirs(results_base, exist_ok=True)

    name_pre = (
        f'{target.name.replace(" ", "_")}{target.cluster_name}_{target.iso_name.upper()}_{target.METHOD}_{target.grids_type}_grid'
        f"_sigma{target.sigma_adjustment_parameter}"
    )

    if extinction != 0:
        name_pre += f"_{target.extinction_type}"

    if any(x is not None for x in target.true_values) and not all(x is not None for x in target.true_values):
        if target.known_Mini is not None:
            name_pre += f'_known_Mini_{target.known_Mini}'
        if target.known_logAge is not None:
            name_pre += f'_known_logAge_{target.known_logAge}'
        if target.known_MH is not None:
            name_pre += f'_known_MH_{target.known_MH}'
        if target.known_omega_ini is not None:
            name_pre += f'_known_omega_ini_{target.known_omega_ini}'
        if target.known_Inclination is not None:
            name_pre += f'_known_Inclination_{target.known_Inclination}'
        if target.known_parallax is not None:
            name_pre += f'_known_parallax_{target.known_parallax}'

    folder_name = name_pre if folder_suffix == "" else name_pre + f" ({folder_suffix})"
    folder_path = os.path.join(results_base, folder_name)

    if not os.path.exists(folder_path):
        new_folder_path = folder_path
        os.makedirs(new_folder_path)
    else:
        count = 2
        while True:
            new_folder_path = f"{folder_path} ({count})"
            if not os.path.exists(new_folder_path):
                os.makedirs(new_folder_path)
                break
            count += 1

    if save_samples:
        print(f"Your plots and samples will be saved at \n\t{new_folder_path}")
    else:
        print(f"Your plots will be saved at \n\t{new_folder_path}")

    suffix = os.path.basename(new_folder_path)

    # -------------------------
    # Extract thinned samples / blobs
    # -------------------------
    available_steps = sampler.iteration - n_discard
    if available_steps <= 0:
        raise ValueError(
            f"No samples remain: sampler.iteration={sampler.iteration}, "
            f"n_discard={n_discard}"
        )

    total_available_samples = available_steps * sampler.nwalkers

    if plot_samples == "all":
        n_thin = 1
        plot_samples = total_available_samples
    elif plot_samples > total_available_samples:
        warnings.warn(
            f"Requested plot_samples ({plot_samples}) exceeds available samples "
            f"after discard ({total_available_samples}). Using all available samples.",
            RuntimeWarning,
        )
        plot_samples = total_available_samples

    needed_steps = math.ceil(plot_samples / sampler.nwalkers)
    n_thin = max(1, available_steps // needed_steps)

    while math.ceil(available_steps / n_thin) * sampler.nwalkers < plot_samples:
        n_thin -= 1
        if n_thin < 1:
            n_thin = 1
            break

    samples = sampler.get_chain(discard=n_discard, thin=n_thin, flat=False)
    samples_flat = sampler.get_chain(discard=n_discard, thin=n_thin, flat=True)

    blobs = sampler.get_blobs(discard=n_discard, thin=n_thin, flat=False)
    blobs_array = _normalize_blobs(blobs)
    blobs_flat = blobs_array.reshape(-1, blobs_array.shape[-1])

    # -------------------------
    # Labels
    # -------------------------
    title_map = {
        "Mini": r"$M_{\mathrm{ini}}\ (M_\odot)$",
        "MH": r"$[\mathrm{M/H}]$",
        "omega_ini": r"$\omega_{\mathrm{ini}}$",
        "parallax": r"$\varpi\ (\mathrm{mas})$",
        "sigma": r"$\sigma_{\mathrm{adjust}}\ (\mathrm{mag})$",
        "sigma_gaia": r"$\sigma_{\mathrm{adjust},\ Gaia}\ (\mathrm{mag})$",
        "sigma_tycho2": r"$\sigma_{\mathrm{adjust,\ Tycho-2}}\ (\mathrm{mag})$",
        "sigma_2mass": r"$\sigma_{\mathrm{adjust,\ 2MASS}}\ (\mathrm{mag})$",
    }

    extinction_map = {
        "A_V": r"$A_V\ (\mathrm{mag})$",
        "EB_V": r"$E(B-V)\ (\mathrm{mag})$",
        "EBP_RP": r"$E(G_{\mathrm{BP}} - G_{\mathrm{RP}})\ (\mathrm{mag})$",
    }

    corner_sampling_variables_list = [
        r"$\log(\tau/\mathrm{yr})$" if v == "logAge" and corner_age_type == "log"
        else r"$\tau\ (\mathrm{Myr})$" if v == "logAge" and corner_age_type == "linear"
        else r"$i\ (\mathrm{rad})$" if v == "Inclination" and corner_inc_type == "radian"
        else r"$i\ (^\circ)$" if v == "Inclination" and corner_inc_type == "degree"
        else extinction_map[target.extinction_type] if v == "extinction"
        else title_map.get(v, v)
        for v in target.sampling_variables_list
    ]

    corner_samples_flat = samples_flat.copy()

    if corner_age_type not in ["log", "linear"]:
        raise ValueError('"corner_age_type" must be either "log" or "linear".')
    if corner_inc_type not in ["radian", "degree"]:
        raise ValueError('"corner_inc_type" must be either "radian" or "degree".')

    if corner_age_type == "linear":
        age_index = target.sampling_variables_list.index("logAge")
        corner_samples_flat[:, age_index] = 10 ** (corner_samples_flat[:, age_index] - 6)
        if true_values_flag:
            corner_true_values[age_index] = 10 ** (corner_true_values[age_index] - 6)

    if corner_inc_type == "degree" and "Inclination" in target.sampling_variables_list:
        inc_index = target.sampling_variables_list.index("Inclination")
        corner_samples_flat[:, inc_index] = np.rad2deg(corner_samples_flat[:, inc_index])
        if true_values_flag:
            corner_true_values[inc_index] = np.rad2deg(corner_true_values[inc_index])

    print(f"True values: {true_values}")

    output_list = target.bands_list.copy()

    if target.METHOD == "DIRECT":
        output_list += [
            "vsini",
            "parallax",
            "chi_squared",
            "vtaneq",
            "Mass",
            "omega",
            "Req",
            "Rpol",
            "lnL0",
            "lnL",
            "u",
        ]
    elif target.METHOD == "PARS":
        output_list += [
            "vsini",
            "parallax",
            "chi_squared",
            "vtaneq",
            "Mass",
            "omega",
            "Req",
            "Rpol",
            "logL",
            "lnL0",
            "lnL",
            "u",
        ]
    else:
        raise ValueError(f"Unknown target.METHOD: {target.METHOD}")

    # -------------------------
    # Validate blobs
    # -------------------------
    n_steps, n_walkers, n_dim = samples.shape

    if blobs_array.shape[0] != n_steps or blobs_array.shape[1] != n_walkers:
        raise ValueError(
            f"samples shape {samples.shape} and blobs shape {blobs_array.shape} do not match."
        )

    n_blob = blobs_array.shape[-1]

    if len(output_list) != n_blob:
        raise ValueError(
            f"output_list length ({len(output_list)}) does not match "
            f"blob dimension ({n_blob}).\n"
            f"output_list={output_list}\n"
            f"blobs_array.shape={blobs_array.shape}"
        )


    # -------------------------
    # Save samples
    # -------------------------
    if save_samples:
        if save_64:
            samples_full = sampler.get_chain()
            lnprob_full = sampler.get_log_prob()
            blobs_full = _normalize_blobs(sampler.get_blobs())
        else:
            samples_full = sampler.get_chain().astype(np.float32)
            lnprob_full = sampler.get_log_prob().astype(np.float32)
            blobs_full = _normalize_blobs(sampler.get_blobs()).astype(np.float32)

        param_names = np.array(target.sampling_variables_list.copy())
        out_npz = Path(new_folder_path) / f"posterior_{suffix}.npz"

        np.savez_compressed(
            out_npz,
            samples=samples_full,
            log_prob=lnprob_full,
            param_names=param_names,
            output_list=np.array(output_list),
            blobs=blobs_full,
            n_discard=np.array(n_discard),
        )

    # -------------------------
    # Recovered-parameter corner plot
    # -------------------------
    if plot_recovered_corner:
        try:
            recovered_samples, recovered_labels, recovered_truths = _make_recovered_corner_samples(
                corner_samples_flat=corner_samples_flat,
                corner_labels=corner_sampling_variables_list,
                sampling_names=target.sampling_variables_list,
                blobs_flat=blobs_flat,
                blob_names=output_list,
                corner_true_values=corner_true_values,
                true_values_flag=true_values_flag,
            )

            fig = corner.corner(
                recovered_samples,
                labels=recovered_labels,
                quantiles=quantiles,
                show_titles=True,
                hist_bin_factor=hist_bin_factor,
                title_fmt=f".{dig_num}f",
                truths=recovered_truths,
            )

            recovered_corner_path = (
                Path(new_folder_path)
                / f"corner_recovered_burnin_{n_discard}_thin_{n_thin}_{suffix}.pdf"
            )

            with PdfPages(recovered_corner_path) as pdf:
                pdf.savefig(fig, dpi=150)

            # if show_fig and running_in_jupyter():
            #     plt.show()

            plt.close(fig)

        except Exception as e:
            print(f"⚠️ Recovered-parameter corner plot failed: {e}")
            try:
                plt.close("all")
            except Exception:
                pass

    # -------------------------
    # Blobs histogram summary
    # -------------------------
    if plot_blobs_hist:
        try:
            blobs_hist_path = (
                    Path(new_folder_path)
                    / f"blobs_hist_burnin_{n_discard}_thin_{n_thin}_{suffix}.pdf"
            )

            _plot_blobs_hist(
                blobs_flat=blobs_flat,
                blob_names=output_list,
                out_path=blobs_hist_path,
                bins=blobs_hist_bins,
                ncols=blobs_hist_ncols,
                digits=dig_num + 1,
            )

        except Exception as e:
            print(f"⚠️ Blobs histogram plot failed: {e}")
            try:
                plt.close("all")
            except Exception:
                pass

    # -------------------------
    # Trace plot
    # -------------------------
    n_dim_total = n_dim + n_blob
    num_groups = math.ceil(n_walkers / group_size)
    x = np.arange(n_steps)

    for g in range(num_groups):
        fig, axes = plt.subplots(
            n_dim_total,
            1,
            figsize=(15, 2.0 * n_dim_total),
            sharex=True,
        )
        plt.subplots_adjust(hspace=0)

        if n_dim_total == 1:
            axes = [axes]

        start = g * group_size
        end = min((g + 1) * group_size, n_walkers)
        walker_indices = range(start, end)

        for dim in range(n_dim_total):
            ax = axes[dim]

            if dim < n_dim:
                for w in walker_indices:
                    ax.plot(x, samples[:, w, dim], alpha=0.5, label=f"walker {w}")

                if true_values_flag:
                    ax.axhline(
                        true_values[dim],
                        color="red",
                        linestyle="--",
                        label="true values",
                    )
                    true_ylabel = f"\ntruth={np.round(true_values[dim], 4)}"
                else:
                    true_ylabel = ""

                if autocorr_flag and autocorr_time is not None:
                    act = int(np.ceil(autocorr_time[dim]))
                    ax.set_ylabel(
                        f"{corner_sampling_variables_list[dim]}{true_ylabel}\n(ACT={act})"
                    )
                else:
                    ax.set_ylabel(f"{corner_sampling_variables_list[dim]}{true_ylabel}")

            else:
                idx = dim - n_dim
                blob_name = output_list[idx]

                if blob_name != "chi_squared":
                    for w in walker_indices:
                        ax.plot(
                            x,
                            blobs_array[:, w, idx],
                            alpha=0.5,
                            label=f"walker {w}",
                        )

                    if idx < len(target.bands_list) + 2:
                        ax.axhline(
                            obs_values[idx],
                            color="black",
                            linestyle="--",
                            label="observed values",
                        )

                        ax.fill_between(
                            x=x,
                            y1=obs_values[idx] - obs_errors[idx],
                            y2=obs_values[idx] + obs_errors[idx],
                            color="gray",
                            alpha=0.3,
                            label="± 1σ uncertainty",
                        )

                        error = np.round(obs_errors[idx], 4)
                        if np.isnan(error):
                            ax.set_ylabel(f"{blob_name}")
                        else:
                            ax.set_ylabel(f"{blob_name}\n($\\sigma$={error})")
                    else:
                        ax.set_ylabel(f"{blob_name}")

                else:
                    dof = len(target.bands_list) + 2 - len(target.sampling_variables_list)

                    for w in walker_indices:
                        chi2 = blobs_array[:, w, idx]
                        reduced_chi2 = chi2 / dof
                        ax.plot(
                            x,
                            reduced_chi2,
                            alpha=0.5,
                            label=f"walker {w}",
                        )

                    min_val = np.round(np.nanmin(blobs_array[:, :, idx]), 2) / dof
                    ax.set_ylabel(f"Reduced $\\chi^2$\n(min={min_val:.4f})")

            ax.set_xlim([np.min(x), np.max(x)])
            ax.tick_params(axis="x", which="both", labelbottom=(dim == n_dim_total - 1))

        axes[-1].set_xlabel("samples" if n_thin == 1 else f"samples (thinning={n_thin})")

        all_handles, all_labels = [], []
        for ax in axes:
            h, l = ax.get_legend_handles_labels()
            all_handles.extend(h)
            all_labels.extend(l)

        unique = dict(zip(all_labels, all_handles))
        axes[0].legend(unique.values(), unique.keys(), loc="upper right")

        trace_path = os.path.join(
            new_folder_path,
            f"trace_group{g}_burnin{n_discard}_thin{n_thin}_{suffix}.pdf",
        )

        plt.savefig(trace_path, bbox_inches="tight")
        plt.close(fig)

    # -------------------------
    # Original corner plot
    # -------------------------
    try:
        if true_values_flag:
            print("==> Overplot true values on the original corner plot.")

        fig = corner.corner(
            corner_samples_flat,
            labels=corner_sampling_variables_list,
            quantiles=quantiles,
            show_titles=True,
            hist_bin_factor=hist_bin_factor,
            title_fmt=f".{dig_num}f",
            truths=(corner_true_values if true_values_flag else None),
        )

        corner_pdf_path = (
            Path(new_folder_path)
            / f"corner_burnin_{n_discard}_thin_{n_thin}_{suffix}.pdf"
        )

        with PdfPages(corner_pdf_path) as pdf:
            pdf.savefig(fig, dpi=150)

        print("Finished saving plots.")
        print(f"Saved folder: {new_folder_path}")

        if show_fig and running_in_jupyter():
            plt.show()

        plt.close(fig)

    except Exception as e:
        print(f"⚠️ Original corner plot failed: {e}")
        try:
            plt.close("all")
        except Exception:
            pass

    # return new_folder_path