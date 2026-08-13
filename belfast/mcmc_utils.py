import os
import time
import math
import json
import platform
import warnings
from pathlib import Path
from datetime import datetime
import numpy as np
from belfast.target_utils import cal_obs_mag, cal_parsec, add_extinction
from belfast.paths import REPO_ROOT, RESULTS_DIR, DATA_DIR

GM_SUN = 1.32712440018e20 # unit='m3 / s2'
R_SUN = 6.957e8 # unit='m'

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

warnings.filterwarnings(
    "ignore",
    message=r"divide by zero encountered in divide",
    category=RuntimeWarning,
    module=r".*surface",
)

warnings.filterwarnings(
    "ignore",
    message=r"invalid value encountered in divide",
    category=RuntimeWarning,
    module=r".*surface",
)

target = None
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
    from scipy.special import erf

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
    from scipy.stats import norm

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
        Age (float): Logarithm of stellar age (yr).
        MH (float): Metallicity [M/H].
        omega (float): Rotation parameter (fraction of critical angular velocity).
        Inclination (float): Inclination angle (deg).
        parallax (float): Stellar parallax (mas).
        extinction (float): Extinction parameter.
        Mini_prior, Age_prior, MH_prior, omega_ini_prior, Inclination_prior, parallax_prior, extinction_prior:
            Individual log-priors for each parameter, based on default or user-defined priors.

    Returns:
        float: Logarithm of the total prior probability (sum of individual priors).
               Returns -np.inf if any variable is outside allowed limits.
    """
    from scipy.stats import norm

    if isinstance(sampling_variables, list):
        sampling_variables = np.array(sampling_variables)

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

    if 'Age' in target.sampling_variables_list:
        Age_index = target.sampling_variables_list.index('Age')
        Age = sampling_variables[Age_index]
        if target.Age_prior == 'default':
            Age_prior = 0
        else:
            Age_prior = target.Age_prior(Age)
    else:
        Age_prior = 0
    if verbose:
        print(f'Age_prior: {Age_prior}')

    if 'MH' in target.sampling_variables_list:
        MH_index = target.sampling_variables_list.index('MH')
        if target.iso_name == 'MIST' and target.correct_MIST_MH_zeropoint:
            sampling_variables[MH_index] += target.MIST_MH_offset
            # print(f'sampling_variables = {sampling_variables}')
        MH = sampling_variables[MH_index]
        # print(f'MH = {MH}')
        if target.MH_prior == 'default':
            # (Caffua+2011) Fe = 7.52 ± 0.06; (Grevesse+2007) Fe = 7.45 ± 0.05
            MH_prior = norm.logpdf(MH, loc=-0.106, scale=0.224)
                # -0.036 +/- 0.210 Gaussian fit of GALAH survey (Buder et al. 2019) [Fe/H] of samples which are < 8 Gyr.
            # MH_prior = norm.logpdf(MH, loc=-0.1127, scale=0.2582)  # non-informative prior from the GALAH survey (Buder et al. 2019) corrected solar abundance zeropoint from Grevesse+07 to Caffau+11
            # MH_prior = norm.logpdf(MH, loc=-0.0427, scale=0.2461) # non-informative prior from the GALAH survey (Buder et al. 2019) Wrong! Not correct solar abundance difference
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

    if np.all((sampling_variables > target.sampling_lower_limit) & (sampling_variables < target.sampling_upper_limit)):
        return Mini_prior + Age_prior + MH_prior + omega_ini_prior + Inclination_prior + parallax_prior + extinction_prior
    else:
        if verbose:
            print('Out of the parameter space of sampling!')
        return -np.inf


def log_likelihood_INTERP(sampling_variables, verbose=False):
    """
    Compute the log-likelihood for the INETRP method.

        Args:
        sampling_variables (list or np.ndarray): Current parameter values being sampled.

    Key variables:
        Mini, Age, MH, omega, Inclination, parallax, extinction (floats): Stellar parameters.
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

    if 'Age' in target.sampling_variables_list:
        Age_index = target.sampling_variables_list.index('Age')
        Age = sampling_variables[Age_index]
    elif target.known_Age is not None:
        Age = target.known_Age

    if 'MH' in target.sampling_variables_list:
        MH_index = target.sampling_variables_list.index('MH')
        if target.iso_name == 'MIST' and target.correct_MIST_MH_zeropoint:
            sampling_variables[MH_index] += target.MIST_MH_offset
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
        return -np.inf, np.ones(len(target.bands_list) + 13)*np.nan

    Mass, omega, logL, logTe, Req, Rpol = y_interpolated[:6]
    abs_mags_interpolated = y_interpolated[6:]


    obs_mags_model = cal_obs_mag(abs_mags_interpolated, parallax)
    if target.extinction_parameter == True:
        obs_mags_model_extincted, obs_mags_model_err_extincted = add_extinction(obs_mags_model, 0, target.bands_list, extinction, extinction_err, target.extinction_type, print_flag=False)
    else:
        obs_mags_model_extincted, obs_mags_model_err_extincted = obs_mags_model, 0
    final_mags = obs_mags_model_extincted

    final_err_mags = np.sqrt(obs_err_mags_new**2 + obs_mags_model_err_extincted**2)

    omega_angular = omega * np.sqrt(GM_SUN * Mass / (R_SUN * 1.5 * Rpol)**3)
    vtaneq = omega_angular * Req * R_SUN / 1000 # km/s
    vsini = vtaneq * np.sin(Inclination)
    # v_keplerian = np.sqrt(GM_SUN * Mass / (R_SUN * Req)) / 1000 # km/s
    # v_critical = np.sqrt(GM_SUN * Mass / (R_SUN * 1.5 * Rpol)) / 1000 # km/s

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
    # blobs.append(parallax)
    blobs.append(chi_squared)
    blobs.append(vtaneq)
    blobs.append(Mass)
    blobs.append(omega)
    blobs.append(logL)
    blobs.append(logTe)
    blobs.append(Req)
    blobs.append(Rpol)
    blobs.append(lnL0)
    blobs.append(lnL)
    if target.iso_name == 'PARSEC':
        u = target.interp_u([Mini, Age, MH, omega_ini, Inclination])
    blobs.append(u)

    # Vrot empirical constraint
    # This is a likelihood term in Vrot space, not a transformed prior on omega_ini.
    if target.Vrot_prior is not None:
        p_Vrot = target.Vrot_prior(vtaneq)

        if not np.isfinite(p_Vrot):
            if verbose:
                print("Failed in Vrot_prior calculation.")
            return -np.inf, np.ones(len(target.bands_list) + 13) * np.nan

        lnL += p_Vrot
    else:
        p_Vrot = 0

    blobs.append(p_Vrot)

    unified_blobs = [float(np.squeeze(x)) for x in blobs]

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
    from scipy.interpolate import interp1d
    from pa.lib import star as st

    if isinstance(sampling_variables, list):
        sampling_variables = np.array(sampling_variables)

    if 'Mini' in target.sampling_variables_list:
        Mini_index = target.sampling_variables_list.index('Mini')
        Mini = sampling_variables[Mini_index]
    elif target.known_Mini is not None:
        Mini = target.known_Mini

    if 'Age' in target.sampling_variables_list:
        Age_index = target.sampling_variables_list.index('Age')
        Age = sampling_variables[Age_index]
    elif target.known_Age is not None:
        Age = target.known_Age

    if 'MH' in target.sampling_variables_list:
        MH_index = target.sampling_variables_list.index('MH')
        if target.iso_name == 'MIST' and target.correct_MIST_MH_zeropoint:
            sampling_variables[MH_index] += target.MIST_MH_offset
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
        Mass, omega, logL, logTe, Req, Rpol = target.Model_interp(sampling_variables[:len(target.interp_variables_list)]).flatten()
        omega_angular = omega * np.sqrt(GM_SUN * Mass/(R_SUN * 1.5 * Rpol)**3)
        omega_keplerian = np.sqrt(GM_SUN * Mass / (R_SUN * Req)**3)
        omega_k = omega_angular / omega_keplerian
        # omega_k = omega_k.value
        v_critical = np.sqrt(GM_SUN * Mass / (R_SUN * 1.5 * Rpol)) / 1000 # km/s
    elif target.iso_name.lower() == 'MIST'.lower():
        Mass, omega, logL, Req = target.Model_interp(sampling_variables[:len(target.interp_variables_list)]).flatten()
        Rpol = Req
        omega_angular = omega * np.sqrt(GM_SUN * Mass/(R_SUN * Req)**3)
        omega_k = omega
        v_critical = np.sqrt(GM_SUN * Mass/(R_SUN * Req)) / 1000 # km/s

    vtaneq = (omega_angular * Req * R_SUN) / 1000 # km/s
    vsini = vtaneq * np.sin(Inclination)
    distance_pc = cal_parsec(parallax)
    distance_cm = distance_pc * 3.0856775814913674e+18 # pc to cm

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
            return -np.inf, np.ones(len(target.bands_list) + 13)*np.nan
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
                return -np.inf, np.ones(len(target.bands_list) + 13)*np.nan

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
    # blobs.append(parallax)
    blobs.append(chi_squared)
    blobs.append(vtaneq)
    blobs.append(Mass)
    blobs.append(omega)
    blobs.append(logL)
    blobs.append(logTe)
    blobs.append(Req)
    blobs.append(Rpol)
    blobs.append(lnL0)
    blobs.append(lnL)
    if target.iso_name == 'PARSEC':
        u = target.interp_u(sampling_variables[:len(target.interp_variables_list)])
    else:
        u = 0
    blobs.append(u)

    # Vrot empirical constraint
    # This is a likelihood term in Vrot space, not a transformed prior on omega_ini.
    if target.Vrot_prior is not None:
        p_Vrot = target.Vrot_prior(vtaneq)

        if not np.isfinite(p_Vrot):
            if verbose:
                print("Failed in Vrot_prior calculation.")
            return -np.inf, np.ones(len(target.bands_list) + 13) * np.nan

        lnL += p_Vrot
    else:
        p_Vrot = 0

    blobs.append(p_Vrot)

    unified_blobs = [float(np.squeeze(x)) for x in blobs]

    return lnL, np.array(unified_blobs)


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
            blobs = np.ones(len(target.bands_list) + 13)*np.nan
        elif target.METHOD == "PARS":
            blobs = np.ones(len(target.bands_list) + 13)*np.nan
        return -np.inf, blobs

    if target.METHOD == 'DIRECT':
        log_lkh, blobs = log_likelihood_INTERP(sampling_variables)
    elif target.METHOD == "PARS":
        log_lkh, blobs = log_likelihood_PARS(sampling_variables)

    log_prob = log_pr + log_lkh

    return log_prob, blobs


def running_in_jupyter():
    try:
        from IPython import get_ipython
        if 'IPKernelApp' in get_ipython().config:
            return True
    except Exception:
        pass


def _validate_emcee_moves(emcee_moves):
    """Validate optional ``[(emcee_move, fraction), ...]`` configuration."""
    if emcee_moves is None:
        return None

    import emcee

    if not isinstance(emcee_moves, (list, tuple)) or not emcee_moves:
        raise ValueError("target.emcee_moves must be None or a non-empty list/tuple of (move, fraction) pairs.")

    total = 0.0
    for i, item in enumerate(emcee_moves):
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(f"Invalid target.emcee_moves entry at index {i}: {item!r}")
        move, fraction = item
        if not isinstance(move, emcee.moves.Move):
            raise TypeError(f"Entry {i} is not an emcee Move: {type(move).__name__}")
        if not isinstance(fraction, (int, float, np.integer, np.floating)):
            raise TypeError(f"Move fraction at index {i} must be numeric.")
        fraction = float(fraction)
        if not np.isfinite(fraction) or fraction < 0:
            raise ValueError(f"Move fraction at index {i} must be finite and non-negative.")
        total += fraction

    if not np.isclose(total, 1.0, rtol=0.0, atol=1e-8):
        raise ValueError(f"The fractions in target.emcee_moves must sum to 1. Got {total:.12g}.")
    return emcee_moves


def _validate_triplet(values, default, name, lower=None, upper=None):
    arr = np.asarray(default if values is None else values, dtype=float)
    if arr.shape != (3,) or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain exactly three finite values.")
    if lower is not None and np.any(arr <= lower):
        raise ValueError(f"All {name} values must be > {lower}.")
    if upper is not None and np.any(arr >= upper):
        raise ValueError(f"All {name} values must be < {upper}.")
    return arr.tolist()


def _validate_age_sampling():
    variables = list(target.sampling_variables_list)
    if "logAge" in variables or "Age" not in variables:
        raise ValueError(f'Sampling must use "Age" in Myr, not "logAge". Got: {variables}')
    age_unit = getattr(target, "age_unit", None)
    if age_unit is not None and str(age_unit).lower() != "myr":
        raise ValueError(f'The target age unit must be "Myr". Got: {age_unit!r}')


def _estimate_late_tau(chain):
    """Return conservative late-chain ACT and its 50%/30% stability difference."""
    import emcee

    n = len(chain)
    tau_05 = emcee.autocorr.integrated_time(chain[int(0.5 * n):], tol=0)
    tau_03 = emcee.autocorr.integrated_time(chain[int(0.7 * n):], tol=0)
    t05 = np.nanmax(tau_05) if np.any(np.isfinite(tau_05)) else np.nan
    t03 = np.nanmax(tau_03) if np.any(np.isfinite(tau_03)) else np.nan
    tau = np.nanmax([t05, t03])
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("No finite ACT estimate.")
    rel_diff = abs(t05 - t03) / max(t05, t03) if np.isfinite(t05) and np.isfinite(t03) else np.inf
    return float(tau), float(t05), float(t03), float(rel_diff)


def _next_check_steps(
    tau, current_ratio, target_ratio, max_check_steps,
    min_check_tau=10.0, seconds_per_step=None, max_check_minutes=None,
    remaining_seconds=None,
):
    """Plan the next checkpoint from the remaining required number of ACTs."""
    missing_tau = max(float(target_ratio) - float(current_ratio), 0.0)
    planned_tau = max(float(min_check_tau), missing_tau)
    requested = max(1, math.ceil(planned_tau * tau))
    limits = [requested, int(max_check_steps)]

    if seconds_per_step and seconds_per_step > 0:
        if max_check_minutes is not None:
            limits.append(max(1, int(max_check_minutes * 60 / seconds_per_step)))
        if remaining_seconds is not None:
            limits.append(max(1, int(remaining_seconds / seconds_per_step)))

    return max(1, min(limits)), missing_tau, planned_tau, requested


def runnning_in_pool(
    initial_steps, max_hours, emcee_moves, pool, dtype, use_tqdm,
    min_length_ratio_list, max_tau_relative_difference_list,
    max_check_steps=10000, min_check_tau=10.0, max_check_minutes=30.0,
):
    """Run adaptive MCMC and check ACT stability plus usable chain length."""
    import emcee

    sampler = emcee.EnsembleSampler(
        target.n_walkers, target.ndim, log_posterior,
        moves=emcee_moves, pool=pool, blobs_dtype=dtype,
    )
    start_total = time.time()
    max_runtime = max_hours * 3600
    stable_count = 0
    previous_stage = None

    if not use_tqdm:
        print(f"Starting Time: {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"🔄 Initial sampling: {initial_steps} steps")
    batch_start = time.time()
    sampler.run_mcmc(target.initial_states, initial_steps, progress=use_tqdm)
    if not use_tqdm:
        dt = int(time.time() - batch_start)
        print(f"Sampling time: {dt//3600:02d}:{(dt%3600)//60:02d}:{dt%60:02d}")

    while True:
        total_steps = sampler.iteration
        elapsed = time.time() - start_total
        if elapsed >= max_runtime:
            raise TimeoutError(f"⏰ Sampling exceeded {max_hours} hours limit.")

        stage = 2 if elapsed > 7200 else 1 if elapsed > 3600 else 0
        min_ratio = min_length_ratio_list[stage]
        max_tau_diff = max_tau_relative_difference_list[stage]
        if stage != previous_stage:
            labels = ("Standard", "Runtime > 1h", "Runtime > 2h")
            print(f"🚀 {labels[stage]} criteria: usable N/τ ≥ {min_ratio:g}, τ_diff < {max_tau_diff:.0%}")
            previous_stage = stage

        tau = None
        try:
            tau, tau_05, tau_03, rel_diff = _estimate_late_tau(sampler.get_chain())
            burnin = int(10 * tau)
            length_ratio = (total_steps - burnin) / tau
            stable_tau = rel_diff < max_tau_diff
            length_ok = length_ratio >= min_ratio
            stable_count = stable_count + 1 if stable_tau and length_ok else 0

            h, m = int(elapsed // 3600), int((elapsed % 3600) // 60)
            print(f"\n🔍 Checkpoint @ {total_steps} steps (elapsed {h}h {m}m)")
            print(f"  τ₀․₅={tau_05:.1f}, τ₀․₃={tau_03:.1f}, τ_diff={rel_diff:.1%}")
            print(f"  Usable N/τ after burn=10τ: {length_ratio:.1f} / {min_ratio:g}")

            if stable_count >= 2:
                print(f"✅ Converged: τ stable for 2 checkpoints and usable N/τ ≥ {min_ratio:g}")
                break

        except (emcee.autocorr.AutocorrError, ValueError, FloatingPointError):
            print("⚠️ ACT estimation failed; using a fallback checkpoint interval.")
            length_ratio = -np.inf

        elapsed = time.time() - start_total
        seconds_per_step = elapsed / max(total_steps, 1)
        remaining_seconds = max_runtime - elapsed

        if tau is None or not np.isfinite(tau):
            check_steps = min(initial_steps, max_check_steps)
            if max_check_minutes is not None and seconds_per_step > 0:
                check_steps = min(check_steps, max(1, int(max_check_minutes * 60 / seconds_per_step)))
            missing_tau = planned_tau = requested = np.nan
        else:
            check_steps, missing_tau, planned_tau, requested = _next_check_steps(
                tau, length_ratio, min_ratio, max_check_steps,
                min_check_tau=min_check_tau,
                seconds_per_step=seconds_per_step,
                max_check_minutes=max_check_minutes,
                remaining_seconds=remaining_seconds,
            )

        if np.isfinite(missing_tau):
            cap_note = f"; capped from {requested}" if check_steps < requested else ""
            print(
                f"⏱ Next: {check_steps} steps "
                f"(missing {missing_tau:.1f}τ, planned {planned_tau:.1f}τ{cap_note})"
            )
        else:
            print(f"⏱ Next: {check_steps} steps (fallback)")

        batch_start = time.time()
        sampler.run_mcmc(None, check_steps, progress=use_tqdm)
        if not use_tqdm:
            dt = int(time.time() - batch_start)
            print(f"Sampling time: {dt//3600:02d}:{(dt%3600)//60:02d}:{dt%60:02d}")

    return sampler


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


def serialize_emcee_moves(emcee_moves):
    import emcee
    import numpy as np

    if emcee_moves is None:
        return None

    move_config = {}
    total_fraction = 0.0

    for move, fraction in emcee_moves:
        fraction = float(fraction)
        total_fraction += fraction

        if isinstance(move, emcee.moves.StretchMove):
            a = getattr(
                move,
                "a",
                getattr(move, "_a", None),
            )

            if a is None:
                raise AttributeError(
                    "Could not read parameter 'a' "
                    "from StretchMove."
                )

            move_name = "StretchMove"
            config = {
                "a": float(a),
                "fraction": fraction,
            }

        elif isinstance(move, emcee.moves.DEMove):
            gamma0 = getattr(
                move,
                "gamma0",
                getattr(move, "_gamma0", None),
            )

            move_name = "DEMove"
            config = {
                "gamma0": (
                    None
                    if gamma0 is None
                    else float(gamma0)
                ),
                "fraction": fraction,
            }

        elif isinstance(move, emcee.moves.DESnookerMove):
            move_name = "DESnookerMove"
            config = {
                "fraction": fraction,
            }

        elif isinstance(move, emcee.moves.KDEMove):
            move_name = "KDEMove"
            config = {
                "fraction": fraction,
            }

        else:
            raise ValueError(
                "Unsupported emcee move type: "
                f"{type(move).__name__}"
            )

        if move_name in move_config:
            raise ValueError(
                f"Duplicate move type: {move_name}"
            )

        move_config[move_name] = config

    if not np.isclose(
        total_fraction,
        1.0,
        atol=1e-8,
    ):
        raise ValueError(
            "Move fractions must sum to 1. "
            f"Got {total_fraction}."
        )

    return move_config

def build_emcee_moves(emcee_move_config):
    import emcee

    if emcee_move_config is None:
        return None

    emcee_moves = []

    for move_name, config in emcee_move_config.items():
        fraction = float(config["fraction"])

        if move_name == "StretchMove":
            move = emcee.moves.StretchMove(
                a=float(config["a"])
            )

        elif move_name == "DEMove":
            move = emcee.moves.DEMove(
                gamma0=config.get(
                    "gamma0",
                    None,
                )
            )

        elif move_name == "DESnookerMove":
            move = emcee.moves.DESnookerMove()

        elif move_name == "KDEMove":
            move = emcee.moves.KDEMove()

        else:
            raise ValueError(
                f"Unknown move type: {move_name}"
            )

        emcee_moves.append(
            (move, fraction)
        )

    return emcee_moves


def run_test_sampling(
    n_steps_test=None,
    n_steps_evaluate=None,
    use_tqdm=True,
    show_fig=True,
    n_processes=8,
    corner_inc_type="degree",
    quantiles=None,
    hist_bin_factor=3,
    dig_num=2,
    save_path=str(RESULTS_DIR),
    save_target=False,
    save_samples=False,
    save_64=False,
):
    """
    Run a shorter MCMC sampling for testing and visualization.

    Notes
    -----
    - ``Age`` is sampled, saved, and plotted directly in Myr.
    - No logarithmic-age conversion is performed.
    - ``corner_inc_type`` controls whether inclination is displayed in
      radians or degrees.
    - Sampler moves are read only from ``target.emcee_moves``.
    - ``target.emcee_moves`` must be either ``None`` or a list/tuple of
      ``(move_object, fraction)`` pairs.
    - This function does not serialize or save the move configuration.

    Special logic
    -------------
    If ``target.omega_ini_prior == "default"`` and
    ``target.Vrot_prior is None``, a preliminary chain is run to estimate
    the rough stellar mass and select the corresponding default Vrot prior.

    Returns
    -------
    sampler_test : emcee.EnsembleSampler
        Sampler from the final test run.

    target : Target
        Updated target object containing, when applicable:

        - rough_mass
        - Vrot_prior
    """
    import sys

    import corner
    import emcee
    import numpy as np
    from IPython.display import display
    from matplotlib import pyplot as plt

    # ============================================================
    # Defaults
    # ============================================================
    if n_steps_test is None:
        n_steps_test = 3000

    if n_steps_evaluate is None:
        n_steps_evaluate = n_steps_test

    if quantiles is None:
        quantiles = [0.16, 0.5, 0.84]

    n_steps_test = int(n_steps_test)
    n_steps_evaluate = int(n_steps_evaluate)
    n_processes = int(n_processes)

    # ============================================================
    # General validation
    # ============================================================
    if n_steps_test <= 0:
        raise ValueError(
            "n_steps_test must be a positive integer."
        )

    if n_steps_evaluate <= 0:
        raise ValueError(
            "n_steps_evaluate must be a positive integer."
        )

    if n_processes <= 0:
        raise ValueError(
            "n_processes must be a positive integer."
        )

    if corner_inc_type not in {
        "radian",
        "degree",
    }:
        raise ValueError(
            '"corner_inc_type" must be either '
            '"radian" or "degree".'
        )

    _validate_age_sampling()

    quantiles = np.asarray(
        quantiles,
        dtype=float,
    )

    if quantiles.ndim != 1 or len(quantiles) != 3:
        raise ValueError(
            "quantiles must contain exactly three values."
        )

    if (
        not np.all(np.isfinite(quantiles))
        or np.any(quantiles < 0)
        or np.any(quantiles > 1)
        or not np.all(np.diff(quantiles) > 0)
    ):
        raise ValueError(
            "quantiles must be finite, strictly increasing, "
            "and between 0 and 1."
        )

    quantiles = quantiles.tolist()

    emcee_moves = _validate_emcee_moves(getattr(target, "emcee_moves", None))

    # ============================================================
    # Save path
    # ============================================================
    save_path = str(save_path)

    if save_samples or save_target or not show_fig:
        os.makedirs(
            save_path,
            exist_ok=True,
        )

    save_path = save_path.rstrip("/\\")

    if not save_path:
        save_path = "."

    # ============================================================
    # Blob output names
    # ============================================================
    output_dict = get_blob_output_dict(
        target
    )

    output_list = list(
        output_dict.keys()
    )

    # ============================================================
    # Helper: determine whether multiprocessing should be disabled
    # ============================================================
    def _should_disable_mp_pool():
        """
        Disable multiprocessing for the original NN/Linux/Jupyter
        special case.
        """
        return (
            target.METHOD == "NN"
            and platform.system() == "Linux"
            and running_in_jupyter()
        )

    # ============================================================
    # Helper: run one sampler
    # ============================================================
    def _run_once(
        progress_flag,
        n_steps,
    ):
        """
        Run one emcee chain.

        Returns
        -------
        sampler : emcee.EnsembleSampler
        flat_samples : ndarray
            Samples after discarding the first 10%.
        """
        dtype = [
            (
                "blob",
                np.ndarray,
            )
        ]

        if _should_disable_mp_pool():
            sampler = emcee.EnsembleSampler(
                target.n_walkers,
                target.ndim,
                log_posterior,
                moves=emcee_moves,
                blobs_dtype=dtype,
            )

            sampler.run_mcmc(
                target.initial_states,
                n_steps,
                progress=progress_flag,
            )

        else:
            import multiprocess as mp

            with mp.Pool(
                processes=n_processes
            ) as pool:
                sampler = emcee.EnsembleSampler(
                    target.n_walkers,
                    target.ndim,
                    log_posterior,
                    moves=emcee_moves,
                    pool=pool,
                    blobs_dtype=dtype,
                )

                sampler.run_mcmc(
                    target.initial_states,
                    n_steps,
                    progress=progress_flag,
                )

        discard = int(
            sampler.iteration / 10
        )

        flat_samples = sampler.get_chain(
            discard=discard,
            thin=1,
            flat=True,
        )

        return sampler, flat_samples

    # ============================================================
    # Helper: estimate rough mass
    # ============================================================
    def _estimate_rough_mass(
        mass_samples,
    ):
        """
        Estimate the rough stellar mass from the marginal posterior.
        """
        mass_samples = np.asarray(
            mass_samples,
            dtype=float,
        )

        mass_samples = mass_samples[
            np.isfinite(mass_samples)
        ]

        if mass_samples.size == 0:
            raise ValueError(
                "No finite mass samples are available."
            )

        qs = np.percentile(
            mass_samples,
            [
                100.0 * q
                for q in quantiles
            ],
        )

        return {
            "median": float(qs[1]),
            "q_low": float(qs[0]),
            "q_high": float(qs[2]),
            "plus": float(
                qs[2] - qs[1]
            ),
            "minus": float(
                qs[1] - qs[0]
            ),
            "samples": mass_samples,
        }

    # ============================================================
    # Output filename prefix
    # ============================================================
    name_pre = (
        f"{target.name}"
        f"{target.cluster_name}_"
        f"{target.iso_name}_"
        f"{target.METHOD}_"
        f"{target.grids_type}_grid"
    )

    if target.sigma_adjustment_parameter in [
        1,
        3,
    ]:
        name_pre += (
            f"_sigma"
            f"{target.sigma_adjustment_parameter}"
        )

    if target.extinction_parameter:
        name_pre += (
            f"_{target.extinction_type}"
        )

    # ============================================================
    # Stage 1: estimate rough mass and select Vrot prior
    # ============================================================
    if (
        target.omega_ini_prior == "default"
        and target.Vrot_prior is None
    ):
        print(
            "Estimating rough mass..."
        )

        (
            sampler_evaluate,
            samples_evaluate,
        ) = _run_once(
            progress_flag=use_tqdm,
            n_steps=n_steps_evaluate,
        )

        blobs_evaluate = (
            sampler_evaluate.get_blobs(
                flat=False
            )
        )

        blobs_array = _normalize_blobs(
            blobs_evaluate
        )

        blobs_flat = blobs_array.reshape(
            -1,
            blobs_array.shape[-1],
        )

        mass_index = output_list.index(
            "Mass"
        )

        mass_samples = blobs_flat[
            :,
            mass_index,
        ]

        rough_mass = _estimate_rough_mass(
            mass_samples
        )

        print(
            "Rough mass is "
            f'{rough_mass["median"]:.2f} '
            "M_sun."
        )

        key, Vrot_prior = (
            select_Vrot_prior_by_rough_mass(
                rough_mass["median"],
                log_prior_Vrot_dict,
            )
        )

        target.rough_mass = (
            rough_mass["median"]
        )

        target.Vrot_prior = Vrot_prior

        if key is None:
            print(
                "====> No default Vrot prior available."
            )
        else:
            print(
                "====> Using default Vrot prior:",
                key,
            )

    # ============================================================
    # Stage 2: final test chain
    # ============================================================
    print(
        f"Running a test with "
        f"{n_steps_test} samples..."
    )

    (
        sampler_test,
        samples_test_emcee,
    ) = _run_once(
        progress_flag=use_tqdm,
        n_steps=n_steps_test,
    )

    # ============================================================
    # Prepare corner samples
    # ============================================================
    # Age is already stored in Myr.
    corner_samples_flat = (
        samples_test_emcee.copy()
    )

    if (
        corner_inc_type == "degree"
        and "Inclination"
        in target.sampling_variables_list
    ):
        inc_index = (
            target.sampling_variables_list.index(
                "Inclination"
            )
        )

        corner_samples_flat[
            :,
            inc_index,
        ] = np.rad2deg(
            corner_samples_flat[
                :,
                inc_index,
            ]
        )

    # ============================================================
    # Plot labels
    # ============================================================
    title_map = {
        "Mini": (
            r"$M_{\mathrm{ini}}\ "
            r"(M_\odot)$"
        ),
        "Age": (
            r"$\tau\ "
            r"(\mathrm{Myr})$"
        ),
        "MH": (
            r"$[\mathrm{M/H}]$"
        ),
        "omega_ini": (
            r"$\omega_{\mathrm{ini}}$"
        ),
        "parallax": (
            r"$\varpi\ "
            r"(\mathrm{mas})$"
        ),
        "sigma": (
            r"$\sigma_{\mathrm{adjust}}\ "
            r"(\mathrm{mag})$"
        ),
        "sigma_gaia": (
            r"$\sigma_{\mathrm{adjust},\ Gaia}\ "
            r"(\mathrm{mag})$"
        ),
        "sigma_tycho2": (
            r"$\sigma_{\mathrm{adjust},\ Tycho-2}\ "
            r"(\mathrm{mag})$"
        ),
        "sigma_2mass": (
            r"$\sigma_{\mathrm{adjust},\ 2MASS}\ "
            r"(\mathrm{mag})$"
        ),
    }

    extinction_map = {
        "A_V": (
            r"$A_V\ "
            r"(\mathrm{mag})$"
        ),
        "EB_V": (
            r"$E(B-V)\ "
            r"(\mathrm{mag})$"
        ),
        "EBP_RP": (
            r"$E(G_{\mathrm{BP}}"
            r"-G_{\mathrm{RP}})\ "
            r"(\mathrm{mag})$"
        ),
    }

    corner_sampling_variables_list = []

    for variable in (
        target.sampling_variables_list
    ):
        if variable == "Inclination":
            if corner_inc_type == "radian":
                label = (
                    r"$i\ "
                    r"(\mathrm{rad})$"
                )
            else:
                label = (
                    r"$i\ "
                    r"(^\circ)$"
                )

        elif variable == "extinction":
            label = extinction_map.get(
                target.extinction_type,
                str(
                    target.extinction_type
                ),
            )

        else:
            label = title_map.get(
                variable,
                variable,
            )

        corner_sampling_variables_list.append(
            label
        )

    # ============================================================
    # Diagnostics
    # ============================================================
    acceptance_fraction_arr = np.asarray(
        sampler_test.acceptance_fraction,
        dtype=float,
    )

    print(
        "acceptance ratio:"
    )

    for i, value in enumerate(
        acceptance_fraction_arr
    ):
        if i % 9 == 0:
            print(
                "\t",
                end="",
            )

        print(
            f"{value:.2f}",
            end=" ",
        )

        if (i + 1) % 9 == 0:
            print()

    if (
        len(acceptance_fraction_arr)
        % 9
        != 0
    ):
        print()

    print(
        "mean acceptance ratio: "
        f"{np.mean(acceptance_fraction_arr):.3f}"
    )

    print(
        f"n_steps/60: "
        f"{n_steps_test / 60:.2f}"
    )

    try:
        autocorr_time = (
            sampler_test.get_autocorr_time(
                tol=0
            )
        )

        print(
            "autocorrelation time:",
            np.ceil(
                autocorr_time
            ).astype(int),
        )

    except emcee.autocorr.AutocorrError as exc:
        autocorr_time = None

        print(
            "autocorrelation time could not "
            "be estimated for this short chain: "
            f"{exc}"
        )

    # ============================================================
    # Save target
    # ============================================================
    if save_target:
        target_path = (
            f"{save_path}/"
            f"testing_target_"
            f"{name_pre}.npz"
        )

        target.save(
            target_path
        )

    # ============================================================
    # Save posterior
    # ============================================================
    if save_samples:
        if save_64:
            samples = (
                sampler_test.get_chain()
            )

            log_prob = (
                sampler_test.get_log_prob()
            )

            blobs_full = _normalize_blobs(
                sampler_test.get_blobs()
            )

            acceptance_fraction_save = (
                acceptance_fraction_arr.astype(
                    np.float64
                )
            )

        else:
            samples = (
                sampler_test.get_chain()
                .astype(np.float32)
            )

            log_prob = (
                sampler_test.get_log_prob()
                .astype(np.float32)
            )

            blobs_full = (
                _normalize_blobs(
                    sampler_test.get_blobs()
                )
                .astype(np.float32)
            )

            acceptance_fraction_save = (
                acceptance_fraction_arr.astype(
                    np.float32
                )
            )

        posterior_path = (
            f"{save_path}/"
            f"testing_posterior_"
            f"{name_pre}_"
            f"steps{n_steps_test}.npz"
        )
        target.posterior_path = posterior_path

        np.savez_compressed(
            posterior_path,
            samples=samples,
            log_prob=log_prob,
            blobs=(
                blobs_full
                if blobs_full is not None
                else np.array(
                    None,
                    dtype=object,
                )
            ),
            param_names=np.array(
                target.sampling_variables_list.copy()
            ),
            age_unit=np.array(
                "Myr"
            ),
            output_list=np.array(
                output_list
            ),

            # Walker acceptance ratios
            acceptance_fraction=(
                acceptance_fraction_save
            ),
            mean_acceptance_fraction=np.array(
                np.mean(
                    acceptance_fraction_arr
                )
            ),
            median_acceptance_fraction=np.array(
                np.median(
                    acceptance_fraction_arr
                )
            ),
            min_acceptance_fraction=np.array(
                np.min(
                    acceptance_fraction_arr
                )
            ),
            max_acceptance_fraction=np.array(
                np.max(
                    acceptance_fraction_arr
                )
            ),

            n_steps_test=np.array(
                n_steps_test
            ),
            n_steps_evaluate=np.array(
                n_steps_evaluate
            ),
            n_walkers=np.array(
                target.n_walkers
            ),
            n_dim=np.array(
                len(
                    target.sampling_variables_list
                )
            ),
            n_discard=np.array(
                0
            ),
            n_discard_total=np.array(
                0
            ),
            n_thin=np.array(
                1
            ),
        )

    # ============================================================
    # Corner plot
    # ============================================================
    fig = corner.corner(
        corner_samples_flat,
        labels=corner_sampling_variables_list,
        quantiles=quantiles,
        show_titles=True,
        hist_bin_factor=hist_bin_factor,
        title_fmt=f".{dig_num}f",
    )

    if not show_fig:
        figure_path = (
            f"{save_path}/"
            f"{name_pre}_testing_"
            f"steps_test{n_steps_test}_"
            f"n_steps_evaluate_"
            f"{n_steps_evaluate}.pdf"
        )

        fig.savefig(
            figure_path,
            bbox_inches="tight",
        )

        plt.close(
            fig
        )

    else:
        sys.stdout.flush()
        sys.stderr.flush()

        display(
            fig
        )

        plt.close(
            fig
        )

    return sampler_test, target


def run_sampling(
    initial_steps=10000,
    use_tqdm=True,
    max_hours=4,
    n_processes=8,
    min_length_ratio=None,
    max_tau_relative_difference=None,
    max_check_steps=10000,
    min_check_tau=10.0,
    max_check_minutes=30.0,
):
    """
    Run full MCMC sampling until convergence or ``max_hours``.

    After the initial run, the next checkpoint follows the ACT deficit:
    usable N/τ=33 with a target of 70 requests another 37τ. Once the
    deficit is < ``min_check_tau`` (default 10τ), checks stay 10τ apart,
    subject to the per-batch step/time safety caps.
    """
    initial_steps = int(initial_steps)
    n_processes = int(n_processes)
    max_hours = float(max_hours)
    max_check_steps = int(max_check_steps)
    min_check_tau = float(min_check_tau)

    if initial_steps <= 0 or n_processes <= 0 or max_check_steps <= 0:
        raise ValueError("initial_steps, n_processes, and max_check_steps must be > 0.")
    if not np.isfinite(max_hours) or max_hours <= 0:
        raise ValueError("max_hours must be finite and > 0.")
    if not np.isfinite(min_check_tau) or min_check_tau <= 0:
        raise ValueError("min_check_tau must be finite and > 0.")
    if max_check_minutes is not None:
        max_check_minutes = float(max_check_minutes)
        if not np.isfinite(max_check_minutes) or max_check_minutes <= 0:
            raise ValueError("max_check_minutes must be None or a finite value > 0.")

    _validate_age_sampling()
    min_length_ratio = _validate_triplet(
        min_length_ratio, [70, 60, 50], "min_length_ratio", lower=0,
    )
    max_tau_relative_difference = _validate_triplet(
        max_tau_relative_difference, [0.20, 0.25, 0.30],
        "max_tau_relative_difference", lower=0, upper=1,
    )
    if min_length_ratio[2] < 50:
        print('Warning: final convergence criterion with usable N/ACT < 50 is not recommended.')

    emcee_moves = _validate_emcee_moves(getattr(target, "emcee_moves", None))
    dtype = [("blob", np.ndarray)]
    start = time.time()
    disable_pool = target.METHOD == "NN" and platform.system() == "Linux" and running_in_jupyter()

    args = (
        initial_steps, max_hours, emcee_moves, None, dtype, use_tqdm,
        min_length_ratio, max_tau_relative_difference,
        max_check_steps, min_check_tau, max_check_minutes,
    )

    if disable_pool:
        sampler = runnning_in_pool(*args)
    else:
        import multiprocess as mp
        with mp.Pool(processes=n_processes) as pool:
            args = list(args)
            args[3] = pool
            sampler = runnning_in_pool(*args)

    elapsed = int(time.time() - start)
    print(f"Total sampling time: {elapsed//3600:02d}:{(elapsed%3600)//60:02d}:{elapsed%60:02d}")
    print(f"Total MCMC steps of each walker: {sampler.iteration}")
    return sampler


def get_blob_output_dict(target):
    """
    Return blob names and plotting labels.

    The insertion order must exactly match the blob array
    returned by log_posterior.
    """
    if target.METHOD not in {
        "DIRECT",
        "PARS",
    }:
        raise ValueError(
            f"Unknown target.METHOD: {target.METHOD}"
        )

    photometric_label_map = {
        # Gaia
        "G": r"$G\ (\mathrm{mag})$",
        "G_BP": r"$G_{\mathrm{BP}}\ (\mathrm{mag})$",
        "G_RP": r"$G_{\mathrm{RP}}\ (\mathrm{mag})$",
        "BP": r"$G_{\mathrm{BP}}\ (\mathrm{mag})$",
        "RP": r"$G_{\mathrm{RP}}\ (\mathrm{mag})$",

        # Tycho-2
        "B_T": r"$B_{\mathrm{T}}\ (\mathrm{mag})$",
        "V_T": r"$V_{\mathrm{T}}\ (\mathrm{mag})$",

        # Johnson
        "U": r"$U\ (\mathrm{mag})$",
        "B": r"$B\ (\mathrm{mag})$",
        "V": r"$V\ (\mathrm{mag})$",
        "R": r"$R\ (\mathrm{mag})$",
        "I": r"$I\ (\mathrm{mag})$",

        # 2MASS
        "J": r"$J\ (\mathrm{mag})$",
        "H": r"$H\ (\mathrm{mag})$",
        "Ks": r"$K_{\mathrm{s}}\ (\mathrm{mag})$",
        "K_s": r"$K_{\mathrm{s}}\ (\mathrm{mag})$",
    }

    output_dict = {
        band: photometric_label_map.get(
            band,
            rf"${band}\ (\mathrm{{mag}})$",
        )
        for band in target.bands_list
    }

    # DIRECT 和 PARS 使用完全相同的 blob 输出
    output_dict.update({
        "vsini": (
            r"$v\sin i\ "
            r"(\mathrm{km}\,\mathrm{s}^{-1})$"
        ),
        "chi_squared": r"$\chi^2$",
        "vtaneq": (
            r"$v_{\mathrm{eq}}\ "
            r"(\mathrm{km}\,\mathrm{s}^{-1})$"
        ),
        "Mass": r"$M\ (M_\odot)$",
        "omega": r"$\omega$",
        "logL": r"$\log_{10}(L/L_\odot)$",
        "logTe": r"$\log_{10}(T_{\mathrm{eff}}/\mathrm{K})$",
        "Req": r"$R_{\mathrm{eq}}\ (R_\odot)$",
        "Rpol": r"$R_{\mathrm{pol}}\ (R_\odot)$",
        "lnL0": r"$\ln \mathcal{L}_0$",
        "lnL": r"$\ln \mathcal{L}$",
        "u": r"$u$",
        "p_Vrot": r"$\ln p(v_{\mathrm{eq}})$",
    })

    return output_dict

def save_and_plot(
    sampler,
    show_fig=True,
    burnin_ACT=10,
    group_size=3,
    plot_samples="all",
    folder_suffix="",
    hist_bin_factor=3,
    dig_num=2,
    quantiles=None,
    corner_inc_type="degree",
    save_path=str(RESULTS_DIR),
    save_target=True,
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

    Notes
    -----
    ``Age`` is sampled, summarized, saved, and plotted directly in Myr.
    This function does not support ``logAge`` and performs no age conversion.
    """
    from matplotlib.backends.backend_pdf import PdfPages
    import corner
    import emcee
    from matplotlib import pyplot as plt

    # -------------------------
    # Defaults
    # -------------------------
    if quantiles is None:
        quantiles = [0.16, 0.5, 0.84]

    sampling_variables = list(target.sampling_variables_list)

    if "logAge" in sampling_variables:
        raise ValueError(
            '"logAge" is no longer supported. '
            'Use "Age" in Myr in target.sampling_variables_list.'
        )

    if "Age" not in sampling_variables:
        raise ValueError(
            'target.sampling_variables_list must contain "Age" in Myr. '
            f"Got: {sampling_variables}"
        )

    age_unit = getattr(target, "age_unit", None)
    if age_unit is not None and str(age_unit).lower() != "myr":
        raise ValueError(
            'The target age unit must be "Myr". '
            f"Got age_unit={age_unit!r}."
        )

    if corner_inc_type not in {"radian", "degree"}:
        raise ValueError(
            '"corner_inc_type" must be either "radian" or "degree".'
        )

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
            blob_label_map,
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
        hist_labels = [
            blob_label_map.get(name, name)
            for name in hist_names
        ]

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
                    ax.set_title(hist_labels[i])
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
                    rf"^{{+{q84 - q50:.{digits}f}}}"
                    rf"_{{-{q50 - q16:.{digits}f}}}$"
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

                q2_low, q2_med, q2_high = np.percentile(x, [2.28, 50.0, 97.72])

                label_2sigma = (
                    rf"$2\sigma: {q2_med:.{digits}f}"
                    rf"^{{+{q2_high - q2_med:.{digits}f}}}"
                    rf"_{{-{q2_med - q2_low:.{digits}f}}}$"
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
                    ax.set_title(hist_labels[i])
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

            ax.set_title(hist_labels[i])
            ax.set_xlabel(hist_labels[i])
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
            blob_label_map,
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
            recovered_labels[idx] = blob_label_map.get(
                "Mass",
                r"$M\ (M_\odot)$",
            )

            # Usually true_values has Mini, not current Mass.
            # Keep the original truth only if user explicitly wants it.
            if recovered_truths is not None:
                recovered_truths[idx] = None

        if "omega_ini" in name_to_idx and "omega" in blob_to_idx:
            idx = name_to_idx["omega_ini"]
            recovered_samples[:, idx] = blobs_flat[:, blob_to_idx["omega"]]
            recovered_labels[idx] = blob_label_map.get(
                "omega",
                r"$\omega$",
            )

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
        # tau_late = np.nanmax([tau05_max, tau03_max])
        tau_late = np.maximum(tau05_max, tau03_max)

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
    results_base = save_path
    os.makedirs(results_base, exist_ok=True)

    name_pre = (
        f'{target.name}{target.cluster_name}_{target.iso_name}_{target.METHOD}_{target.grids_type}_grid'
    )

    if target.sigma_adjustment_parameter in [1, 3]:
        name_pre += f"_sigma{target.sigma_adjustment_parameter}"

    if target.extinction_parameter:
        name_pre += f"_{target.extinction_type}"

    if any(x is not None for x in target.true_values) and not all(x is not None for x in target.true_values):
        if target.known_Mini is not None:
            name_pre += f'_known_Mini_{target.known_Mini}'
        if target.known_Age is not None:
            name_pre += f'_known_Age_{target.known_Age}Myr'
        if target.known_MH is not None:
            name_pre += f'_known_MH_{target.known_MH}'
        if target.known_omega_ini is not None:
            name_pre += f'_known_omega_ini_{target.known_omega_ini}'
        if target.known_Inclination is not None:
            name_pre += f'_known_Inclination_{target.known_Inclination}'
        if target.known_parallax is not None:
            name_pre += f'_known_parallax_{target.known_parallax}'

    # User-provided suffix: spaces are converted to underscores.
    # No parentheses are used in directory or file names.
    suffix_clean = str(folder_suffix).strip().replace(" ", "_")

    if suffix_clean:
        folder_name = f"{name_pre}_{suffix_clean}"
    else:
        folder_name = name_pre

    folder_path = os.path.join(results_base, folder_name)

    if not os.path.exists(folder_path):
        new_folder_path = folder_path
        os.makedirs(new_folder_path)
    else:
        count = 2
        while True:
            candidate_path = f"{folder_path}_{count}"
            if not os.path.exists(candidate_path):
                new_folder_path = candidate_path
                os.makedirs(new_folder_path)
                break
            count += 1

    print(
        "Results and plots will be saved at\n"
        f"\t{os.path.abspath(new_folder_path)}"
    )

    # Final unique run directory name, for metadata when needed.
    run_name = os.path.basename(new_folder_path)

    # -------------------------
    # Extract thinned samples / blobs
    # -------------------------
    available_steps = sampler.iteration - n_discard
    if available_steps <= 0:
        raise ValueError(
            f"No samples remain: sampler.iteration={sampler.iteration}, "
            f"n_discard={n_discard}"
        )

    total_available_samples = available_steps * target.n_walkers

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

    needed_steps = math.ceil(plot_samples / target.n_walkers)
    n_thin = max(1, available_steps // needed_steps)

    while math.ceil(available_steps / n_thin) * target.n_walkers < plot_samples:
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
    # Labels and display-unit preparation
    # -------------------------
    title_map = {
        "Mini": r"$M_{\mathrm{ini}}\ (M_\odot)$",
        "Age": r"$\tau\ (\mathrm{Myr})$",
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
        r"$i\ (\mathrm{rad})$"
        if v == "Inclination" and corner_inc_type == "radian"
        else r"$i\ (^\circ)$"
        if v == "Inclination" and corner_inc_type == "degree"
        else extinction_map.get(target.extinction_type, target.extinction_type)
        if v == "extinction"
        else title_map.get(v, v)
        for v in target.sampling_variables_list
    ]

    # Age is already stored in Myr. Only inclination may need a display-unit
    # conversion for corner plots, trace plots, and text summaries.
    corner_samples_flat = samples_flat.copy()
    trace_samples = samples.copy()

    trace_true_values = (
        true_values.copy()
        if isinstance(true_values, list)
        else true_values
    )

    if corner_inc_type == "degree" and "Inclination" in target.sampling_variables_list:
        inc_index = target.sampling_variables_list.index("Inclination")

        corner_samples_flat[:, inc_index] = np.rad2deg(
            corner_samples_flat[:, inc_index]
        )
        trace_samples[:, :, inc_index] = np.rad2deg(
            trace_samples[:, :, inc_index]
        )

        if true_values_flag:
            corner_true_values[inc_index] = np.rad2deg(
                corner_true_values[inc_index]
            )
            trace_true_values[inc_index] = np.rad2deg(
                trace_true_values[inc_index]
            )

    print(f"True values: {true_values}")

    # -------------------------
    # Blob names and plotting labels
    # -------------------------
    output_dict = get_blob_output_dict(
        target
    )

    output_list = list(
        output_dict.keys()
    )

    output_label_list = list(
        output_dict.values()
    )

    output_label_map = output_dict

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
    # Text summary
    # -------------------------

    def _fmt_name(name, width=16):
        """Format parameter name with fixed width."""
        name = str(name)
        if len(name) > width:
            name = name[:width]
        return f"{name:<{width}}"

    def _fmt_text(text, width=14):
        """Format text with fixed width."""
        text = "" if text is None else str(text)
        if len(text) > width:
            text = text[:width]
        return f"{text:<{width}}"

    def _fmt_val(value, width=10):
        """Format value with ~2 significant digits, no scientific notation, left-aligned."""
        if value is None:
            return f"{'':<{width}}"

        try:
            if not np.isfinite(value):
                return f"{'nan':<{width}}"
        except TypeError:
            return f"{str(value):<{width}}"

        value = float(value)
        abs_val = abs(value)

        if abs_val == 0:
            text = "0"
        elif abs_val >= 100:
            text = f"{value:.0f}"
        elif abs_val >= 10:
            text = f"{value:.1f}"
        elif abs_val >= 1:
            text = f"{value:.2f}"
        elif abs_val >= 0.1:
            text = f"{value:.3f}"
        elif abs_val >= 0.01:
            text = f"{value:.4f}"
        elif abs_val >= 0.001:
            text = f"{value:.5f}"
        else:
            text = f"{value:.6f}"

        return f"{text:<{width}}"

    def _prior_type_from_value(value):
        """
        Return prior type label.

        None or 'default' -> default
        anything else     -> user-specific
        """
        if value is None:
            return "default"

        if isinstance(value, str) and value.lower() == "default":
            return "default"

        return "user-specific"

    def _sampling_prior_type(raw_name):
        """
        Determine prior type for sampling parameters.
        """
        prior_attr_map = {
            "Mini": "mini_prior",
            "Age": "Age_prior",
            "MH": "MH_prior",
            "omega_ini": "omega_ini_prior",
            "Inclination": "inclination_prior",
            "parallax": "parallax_prior",
            "extinction": "extinction_prior",
        }

        # Sigma parameters are fixed to uniform/default by design.
        if raw_name in ["sigma", "sigma_gaia", "sigma_tycho2", "sigma_2mass"]:
            return "default"

        attr = prior_attr_map.get(raw_name, None)

        if attr is None:
            return "default"

        return _prior_type_from_value(getattr(target, attr, "default"))

    def _vrot_prior_type(vrot_prior):
        """
        Determine whether Vrot_prior is default or user-specific.

        Default cases:
            None
            'default'
            function generated by _make_log_prior_Vrot with v_max=500.0
        """
        if vrot_prior is None:
            return "default"

        if isinstance(vrot_prior, str):
            if vrot_prior.lower() == "default":
                return "default"
            return "user-specific"

        if callable(vrot_prior):
            import inspect

            qualname = getattr(vrot_prior, "__qualname__", "")

            try:
                sig = inspect.signature(vrot_prior)
                params = sig.parameters

                if "v_max" in params:
                    v_max_default = params["v_max"].default

                    if (
                            "_make_log_prior_Vrot" in qualname
                            and v_max_default == 500.0
                    ):
                        return "default"

                    return "user-specific"

            except Exception:
                pass

            return "user-specific"

        return "user-specific"

    def _derived_prior_type(name):
        """
        Determine prior type for derived parameters.

        Vrot_prior is not itself a sampling parameter, but it constrains vtaneq.
        Other derived parameters do not have priors, so leave Type blank.
        """
        if name == "vtaneq":
            return _vrot_prior_type(getattr(target, "Vrot_prior", None))

        return ""

    def _summary_lines(
            section_title,
            names,
            values_2d,
            raw_names=None,
            lower_limits=None,
            upper_limits=None,
            prior_types=None,
            name_width=16,
            val_width=10,
            type_width=14,
            include_limits=True,
    ):
        """
        Create aligned percentile summary lines.

        values_2d should have shape (n_samples, n_parameters).
        """
        values_2d = np.asarray(values_2d, dtype=float)

        if values_2d.ndim == 1:
            values_2d = values_2d[:, None]

        if values_2d.shape[1] != len(names):
            raise ValueError(
                f"{section_title}: values_2d has {values_2d.shape[1]} columns, "
                f"but names has {len(names)} entries."
            )

        if raw_names is None:
            raw_names = names

        if lower_limits is None:
            lower_limits = [None] * len(names)

        if upper_limits is None:
            upper_limits = [None] * len(names)

        if prior_types is None:
            prior_types = ["default"] * len(names)

        lines = []
        lines.append(section_title)

        header = (
            f"{'parameter':<{name_width}}"
            f"{'2.5%':<{val_width}}"
            f"{'16%':<{val_width}}"
            f"{'50%':<{val_width}}"
            f"{'84%':<{val_width}}"
            f"{'97.5%':<{val_width}}"
        )

        if include_limits:
            header += (
                f"{'Min':<{val_width}}"
                f"{'Max':<{val_width}}"
                f"{'Prior type':<{type_width}}"
            )

        lines.append(header)

        for i, name in enumerate(names):
            x = values_2d[:, i]
            x = x[np.isfinite(x)]

            if x.size == 0:
                q = [np.nan, np.nan, np.nan, np.nan, np.nan]
            else:
                q = np.percentile(x, [2.5, 16, 50, 84, 97.5])

            line = _fmt_name(name, width=name_width)
            line += "".join(_fmt_val(v, width=val_width) for v in q)

            if include_limits:
                line += _fmt_val(lower_limits[i], width=val_width)
                line += _fmt_val(upper_limits[i], width=val_width)
                line += _fmt_text(prior_types[i], width=type_width)

            lines.append(line)

        return lines

    # -------------------------
    # Write summary text file
    # -------------------------

    summary_filename = "summary.txt"

    summary_path = os.path.join(
        new_folder_path,
        summary_filename,
    )

    # Age values are already in Myr. corner_samples_flat differs from the raw
    # chain only when inclination is displayed in degrees.
    summary_raw_param_names = list(
        target.sampling_variables_list
    )

    summary_param_names = [
        "Age_Myr" if name == "Age" else name
        for name in target.sampling_variables_list
    ]

    if (
            corner_inc_type == "degree"
            and "Inclination" in summary_param_names
    ):
        summary_param_names = [
            "Incl_deg" if name == "Inclination" else name
            for name in summary_param_names
        ]

    sampling_lower_limits = list(
        getattr(
            target,
            "sampling_lower_limit",
            [],
        )
    )

    sampling_upper_limits = list(
        getattr(
            target,
            "sampling_upper_limit",
            [],
        )
    )

    if (
            len(sampling_lower_limits)
            != len(summary_raw_param_names)
    ):
        sampling_lower_limits = [
                                    None
                                ] * len(summary_raw_param_names)

    if (
            len(sampling_upper_limits)
            != len(summary_raw_param_names)
    ):
        sampling_upper_limits = [
                                    None
                                ] * len(summary_raw_param_names)

    if (
            corner_inc_type == "degree"
            and "Inclination" in summary_raw_param_names
    ):
        inc_index = summary_raw_param_names.index(
            "Inclination"
        )

        if sampling_lower_limits[inc_index] is not None:
            sampling_lower_limits[inc_index] = (
                np.rad2deg(
                    sampling_lower_limits[inc_index]
                )
            )

        if sampling_upper_limits[inc_index] is not None:
            sampling_upper_limits[inc_index] = (
                np.rad2deg(
                    sampling_upper_limits[inc_index]
                )
            )

    sampling_prior_types = [
        _sampling_prior_type(name)
        for name in summary_raw_param_names
    ]

    derived_lower_limits = [
                               None
                           ] * len(output_list)

    derived_upper_limits = [
                               None
                           ] * len(output_list)

    derived_prior_types = [
        _derived_prior_type(name)
        for name in output_list
    ]

    # ============================================================
    # Walker diagnostics
    # ============================================================

    # Use the unthinned post-burn-in chain for walker diagnostics.
    # Single-walker ACT is used only as an auxiliary diagnostic.
    walker_chain_post = sampler.get_chain(
        discard=n_discard,
        thin=1,
        flat=False,
    )

    walker_log_prob_post = sampler.get_log_prob(
        discard=n_discard,
        thin=1,
        flat=False,
    )

    if walker_chain_post.ndim != 3:
        raise ValueError(
            "Expected post-burn-in chain shape "
            "(n_steps, n_walkers, n_dim), "
            f"but got {walker_chain_post.shape}."
        )

    if walker_log_prob_post.ndim != 2:
        raise ValueError(
            "Expected post-burn-in log-probability shape "
            "(n_steps, n_walkers), "
            f"but got {walker_log_prob_post.shape}."
        )

    n_steps_post = walker_chain_post.shape[0]
    n_walkers_post = walker_chain_post.shape[1]
    n_dim_post = walker_chain_post.shape[2]

    if n_steps_post <= 0:
        raise ValueError(
            "No post-burn-in steps are available for "
            "walker diagnostics."
        )

    if n_walkers_post != target.n_walkers:
        raise ValueError(
            "Walker-count mismatch: "
            f"chain has {n_walkers_post}, "
            f"target.n_walkers={target.n_walkers}."
        )

    if n_dim_post != len(summary_raw_param_names):
        raise ValueError(
            "Parameter-count mismatch: "
            f"chain has {n_dim_post} dimensions, "
            f"but sampling_variables_list has "
            f"{len(summary_raw_param_names)} entries."
        )

    acceptance_fraction = np.asarray(
        sampler.acceptance_fraction,
        dtype=float,
    )

    if acceptance_fraction.shape != (
            n_walkers_post,
    ):
        raise ValueError(
            "Unexpected acceptance_fraction shape: "
            f"{acceptance_fraction.shape}; expected "
            f"({n_walkers_post},)."
        )

    finite_log_prob_values = (
        walker_log_prob_post[
            np.isfinite(walker_log_prob_post)
        ]
    )

    if finite_log_prob_values.size > 0:
        global_median_log_prob = float(
            np.median(finite_log_prob_values)
        )
    else:
        global_median_log_prob = np.nan

    walker_diagnostics = []

    for walker_index in range(n_walkers_post):
        walker_chain = walker_chain_post[
            :,
            walker_index,
            :,
        ]

        walker_log_prob = walker_log_prob_post[
            :,
            walker_index,
        ]

        # --------------------------------------------
        # Single-walker autocorrelation time
        # --------------------------------------------
        max_act = np.nan
        slowest_parameter = "N/A"
        min_ess = np.nan

        try:
            walker_tau = (
                emcee.autocorr.integrated_time(
                    walker_chain,
                    tol=0,
                    quiet=True,
                )
            )

            walker_tau = np.asarray(
                walker_tau,
                dtype=float,
            ).reshape(-1)

            finite_tau_mask = (
                    np.isfinite(walker_tau)
                    & (walker_tau > 0)
            )

            if np.any(finite_tau_mask):
                finite_tau_indices = np.where(
                    finite_tau_mask
                )[0]

                slowest_local_index = int(
                    finite_tau_indices[
                        np.argmax(
                            walker_tau[
                                finite_tau_indices
                            ]
                        )
                    ]
                )

                max_act = float(
                    walker_tau[
                        slowest_local_index
                    ]
                )

                slowest_parameter = (
                    summary_raw_param_names[
                        slowest_local_index
                    ]
                )

                min_ess = float(
                    n_steps_post / max_act
                )

        except Exception:
            # ACT may fail for a short or poorly mixed
            # individual-walker chain.
            max_act = np.nan
            slowest_parameter = "N/A"
            min_ess = np.nan

        # --------------------------------------------
        # Walker log-probability statistics
        # --------------------------------------------
        finite_walker_log_prob = walker_log_prob[
            np.isfinite(walker_log_prob)
        ]

        if finite_walker_log_prob.size > 0:
            median_log_prob = float(
                np.median(
                    finite_walker_log_prob
                )
            )
        else:
            median_log_prob = np.nan

        if (
                np.isfinite(median_log_prob)
                and np.isfinite(
            global_median_log_prob
        )
        ):
            delta_log_prob = (
                    median_log_prob
                    - global_median_log_prob
            )
        else:
            delta_log_prob = np.nan

        walker_diagnostics.append({
            "walker": walker_index,
            "acceptance": float(
                acceptance_fraction[
                    walker_index
                ]
            ),
            "max_ACT": max_act,
            "slowest_param": (
                slowest_parameter
            ),
            "min_ESS": min_ess,
            "median_logP": median_log_prob,
            "delta_logP": delta_log_prob,
        })

    # ============================================================
    # Build summary
    # ============================================================

    summary_lines = []

    summary_lines += _summary_lines(
        section_title="Sampling parameters",
        names=summary_param_names,
        values_2d=corner_samples_flat,
        raw_names=summary_raw_param_names,
        lower_limits=sampling_lower_limits,
        upper_limits=sampling_upper_limits,
        prior_types=sampling_prior_types,
        name_width=16,
        val_width=10,
        type_width=14,
        include_limits=True,
    )

    summary_lines.append("")

    summary_lines += _summary_lines(
        section_title="Derived parameters",
        names=output_list,
        values_2d=blobs_flat,
        raw_names=output_list,
        lower_limits=derived_lower_limits,
        upper_limits=derived_upper_limits,
        prior_types=derived_prior_types,
        name_width=16,
        val_width=10,
        type_width=14,
        include_limits=True,
    )

    # ============================================================
    # Append walker diagnostics
    # ============================================================

    summary_lines.append("")
    summary_lines.append(
        "Walker diagnostics"
    )

    walker_header = (
        f"{'walker':<10}"
        f"{'acceptance':<14}"
        f"{'max_ACT':<12}"
        f"{'slowest_param':<18}"
        f"{'min_ESS':<12}"
        f"{'median_logP':<16}"
        f"{'delta_logP':<14}"
    )

    summary_lines.append(
        walker_header
    )

    for item in walker_diagnostics:
        max_act_text = (
            f"{item['max_ACT']:.1f}"
            if np.isfinite(
                item["max_ACT"]
            )
            else "nan"
        )

        min_ess_text = (
            f"{item['min_ESS']:.1f}"
            if np.isfinite(
                item["min_ESS"]
            )
            else "nan"
        )

        median_logp_text = (
            f"{item['median_logP']:.3f}"
            if np.isfinite(
                item["median_logP"]
            )
            else "nan"
        )

        delta_logp_text = (
            f"{item['delta_logP']:.3f}"
            if np.isfinite(
                item["delta_logP"]
            )
            else "nan"
        )

        summary_lines.append(
            f"{item['walker']:<10d}"
            f"{item['acceptance']:<14.3f}"
            f"{max_act_text:<12}"
            f"{item['slowest_param']:<18}"
            f"{min_ess_text:<12}"
            f"{median_logp_text:<16}"
            f"{delta_logp_text:<14}"
        )

    # ============================================================
    # Write summary file
    # ============================================================

    with open(
            summary_path,
            "w",
            encoding="utf-8",
    ) as f:
        f.write(
            "\n".join(summary_lines)
        )
        f.write("\n")

    # print(f"Saved summary text file: {summary_path}")

    # -------------------------
    # Saving
    # -------------------------

    from pathlib import Path
    import shutil

    if save_target:
        target_save_path = Path(new_folder_path) / "target.npz"

        if not getattr(target, "load_flag", None):
            target.save(target_save_path, verbose=False)
        else:
            source_path = Path(target.load_flag)

            if not source_path.exists():
                raise FileNotFoundError(
                    f"Loaded target file does not exist: {source_path}"
                )

            shutil.copy2(source_path, target_save_path)

    if save_samples:
        if save_64:
            samples_full = sampler.get_chain()
            lnprob_full = sampler.get_log_prob()
            blobs_full = _normalize_blobs(
                sampler.get_blobs()
            )
            acceptance_fraction = np.asarray(
                sampler.acceptance_fraction,
                dtype=np.float64,
            )
        else:
            samples_full = (
                sampler.get_chain()
                .astype(np.float32)
            )
            lnprob_full = (
                sampler.get_log_prob()
                .astype(np.float32)
            )
            blobs_full = (
                _normalize_blobs(
                    sampler.get_blobs()
                )
                .astype(np.float32)
            )
            acceptance_fraction = np.asarray(
                sampler.acceptance_fraction,
                dtype=np.float32,
            )

        param_names = np.array(
            target.sampling_variables_list.copy()
        )

        posterior_path = Path(new_folder_path) / "posterior.npz"
        target.posterior_path = posterior_path

        np.savez_compressed(
            posterior_path,
            samples=samples_full,
            log_prob=lnprob_full,
            param_names=param_names,
            age_unit=np.array("Myr"),
            output_list=np.array(output_list),
            output_labels=np.array(output_label_list),
            blobs=blobs_full,
            acceptance_fraction=acceptance_fraction,
            run_name=np.array(run_name),
            user_suffix=np.array(suffix_clean),
            n_steps_thin_plot=n_steps,
            n_walkers=n_walkers,
            n_dim=n_dim,
            n_discard=n_discard,
            n_discard_total=n_discard * n_walkers,
            n_thin=n_thin,
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
                blob_label_map=output_label_map,
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
                    / "corner_recovered_parameters.pdf"
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
                    / "blobs_hist.pdf"
            )

            _plot_blobs_hist(
                blobs_flat=blobs_flat,
                blob_names=output_list,
                blob_label_map=output_label_map,
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
                    ax.plot(x, trace_samples[:, w, dim], alpha=0.5, label=f"walker {w}")

                if true_values_flag:
                    ax.axhline(
                        trace_true_values[dim],
                        color="red",
                        linestyle="--",
                        label="true values",
                    )
                    true_ylabel = f"\ntruth={np.round(trace_true_values[dim], 4)}"
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
                blob_label = output_label_map.get(
                    blob_name,
                    blob_name,
                )

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
                            ax.set_ylabel(blob_label)
                        else:
                            ax.set_ylabel(f"{blob_label}\n($\\sigma$={error})")
                    else:
                        ax.set_ylabel(blob_label)

                else:
                    # Number of observational constraints
                    n_data = len(target.bands_list) + 1  # photometry + parallax

                    if (
                            np.isfinite(target.vsini_obs)
                            and np.isfinite(target.vsini_obs_err)
                    ):
                        n_data += 1  # add vsini constraint

                    # Number of free sampled parameters
                    n_free = len(target.sampling_variables_list)

                    dof = n_data - n_free

                    if dof <= 0:
                        warnings.warn(
                            f"Non-positive degrees of freedom: "
                            f"n_data={n_data}, n_free={n_free}, dof={dof}. "
                            "Reduced chi-square will not be plotted.",
                            RuntimeWarning,
                        )

                        for w in walker_indices:
                            chi2 = blobs_array[:, w, idx]
                            ax.plot(
                                x,
                                chi2,
                                alpha=0.5,
                                label=f"walker {w}",
                            )

                        min_val = np.nanmin(blobs_array[:, :, idx])

                        ax.set_ylabel(
                            r"$\chi^2$"
                            f"\n(min={min_val:.4f})"
                        )

                    else:
                        for w in walker_indices:
                            chi2 = blobs_array[:, w, idx]
                            reduced_chi2 = chi2 / dof

                            ax.plot(
                                x,
                                reduced_chi2,
                                alpha=0.5,
                                label=f"walker {w}",
                            )

                        min_chi2 = np.nanmin(
                            blobs_array[:, :, idx]
                        )
                        min_reduced_chi2 = min_chi2 / dof

                        ax.set_ylabel(
                            r"Reduced $\chi_\nu^2$"
                            f"\n(min={min_reduced_chi2:.4f}, "
                            rf"$\nu$={dof})"
                        )

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
            f"trace_group{g}_burnin{n_discard}_thin{n_thin}.pdf",
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
                / "corner_sampling_parameters.pdf"
        )

        with PdfPages(corner_pdf_path) as pdf:
            pdf.savefig(fig, dpi=150)

        print("Finished saving plots.")
        # print(f"Saved folder: {os.path.abspath(new_folder_path)}")

        if show_fig and running_in_jupyter():
            plt.show()

        plt.close(fig)

    except Exception as e:
        print(f"⚠️ Original corner plot failed: {e}")
        try:
            plt.close("all")
        except Exception:
            pass

    return target