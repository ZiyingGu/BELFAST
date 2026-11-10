import os
import time
import math
import json
import platform
import warnings
from pathlib import Path
from scipy.stats import norm
from datetime import datetime
import numpy as np
from belfast.target_utils import cal_obs_mag, cal_parsec, add_extinction
from belfast.paths import REPO_ROOT, RESULTS_DIR, DATA_DIR
from belfast.config import GM_SUN, R_SUN, PC_TO_CM, _VROT_PRIOR_PARAMS

target = None

# Ignore RuntimeWarnings caused by invalid scalar subtraction during the initial MCMC steps.
# Since walkers are initialized randomly, some initial positions may have a posterior log-probability of -inf.
# When emcee compares two invalid log-probabilities, it may evaluate -inf - (-inf), resulting in this warning.
# This behavior has been tested and validated:
# all walkers eventually reach finite posterior log-probabilities, mand the sampling proceeds normally.
# Therefore, we suppress this warning rather than changing the initialization strategy.
warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in scalar subtract",
    category=RuntimeWarning,
)

def _logsumexp_two(a, b):
    """
    Compute log(exp(a) + exp(b)) in a numerically stable way.

    Args:
        a (array-like): First input value or array.
        b (array-like): Second input value or array.

    Returns:
        numpy.ndarray: Logarithm of the sum of exponentials.
    """
    return np.logaddexp(a, b)


def _maxwell_cdf(x, alpha):
    """
    Compute the cumulative distribution function of a Maxwell distribution.

    Args:
        x (float or array-like): Velocity values, physically supported for x >= 0.
        alpha (float): Maxwell distribution scale parameter.

    Returns:
        numpy.ndarray: Cumulative probabilities clipped to [0, 1].
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


def _log_prior_maxwell_truncated(v, alpha, v_max=600.0):
    """
    Compute the normalized log-PDF of a truncated Maxwell distribution.

    Args:
        v (float or array-like): Equatorial rotational velocity in km/s.
        alpha (float): Maxwell distribution scale parameter.
        v_max (float): Upper velocity cutoff in km/s. Defaults to 500.0.

    Returns:
        float or numpy.ndarray: Log-probability density within [0, v_max],
            or -inf for values outside this range.

    Raises:
        ValueError: If the normalization constant is non-positive.
    """

    v = np.asarray(v, dtype=float)

    out = np.full_like(v, -np.inf, dtype=float)
    ok = (v > 0.0) & (v <= v_max)

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


def _log_prior_lagged_maxwell_truncated(v, alpha, ell, v_max=600.0):
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
    ok = (v > ell) & (v <= v_max)

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
    v, slow_frac, slow_mu, fast_frac, fast_sqrt2alpha, fast_ell, v_max=600.0
):
    """
    Compute the normalized log-prior of a truncated Maxwellian mixture.

    Args:
        v (float or array-like): Equatorial rotational velocity in km/s.
        slow_frac (float): Fraction of slow rotators.
        slow_mu (float or None): Mode of the slow Maxwellian in km/s.
        fast_frac (float): Fraction of fast rotators.
        fast_sqrt2alpha (float): sqrt(2) times the fast Maxwellian scale parameter.
        fast_ell (float): Lag parameter for the fast Maxwellian in km/s.
        v_max (float): Upper velocity cutoff in km/s. Defaults to 500.0.

    Returns:
        float or numpy.ndarray: Log-probability density of the mixture, with
            components normalized over their respective truncated ranges.

    Raises:
        ValueError: If the mixture fractions do not sum to 1 or a required
            slow_mu is None.
    """
    v = np.asarray(v, dtype=float)
    frac_sum = slow_frac + fast_frac
    if not np.isclose(frac_sum, 1.0, atol=1e-12):
        raise ValueError(f"slow_frac + fast_frac must equal 1. Got {frac_sum}")

    log_terms = []
    if slow_frac > 0:
        if slow_mu is None:
            raise ValueError("slow_mu cannot be None when slow_frac > 0")
        alpha_slow = slow_mu / np.sqrt(2.0)
        log_terms.append(
            np.log(slow_frac) + _log_prior_maxwell_truncated(v, alpha_slow, v_max)
        )

    if fast_frac > 0:
        alpha_fast = fast_sqrt2alpha / np.sqrt(2.0)
        log_terms.append(
            np.log(fast_frac)
            + _log_prior_lagged_maxwell_truncated(v, alpha_fast, fast_ell, v_max)
        )

    if not log_terms:
        out = np.full_like(v, -np.inf, dtype=float)
    elif len(log_terms) == 1:
        out = log_terms[0]
    else:
        out = _logsumexp_two(log_terms[0], log_terms[1])

    return out if out.ndim else float(out)



def _make_log_prior_Vrot(params):
    """
    Create a rotational-velocity log-prior function from parameter settings.

    Args:
        params (dict): Parameters defining the slow and fast rotator mixture.

    Returns:
        callable: Function accepting velocity and an optional upper cutoff.
    """
    def _f(v, v_max=600.0):
        return _log_prior_mixture_maxwell_lagged_truncated(
            v, slow_frac=params["slow_frac"], slow_mu=params["slow_mu"],
            fast_frac=params["fast_frac"], fast_sqrt2alpha=params["fast_sqrt2alpha"],
            fast_ell=params["fast_ell"], v_max=v_max,
        )
    return _f


log_prior_Vrot_dict = {
    key: _make_log_prior_Vrot(params)
    for key, params in _VROT_PRIOR_PARAMS.items()
}


def parse_mass_range(key):
    """
    Parse the minimum and maximum masses from a mass-range string.

    Args:
        key (str): Mass range, e.g. '1.60-2.00 Msun'.

    Returns:
        tuple[float, float]: Lower and upper mass limits in solar masses.
    """
    m_min, m_max = key.replace("Msun", "").strip().split("-")
    return float(m_min), float(m_max)


def select_Vrot_prior_by_rough_mass(rough_mass, log_prior_Vrot_dict):
    """
    Select the rotational-velocity prior for a rough mass estimate.

    Selects the prior whose bin contains the estimated mass and whose center
    is closest to it. Returns (None, None) if no bin contains the mass.

    Args:
        rough_mass (float): Rough stellar mass estimate in solar masses.
        log_prior_Vrot_dict (dict): Mapping of mass-bin labels to prior functions.

    Returns:
        tuple: Selected mass-bin label and prior function, or (None, None)
            if the mass is not covered by any bin.
    """
    best_key, best_dist = None, np.inf

    for key in log_prior_Vrot_dict:
        m_min, m_max = parse_mass_range(key)
        if m_min <= rough_mass <= m_max:
            dist = abs(rough_mass - 0.5 * (m_min + m_max))
            if dist < best_dist:
                best_key, best_dist = key, dist

    return (best_key, log_prior_Vrot_dict[best_key]) if best_key is not None else (None, None)



def log_prior(sampling_variables, verbose=False):
    """
    Compute the total log-prior for the sampling variables.

    Args:
        sampling_variables (list or np.ndarray): Current parameter values. Inclination is in radians.
        verbose (bool): Whether to print individual log-priors.

    Returns:
        float: Total log-prior, or -np.inf if any parameter is outside its allowed bounds.
    """
    sampling_variables = np.asarray(sampling_variables, dtype=float)
    names = target.sampling_variables_list

    if not np.all((sampling_variables > target.sampling_lower_limit) &
                  (sampling_variables < target.sampling_upper_limit)):
        if verbose:
            print('Out of the parameter space of sampling!')
        return -np.inf

    params = dict(zip(names, sampling_variables))
    priors = {}

    for name, attr in [
        ('Mini', 'Mini_prior'),
        ('Age', 'Age_prior'),
        ('MH', 'MH_prior'),
        ('omega_ini', 'omega_ini_prior'),
        ('Inclination', 'inclination_prior'),
        ('parallax', 'parallax_prior'),
        ('extinction', 'extinction_prior'),
    ]:
        if name not in params:
            priors[name] = 0.0
            continue

        value = params[name]
        prior = getattr(target, attr)

        if name == 'Mini' and prior == 'default':
            priors[name] = -2.3 * np.log(np.maximum(1e-8, value))
        elif name == 'Age' and prior == 'default':
            priors[name] = 0.0
        elif name == 'MH':
            if target.iso_name == 'MIST' and target.correct_MIST_MH_zeropoint:
                value += target.MIST_MH_offset
            if prior == 'default':
                priors[name] = norm.logpdf(value, loc=-0.106, scale=0.224)
            else:
                priors[name] = prior(value)
        elif name == 'omega_ini' and prior == 'default':
            priors[name] = 0.0
        elif name == 'Inclination' and prior == 'default':
            priors[name] = np.log(np.maximum(1e-8, np.sin(value)))
        elif name == 'parallax' and prior == 'default':
            priors[name] = -4.0 * np.log(np.maximum(1e-8, value))
        elif name == 'extinction' and prior == 'default':
            priors[name] = 0.0
        else:
            priors[name] = prior(value)

    if verbose:
        for name, value in priors.items():
            print(f'{name}_prior: {value}')

    return sum(priors.values())




def log_likelihood_INTERP(sampling_variables, verbose=False):
    """
    Compute the log-likelihood using the interpolation method.

    Args:
        sampling_variables (list or np.ndarray): Current sampled parameter values.
        verbose (bool): Whether to print diagnostic information.

    Returns:
        tuple: Total log-likelihood and diagnostic blobs. The returned log-likelihood includes
            p_Vrot, while the lnL value stored in blobs does not. Returns -np.inf with NaN blobs
            if interpolation or the Vrot constraint fails.
    """
    sampling_variables = np.asarray(sampling_variables, dtype=float)
    names = target.sampling_variables_list
    params = dict(zip(names, sampling_variables))

    if 'MH' in params and target.iso_name == 'MIST' and target.correct_MIST_MH_zeropoint:
        sampling_variables = sampling_variables.copy()
        mh_idx = names.index('MH')
        sampling_variables[mh_idx] += target.MIST_MH_offset
        params['MH'] = sampling_variables[mh_idx]

    Mini = params.get('Mini', target.known_Mini)
    Age = params.get('Age', target.known_Age)
    MH = params.get('MH', target.known_MH)
    omega_ini = params.get('omega_ini', target.known_omega_ini)
    Inclination = params.get('Inclination', target.known_Inclination)
    parallax = params.get('parallax', target.known_parallax)

    if 'extinction' in params:
        extinction, extinction_err = params['extinction'], 0
    else:
        extinction, extinction_err = target.extinction, target.extinction_err

    if target.sigma_adjustment_parameter == 1:
        sigma = params['sigma']
        obs_err_mags_new = np.sqrt(target.obs_err_mags**2 + sigma**2)
    elif target.sigma_adjustment_parameter == 3:
        sigma_gaia = params['sigma_gaia']
        sigma_tycho2 = params['sigma_tycho2']
        sigma_2mass = params['sigma_2mass']
        obs_err_mags_sigma = (
            np.array([1, 1, 1, 0, 0, 0, 0, 0]) * sigma_gaia
            + np.array([0, 0, 0, 1, 1, 0, 0, 0]) * sigma_tycho2
            + np.array([0, 0, 0, 0, 0, 1, 1, 1]) * sigma_2mass
        )
        obs_err_mags_new = np.sqrt(target.obs_err_mags**2 + obs_err_mags_sigma**2)
    elif target.sigma_adjustment_parameter == 0:
        obs_err_mags_new = target.obs_err_mags
    else:
        raise ValueError('sigma_adjustment_parameter must be 0, 1, or 3.')

    if target.fixed_parameter_indices:
        for idx in sorted(target.fixed_parameter_indices):
            sampling_variables = np.insert(sampling_variables, idx, target.true_values[idx])

    y_interpolated = target.Model_interp(
        sampling_variables[:len(target.interp_variables_list)]
    ).flatten()

    if not np.all(np.isfinite(y_interpolated)):
        if verbose:
            print('Failed in interpolation.')
        return -np.inf, np.full(len(target.bands_list) + 13, np.nan)

    Mass, omega, logL, logTe, Req, Rpol = y_interpolated[:6]
    abs_mags_interpolated = y_interpolated[6:]

    obs_mags_model = cal_obs_mag(abs_mags_interpolated, parallax)
    obs_err_mags_model = np.zeros_like(obs_mags_model)

    if target.extinction_parameter:
        final_mags, obs_err_mags_model = add_extinction(
            obs_mags_model, obs_err_mags_model, target.bands_list, extinction, extinction_err,
            target.extinction_type, verbose=False
        )
    else:
        final_mags = obs_mags_model

    final_err_mags = np.sqrt(obs_err_mags_new**2 + np.asarray(obs_err_mags_model)**2)

    omega_angular = omega * np.sqrt(GM_SUN * Mass / (R_SUN * 1.5 * Rpol)**3)
    vtaneq = omega_angular * Req * R_SUN / 1000
    vsini = vtaneq * np.sin(Inclination)

    chi_photo = np.sum(((final_mags - target.obs_mags) / final_err_mags)**2)
    chi_parallax = ((parallax - target.parallax_obs) / target.parallax_obs_err)**2

    if np.isnan(target.vsini_obs) or np.isnan(target.vsini_obs_err):
        chi_vsini = 0
    else:
        chi_vsini = ((vsini - target.vsini_obs) / target.vsini_obs_err)**2

    if (any(x is not None for x in target.true_values)
            and not all(x is not None for x in target.true_values)
            and target.known_omega_ini is not None
            and target.known_Inclination is not None
            and np.isclose(target.known_omega_ini * np.sin(target.known_Inclination), 0)):
        chi_vsini = 0

    if verbose:
        print(f'photometric Chi_square: {chi_photo}')
        print(f'parallax Chi_square: {chi_parallax}')
        print(f'vsini Chi_square: {chi_vsini}')

    chi_squared = float(chi_photo + chi_parallax + chi_vsini)
    if np.isnan(chi_squared):
        chi_squared = np.inf

    lnL0 = -0.5 * chi_squared
    if target.sigma_adjustment_parameter in (1, 3):
        lnL = lnL0 - np.sum(np.log(final_err_mags))
    else:
        lnL = lnL0

    u = target.interp_u([Mini, Age, MH, omega_ini, Inclination]) if target.iso_name == 'PARSEC' else 0

    p_Vrot = 0
    if target.Vrot_prior is not None:
        p_Vrot = target.Vrot_prior(vtaneq)
        if not np.isfinite(p_Vrot):
            if verbose:
                print('Failed in Vrot_prior calculation.')
            return -np.inf, np.full(len(target.bands_list) + 13, np.nan)

    lnL_total = lnL + p_Vrot

    blobs = final_mags.tolist() + [
        vsini, chi_squared, vtaneq, Mass, omega, logL, logTe, Req, Rpol, lnL0, lnL, u, p_Vrot
    ]

    return lnL_total, np.asarray([float(np.squeeze(x)) for x in blobs])




def log_likelihood_PARS(sampling_variables, verbose=False):
    """
    Compute the log-likelihood using PARS photometry.

    Args:
        sampling_variables (list or np.ndarray): Current sampled parameter values.
        verbose (bool): Whether to print diagnostic information.

    Returns:
        tuple: Total log-likelihood and diagnostic blobs. The returned log-likelihood includes
            p_Vrot, while the lnL value stored in blobs does not. Returns -np.inf with NaN blobs
            if interpolation or PARS calculations fail.
    """
    from scipy.interpolate import interp1d
    from pa.lib import star as st

    sampling_variables = np.asarray(sampling_variables, dtype=float)
    names = target.sampling_variables_list
    params = dict(zip(names, sampling_variables))

    if 'MH' in params and target.iso_name == 'MIST' and target.correct_MIST_MH_zeropoint:
        sampling_variables = sampling_variables.copy()
        mh_idx = names.index('MH')
        sampling_variables[mh_idx] += target.MIST_MH_offset
        params['MH'] = sampling_variables[mh_idx]

    Mini = params.get('Mini', target.known_Mini)
    Age = params.get('Age', target.known_Age)
    MH = params.get('MH', target.known_MH)
    omega_ini = params.get('omega_ini', target.known_omega_ini)
    Inclination = params.get('Inclination', target.known_Inclination)
    parallax = params.get('parallax', target.known_parallax)

    if 'extinction' in params:
        extinction, extinction_err = params['extinction'], 0
    else:
        extinction, extinction_err = target.extinction, target.extinction_err

    if target.sigma_adjustment_parameter == 1:
        sigma = params['sigma']
        obs_err_mags_new = np.sqrt(target.obs_err_mags**2 + sigma**2)
    elif target.sigma_adjustment_parameter == 3:
        sigma_gaia = params['sigma_gaia']
        sigma_tycho2 = params['sigma_tycho2']
        sigma_2mass = params['sigma_2mass']
        obs_err_mags_sigma = (
            np.array([1, 1, 1, 0, 0, 0, 0, 0]) * sigma_gaia
            + np.array([0, 0, 0, 1, 1, 0, 0, 0]) * sigma_tycho2
            + np.array([0, 0, 0, 0, 0, 1, 1, 1]) * sigma_2mass
        )
        obs_err_mags_new = np.sqrt(target.obs_err_mags**2 + obs_err_mags_sigma**2)
    elif target.sigma_adjustment_parameter == 0:
        obs_err_mags_new = target.obs_err_mags
    else:
        raise ValueError('sigma_adjustment_parameter must be 0, 1, or 3.')

    if target.fixed_parameter_indices:
        for idx in sorted(target.fixed_parameter_indices):
            sampling_variables = np.insert(sampling_variables, idx, target.true_values[idx])

    interp_input = sampling_variables[:len(target.interp_variables_list)]
    model_output = target.Model_interp(interp_input).flatten()

    if not np.all(np.isfinite(model_output)):
        if verbose:
            print('Failed in interpolation.')
        return -np.inf, np.full(len(target.bands_list) + 13, np.nan)

    if target.iso_name.upper() == 'PARSEC':
        Mass, omega, logL, logTe, Req, Rpol = model_output
        omega_angular = omega * np.sqrt(GM_SUN * Mass / (R_SUN * 1.5 * Rpol)**3)
        omega_keplerian = np.sqrt(GM_SUN * Mass / (R_SUN * Req)**3)
        omega_k = omega_angular / omega_keplerian
        # v_critical = np.sqrt(GM_SUN * Mass / (R_SUN * 1.5 * Rpol)) / 1000
    elif target.iso_name.upper() == 'MIST':
        Mass, omega, logL, logTe, Req = model_output
        Rpol = Req
        omega_angular = omega * np.sqrt(GM_SUN * Mass / (R_SUN * Req)**3)
        omega_k = omega
        # v_critical = np.sqrt(GM_SUN * Mass / (R_SUN * Req)) / 1000
    else:
        raise ValueError(f'Unsupported isochrone model: {target.iso_name}')

    vtaneq = omega_angular * Req * R_SUN / 1000
    vsini = vtaneq * np.sin(Inclination)
    distance_pc = cal_parsec(parallax)
    distance_cm = distance_pc * PC_TO_CM

    output_rearrange_index = [
        list(target.made_limb_darkening_dict.values())[0].bands.index(band)
        for band in target.bands_list
    ]
    MH_PARS = MH + target.PARS_MH_offset
    ld_dict = target.made_limb_darkening_dict

    def calculate_PARS_mags(limb_darkening_file):
        star = st.Star(
            omega_k, 10**logL, Mass, Req, distance_cm, target.PARS_nz, limb_darkening_file
        )
        original_mags = star.integrate(Inclination)
        return np.asarray([original_mags[i] for i in output_rearrange_index])

    if MH_PARS in ld_dict:
        try:
            obs_mags_model = calculate_PARS_mags(ld_dict[MH_PARS])
        except Exception:
            if verbose:
                print('Failed in PARS calculation.')
            return -np.inf, np.full(len(target.bands_list) + 13, np.nan)
    else:
        sorted_MH = sorted(ld_dict)
        lower_MH = max((v for v in sorted_MH if v < MH_PARS), default=None)
        upper_MH = min((v for v in sorted_MH if v > MH_PARS), default=None)
        closest_dict = {v: ld_dict[v] for v in (lower_MH, upper_MH) if v is not None}
        PARS_dict = {}

        for mh, limb_darkening_file in closest_dict.items():
            try:
                PARS_dict[mh] = calculate_PARS_mags(limb_darkening_file)
            except Exception:
                if verbose:
                    print('Failed in PARS calculation.')
                return -np.inf, np.full(len(target.bands_list) + 13, np.nan)

        try:
            interp_MH = interp1d(
                np.array(list(PARS_dict.keys())),
                np.array(list(PARS_dict.values())).T,
                axis=-1
            )
            obs_mags_model = interp_MH(MH_PARS)
        except Exception:
            if verbose:
                print('Failed in PARS metallicity interpolation.')
            return -np.inf, np.full(len(target.bands_list) + 13, np.nan)

    if not np.all(np.isfinite(obs_mags_model)):
        if verbose:
            print('Invalid PARS magnitudes.')
        return -np.inf, np.full(len(target.bands_list) + 13, np.nan)

    obs_err_mags_model = np.zeros_like(obs_mags_model)

    if target.extinction_parameter:
        final_mags, obs_err_mags_model = add_extinction(
            obs_mags_model, obs_err_mags_model, target.bands_list, extinction, extinction_err,
            target.extinction_type, verbose=False
        )
    else:
        final_mags, obs_err_mags_model = obs_mags_model, 0

    final_err_mags = np.sqrt(obs_err_mags_new**2 + np.asarray(obs_err_mags_model)**2)

    chi_photo = np.sum(((final_mags - target.obs_mags) / final_err_mags)**2)
    chi_parallax = ((parallax - target.parallax_obs) / target.parallax_obs_err)**2

    if np.isnan(target.vsini_obs) or np.isnan(target.vsini_obs_err):
        chi_vsini = 0
    else:
        chi_vsini = ((vsini - target.vsini_obs) / target.vsini_obs_err)**2

    if (any(x is not None for x in target.true_values)
            and not all(x is not None for x in target.true_values)
            and target.known_omega_ini is not None
            and target.known_Inclination is not None
            and np.isclose(target.known_omega_ini * np.sin(target.known_Inclination), 0)):
        chi_vsini = 0

    if verbose:
        print(f'photometric Chi_square: {chi_photo}')
        print(f'parallax Chi_square: {chi_parallax}')
        print(f'vsini Chi_square: {chi_vsini}')

    chi_squared = float(chi_photo + chi_parallax + chi_vsini)
    if np.isnan(chi_squared):
        chi_squared = np.inf

    lnL0 = -0.5 * chi_squared
    if target.sigma_adjustment_parameter in (1, 3):
        lnL = lnL0 - np.sum(np.log(final_err_mags))
    else:
        lnL = lnL0

    u = target.interp_u(interp_input) if target.iso_name.upper() == 'PARSEC' else 0

    p_Vrot = 0
    if target.Vrot_prior is not None:
        p_Vrot = target.Vrot_prior(vtaneq)
        if not np.isfinite(p_Vrot):
            if verbose:
                print('Failed in Vrot_prior calculation.')
            return -np.inf, np.full(len(target.bands_list) + 13, np.nan)

    lnL_total = lnL + p_Vrot

    blobs = final_mags.tolist() + [
        vsini, chi_squared, vtaneq, Mass, omega, logL, logTe, Req, Rpol, lnL0, lnL, u, p_Vrot
    ]

    return lnL_total, np.asarray([float(np.squeeze(x)) for x in blobs])




def log_posterior(sampling_variables):
    """
    Compute the log-posterior probability and diagnostic blobs.

    Args:
        sampling_variables (list or np.ndarray): Current parameter values.

    Returns:
        tuple: Log-posterior probability and diagnostic blobs.
    """
    log_pr = log_prior(sampling_variables)
    if not np.isfinite(log_pr):
        return -np.inf, np.full(len(target.bands_list) + 13, np.nan)

    if target.METHOD == "DIRECT":
        log_lkh, blobs = log_likelihood_INTERP(sampling_variables)
    elif target.METHOD == "PARS":
        log_lkh, blobs = log_likelihood_PARS(sampling_variables)
    else:
        raise ValueError(f"Unknown METHOD: {target.METHOD}")

    return log_pr + log_lkh, blobs




def running_in_jupyter():
    """
    Return whether the current Python session runs in a Jupyter kernel.
    """
    try:
        from IPython import get_ipython
        shell = get_ipython()
        return shell is not None and "IPKernelApp" in shell.config
    except (ImportError, AttributeError):
        return False



def _validate_emcee_moves(emcee_moves):
    """
    Validate the emcee move configuration.

    Args:
        emcee_moves: A list or tuple of ``(move, fraction)`` pairs, or None.

    Returns:
        The validated move configuration, or None if no configuration is provided.

    Raises:
        TypeError: If a move or its fraction has an invalid type.
        ValueError: If the configuration or fractions are invalid.
    """
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
    """
    Validate a sequence of exactly three finite values.

    Args:
        values: Values to validate, or None to use the default.
        default: Default values used when values is None.
        name: Parameter name used in error messages.
        lower: Optional exclusive lower bound for all values.
        upper: Optional exclusive upper bound for all values.

    Returns:
        A list containing the three validated values.

    Raises:
        ValueError: If the values have the wrong shape, contain non-finite values,
            or violate the specified bounds.
    """
    arr = np.asarray(default if values is None else values, dtype=float)

    if arr.shape != (3,) or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain exactly three finite values.")
    if lower is not None and np.any(arr <= lower):
        raise ValueError(f"All {name} values must be > {lower}.")
    if upper is not None and np.any(arr >= upper):
        raise ValueError(f"All {name} values must be < {upper}.")

    return arr.tolist()


def _validate_age_sampling():
    """
    Validate that the target samples stellar age directly in Myr.

    Raises:
        ValueError: If ``logAge`` is sampled, ``Age`` is missing, or the age unit is not Myr.
    """
    variables = list(target.sampling_variables_list)
    if "logAge" in variables or "Age" not in variables:
        raise ValueError(f'Sampling must use "Age" in Myr, not "logAge". Got: {variables}')

    age_unit = getattr(target, "age_unit", None)
    if age_unit is not None and str(age_unit).lower() != "myr":
        raise ValueError(f'The target age unit must be "Myr". Got: {age_unit!r}')



def _estimate_late_tau(chain):
    """
    Estimate conservative ACT and its relative change between two late-chain segments.

    Args:
        chain: MCMC chain with shape (n_steps, n_walkers, n_dim).

    Returns:
        A tuple containing the conservative maximum ACT, the ACT estimates from the
        latter 50% and 30% of the chain, and their relative difference.

    Raises:
        ValueError: If no finite positive ACT estimate is available.
    """
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
    """
    Calculate the number of steps until the next MCMC checkpoint.

    Args:
        tau: Estimated integrated autocorrelation time (ACT), in steps.
        current_ratio: Current usable chain length divided by ACT.
        target_ratio: Required usable chain length divided by ACT.
        max_check_steps: Maximum number of steps per checkpoint.
        min_check_tau: Minimum planned checkpoint interval in units of ACT.
        seconds_per_step: Estimated runtime per MCMC step, if available.
        max_check_minutes: Maximum estimated duration of a checkpoint, in minutes.
        remaining_seconds: Remaining runtime budget, in seconds.

    Returns:
        A tuple containing the next step count, missing ACT ratio, planned ACT interval,
        and requested step count before applying limits.
    """
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



def running_in_pool(
    initial_steps, max_hours, emcee_moves, pool, dtype, show_progress,
    min_length_ratio_list, max_tau_relative_difference_list,
    max_check_steps=10000, min_check_tau=10.0, max_check_minutes=30.0,
):
    """
    Run adaptive MCMC sampling until convergence or the runtime limit is reached.

    Args:
        initial_steps: Number of steps in the initial sampling batch.
        max_hours: Maximum allowed runtime in hours.
        emcee_moves: Validated emcee move configuration.
        pool: Multiprocessing pool, or None to run without multiprocessing.
        dtype: Blob data type passed to the emcee sampler.
        show_progress: Whether to display the sampling progress bar.
        min_length_ratio_list: Minimum usable chain length divided by ACT for each runtime stage.
        max_tau_relative_difference_list: Maximum allowed relative ACT difference for each stage.
        max_check_steps: Maximum number of steps per checkpoint.
        min_check_tau: Minimum planned checkpoint interval in units of ACT.
        max_check_minutes: Maximum estimated duration of a checkpoint, in minutes.

    Returns:
        The completed emcee EnsembleSampler.

    Raises:
        TimeoutError: If the sampling runtime reaches the configured limit.
    """
    import emcee

    sampler = emcee.EnsembleSampler(
        target.n_walkers, target.ndim, log_posterior,
        moves=emcee_moves, pool=pool, blobs_dtype=dtype,
    )
    start_total = time.time()
    max_runtime = max_hours * 3600
    stable_count = 0
    previous_stage = None

    if not show_progress:
        print(f"Starting Time: {datetime.now():%Y-%m-%d %H:%M:%S}")

    print(f"🔄 Initial sampling: {initial_steps} steps")
    batch_start = time.time()
    sampler.run_mcmc(target.initial_states, initial_steps, progress=show_progress)

    if not show_progress:
        dt = int(time.time() - batch_start)
        print(f"Sampling time: {dt // 3600:02d}:{(dt % 3600) // 60:02d}:{dt % 60:02d}")

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
            print(f"⏱ Next: {check_steps} steps (missing {missing_tau:.1f}τ, planned {planned_tau:.1f}τ{cap_note})")
        else:
            print(f"⏱ Next: {check_steps} steps (fallback)")

        batch_start = time.time()
        sampler.run_mcmc(None, check_steps, progress=show_progress)

        if not show_progress:
            dt = int(time.time() - batch_start)
            print(f"Sampling time: {dt // 3600:02d}:{(dt % 3600) // 60:02d}:{dt % 60:02d}")

    return sampler



def _normalize_blobs(blobs_in):
    """
    Convert sampler blobs to a float array with three dimensions.

    Args:
        blobs_in: Blobs returned by ``sampler.get_blobs()``, potentially as a structured
            array or an object array containing individual blob values.

    Returns:
        A float64 array with shape ``(n_steps, n_walkers, n_blob)``.

    Raises:
        ValueError: If blobs are missing, have an unexpected structure, or cannot be
            normalized to the required three-dimensional shape.
    """
    if blobs_in is None:
        raise ValueError("sampler.get_blobs() returned None.")

    arr = np.asarray(blobs_in)

    if arr.dtype.names is not None:
        if "blob" not in arr.dtype.names:
            raise ValueError(f"Unexpected blob dtype names: {arr.dtype.names}")
        arr = arr["blob"]

    if arr.dtype == object:
        if arr.ndim == 2:
            arr = np.stack([
                np.stack([arr[i, j] for j in range(arr.shape[1])], axis=0)
                for i in range(arr.shape[0])
            ], axis=0)
        elif arr.ndim == 1:
            arr = np.stack(arr, axis=0)
        else:
            arr = np.asarray(arr.tolist())

    arr = np.asarray(arr, dtype=np.float64)

    if arr.ndim != 3:
        raise ValueError(f"Expected blobs_array shape (n_steps, n_walkers, n_blob), but got {arr.shape}.")

    return arr



def serialize_emcee_moves(emcee_moves):
    """
    Serialize emcee move objects into a dictionary configuration.

    Args:
        emcee_moves: A sequence of ``(move, fraction)`` pairs, or None.

    Returns:
        A dictionary mapping move type names to their parameters and fractions,
        or None if no move configuration is provided.

    Raises:
        AttributeError: If a required move parameter cannot be read.
        ValueError: If a move type is unsupported, duplicated, or fractions do not sum to 1.
    """
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
            a = getattr(move, "a", getattr(move, "_a", None))
            if a is None:
                raise AttributeError("Could not read parameter 'a' from StretchMove.")
            move_name, config = "StretchMove", {"a": float(a), "fraction": fraction}

        elif isinstance(move, emcee.moves.DEMove):
            gamma0 = getattr(move, "gamma0", getattr(move, "_gamma0", None))
            move_name = "DEMove"
            config = {"gamma0": None if gamma0 is None else float(gamma0), "fraction": fraction}

        elif isinstance(move, emcee.moves.DESnookerMove):
            move_name, config = "DESnookerMove", {"fraction": fraction}

        elif isinstance(move, emcee.moves.KDEMove):
            move_name, config = "KDEMove", {"fraction": fraction}

        else:
            raise ValueError(f"Unsupported emcee move type: {type(move).__name__}")

        if move_name in move_config:
            raise ValueError(f"Duplicate move type: {move_name}")

        move_config[move_name] = config

    if not np.isclose(total_fraction, 1.0, rtol=0.0, atol=1e-8):
        raise ValueError(f"Move fractions must sum to 1. Got {total_fraction}.")

    return move_config



def build_emcee_moves(emcee_move_config):
    """
    Build emcee move objects from a serialized configuration.

    Args:
        emcee_move_config: Dictionary mapping move type names to their parameters and fractions,
            or None.

    Returns:
        A list of ``(move, fraction)`` pairs, or None if no configuration is provided.

    Raises:
        KeyError: If a required configuration parameter is missing.
        ValueError: If a move type is unsupported.
    """
    import emcee

    if emcee_move_config is None:
        return None

    emcee_moves = []
    for move_name, config in emcee_move_config.items():
        fraction = float(config["fraction"])

        if move_name == "StretchMove":
            move = emcee.moves.StretchMove(a=float(config["a"]))
        elif move_name == "DEMove":
            move = emcee.moves.DEMove(gamma0=config.get("gamma0"))
        elif move_name == "DESnookerMove":
            move = emcee.moves.DESnookerMove()
        elif move_name == "KDEMove":
            move = emcee.moves.KDEMove()
        else:
            raise ValueError(f"Unknown move type: {move_name}")

        emcee_moves.append((move, fraction))

    return emcee_moves



def run_test_sampling(
    n_steps_test=None, n_steps_evaluate=None, show_progress=True, show_fig=True, n_processes=8,
    corner_inc_type="degree", quantiles=None, hist_bin_factor=3, dig_num=2,
    save_path=RESULTS_DIR, save_target=False, save_samples=False, save_64=False,
):
    """
    Run a short MCMC sampling run for testing and visualization.

    Notes:
        Age is sampled, saved, and plotted directly in Myr.
        Inclination is displayed in radians or degrees according to corner_inc_type.
        Moves are read from target.emcee_moves and are not serialized.
        If omega_ini_prior is "default" and Vrot_prior is None, a preliminary chain
        estimates the rough mass and selects the corresponding default Vrot prior.

    Args:
        n_steps_test: Number of steps in the final test chain. Defaults to 3000.
        n_steps_evaluate: Number of preliminary steps for estimating mass. Defaults to n_steps_test.
        show_progress: Whether to display the sampling progress bar.
        show_fig: Whether to display the corner plot instead of saving it.
        n_processes: Number of worker processes for multiprocessing.
        corner_inc_type: Inclination display unit, either "radian" or "degree".
        quantiles: Three increasing quantiles used in the corner plot and mass summary.
        hist_bin_factor: Multiplicative factor controlling histogram binning.
        dig_num: Number of decimal places displayed in corner-plot titles.
        save_path: Directory for saved target, posterior, and plot files.
        save_target: Whether to save the updated target.
        save_samples: Whether to save the full final chain and associated data.
        save_64: Whether to preserve the original floating-point precision when saving.

    Returns:
        A tuple containing the final emcee EnsembleSampler and the updated target.

    Raises:
        ValueError: If sampling parameters, inclination units, quantiles, or move settings
            are invalid.
        TimeoutError: Not raised directly by this function; sampling errors propagate.
    """
    import sys

    import corner
    import emcee
    import numpy as np
    from IPython.display import display
    from matplotlib import pyplot as plt

    if n_steps_test is None:
        n_steps_test = 3000
    if n_steps_evaluate is None:
        n_steps_evaluate = n_steps_test
    if quantiles is None:
        quantiles = [0.16, 0.5, 0.84]

    n_steps_test, n_steps_evaluate, n_processes = int(n_steps_test), int(n_steps_evaluate), int(n_processes)

    if n_steps_test <= 0:
        raise ValueError("n_steps_test must be a positive integer.")
    if n_steps_evaluate <= 0:
        raise ValueError("n_steps_evaluate must be a positive integer.")
    if n_processes <= 0:
        raise ValueError("n_processes must be a positive integer.")
    if corner_inc_type not in {"radian", "degree"}:
        raise ValueError('"corner_inc_type" must be either "radian" or "degree".')

    _validate_age_sampling()
    quantiles = np.asarray(quantiles, dtype=float)

    if quantiles.ndim != 1 or len(quantiles) != 3:
        raise ValueError("quantiles must contain exactly three values.")
    if (not np.all(np.isfinite(quantiles)) or np.any(quantiles < 0) or np.any(quantiles > 1)
            or not np.all(np.diff(quantiles) > 0)):
        raise ValueError("quantiles must be finite, strictly increasing, and between 0 and 1.")

    quantiles = quantiles.tolist()
    emcee_moves = _validate_emcee_moves(getattr(target, "emcee_moves", None))

    save_path = os.fspath(save_path)
    if save_samples or save_target or not show_fig:
        os.makedirs(save_path, exist_ok=True)
    # save_path = save_path.rstrip("/\\") or "."

    output_dict = get_blob_output_dict(target)
    output_list = list(output_dict.keys())

    def _should_disable_mp_pool():
        """Return whether multiprocessing should be disabled for NN on Linux in Jupyter."""
        return target.METHOD == "NN" and platform.system() == "Linux" and running_in_jupyter()

    def _run_once(progress_flag, n_steps):
        """Run one emcee chain and return the sampler and flattened post-burn-in samples."""
        dtype = [("blob", np.ndarray)]

        if _should_disable_mp_pool():
            sampler = emcee.EnsembleSampler(
                target.n_walkers, target.ndim, log_posterior, moves=emcee_moves, blobs_dtype=dtype
            )
            sampler.run_mcmc(target.initial_states, n_steps, progress=progress_flag)
        else:
            import multiprocess as mp

            with mp.Pool(processes=n_processes) as pool:
                sampler = emcee.EnsembleSampler(
                    target.n_walkers, target.ndim, log_posterior, moves=emcee_moves, pool=pool,
                    blobs_dtype=dtype,
                )
                sampler.run_mcmc(target.initial_states, n_steps, progress=progress_flag)

        discard = int(sampler.iteration / 10)
        flat_samples = sampler.get_chain(discard=discard, thin=1, flat=True)
        return sampler, flat_samples

    def _estimate_rough_mass(mass_samples):
        """Estimate the rough stellar mass from its marginal posterior."""
        mass_samples = np.asarray(mass_samples, dtype=float)
        mass_samples = mass_samples[np.isfinite(mass_samples)]

        if mass_samples.size == 0:
            raise ValueError("No finite mass samples are available.")

        qs = np.percentile(mass_samples, [100.0 * q for q in quantiles])
        return {
            "median": float(qs[1]),
            "q_low": float(qs[0]),
            "q_high": float(qs[2]),
            "plus": float(qs[2] - qs[1]),
            "minus": float(qs[1] - qs[0]),
            "samples": mass_samples,
        }

    name_pre = f"{target.name}{target.cluster_name}_{target.iso_name}_{target.METHOD}_{target.regrid_type}_grid"
    if target.sigma_adjustment_parameter in [1, 3]:
        name_pre += f"_sigma{target.sigma_adjustment_parameter}"
    if target.extinction_parameter:
        name_pre += f"_{target.extinction_type}"

    if target.omega_ini_prior == "default" and target.Vrot_prior is None:
        print("Estimating rough mass...")
        sampler_evaluate, samples_evaluate = _run_once(show_progress, n_steps_evaluate)

        blobs_array = _normalize_blobs(sampler_evaluate.get_blobs(flat=False))
        blobs_flat = blobs_array.reshape(-1, blobs_array.shape[-1])
        mass_samples = blobs_flat[:, output_list.index("Mass")]
        rough_mass = _estimate_rough_mass(mass_samples)

        print(f'Rough mass is {rough_mass["median"]:.2f} M_sun.')
        key, Vrot_prior = select_Vrot_prior_by_rough_mass(
            rough_mass["median"], log_prior_Vrot_dict
        )
        target.rough_mass = rough_mass["median"]
        target.Vrot_prior = Vrot_prior

        if key is None:
            print("====> No default Vrot prior available.")
        else:
            print("====> Using default Vrot prior:", key)

    print(f"Running a test with {n_steps_test} samples...")
    sampler_test, samples_test_emcee = _run_once(show_progress, n_steps_test)

    # Age is already stored in Myr.
    corner_samples_flat = samples_test_emcee.copy()
    if corner_inc_type == "degree" and "Inclination" in target.sampling_variables_list:
        inc_index = target.sampling_variables_list.index("Inclination")
        corner_samples_flat[:, inc_index] = np.rad2deg(corner_samples_flat[:, inc_index])

    title_map = {
        "Mini": r"$M_{\mathrm{ini}}\ (M_\odot)$",
        "Age": r"$\tau\ (\mathrm{Myr})$",
        "MH": r"$[\mathrm{M/H}]$",
        "omega_ini": r"$\omega_{\mathrm{ini}}$",
        "parallax": r"$\varpi\ (\mathrm{mas})$",
        "sigma": r"$\sigma_{\mathrm{adjust}}\ (\mathrm{mag})$",
        "sigma_gaia": r"$\sigma_{\mathrm{adjust},\ Gaia}\ (\mathrm{mag})$",
        "sigma_tycho2": r"$\sigma_{\mathrm{adjust},\ Tycho-2}\ (\mathrm{mag})$",
        "sigma_2mass": r"$\sigma_{\mathrm{adjust},\ 2MASS}\ (\mathrm{mag})$",
    }
    extinction_map = {
        "A_V": r"$A_V\ (\mathrm{mag})$",
        "EB_V": r"$E(B-V)\ (\mathrm{mag})$",
        "EBP_RP": r"$E(G_{\mathrm{BP}}-G_{\mathrm{RP}})\ (\mathrm{mag})$",
    }

    corner_sampling_variables_list = []
    for variable in target.sampling_variables_list:
        if variable == "Inclination":
            label = r"$i\ (\mathrm{rad})$" if corner_inc_type == "radian" else r"$i\ (^\circ)$"
        elif variable == "extinction":
            label = extinction_map.get(target.extinction_type, str(target.extinction_type))
        else:
            label = title_map.get(variable, variable)
        corner_sampling_variables_list.append(label)

    acceptance_fraction_arr = np.asarray(sampler_test.acceptance_fraction, dtype=float)
    print("acceptance ratio:")
    for i, value in enumerate(acceptance_fraction_arr):
        if i % 9 == 0:
            print("\t", end="")
        print(f"{value:.2f}", end=" ")
        if (i + 1) % 9 == 0:
            print()
    if len(acceptance_fraction_arr) % 9 != 0:
        print()

    print(f"mean acceptance ratio: {np.mean(acceptance_fraction_arr):.3f}")
    print(f"n_steps/60: {n_steps_test / 60:.2f}")

    try:
        autocorr_time = sampler_test.get_autocorr_time(tol=0)
        print("autocorrelation time:", np.ceil(autocorr_time).astype(int))
    except emcee.autocorr.AutocorrError as exc:
        autocorr_time = None
        print(f"autocorrelation time could not be estimated for this short chain: {exc}")

    if save_target:
        target_path = f"{save_path}/testing_target_{name_pre}.npz"
        target.save(target_path)

    if save_samples:
        if save_64:
            samples = sampler_test.get_chain()
            log_prob = sampler_test.get_log_prob()
            blobs_full = _normalize_blobs(sampler_test.get_blobs())
            acceptance_fraction_save = acceptance_fraction_arr.astype(np.float64)
        else:
            samples = sampler_test.get_chain().astype(np.float32)
            log_prob = sampler_test.get_log_prob().astype(np.float32)
            blobs_full = _normalize_blobs(sampler_test.get_blobs()).astype(np.float32)
            acceptance_fraction_save = acceptance_fraction_arr.astype(np.float32)

        posterior_path = f"{save_path}/testing_posterior_{name_pre}_steps{n_steps_test}.npz"
        target.posterior_path = posterior_path

        np.savez_compressed(
            posterior_path,
            true_values=target.true_values,
            samples=samples,
            log_prob=log_prob,
            blobs=blobs_full if blobs_full is not None else np.array(None, dtype=object),
            param_names=np.array(target.sampling_variables_list.copy()),
            age_unit=np.array("Myr"),
            output_list=np.array(output_list),
            acceptance_fraction=acceptance_fraction_save,
            mean_acceptance_fraction=np.array(np.mean(acceptance_fraction_arr)),
            median_acceptance_fraction=np.array(np.median(acceptance_fraction_arr)),
            min_acceptance_fraction=np.array(np.min(acceptance_fraction_arr)),
            max_acceptance_fraction=np.array(np.max(acceptance_fraction_arr)),
            n_steps_test=np.array(n_steps_test),
            n_steps_evaluate=np.array(n_steps_evaluate),
            n_walkers=np.array(target.n_walkers),
            n_dim=np.array(len(target.sampling_variables_list)),
            n_discard=np.nan,
            n_discard_total=np.nan,
            n_thin=np.array(1),
        )

    fig = corner.corner(
        corner_samples_flat, labels=corner_sampling_variables_list, quantiles=quantiles,
        show_titles=True, hist_bin_factor=hist_bin_factor, title_fmt=f".{dig_num}f",
    )

    if not show_fig:
        figure_path = (
            f"{save_path}/{name_pre}_testing_steps_test{n_steps_test}_"
            f"n_steps_evaluate_{n_steps_evaluate}.pdf"
        )
        fig.savefig(figure_path, bbox_inches="tight")
        plt.close(fig)
    else:
        sys.stdout.flush()
        sys.stderr.flush()
        display(fig)
        plt.close(fig)

    return sampler_test, target



def run_sampling(
    initial_steps=10000,
    show_progress=True,
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
        initial_steps, max_hours, emcee_moves, None, dtype, show_progress,
        min_length_ratio, max_tau_relative_difference,
        max_check_steps, min_check_tau, max_check_minutes,
    )

    if disable_pool:
        sampler = running_in_pool(*args)
    else:
        import multiprocess as mp
        with mp.Pool(processes=n_processes) as pool:
            args = list(args)
            args[3] = pool
            sampler = running_in_pool(*args)

    elapsed = int(time.time() - start)
    print(f"Total sampling time: {elapsed//3600:02d}:{(elapsed%3600)//60:02d}:{elapsed%60:02d}")
    print(f"Total MCMC steps of each walker: {sampler.iteration}")
    return sampler



def get_blob_output_dict(target):
    """
    Return blob names and plotting labels in the order expected by ``log_posterior``.

    Args:
        target: Target object containing ``METHOD`` and ``bands_list``.

    Returns:
        A dictionary mapping blob names to plotting labels.
    """
    if target.METHOD not in {"DIRECT", "PARS"}:
        raise ValueError(f"Unknown target.METHOD: {target.METHOD}")

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
        band: photometric_label_map.get(band, rf"${band}\ (\mathrm{{mag}})$")
        for band in target.bands_list
    }

    # DIRECT and PARS use the same blob outputs.
    output_dict.update({
        "vsini": r"$v\sin i\ (\mathrm{km}\,\mathrm{s}^{-1})$",
        "chi_squared": r"$\chi^2$",
        "vtaneq": r"$v_{\mathrm{eq}}\ (\mathrm{km}\,\mathrm{s}^{-1})$",
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
    save_path=RESULTS_DIR,
    save_target=True,
    save_samples=True,
    save_64=False,
    plot_recovered_corner=True,
    plot_blobs_hist=True,
    blobs_hist_bins=80,
    blobs_hist_ncols=6,
):
    """Save MCMC results, diagnostic plots, and posterior summaries.

    Args:
        sampler: The emcee sampler containing chains, log probabilities, and blobs.
        show_fig: Whether to display the original corner plot in Jupyter.
        burnin_ACT: Multiplier applied to the estimated autocorrelation time for burn-in.
        group_size: Number of walkers displayed in each trace plot.
        plot_samples: Maximum number of samples used for plots, or "all".
        folder_suffix: Optional suffix appended to the output directory name.
        hist_bin_factor: Multiplicative factor controlling corner-plot histogram bins.
        dig_num: Number of decimal places shown in corner-plot titles.
        quantiles: Quantiles displayed in corner plots.
        corner_inc_type: Inclination display unit, either "radian" or "degree".
        save_path: Base directory for saving results.
        save_target: Whether to save the target object.
        save_samples: Whether to save the full posterior chains.
        save_64: Whether to save posterior arrays in float64 rather than float32.
        plot_recovered_corner: Whether to save the recovered-parameter corner plot.
        plot_blobs_hist: Whether to save histograms of derived parameters.
        blobs_hist_bins: Number of bins used in blob histograms.
        blobs_hist_ncols: Number of columns in the blob histogram figure.

    Returns:
        The target object with its posterior path updated when samples are saved.

    Notes:
        ``Age`` is sampled and stored directly in Myr. ``logAge`` is not supported.
        The expected blob shape is ``(n_steps, n_walkers, n_blob)``.
    """
    from pathlib import Path
    from matplotlib.backends.backend_pdf import PdfPages
    import corner
    import emcee
    import shutil
    from matplotlib import pyplot as plt

    if quantiles is None:
        quantiles = [0.16, 0.5, 0.84]

    sampling_variables = list(target.sampling_variables_list)

    if "logAge" in sampling_variables:
        raise ValueError('"logAge" is not supported. Use "Age" in Myr.')
    if "Age" not in sampling_variables:
        raise ValueError(f'target.sampling_variables_list must contain "Age": {sampling_variables}')
    if getattr(target, "age_unit", None) is not None and str(target.age_unit).lower() != "myr":
        raise ValueError(f'The target age unit must be "Myr", got {target.age_unit!r}.')
    if corner_inc_type not in {"radian", "degree"}:
        raise ValueError('"corner_inc_type" must be "radian" or "degree".')
    if plot_samples != "all" and (
        isinstance(plot_samples, (bool, np.bool_))
        or not isinstance(plot_samples, (int, np.integer))
        or plot_samples <= 0
    ):
        raise ValueError('"plot_samples" must be a positive integer or "all".')
    if group_size <= 0:
        raise ValueError('"group_size" must be a positive integer.')
    if blobs_hist_ncols <= 0:
        raise ValueError('"blobs_hist_ncols" must be a positive integer.')

    def running_in_jupyter():
        try:
            from IPython import get_ipython
            return get_ipython().__class__.__name__ == "ZMQInteractiveShell"
        except Exception:
            return False

    def _latex_summary(x, digits=3):
        x = np.asarray(x, dtype=float)
        x = x[np.isfinite(x)]
        if not x.size:
            return None, None, None, None

        q16, q50, q84 = np.percentile(x, [16, 50, 84])
        label = rf"${q50:.{digits}f}^{{+{q84-q50:.{digits}f}}}_{{-{q50-q16:.{digits}f}}}$"
        return q16, q50, q84, label

    def _plot_blobs_hist(blobs_flat, blob_names, blob_label_map, out_path, bins=80, ncols=4,
                         digits=3, color="#6BAED6", shade_color="grey"):
        hist_data = np.asarray(blobs_flat, dtype=np.float64)
        if hist_data.ndim == 1:
            hist_data = hist_data[:, None]

        hist_names = list(blob_names)
        hist_labels = [blob_label_map.get(name, name) for name in hist_names]
        if hist_data.shape[1] != len(hist_names):
            raise ValueError(
                f"blobs_flat has {hist_data.shape[1]} columns, but blob_names has {len(hist_names)}."
            )
        if not hist_names:
            return

        n_hist = len(hist_names)
        nrows = math.ceil(n_hist / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), squeeze=False)
        axes = axes.ravel()

        for i, name in enumerate(hist_names):
            ax = axes[i]
            x = hist_data[:, i]
            x = x[np.isfinite(x)]
            ax.set_title(hist_labels[i])
            ax.set_xlabel(hist_labels[i])
            ax.set_ylabel("Density")

            if not x.size:
                ax.text(0.5, 0.5, "No finite samples", ha="center", va="center",
                        transform=ax.transAxes)
                continue

            q16, q50, q84, label = _latex_summary(x, digits=digits)
            ax.hist(x, bins=bins, density=True, alpha=0.65, color=color, edgecolor="none")
            ax.axvspan(q16, q84, color=shade_color, alpha=0.25, zorder=3, label=label)
            ax.axvline(q50, color="black", linestyle="--", linewidth=1.5)

            if name == "u":
                q2_low, q2_med, q2_high = np.percentile(x, [2.28, 50.0, 97.72])
                label_2sigma = (
                    rf"$2\sigma: {q2_med:.{digits}f}"
                    rf"^{{+{q2_high-q2_med:.{digits}f}}}"
                    rf"_{{-{q2_med-q2_low:.{digits}f}}}$"
                )
                ax.axvspan(q2_low, q2_high, color=shade_color, alpha=0.12, zorder=2,
                           label=label_2sigma)
                ax.axvline(q2_low, color="black", linestyle=":", linewidth=1.2)
                ax.axvline(q2_high, color="black", linestyle=":", linewidth=1.2)

            ax.legend(frameon=False, fontsize=9)

        for ax in axes[n_hist:]:
            fig.delaxes(ax)

        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight", dpi=150)
        plt.close(fig)

    def _make_recovered_corner_samples(corner_samples_flat, corner_labels, sampling_names,
                                       blobs_flat, blob_names, blob_label_map,
                                       corner_true_values=None, true_values_flag=False):
        """Replace initial mass and rotation parameters with their recovered blob values."""
        recovered_samples = corner_samples_flat.copy()
        recovered_labels = list(corner_labels)
        recovered_truths = list(corner_true_values) if (
            true_values_flag and corner_true_values is not None
        ) else None

        name_to_idx = {name: i for i, name in enumerate(sampling_names)}
        blob_to_idx = {name: i for i, name in enumerate(blob_names)}

        for parameter, blob_name, fallback_label in (
            ("Mini", "Mass", r"$M\ (M_\odot)$"),
            ("omega_ini", "omega", r"$\omega$"),
        ):
            if parameter not in name_to_idx or blob_name not in blob_to_idx:
                continue

            idx = name_to_idx[parameter]
            recovered_samples[:, idx] = blobs_flat[:, blob_to_idx[blob_name]]
            recovered_labels[idx] = blob_label_map.get(blob_name, fallback_label)
            if recovered_truths is not None:
                recovered_truths[idx] = None

        if recovered_truths is not None and any(v is None for v in recovered_truths):
            recovered_truths = None

        return recovered_samples, recovered_labels, recovered_truths

    # Observed values used in trace plots.
    obs_values = target.obs_mags.tolist() + [target.vsini_obs, target.parallax_obs]
    obs_errors = target.obs_err_mags.tolist() + [target.vsini_obs_err, target.parallax_obs_err]

    # Normalize true values for plotting and summary output.
    raw_true_values = target.true_values
    if isinstance(raw_true_values, np.ndarray):
        true_values = raw_true_values.tolist()
    elif isinstance(raw_true_values, list):
        true_values = raw_true_values.copy()
    else:
        true_values = raw_true_values

    corner_true_values = true_values.copy() if isinstance(true_values, list) else true_values
    true_values_flag = False

    if isinstance(true_values, list):
        if any(v is None for v in true_values):
            pass
        elif all(isinstance(v, (int, float, np.integer, np.floating)) for v in true_values):
            true_values_flag = True
            if target.extinction_parameter:
                true_values.append(target.extinction)
                corner_true_values.append(target.extinction)
            if target.sigma_adjustment_parameter == 1:
                true_values.append(0)
                corner_true_values.append(0)
            elif target.sigma_adjustment_parameter == 3:
                true_values.extend([0, 0, 0])
                corner_true_values.extend([0, 0, 0])
        else:
            raise ValueError("'true_values' must contain only numbers or None.")

    # Estimate burn-in from autocorrelation times measured in the latter parts of the chain.
    chain = sampler.get_chain()
    n_steps_full = chain.shape[0]
    if n_steps_full == 0:
        raise ValueError("The sampler contains no chain steps.")

    try:
        tau_05 = emcee.autocorr.integrated_time(chain[int(n_steps_full * 0.5):], tol=0)
        tau_03 = emcee.autocorr.integrated_time(chain[int(n_steps_full * 0.7):], tol=0)
        tau05_max = float(np.nanmax(tau_05))
        tau03_max = float(np.nanmax(tau_03))
        tau_late = max(tau05_max, tau03_max)
        if not np.isfinite(tau_late) or tau_late < 0:
            raise ValueError(f"Invalid autocorrelation time: {tau_late}")

        n_discard = int(burnin_ACT * tau_late)
        autocorr_time = np.maximum(tau_05, tau_03)
        autocorr_flag = True
        print(f"📏 τ₀․₅={tau05_max:.1f}, τ₀․₃={tau03_max:.1f}, "
              f"τ_late={tau_late:.1f} → burn-in={n_discard}")
    except Exception as e:
        print(f"⚠️ Failed to compute τ adaptively ({e}). Trying get_autocorr_time().")
        try:
            tau_fallback = float(np.nanmax(sampler.get_autocorr_time()))
            if not np.isfinite(tau_fallback) or tau_fallback < 0:
                raise ValueError(f"Invalid autocorrelation time: {tau_fallback}")
            n_discard = int(burnin_ACT * tau_fallback)
            autocorr_time = None
            autocorr_flag = False
        except Exception as fallback_error:
            warnings.warn(
                f"Autocorrelation estimates failed ({fallback_error}); using no burn-in.",
                RuntimeWarning,
            )
            n_discard = 0
            autocorr_time = None
            autocorr_flag = False

    # Determine the number of samples available after burn-in.
    available_steps = sampler.iteration - n_discard
    if available_steps <= 0:
        raise ValueError(
            f"No samples remain: sampler.iteration={sampler.iteration}, n_discard={n_discard}"
        )

    total_available_samples = available_steps * target.n_walkers
    if plot_samples == "all":
        plot_samples = total_available_samples
        n_thin = 1
    else:
        if plot_samples > total_available_samples:
            warnings.warn(
                f"Requested plot_samples ({plot_samples}) exceeds available samples "
                f"after discard ({total_available_samples}); using all available samples.",
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

    # Create a unique output directory.
    results_base = os.fspath(save_path)
    os.makedirs(results_base, exist_ok=True)

    name_pre = (
        f"{target.name}{target.cluster_name}_{target.iso_name}_"
        f"{target.METHOD}_{target.regrid_type}_grid"
    )
    if target.extinction_parameter:
        name_pre += f"_{target.extinction_type}"
    if target.sigma_adjustment_parameter in (1, 3):
        name_pre += f"_sigma{target.sigma_adjustment_parameter}"

    if isinstance(raw_true_values, (list, tuple, np.ndarray)):
        if any(x is not None for x in raw_true_values) and not all(
            x is not None for x in raw_true_values
        ):
            for attr, suffix in (
                ("known_Mini", "known_Mini"),
                ("known_Age", "known_Age"),
                ("known_MH", "known_MH"),
                ("known_omega_ini", "known_omega_ini"),
                ("known_Inclination", "known_Inclination"),
                ("known_parallax", "known_parallax"),
            ):
                value = getattr(target, attr, None)
                if value is not None:
                    unit = "Myr" if attr == "known_Age" else ""
                    name_pre += f"_{suffix}_{value}{unit}"

    suffix_clean = str(folder_suffix).strip().replace(" ", "_")
    folder_name = f"{name_pre}_{suffix_clean}" if suffix_clean else name_pre
    folder_path = os.path.join(results_base, folder_name)

    new_folder_path = folder_path
    count = 2
    while os.path.exists(new_folder_path):
        new_folder_path = f"{folder_path}_{count}"
        count += 1
    os.makedirs(new_folder_path)

    print(f"Results and plots will be saved at\n\t{os.path.abspath(new_folder_path)}")
    run_name = os.path.basename(new_folder_path)

    # Extract thinned samples and blobs for plotting.
    samples = sampler.get_chain(discard=n_discard, thin=n_thin, flat=False)
    samples_flat = sampler.get_chain(discard=n_discard, thin=n_thin, flat=True)
    blobs = sampler.get_blobs(discard=n_discard, thin=n_thin, flat=False)
    blobs_array = _normalize_blobs(blobs)
    blobs_flat = blobs_array.reshape(-1, blobs_array.shape[-1])

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
        r"$i\ (\mathrm{rad})$" if v == "Inclination" and corner_inc_type == "radian"
        else r"$i\ (^\circ)$" if v == "Inclination"
        else extinction_map.get(target.extinction_type, target.extinction_type)
        if v == "extinction" else title_map.get(v, v)
        for v in sampling_variables
    ]

    corner_samples_flat = samples_flat.copy()
    trace_samples = samples.copy()
    trace_true_values = true_values.copy() if isinstance(true_values, list) else true_values

    if corner_inc_type == "degree" and "Inclination" in sampling_variables:
        inc_index = sampling_variables.index("Inclination")
        corner_samples_flat[:, inc_index] = np.rad2deg(corner_samples_flat[:, inc_index])
        trace_samples[:, :, inc_index] = np.rad2deg(trace_samples[:, :, inc_index])
        if true_values_flag:
            corner_true_values[inc_index] = np.rad2deg(corner_true_values[inc_index])
            trace_true_values[inc_index] = np.rad2deg(trace_true_values[inc_index])

    print(f"True values: {true_values}")

    output_dict = get_blob_output_dict(target)
    output_list = list(output_dict)
    output_label_list = list(output_dict.values())
    output_label_map = output_dict

    n_steps, n_walkers, n_dim = samples.shape
    if blobs_array.shape[:2] != (n_steps, n_walkers):
        raise ValueError(f"samples shape {samples.shape} and blobs shape {blobs_array.shape} do not match.")
    n_blob = blobs_array.shape[-1]
    if len(output_list) != n_blob:
        raise ValueError(
            f"output_list length ({len(output_list)}) does not match blob dimension ({n_blob}). "
            f"output_list={output_list}, blobs_array.shape={blobs_array.shape}"
        )

    # Text-summary formatting helpers.
    def _fmt_name(name, width=16):
        return f"{str(name)[:width]:<{width}}"

    def _fmt_text(value, width=14):
        return f"{('' if value is None else str(value))[:width]:<{width}}"

    def _fmt_val(value, width=10):
        if value is None:
            return f"{'':<{width}}"
        try:
            if not np.isfinite(value):
                return f"{'nan':<{width}}"
        except (TypeError, ValueError):
            return f"{str(value):<{width}}"

        value = float(value)
        a = abs(value)
        if a == 0:
            text = "0"
        elif a >= 100:
            text = f"{value:.0f}"
        elif a >= 10:
            text = f"{value:.1f}"
        elif a >= 1:
            text = f"{value:.2f}"
        elif a >= 0.1:
            text = f"{value:.3f}"
        elif a >= 0.01:
            text = f"{value:.4f}"
        elif a >= 0.001:
            text = f"{value:.5f}"
        else:
            text = f"{value:.6f}"
        return f"{text:<{width}}"

    def _prior_type_from_value(value):
        return "default" if value is None or (
            isinstance(value, str) and value.lower() == "default"
        ) else "user-specific"

    def _sampling_prior_type(raw_name):
        prior_attr_map = {
            "Mini": "Mini_prior", "Age": "Age_prior", "MH": "MH_prior",
            "omega_ini": "omega_ini_prior", "Inclination": "inclination_prior",
            "parallax": "parallax_prior", "extinction": "extinction_prior",
        }
        if raw_name in {"sigma", "sigma_gaia", "sigma_tycho2", "sigma_2mass"}:
            return "default"
        attr = prior_attr_map.get(raw_name)
        return _prior_type_from_value(getattr(target, attr, "default")) if attr else "default"

    def _vrot_prior_type(vrot_prior):
        """Classify the rotational-velocity prior as default or user-specific."""
        if vrot_prior is None or (
            isinstance(vrot_prior, str) and vrot_prior.lower() == "default"
        ):
            return "default"
        if isinstance(vrot_prior, str) or not callable(vrot_prior):
            return "user-specific"

        import inspect
        qualname = getattr(vrot_prior, "__qualname__", "")
        try:
            params = inspect.signature(vrot_prior).parameters
            if "v_max" in params:
                return "default" if (
                    "_make_log_prior_Vrot" in qualname and params["v_max"].default == 600.0
                ) else "user-specific"
        except (TypeError, ValueError):
            pass
        return "user-specific"

    def _derived_prior_type(name):
        return _vrot_prior_type(getattr(target, "Vrot_prior", None)) if name == "vtaneq" else ""

    def _summary_lines(section_title, names, values_2d, raw_names=None, lower_limits=None,
                       upper_limits=None, prior_types=None, true_values=None, name_width=16,
                       val_width=10, type_width=16, true_value_width=28, include_limits=True):
        """Create aligned percentile summary lines."""
        values_2d = np.asarray(values_2d, dtype=float)
        if values_2d.ndim == 1:
            values_2d = values_2d[:, None]
        if values_2d.shape[1] != len(names):
            raise ValueError(
                f"{section_title}: values_2d has {values_2d.shape[1]} columns, "
                f"but names has {len(names)} entries."
            )

        lower_limits = [None] * len(names) if lower_limits is None else lower_limits
        upper_limits = [None] * len(names) if upper_limits is None else upper_limits
        prior_types = ["default"] * len(names) if prior_types is None else prior_types
        true_values = [None] * len(names) if true_values is None else true_values
        if len(true_values) != len(names):
            raise ValueError(f"{section_title}: true_values must have {len(names)} entries.")

        header = (
            f"{'Parameter':<{name_width}}"
            f"{'2.5%':<{val_width}}{'16%':<{val_width}}{'50%':<{val_width}}"
            f"{'84%':<{val_width}}{'97.5%':<{val_width}}"
        )
        if include_limits:
            header += (
                f"{'Min':<{val_width}}{'Max':<{val_width}}{'Prior type':<{type_width}}"
            )
        header += f"{'True value':<{true_value_width}}"
        lines = [section_title, header]

        for i, name in enumerate(names):
            x = values_2d[:, i]
            x = x[np.isfinite(x)]
            q = np.percentile(x, [2.5, 16, 50, 84, 97.5]) if x.size else [np.nan] * 5
            line = _fmt_name(name, name_width) + "".join(_fmt_val(v, val_width) for v in q)
            if include_limits:
                line += _fmt_val(lower_limits[i], val_width) + _fmt_val(upper_limits[i], val_width)
                line += _fmt_text(prior_types[i], type_width)

            value = true_values[i]
            if value is None:
                value_text = "None"
            elif isinstance(value, str):
                value_text = value
            else:
                try:
                    value_text = _fmt_val(value, width=0).strip() if np.isfinite(value) else str(value)
                except (TypeError, ValueError):
                    value_text = str(value)
            lines.append(line + f"{value_text:<{true_value_width}}")

        return lines

    summary_raw_param_names = sampling_variables.copy()
    extinction_name_map = {"A_V": "A_V", "EB_V": "E_BV", "EBP_RP": "E_BP_RP"}
    summary_param_names = [
        "Age_Myr" if name == "Age" else extinction_name_map[target.extinction_type]
        if name == "extinction" else name
        for name in sampling_variables
    ]
    if corner_inc_type == "degree":
        summary_param_names = [
            "Incl_deg" if name == "Inclination" else name for name in summary_param_names
        ]

    true_value_param_names = ["Mini", "Age", "MH", "omega_ini", "Inclination", "parallax"]
    if raw_true_values is False or raw_true_values is None:
        true_value_map = dict.fromkeys(true_value_param_names)
    elif isinstance(raw_true_values, (list, tuple, np.ndarray)):
        raw_true_values = list(raw_true_values)
        if len(raw_true_values) != 6:
            raise ValueError(
                "target.true_values must contain exactly 6 values in the order "
                f"{true_value_param_names}; got {len(raw_true_values)}."
            )
        true_value_map = dict(zip(true_value_param_names, raw_true_values))
    else:
        raise ValueError("target.true_values must be False, None, or a sequence of 6 values.")

    if corner_inc_type == "degree" and true_value_map["Inclination"] is not None:
        true_value_map["Inclination"] = np.rad2deg(true_value_map["Inclination"])

    summary_true_values = []
    for name in summary_raw_param_names:
        if name in true_value_map:
            value = true_value_map[name]
        elif name == "extinction":
            extinction_value = getattr(target, "extinction", None)
            value = None if extinction_value is None else f"{extinction_value} (input)"
        elif name in {"sigma", "sigma_gaia", "sigma_tycho2", "sigma_2mass"}:
            value = "Not applicable"
        else:
            value = None
        summary_true_values.append(value)

    sampling_lower_limits = list(getattr(target, "sampling_lower_limit", []))
    sampling_upper_limits = list(getattr(target, "sampling_upper_limit", []))
    if len(sampling_lower_limits) != len(summary_raw_param_names):
        sampling_lower_limits = [None] * len(summary_raw_param_names)
    if len(sampling_upper_limits) != len(summary_raw_param_names):
        sampling_upper_limits = [None] * len(summary_raw_param_names)

    if corner_inc_type == "degree" and "Inclination" in summary_raw_param_names:
        inc_index = summary_raw_param_names.index("Inclination")
        for limits in (sampling_lower_limits, sampling_upper_limits):
            if limits[inc_index] is not None:
                limits[inc_index] = np.rad2deg(limits[inc_index])

    sampling_prior_types = [_sampling_prior_type(name) for name in summary_raw_param_names]
    derived_lower_limits = [None] * len(output_list)
    derived_upper_limits = [None] * len(output_list)
    derived_prior_types = [_derived_prior_type(name) for name in output_list]

    # Walker diagnostics use the unthinned post-burn-in chain.
    walker_chain_post = sampler.get_chain(discard=n_discard, thin=1, flat=False)
    walker_log_prob_post = sampler.get_log_prob(discard=n_discard, thin=1, flat=False)

    if walker_chain_post.ndim != 3:
        raise ValueError(f"Expected chain shape (n_steps, n_walkers, n_dim), got {walker_chain_post.shape}.")
    if walker_log_prob_post.ndim != 2:
        raise ValueError(
            f"Expected log-probability shape (n_steps, n_walkers), got {walker_log_prob_post.shape}."
        )

    n_steps_post, n_walkers_post, n_dim_post = walker_chain_post.shape
    if n_steps_post <= 0:
        raise ValueError("No post-burn-in steps are available for walker diagnostics.")
    if n_walkers_post != target.n_walkers:
        raise ValueError(
            f"Walker-count mismatch: chain has {n_walkers_post}, target.n_walkers={target.n_walkers}."
        )
    if n_dim_post != len(summary_raw_param_names):
        raise ValueError(
            f"Parameter-count mismatch: chain has {n_dim_post} dimensions, "
            f"but sampling_variables_list has {len(summary_raw_param_names)} entries."
        )
    if walker_log_prob_post.shape != (n_steps_post, n_walkers_post):
        raise ValueError("The post-burn-in chain and log-probability shapes do not match.")

    acceptance_fraction = np.asarray(sampler.acceptance_fraction, dtype=float)
    if acceptance_fraction.shape != (n_walkers_post,):
        raise ValueError(
            f"Unexpected acceptance_fraction shape {acceptance_fraction.shape}; "
            f"expected ({n_walkers_post},)."
        )

    finite_log_prob_values = walker_log_prob_post[np.isfinite(walker_log_prob_post)]
    global_median_log_prob = float(np.median(finite_log_prob_values)) if finite_log_prob_values.size else np.nan
    walker_diagnostics = []

    for walker_index in range(n_walkers_post):
        walker_chain = walker_chain_post[:, walker_index, :]
        walker_log_prob = walker_log_prob_post[:, walker_index]
        max_act, min_ess, slowest_parameter = np.nan, np.nan, "N/A"

        try:
            walker_tau = np.asarray(
                emcee.autocorr.integrated_time(walker_chain, tol=0, quiet=True), dtype=float
            ).reshape(-1)
            finite_tau_indices = np.flatnonzero(np.isfinite(walker_tau) & (walker_tau > 0))
            if finite_tau_indices.size:
                slowest_local_index = finite_tau_indices[np.argmax(walker_tau[finite_tau_indices])]
                max_act = float(walker_tau[slowest_local_index])
                slowest_parameter = summary_raw_param_names[slowest_local_index]
                min_ess = n_steps_post / max_act
        except Exception:
            pass

        finite_walker_log_prob = walker_log_prob[np.isfinite(walker_log_prob)]
        median_log_prob = float(np.median(finite_walker_log_prob)) if finite_walker_log_prob.size else np.nan
        delta_log_prob = (
            median_log_prob - global_median_log_prob
            if np.isfinite(median_log_prob) and np.isfinite(global_median_log_prob)
            else np.nan
        )
        walker_diagnostics.append({
            "walker": walker_index,
            "acceptance": float(acceptance_fraction[walker_index]),
            "max_ACT": max_act,
            "slowest_param": slowest_parameter,
            "min_ESS": min_ess,
            "median_logP": median_log_prob,
            "delta_logP": delta_log_prob,
        })

    summary_lines = _summary_lines(
        "Sampling parameters", summary_param_names, corner_samples_flat,
        raw_names=summary_raw_param_names, lower_limits=sampling_lower_limits,
        upper_limits=sampling_upper_limits, prior_types=sampling_prior_types,
        true_values=summary_true_values,
    )
    summary_lines += [""] + _summary_lines(
        "Derived parameters", output_list, blobs_flat,
        lower_limits=derived_lower_limits, upper_limits=derived_upper_limits,
        prior_types=derived_prior_types, true_values=[""] * len(output_list),
    )

    summary_lines += [
        "",
        "Walker diagnostics",
        f"{'walker':<10}{'acceptance':<14}{'max_ACT':<12}{'slowest_param':<18}"
        f"{'min_ESS':<12}{'median_logP':<16}{'delta_logP':<14}",
    ]
    for item in walker_diagnostics:
        max_act_text = f"{item['max_ACT']:.1f}" if np.isfinite(item["max_ACT"]) else "nan"
        min_ess_text = f"{item['min_ESS']:.1f}" if np.isfinite(item["min_ESS"]) else "nan"
        median_logp_text = f"{item['median_logP']:.3f}" if np.isfinite(item["median_logP"]) else "nan"
        delta_logp_text = f"{item['delta_logP']:.3f}" if np.isfinite(item["delta_logP"]) else "nan"
        summary_lines.append(
            f"{item['walker']:<10d}{item['acceptance']:<14.3f}{max_act_text:<12}"
            f"{item['slowest_param']:<18}{min_ess_text:<12}{median_logp_text:<16}{delta_logp_text:<14}"
        )

    summary_path = Path(new_folder_path) / "summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    # Save the target and full posterior chain.
    if save_target:
        target_save_path = Path(new_folder_path) / "target.npz"
        if not getattr(target, "load_flag", None):
            target.save(target_save_path, verbose=False)
        else:
            source_path = Path(target.load_flag)
            if not source_path.exists():
                raise FileNotFoundError(f"Loaded target file does not exist: {source_path}")
            shutil.copy2(source_path, target_save_path)

    if save_samples:
        dtype = np.float64 if save_64 else np.float32
        samples_full = sampler.get_chain().astype(dtype)
        lnprob_full = sampler.get_log_prob().astype(dtype)
        blobs_full = _normalize_blobs(sampler.get_blobs()).astype(dtype)
        acceptance_fraction = np.asarray(sampler.acceptance_fraction, dtype=dtype)
        param_names = np.array(sampling_variables)

        posterior_path = Path(new_folder_path) / "posterior.npz"
        target.posterior_path = posterior_path
        np.savez_compressed(
            posterior_path,
            true_values=target.true_values,
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

    # Recovered-parameter corner plot.
    if plot_recovered_corner:
        try:
            recovered_samples, recovered_labels, recovered_truths = _make_recovered_corner_samples(
                corner_samples_flat, corner_sampling_variables_list, sampling_variables,
                blobs_flat, output_list, output_label_map, corner_true_values, true_values_flag,
            )
            fig = corner.corner(
                recovered_samples, labels=recovered_labels, quantiles=quantiles,
                show_titles=True, hist_bin_factor=hist_bin_factor,
                title_fmt=f".{dig_num}f", truths=recovered_truths,
            )
            with PdfPages(Path(new_folder_path) / "corner_recovered_parameters.pdf") as pdf:
                pdf.savefig(fig, dpi=150)
            plt.close(fig)
        except Exception as e:
            print(f"⚠️ Recovered-parameter corner plot failed: {e}")
            plt.close("all")

    # Blob histogram summary.
    if plot_blobs_hist:
        try:
            _plot_blobs_hist(
                blobs_flat, output_list, output_label_map,
                Path(new_folder_path) / "blobs_hist.pdf",
                bins=blobs_hist_bins, ncols=blobs_hist_ncols, digits=dig_num + 1,
            )
        except Exception as e:
            print(f"⚠️ Blobs histogram plot failed: {e}")
            plt.close("all")

    # Trace plots, grouped by walker.
    n_dim_total = n_dim + n_blob
    num_groups = math.ceil(n_walkers / group_size)
    x = np.arange(n_steps)

    for g in range(num_groups):
        fig, axes = plt.subplots(n_dim_total, 1, figsize=(15, 2.0 * n_dim_total), sharex=True)
        axes = np.atleast_1d(axes)
        plt.subplots_adjust(hspace=0)
        walker_indices = range(g * group_size, min((g + 1) * group_size, n_walkers))

        for dim, ax in enumerate(axes):
            if dim < n_dim:
                for w in walker_indices:
                    ax.plot(x, trace_samples[:, w, dim], alpha=0.5, label=f"walker {w}")

                true_ylabel = ""
                if true_values_flag:
                    ax.axhline(trace_true_values[dim], color="red", linestyle="--", label="true values")
                    true_ylabel = f"\ntruth={np.round(trace_true_values[dim], 4)}"

                if autocorr_flag and autocorr_time is not None:
                    act = int(np.ceil(autocorr_time[dim]))
                    ax.set_ylabel(f"{corner_sampling_variables_list[dim]}{true_ylabel}\n(ACT={act})")
                else:
                    ax.set_ylabel(f"{corner_sampling_variables_list[dim]}{true_ylabel}")
            else:
                idx = dim - n_dim
                blob_name = output_list[idx]
                blob_label = output_label_map.get(blob_name, blob_name)

                if blob_name != "chi_squared":
                    for w in walker_indices:
                        ax.plot(x, blobs_array[:, w, idx], alpha=0.5, label=f"walker {w}")

                    if idx < len(target.bands_list) + 2:
                        ax.axhline(obs_values[idx], color="black", linestyle="--", label="observed values")
                        ax.fill_between(
                            x, obs_values[idx] - obs_errors[idx], obs_values[idx] + obs_errors[idx],
                            color="gray", alpha=0.3, label="± 1σ uncertainty",
                        )
                        error = np.round(obs_errors[idx], 4)
                        ax.set_ylabel(
                            blob_label if np.isnan(error) else f"{blob_label}\n($\\sigma$={error})"
                        )
                    else:
                        ax.set_ylabel(blob_label)
                else:
                    n_data = len(target.bands_list) + 1
                    if np.isfinite(target.vsini_obs) and np.isfinite(target.vsini_obs_err):
                        n_data += 1
                    dof = n_data - len(sampling_variables)
                    chi2_all = blobs_array[:, :, idx]
                    min_chi2 = np.nanmin(chi2_all)

                    if dof <= 0:
                        warnings.warn(
                            f"Non-positive degrees of freedom: n_data={n_data}, "
                            f"n_free={len(sampling_variables)}, dof={dof}. "
                            "Reduced chi-square will not be plotted.",
                            RuntimeWarning,
                        )
                        for w in walker_indices:
                            ax.plot(x, chi2_all[:, w], alpha=0.5, label=f"walker {w}")
                        ax.set_ylabel(r"$\chi^2$" + f"\n(min={min_chi2:.4f})")
                    else:
                        for w in walker_indices:
                            ax.plot(x, chi2_all[:, w] / dof, alpha=0.5, label=f"walker {w}")
                        ax.set_ylabel(
                            r"Reduced $\chi_\nu^2$" + f"\n(min={min_chi2/dof:.4f}, "
                            rf"$\nu$={dof})"
                        )

            ax.set_xlim([x.min(), x.max()])
            ax.tick_params(axis="x", which="both", labelbottom=(dim == n_dim_total - 1))

        axes[-1].set_xlabel("samples" if n_thin == 1 else f"samples (thinning={n_thin})")
        all_handles, all_labels = [], []
        for ax in axes:
            handles, labels = ax.get_legend_handles_labels()
            all_handles.extend(handles)
            all_labels.extend(labels)
        unique = dict(zip(all_labels, all_handles))
        if unique:
            axes[0].legend(unique.values(), unique.keys(), loc="upper right")

        trace_path = Path(new_folder_path) / f"trace_group{g}_burnin{n_discard}_thin{n_thin}.pdf"
        fig.savefig(trace_path, bbox_inches="tight")
        plt.close(fig)

    # Original corner plot.
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
            truths=corner_true_values if true_values_flag else None,
        )
        with PdfPages(Path(new_folder_path) / "corner_sampling_parameters.pdf") as pdf:
            pdf.savefig(fig, dpi=150)

        print("Finished saving plots.")
        if show_fig and running_in_jupyter():
            plt.show()
        plt.close(fig)
    except Exception as e:
        print(f"⚠️ Original corner plot failed: {e}")
        plt.close("all")

    return target