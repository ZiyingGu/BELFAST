import os
import json
import emcee
import inspect
import textwrap
import numpy as np
import pandas as pd
from glob import glob
from pathlib import Path
from scipy.stats import norm

from belfast.paths import DATA_DIR, KLD_DIR, PARS_FILTER_DIR
from belfast.PARS_utils import extract_MH, make_limb_darkening_files
from belfast.target_utils import (
    cal_abs_mag, remove_extinction, classify_posterior_samples_safety,
)
from belfast.mcmc_utils import (
    select_Vrot_prior_by_rough_mass, log_prior_Vrot_dict,
    _log_prior_mixture_maxwell_lagged_truncated,
)
from belfast.isochrone_utils import (
    load_default_regularized_PARSECv2_Mini_interpolator,
    load_default_regularized_PARSECv2_u_interpolator,
    load_default_regularized_MIST_Mini_interpolator,
)


class Target:
    """Store target observations and configuration for MCMC analysis."""

    def __init__(self, name, bands_list, obs_mags, obs_err_mags, vsini_obs, vsini_obs_err,
                 parallax_obs, parallax_obs_err, sigma_adjustment_parameter=False,
                 extinction_parameter=False, true_values=False):
        """
        Initialize a target and validate its configuration.

        Args:
            name (str): Target name or identifier.
            bands_list (list): Photometric band names in observation order.
            obs_mags (list or np.ndarray): Observed magnitudes, ordered as bands_list.
            obs_err_mags (list or np.ndarray): Magnitude uncertainties in the same order.
            vsini_obs (float): Observed projected rotational velocity.
            vsini_obs_err (float): Uncertainty in the projected rotational velocity.
            parallax_obs (float): Observed parallax.
            parallax_obs_err (float): Uncertainty in the observed parallax.
            sigma_adjustment_parameter (bool or int): Number of uncertainty inflation parameters.
                Allowed values are False, 0, 1, and 3.
                False or 0: Disable uncertainty inflation.
                1: Use one uncertainty inflation parameter for all eight photometric bands.
                3: Use three uncertainty inflation parameters, one for each of Gaia DR3,Tycho-2, and 2MASS.
            extinction_parameter (bool): Whether to sample extinction as a parameter.
            true_values (list or bool): True parameter values for mock data, or False if unavailable.

        Raises:
            ValueError: If the configuration or photometric input dimensions are invalid.
        """
        self.current_path = str(Path.cwd())

        if isinstance(sigma_adjustment_parameter, (bool, np.bool_)):
            if sigma_adjustment_parameter is True:
                raise ValueError("sigma_adjustment_parameter=True is not supported; use 1 explicitly.")
        elif sigma_adjustment_parameter not in (False, 0, 1, 3):
            raise ValueError(
                f"sigma_adjustment_parameter must be False, 0, 1, or 3, "
                f"but got {sigma_adjustment_parameter!r}."
            )

        if not isinstance(extinction_parameter, (bool, np.bool_)):
            raise ValueError("extinction_parameter must be a boolean.")

        if sigma_adjustment_parameter == 3 and extinction_parameter:
            raise ValueError(
                'Invalid combination: sigma_adjustment_parameter=3 cannot be used '
                'with extinction_parameter=True because the degrees of freedom would be zero.'
            )

        self.load_flag = False  # True if loaded from files; False if defined directly.
        self.name = name.strip().replace(" ", "_")
        self.bands_list = list(bands_list)
        self.obs_mags = np.asarray(obs_mags, dtype=float)
        self.obs_err_mags = np.asarray(obs_err_mags, dtype=float)

        if (self.obs_mags.ndim != 1 or self.obs_err_mags.ndim != 1
                or len(self.bands_list) != len(self.obs_mags)
                or len(self.obs_mags) != len(self.obs_err_mags)):
            raise ValueError(
                "bands_list, obs_mags, and obs_err_mags must be one-dimensional "
                "and have the same length."
            )

        self.vsini_obs = vsini_obs
        self.vsini_obs_err = vsini_obs_err
        self.parallax_obs = parallax_obs
        self.parallax_obs_err = parallax_obs_err
        self.sigma_adjustment_parameter = sigma_adjustment_parameter
        self.extinction_parameter = extinction_parameter
        self.true_values = true_values


    def add_info(self, cluster='', extinction=0, extinction_err=0, extinction_type=None,
                 interp_variables_list=None, sampling_variables_list=None, verbose=False,
                 max_sigma_adjustment_parameter=None):
        """
        Add target metadata, configure variables, and calculate corrected magnitudes.

        Args:
            cluster (str): Cluster name, if applicable.
            extinction (float): Extinction value.
            extinction_err (float): Extinction uncertainty.
            extinction_type (str): Extinction type: 'A_V', 'EB_V', or 'EBP_RP'.
            interp_variables_list (list, optional): Variables used for interpolation.
            sampling_variables_list (list, optional): Variables sampled during inference.
            verbose (bool): Whether to print status messages.
            max_sigma_adjustment_parameter (float, optional): Maximum uncertainty adjustment.

        """
        self.cluster = cluster.strip().replace(' ', '_')

        if self.extinction_parameter and extinction_type not in ('A_V', 'EB_V', 'EBP_RP'):
            raise ValueError("extinction_type must be 'A_V', 'EB_V', or 'EBP_RP' when extinction is sampled.")

        self.extinction = extinction
        self.extinction_err = extinction_err
        self.extinction_type = extinction_type
        self.max_sigma_adjustment_parameter = max_sigma_adjustment_parameter

        if interp_variables_list is None:
            interp_variables_list = ['Mini', 'Age', 'MH', 'omega_ini']
        else:
            interp_variables_list = list(interp_variables_list)
        if sampling_variables_list is None:
            sampling_variables_list = ['Mini', 'Age', 'MH', 'omega_ini', 'Inclination', 'parallax']
        else:
            sampling_variables_list = list(sampling_variables_list)

        if self.extinction_parameter:
            sampling_variables_list.append('extinction')

        if extinction == 0:
            print('Assuming no extinction.')

        sigma_variables = {
            0: [],
            1: ['sigma'],
            3: ['sigma_gaia', 'sigma_tycho2', 'sigma_2mass'],
        }
        sampling_variables_list.extend(sigma_variables[self.sigma_adjustment_parameter])

        self.interp_variables_list = interp_variables_list
        self.sampling_variables_list = sampling_variables_list
        self.obs_mags_corrected, self.obs_err_mags_corrected = remove_extinction(
            self.obs_mags, self.obs_err_mags, self.bands_list, extinction, extinction_err,
            extinction_type, verbose=verbose
        )
        self.abs_mags, self.abs_err_mags = cal_abs_mag(
            self.obs_mags_corrected, self.obs_err_mags_corrected, self.parallax_obs,
            self.parallax_obs_err, max_sigma_adjustment_parameter=max_sigma_adjustment_parameter
        )

        self.known_Mini = self.known_Age = self.known_MH = self.known_omega_ini = None
        self.known_Inclination = self.known_parallax = None
        self.true_values = np.array([None] * 6, dtype=object)


    def known_parameter(self, true_values=None, known_Mini=None, known_Age=None, known_MH=None,
                        known_omega_ini=None, known_Inclination=None, known_parallax=None):
        """
        Specify known or true parameter values for the target.

        Args:
            true_values (array-like, optional): Six values ordered as [Mini, Age, MH, omega_ini,
                Inclination, parallax].
            known_Mini (float, optional): Known initial stellar mass.
            known_Age (float, optional): Known log(age).
            known_MH (float, optional): Known metallicity.
            known_omega_ini (float, optional): Known initial rotation parameter.
            known_Inclination (float, optional): Known inclination.
            known_parallax (float, optional): Known parallax.

        Raises:
            ValueError: If true_values does not contain exactly six values.
        """
        if true_values is not None:
            true_values = np.asarray(true_values, dtype=object)
            if true_values.shape != (6,):
                raise ValueError(
                    "`true_values` must have length 6 in the order "
                    "[Mini, Age, MH, omega_ini, Inclination, parallax]."
                )
        else:
            true_values = np.array(
                [known_Mini, known_Age, known_MH, known_omega_ini, known_Inclination, known_parallax],
                dtype=object
            )

        (self.known_Mini, self.known_Age, self.known_MH, self.known_omega_ini,
         self.known_Inclination, self.known_parallax) = true_values
        self.true_values = true_values



    def set_method(self,
                   grids_path=None, regrid_type='Mini', METHOD='DIRECT', iso_name='PARSEC',
                   label_list=None, flag_64=False,
                   # MIST
                   correct_MIST_MH_zeropoint=True, MIST_MH_offset=0.018,
                   # PARS
                   show_PARS_progress=True, PARS_limb_darkening_path=KLD_DIR / 'i[pm]*k*.pck*',
                   PARS_filters=PARS_FILTER_DIR / '*.dat', PARS_nz=100, PARS_MH_offset=-0.046
                   ):
        """
        Configure the isochrone model, interpolation method, and related settings.

        Args:
            grids_path (str or Path, optional): Path to the precomputed interpolation grid file.
            regrid_type (str): Grid parameterization. Must be 'Mini' or 'u'.
            METHOD (str): Interpolation method. Must be 'DIRECT' or 'PARS'.
            iso_name (str): Isochrone model. Must be 'PARSEC' or 'MIST'.
            label_list (list, optional): PARSEC grid labels to load. Defaults to [1].
            flag_64 (bool): Whether to force float64 precision when loading the grid.
            correct_MIST_MH_zeropoint (bool): Whether to correct the MIST metallicity zeropoint.
            MIST_MH_offset (float): Metallicity offset applied to the MIST model.
            show_PARS_progress (bool): Whether to display progress during PARS setup.
            PARS_limb_darkening_path (str or Path): Pattern for locating limb-darkening files.
            PARS_filters (str or Path): Pattern for locating photometric filter files.
            PARS_nz (int): Number of points used for the PARS z-coordinate grid.
            PARS_MH_offset (float): Metallicity offset applied when selecting limb-darkening models.

        Raises:
            ValueError: If the model, grid type, method, or required PARS settings are invalid,
                or if the loaded grid is incompatible with the selected method.

        Returns:
            None.
        """

        iso_name, METHOD = iso_name.upper(), METHOD.upper()

        try:
            regrid_type = {'mini': 'Mini', 'u': 'u'}[regrid_type.lower()]
        except (AttributeError, KeyError):
            raise ValueError('regrid_type must be "Mini" or "u".') from None

        if iso_name not in {'PARSEC', 'MIST'}:
            raise ValueError('iso_name must be "PARSEC" or "MIST".')
        if METHOD == "NN":
            raise NotImplementedError(
                "The NN method is not currently supported. "
                "It will be implemented in a future release."
            )
        if METHOD not in {'DIRECT', 'PARS'}:
            raise ValueError('METHOD must be "DIRECT" or "PARS".')
        if iso_name == 'MIST' and (regrid_type != 'Mini' or METHOD != 'PARS'):
            raise ValueError('MIST only supports regrid_type="Mini" with METHOD="PARS".')

        label_list = [1] if label_list is None else label_list
        grid_type = 'with_inc' if METHOD == 'DIRECT' else 'without_inc'
        PARS_limb_darkening_path = str(PARS_limb_darkening_path) if isinstance(
            PARS_limb_darkening_path, os.PathLike) else PARS_limb_darkening_path
        PARS_filters = str(PARS_filters) if isinstance(PARS_filters, os.PathLike) else PARS_filters

        self.cluster_name = f'_{self.cluster}' if self.cluster else ''
        self.iso_name, self.regrid_type, self.METHOD = iso_name, regrid_type, METHOD
        self.PARS_limb_darkening_path, self.PARS_filters = PARS_limb_darkening_path, PARS_filters
        self.correct_MIST_MH_zeropoint, self.MIST_MH_offset, self.flag_64 = (
            correct_MIST_MH_zeropoint, MIST_MH_offset, flag_64
        )

        print('*' * 100)
        print(f'Using {iso_name} isochrones with {regrid_type} grids and {METHOD} method.')
        print('*' * 100)

        if iso_name == 'PARSEC':
            loader = (load_default_regularized_PARSECv2_Mini_interpolator if regrid_type == 'Mini'
                      else load_default_regularized_PARSECv2_u_interpolator)
            (self.Model_interp, self.interp_u, self.interp_u_to_outputs, self.interp_u_bounds,
             self.interp_variables_list, self.interp_output_list, self.data) = loader(
                npz_path=grids_path, label_list=label_list, grid_type=grid_type, force_float64=flag_64
            )

            actual_grid_type = str(self.data['grid_type'])
            if actual_grid_type != grid_type:
                raise ValueError(f'{METHOD} requires a "{grid_type}" grid, '
                                 f'but the loaded file is "{actual_grid_type}".')
            self.label_list = (self.data['label_list'].tolist()
                               if 'label_list' in self.data.files else list(label_list))
        else:
            (self.Model_interp, self.interp_variables_list, self.interp_output_list,
             self.data) = load_default_regularized_MIST_Mini_interpolator(
                npz_path=grids_path, force_float64=flag_64
            )
            self.interp_u = self.interp_u_to_outputs = self.interp_u_bounds = self.label_list = None

        loaded_path = getattr(getattr(self.data, 'zip', None), 'filename', None)
        self.grids_path = str(loaded_path or grids_path) if (loaded_path or grids_path) else None

        lower_limits, upper_limits = [], []
        for var in self.interp_variables_list:
            if regrid_type == 'u' and var == 'Mini':
                bounds = self.data['grid_bounds']
                lower_limits.append(np.nanmin(bounds[..., 0]))
                upper_limits.append(np.nanmax(bounds[..., 1]))
            else:
                grid = self.data[f'{var}_grid']
                lower_limits.append(np.nanmin(grid))
                upper_limits.append(np.nanmax(grid))

        self.interp_lower_limit = np.asarray(lower_limits)
        self.interp_upper_limit = np.asarray(upper_limits)

        if METHOD == 'PARS':
            if PARS_limb_darkening_path is None:
                raise ValueError('PARS_limb_darkening_path must be defined.')
            if PARS_filters is None:
                raise ValueError('PARS_filters must be defined.')

            limb_darkening_MH_dic = {
                file: mh for file in sorted(glob(PARS_limb_darkening_path))
                if (mh := extract_MH(file)) is not None
            }
            print("Loading limb darkening files from Kurucz's model.")

            MH_index = self.interp_variables_list.index('MH')
            MH_lower = self.interp_lower_limit[MH_index] + PARS_MH_offset
            MH_upper = self.interp_upper_limit[MH_index] + PARS_MH_offset
            self.made_limb_darkening_dict = make_limb_darkening_files(
                limb_darkening_MH_dic, PARS_filters, MH_lower, MH_upper, show_progress=show_PARS_progress
            )
            self.PARS_nz, self.PARS_MH_offset = PARS_nz, PARS_MH_offset

        self.fixed_parameter_indices = [
            i for i, var in enumerate(self.interp_variables_list)
            if var not in self.sampling_variables_list
        ]




    def set_sampling(self, n_walkers=30, age_limit=None, parallax_limit=None, sigma_limit=None,
                     extinction_limit=None, emcee_moves=None, Vrot_prior=None, Mini_prior='default',
                     Age_prior='default', MH_prior='default', omega_ini_prior='default',
                     inclination_prior='default', parallax_prior='default', extinction_prior='default'):
        """
        Configure sampling priors, parameter bounds, and initial walker positions.

        Args:
            n_walkers (int): Number of MCMC walkers.
            age_limit (list, optional): Age bounds [min, max].
            parallax_limit (list, optional): Parallax bounds [min, max].
            sigma_limit (list, optional): Uncertainty inflation bounds [min, max], or three
                bounds pairs for Gaia DR3, Tycho-2, and 2MASS when using three sigma parameters.
            extinction_limit (list, optional): Extinction bounds [min, max].
            emcee_moves (list, optional): emcee move objects and their weights.
            Vrot_prior (callable, optional): Custom projected rotational velocity prior.
            Mini_prior (str or callable): Initial mass prior, 'default' or a custom function.
            Age_prior (str or callable): Age prior, 'default' or a custom function.
            MH_prior (str or callable): Metallicity prior, 'default' or a custom function.
            omega_ini_prior (str or callable): Initial rotation prior, 'default' or a custom function.
            inclination_prior (str or callable): Inclination prior, 'default' or a custom function.
            parallax_prior (str or callable): Parallax prior, 'default' or a custom function.
            extinction_prior (str or callable): Extinction prior, 'default' or a custom function.

        Raises:
            ValueError: If any prior is neither 'default' nor a callable.
        """
        parallax_limit = [self.parallax_obs - 5, self.parallax_obs + 5] if parallax_limit is None else parallax_limit
        sigma_limit = [1e-8, 0.1] if sigma_limit is None else sigma_limit
        extinction_limit = [1e-8, 1] if extinction_limit is None else extinction_limit

        self.emcee_moves = emcee_moves if emcee_moves is not None else [
            (emcee.moves.StretchMove(a=2.0), 0.50),
            (emcee.moves.DEMove(gamma0=0.40), 0.20),
            (emcee.moves.DESnookerMove(), 0.15),
            (emcee.moves.KDEMove(), 0.15),
        ]
        self.age_limit, self.parallax_limit = age_limit, parallax_limit
        self.sigma_limit, self.extinction_limit = sigma_limit, extinction_limit

        # if using non-rotating model
        if any(x is not None for x in self.true_values) and not all(x is not None for x in self.true_values):
            if (self.known_omega_ini is not None and self.known_Inclination is not None
                    and np.isclose(self.known_omega_ini * np.sin(self.known_Inclination), 0)):
                Vrot_prior = lambda x: 0

        priors = {
            'Mini_prior': Mini_prior, 'Age_prior': Age_prior, 'MH_prior': MH_prior,
            'omega_ini_prior': omega_ini_prior, 'inclination_prior': inclination_prior,
            'parallax_prior': parallax_prior, 'extinction_prior': extinction_prior,
            'Vrot_prior': Vrot_prior,
        }
        invalid_keys = [
            name for name, prior in priors.items()
            if (prior is not None if name == 'Vrot_prior' else prior != 'default') and not callable(prior)
        ]
        if invalid_keys:
            raise ValueError(
                f"❌ The following priors are invalid: {invalid_keys}\n\n"
                "✅ Each prior must be either:\n"
                "    - The string 'default', or\n"
                "    - A callable custom function."
            )

        for name, prior in priors.items():
            setattr(self, name, prior)
        self.rough_mass = None

        sampling_vars = self.sampling_variables_list
        sampling_indices = {var: i for i, var in enumerate(sampling_vars)}
        interp_indices = {var: i for i, var in enumerate(self.interp_variables_list)}
        lower = np.zeros(len(sampling_vars))
        upper = np.zeros(len(sampling_vars))

        for var, i in sampling_indices.items():
            if var == 'Inclination':
                lower[i], upper[i] = 0.0, np.deg2rad(90.0)
            elif var == 'parallax':
                lower[i], upper[i] = parallax_limit
            elif var in interp_indices:
                j = interp_indices[var]
                lower[i], upper[i] = self.interp_lower_limit[j], self.interp_upper_limit[j]

        if self.sigma_adjustment_parameter == 1:
            i = sampling_indices['sigma']
            lower[i], upper[i] = sigma_limit
        elif self.sigma_adjustment_parameter == 3:
            if len(sigma_limit) == 2 and all(np.isscalar(x) for x in sigma_limit):
                sigma_limit = [sigma_limit] * 3
            for var, limits in zip(('sigma_gaia', 'sigma_tycho2', 'sigma_2mass'), sigma_limit):
                i = sampling_indices[var]
                lower[i], upper[i] = limits

        if self.extinction_parameter:
            i = sampling_indices['extinction']
            lower[i], upper[i] = extinction_limit
        if age_limit is not None:
            i = sampling_indices['Age']
            lower[i], upper[i] = age_limit

        self.sampling_lower_limit, self.sampling_upper_limit = lower, upper
        self.n_walkers, self.ndim = n_walkers, len(sampling_vars)
        self.initial_states = np.random.rand(n_walkers, self.ndim) * (upper - lower) + lower

        print('Sampling parameter space:')
        print(f"{'Parameter':<12} {'Min':>10} {'Max':>10}")
        print("-" * 34)
        for name, low, high in zip(sampling_vars, lower, upper):
            low_str = f"{low:.4f}" if np.isfinite(low) else "nan"
            high_str = f"{high:.4f}" if np.isfinite(high) else "nan"
            print(f"{name:<12} {low_str:>10} {high_str:>10}")


    def check_risk(self, critical_boundary_path=None, return_flag=False,
                   ignore_out_of_range_mini=False, show_progress=True):
        """
        Classify posterior samples and optionally update the summary file.

        Args:
            critical_boundary_path (str or Path, optional): Path to the critical boundary CSV file.
                Defaults to DATA_DIR / 'critical_boundary.csv'.
            return_flag (bool): Whether to return the classification results.
            ignore_out_of_range_mini (bool): Whether to ignore samples with Mini outside the supported
                mass range.
            show_progress (bool): Whether to display the classification progress bar.

        Returns:
            dict or None: Classification results if return_flag is True; otherwise, None.

        Notes:
            If summary.txt exists alongside the posterior file, update its SGB contamination
            probability. Failures during the summary update are reported as warnings and do not
            affect the classification results.
        """
        if critical_boundary_path is None:
            critical_boundary_path = DATA_DIR / "critical_boundary.csv"

        path = self.posterior_path
        result = classify_posterior_samples_safety(
            np.load(path, allow_pickle=True),
            pd.read_csv(critical_boundary_path),
            batch_size=5000,
            ignore_out_of_range_mini=ignore_out_of_range_mini,
            show_progress=show_progress,
        )

        try:
            summary_path = path.parent / "summary.txt"
            if summary_path.is_file():
                marker = "Probability of contamination by SGB stars:"
                original_text = summary_path.read_text(encoding="utf-8")
                if original_text.startswith(marker):
                    original_text = original_text.split("\n\n", 1)[-1]

                risk_text = (
                    f"{marker}\n"
                    f"P_risk = {np.clip(result['summary']['P_risky'], 0.0, 1.0):.3f}\n\n"
                )
                summary_path.write_text(
                    risk_text + original_text.lstrip("\n"),
                    encoding="utf-8",
                )
        except Exception as e:
            print(f"Warning: Failed to update summary.txt: {e}")

        if return_flag:
            return result



    def save(self, filename, verbose=True):
        """
        Save the Target configuration and numerical arrays to an NPZ file.

        Runtime-only objects are excluded. Custom prior functions are saved as source strings,
        and emcee move objects are converted to a JSON-compatible configuration.

        Args:
            filename (str or Path): Output NPZ file path.
            verbose (bool): Whether to print the saved file path.

        Raises:
            TypeError: If an attribute or prior has an unsupported type.
            ValueError: If the emcee move configuration is invalid.
            RuntimeError: If the source code of a custom prior cannot be retrieved.
        """

        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)

        skip_keys = {
            'Model_interp', 'interp_u', 'interp_u_to_outputs', 'interp_u_bounds',
            'data', 'made_limb_darkening_dict', 'emcee_moves', 'emcee_move_config'
        }
        prior_keys = {
            'Vrot_prior', 'Mini_prior', 'Age_prior', 'MH_prior', 'omega_ini_prior',
            'inclination_prior', 'parallax_prior', 'extinction_prior'
        }
        arrays_to_save, config = {}, {}

        def to_jsonable(value):
            """Convert supported Python and NumPy values to JSON-compatible objects."""
            if value is None or isinstance(value, (str, int, float, bool)):
                return value
            if isinstance(value, (np.integer, np.floating, np.bool_)):
                return value.item()
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, (list, tuple)):
                return [to_jsonable(item) for item in value]
            if isinstance(value, dict):
                return {str(key): to_jsonable(item) for key, item in value.items()}
            raise TypeError(f'Cannot convert object of type {type(value)} to JSON.')

        def serialize_prior(name, prior):
            """Serialize a prior as None, a string, or a custom function's source code."""
            if prior is None or isinstance(prior, str):
                return prior
            if not callable(prior):
                raise TypeError(f'Prior {name!r} must be None, a string, or callable.')

            try:
                source = inspect.getsource(prior)
            except (OSError, TypeError) as exc:
                raise RuntimeError(
                    f'Could not retrieve source for prior {name!r}. '
                    'Define it as a named function or lambda in a source file.'
                ) from exc

            return {'type': 'custom_source', 'name': getattr(prior, '__name__', name),
                    'source': source}

        def serialize_emcee_moves(moves):
            """Serialize supported emcee moves and their weights."""
            if moves is None:
                return None
            if not isinstance(moves, (list, tuple)) or not moves:
                raise ValueError('emcee_moves must be None or a non-empty list/tuple.')

            result, total = {}, 0.0
            for i, item in enumerate(moves):
                if not isinstance(item, (list, tuple)) or len(item) != 2:
                    raise ValueError(f'Invalid emcee_moves entry at index {i}: {item!r}')

                move, fraction = item
                if not isinstance(move, emcee.moves.Move):
                    raise TypeError(f'Entry {i} must contain an emcee Move object.')
                if not isinstance(fraction, (int, float, np.integer, np.floating)):
                    raise TypeError(f'Move fraction at index {i} must be numeric.')

                fraction = float(fraction)
                if not np.isfinite(fraction) or fraction <= 0:
                    raise ValueError(f'Move fraction at index {i} must be finite and positive.')

                total += fraction
                if isinstance(move, emcee.moves.StretchMove):
                    name = 'StretchMove'
                    a = getattr(move, 'a', getattr(move, '_a', None))
                    if a is None:
                        raise AttributeError('Could not read parameter a from StretchMove.')
                    params = {'a': float(a)}
                elif isinstance(move, emcee.moves.DEMove):
                    name = 'DEMove'
                    gamma0 = getattr(move, 'gamma0', getattr(move, '_gamma0', None))
                    params = {'gamma0': None if gamma0 is None else float(gamma0)}
                elif isinstance(move, emcee.moves.DESnookerMove):
                    name, params = 'DESnookerMove', {}
                elif isinstance(move, emcee.moves.KDEMove):
                    name, params = 'KDEMove', {}
                else:
                    raise ValueError(f'Unsupported emcee move type: {type(move).__name__}')

                if name in result:
                    raise ValueError(f'Duplicate emcee move type: {name}.')
                result[name] = {**params, 'fraction': fraction}

            if not np.isclose(total, 1.0, rtol=0.0, atol=1e-8):
                raise ValueError(f'Move fractions must sum to 1; got {total:.12g}.')
            return result

        config['emcee_move_config'] = serialize_emcee_moves(
            getattr(self, 'emcee_moves', None)
        )

        for key, value in self.__dict__.items():
            if key in skip_keys:
                continue
            if key in prior_keys:
                config[key] = serialize_prior(key, value)
            elif isinstance(value, np.ndarray) and value.dtype != object:
                arrays_to_save[key] = value
            elif isinstance(value, (np.ndarray, str, int, float, bool, type(None),
                                    list, tuple, dict, Path, np.integer, np.floating, np.bool_)):
                config[key] = to_jsonable(value)
            else:
                raise TypeError(
                    f'Attribute {key!r} has unsupported type {type(value)}. '
                    'Add it to skip_keys or define how to serialize it.'
                )

        arrays_to_save['config_json'] = np.array(json.dumps(config, indent=2))
        np.savez(filename, **arrays_to_save)

        if verbose:
            print(f'Saved object to: {os.path.abspath(filename)}')

    @classmethod
    def load(cls, filename, grids_path=None):
        """Load a Target object from an NPZ file and rebuild its runtime objects.

        Args:
            filename (str or Path): Path to the saved Target NPZ file.
            grids_path (str or Path, optional): Fallback grid path for older files that
                do not contain ``grids_path`` in their saved configuration.

        Returns:
            Target: Restored Target object with interpolation objects, priors, and emcee moves.

        Raises:
            KeyError: If required configuration entries are missing.
            FileNotFoundError: If the saved working directory does not exist.
            TypeError: If saved prior or emcee move configurations have invalid types.
            ValueError: If emcee move fractions or move types are invalid.
            RuntimeError: If a custom prior cannot be reconstructed.
        """

        filename = Path(filename)
        data = np.load(filename, allow_pickle=False)
        if "config_json" not in data.files:
            raise KeyError("`config_json` was not found in the saved NPZ file.")
        config = json.loads(str(data["config_json"]))
        obj = cls.__new__(cls)

        prior_keys = {
            "Vrot_prior", "Mini_prior", "Age_prior", "MH_prior", "omega_ini_prior",
            "inclination_prior", "parallax_prior", "extinction_prior"
        }
        object_array_keys = {"true_values"}

        def clean_source(source):
            """Clean source code retrieved by inspect.getsource()."""
            lines = textwrap.dedent(source).strip().splitlines()
            if len(lines) == 1 and "lambda" in lines[0] and lines[0].rstrip().endswith(","):
                lines[0] = lines[0].rstrip().rstrip(",")
            return "\n".join(lines)

        def restore_prior(name, value):
            """Restore a prior from its saved configuration."""
            if value is None or isinstance(value, str):
                return value
            if not isinstance(value, dict) or value.get("type") != "custom_source":
                raise TypeError(f"Cannot restore prior `{name}` from type {type(value)}.")

            source = clean_source(value["source"])
            func_name = value.get("name", name)

            if name == "Vrot_prior":
                rough_mass = getattr(obj, "rough_mass", getattr(obj, "rough_mini", None))
                if rough_mass is None:
                    return None
                return select_Vrot_prior_by_rough_mass(rough_mass, log_prior_Vrot_dict)[1]

            namespace = {
                "np": np, "numpy": np, "norm": norm,
                "extinction": getattr(obj, "extinction", None),
                "extinction_err": getattr(obj, "extinction_err", None),
                "true_values": getattr(obj, "true_values", None),
                "_log_prior_mixture_maxwell_lagged_truncated":
                    _log_prior_mixture_maxwell_lagged_truncated,
            }
            exec(source, namespace)

            function = namespace.get(func_name, namespace.get(name))
            if isinstance(function, tuple) and len(function) == 1 and callable(function[0]):
                function = function[0]
            if not callable(function):
                raise RuntimeError(f"Could not restore callable prior `{name}`.")
            return function

        def restore_saved_attributes():
            """Restore saved configuration attributes, arrays, and priors."""
            for key, value in config.items():
                if key in prior_keys:
                    continue
                setattr(obj, key, np.array(value, dtype=object) if key in object_array_keys else value)

            for key in data.files:
                if key != "config_json":
                    setattr(obj, key, data[key])

            for key in prior_keys:
                if key in config:
                    setattr(obj, key, restore_prior(key, config[key]))

        def build_emcee_moves(move_config):
            """Rebuild emcee moves from their saved configuration."""
            if move_config is None:
                return None
            if not isinstance(move_config, dict):
                raise TypeError("emcee_move_config must be a dictionary or None.")

            moves, total_fraction = [], 0.0
            for name, settings in move_config.items():
                if not isinstance(settings, dict):
                    raise TypeError(f"Configuration for `{name}` must be a dictionary.")
                if "fraction" not in settings:
                    raise KeyError(f"`fraction` is missing for move `{name}`.")

                fraction = float(settings["fraction"])
                if not np.isfinite(fraction) or fraction <= 0:
                    raise ValueError(f"Invalid fraction for `{name}`: {fraction}")
                total_fraction += fraction

                if name == "StretchMove":
                    if "a" not in settings:
                        raise KeyError("`a` is missing for StretchMove.")
                    move = emcee.moves.StretchMove(a=float(settings["a"]))
                elif name == "DEMove":
                    gamma0 = settings.get("gamma0")
                    move = emcee.moves.DEMove(gamma0=None if gamma0 is None else float(gamma0))
                elif name == "DESnookerMove":
                    move = emcee.moves.DESnookerMove()
                elif name == "KDEMove":
                    move = emcee.moves.KDEMove()
                else:
                    raise ValueError(f"Unknown emcee move type: {name}")
                moves.append((move, fraction))

            if not np.isclose(total_fraction, 1.0, rtol=0.0, atol=1e-8):
                raise ValueError(f"Saved move fractions must sum to 1. Got {total_fraction:.12g}.")
            return moves

        # Restore the saved working directory before rebuilding runtime objects.
        saved_current_path = config.get("current_path")
        if saved_current_path is not None:
            obj.current_path = str(saved_current_path)
            if not os.path.isdir(obj.current_path):
                raise FileNotFoundError(f"The saved current_path does not exist: {obj.current_path}")
            os.chdir(obj.current_path)

        for key in ("cluster", "sampling_variables_list"):
            if key in config:
                setattr(obj, key, config[key])

        saved_grids_path = config.get("grids_path") or grids_path
        if saved_grids_path is None:
            raise KeyError(
                "`grids_path` was not found in the saved target configuration. "
                "Provide it explicitly: Target.load(filename, grids_path='path/to/grid.npz')"
            )

        for key in ("regrid_type", "METHOD"):
            if key not in config:
                raise KeyError(f"`{key}` is required to rebuild runtime objects with set_method().")

        set_method_kwargs = {
            "grids_path": saved_grids_path,
            "regrid_type": config["regrid_type"],
            "METHOD": config["METHOD"],
            "iso_name": config.get("iso_name", "PARSEC"),
            "flag_64": config.get("flag_64", False),
            "PARS_limb_darkening_path": config.get("PARS_limb_darkening_path"),
            "PARS_filters": config.get("PARS_filters"),
            "PARS_nz": config.get("PARS_nz", 100),
            "PARS_MH_offset": config.get("PARS_MH_offset", -0.046),
        }
        obj.set_method(**set_method_kwargs)

        restore_saved_attributes()
        obj.grids_path = str(saved_grids_path)
        obj.emcee_move_config = config.get("emcee_move_config")
        obj.emcee_moves = build_emcee_moves(obj.emcee_move_config)
        obj.load_flag = str(filename)
        return obj