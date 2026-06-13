# import necessary packages
import os
# import pandas as pd
# import warnings
import numpy as np
# from tqdm import tqdm
# import re
from glob import glob
from pathlib import Path
import json
import inspect
# from itertools import product
# from belfast.isochrone_utils import read_isochrones, PARSECv2_collect_data, MIST_collect_data, make_model
from belfast.isochrone_utils import PARSECv2_collect_data, load_regularized_Mini_interpolator, load_regularized_u_interpolator, load_regularized_MIST_interpolator
from belfast.target_utils import cal_abs_mag, cal_obs_mag, remove_extinction
from belfast.PARS_utils import extract_MH, make_limb_darkening_files


class Target:
    def __init__(self, name, bands_list, obs_mags, obs_err_mags,
                 vsini_obs, vsini_obs_err, parallax_obs, parallax_obs_err,
                 sigma_adjustment_parameter=False, extinction_parameter=False,
                 true_values=False):
        """
        Initialize a Target object with observed and required parameters.

        Args:
            name (str): Name or ID of the target.
            bands_list (list): List of photometric bands (e.g., ["G", "BP", "RP"]).
            obs_mags (list or np.ndarray): Observed magnitudes in bands_list order.
            obs_err_mags (list or np.ndarray): Uncertainties of observed magnitudes.
            vsini_obs (float): Observed projected rotational velocity (v*sin(i)).
            vsini_obs_err (float): Uncertainty in vsini.
            parallax_obs (float): Observed parallax.
            parallax_obs_err (float): Uncertainty of observed parallax.
            sigma_adjustment_parameter (bool/int): Whether to include sigma adjustment and how (0, 1, 3, or False).
            extinction_parameter (bool): Whether to include extinction as a parameter.
            true_values (list/bool): True parameter values for mock data, if available.

        Sets up the object's main attributes and checks for valid parameter combinations.
        """
        self.current_path = str(Path.cwd())
        allowed_sigma_adjustment_parameter = {0, 1, 3}
        if sigma_adjustment_parameter not in allowed_sigma_adjustment_parameter:
            raise ValueError(f'"sigma_adjustment_parameter" must be one of {allowed_sigma_adjustment_parameter}, '
                             f'but got {sigma_adjustment_parameter}.')

        if sigma_adjustment_parameter == 3 and extinction_parameter == True:
            raise ValueError('Invalid combination: "sigma_adjustment_parameter=3" cannot be used with "extinction_parameter=True".\n'
                             'The degree of freedom will be ZERO!')

        self.load_flag = False # False: it is defined, not loaded from files
        self.name = name.strip()
        self.bands_list = bands_list
        if isinstance(obs_mags, list):
            obs_mags = np.array(obs_mags)
        self.obs_mags = obs_mags
        if isinstance(obs_err_mags, list):
            obs_err_mags = np.array(obs_err_mags)
        self.obs_err_mags = obs_err_mags
        self.vsini_obs = vsini_obs
        self.vsini_obs_err = vsini_obs_err
        self.parallax_obs = parallax_obs
        self.parallax_obs_err = parallax_obs_err
        self.sigma_adjustment_parameter = sigma_adjustment_parameter
        self.extinction_parameter = extinction_parameter
        self.true_values = true_values


    def add_info(self, cluster='', extinction=0, extinction_err=0, extinction_type=None,
                 # collect_variables_list=None,
                 interp_variables_list=None,
                 sampling_variables_list=None, print_flag=False,
                 max_sigma_adjustment_parameter=None,):
        """
        Add additional information and settings to the Target object.

        Args:
            cluster (str): Name of cluster (if any).
            extinction (float): Value of extinction.
            extinction_err (float): Uncertainty of extinction.
            extinction_type (str): Extinction type ('A_V', 'EB_V', or 'EBP_RP').
            # collect_variables_list (list): Variables used to collect data from isochrones.
            interp_variables_list (list): Variables to use for interpolation.
            sampling_variables_list (list): Variables to sample in inference.
            print_flag (bool): Whether to print status messages.

        Sets up extinction, variables lists, and computes extinction-corrected and absolute magnitudes.
        """
        self.cluster = cluster
        if self.extinction_parameter == True:
            if extinction_type is None:
                # Here A_V is the extinction of the Johnson V band
                raise ValueError("You must set the 'extinction_type'. Available types: 'A_V', 'EB_V', or 'EBP_RP'.")
            if extinction_type not in ['A_V', 'EB_V', 'EBP_RP']:
                raise ValueError("Wrong type! Available types: 'A_V', 'EB_V', or 'EBP_RP'.")
        self.extinction = extinction
        self.extinction_err = extinction_err
        self.extinction_type = extinction_type
        self.max_sigma_adjustment_parameter = max_sigma_adjustment_parameter

        # if collect_variables_list is None:
        #     collect_variables_list = ['Mini', 'logAge', 'MH', 'logTe', 'logL', 'logg', 'Req', 'Rpol', 'vtaneq', 'Inclination', 'omega_ini'] # Do not change the order of this list; Delete if needed
        if interp_variables_list is None:
            interp_variables_list = ['Mini', 'logAge', 'MH', 'omega_ini'] # Do not change the order of this list; Delete if needed
        if sampling_variables_list is None:
            sampling_variables_list = ['Mini', 'logAge', 'MH', 'omega_ini', 'Inclination', 'parallax'] # Do not change the order of this list; Delete if needed

        if self.sigma_adjustment_parameter in {0, False}:
            sampling_variables_list = sampling_variables_list
        elif self.sigma_adjustment_parameter == 1:
            sampling_variables_list.append('sigma')
        elif self.sigma_adjustment_parameter == 3:
            sampling_variables_list.append('sigma_gaia')
            sampling_variables_list.append('sigma_tycho2')
            sampling_variables_list.append('sigma_2mass')

        if self.extinction_parameter == True:
            sampling_variables_list.append('extinction')

        if extinction == 0:
            print('Assuming no extinction.')

        # self.collect_variables_list = collect_variables_list
        self.interp_variables_list = interp_variables_list
        self.sampling_variables_list = sampling_variables_list
        self.obs_mags_corrected, self.obs_err_mags_corrected = remove_extinction(self.obs_mags, self.obs_err_mags, self.bands_list,
                                                                                 extinction, extinction_err, extinction_type, print_flag=print_flag)
        self.abs_mags, self.abs_err_mags = cal_abs_mag(self.obs_mags_corrected, self.obs_err_mags_corrected, self.parallax_obs, self.parallax_obs_err,
                                                       max_sigma_adjustment_parameter=max_sigma_adjustment_parameter)

    def known_parameter(
            self,
            true_values=None,
            known_Mini=None,
            known_logAge=None,
            known_MH=None,
            known_omega_ini=None,
            known_Inclination=None,
            known_parallax=None,
    ):
        """
        Specify known or true parameter values for the target.

        Parameters can be provided in two ways:

        1. Provide `true_values` directly:
           `true_values` must contain six values in the order:
           [Mini, logAge, MH, omega_ini, Inclination, parallax].

        2. Provide individual known values:
           known_Mini, known_logAge, known_MH, known_omega_ini,
           known_Inclination, and known_parallax.

        If neither `true_values` nor an individual value is provided for a
        parameter, that parameter is stored as None.

        Args:
            true_values (array-like, optional):
                True parameter values in the order
                [Mini, logAge, MH, omega_ini, Inclination, parallax].
            known_Mini (float, optional): Known initial stellar mass.
            known_logAge (float, optional): Known log(age).
            known_MH (float, optional): Known metallicity.
            known_omega_ini (float, optional): Known initial rotation parameter.
            known_Inclination (float, optional): Known inclination.
            known_parallax (float, optional): Known parallax.
        """

        if true_values is not None:
            true_values = np.asarray(true_values, dtype=object)

            if true_values.shape != (6,):
                raise ValueError(
                    "`true_values` must have length 6 in the order "
                    "[Mini, logAge, MH, omega_ini, Inclination, parallax]."
                )

            (
                self.known_Mini,
                self.known_logAge,
                self.known_MH,
                self.known_omega_ini,
                self.known_Inclination,
                self.known_parallax,
            ) = true_values

            self.true_values = true_values

        else:
            self.known_Mini = known_Mini
            self.known_logAge = known_logAge
            self.known_MH = known_MH
            self.known_omega_ini = known_omega_ini
            self.known_Inclination = known_Inclination
            self.known_parallax = known_parallax

            self.true_values = np.array(
                [
                    known_Mini,
                    known_logAge,
                    known_MH,
                    known_omega_ini,
                    known_Inclination,
                    known_parallax,
                ],
                dtype=object,
            )


    def set_method(self,
                   # general setting
                   grids_path,
                   grids_type,
                   METHOD,
                   iso_name='PARSEC',
                   flag_64=False,
                    # PARS setting
                   PARS_limb_darkening_path=None,
                   PARS_filters=None,
                   PARS_nz=100,
                   PARS_MH_offset=-0.046, # [M/H] zeropoint converison from Caffau et. (2011) to Grevesse & Sauval (1998)
                   # NN setting
                   # NN_prepare_data=False,
                   # NN_photo_err_scaler=40,
                   ):

        available_grids_type = ['Mini', 'u']
        available_METHOD = ['DIRECT', 'PARS', 'NN']
        if not isinstance(grids_path, str) or grids_type not in available_grids_type:
            raise Exception(f'Grids path must be a string and one of {available_grids_type}')
        if not isinstance(METHOD, str) or METHOD not in available_METHOD:
            raise TypeError(f'METHOD must be a string and one of {available_METHOD}')

        if self.cluster != '':
            self.cluster_name = '_' + self.cluster
        else:
            self.cluster_name = self.cluster

        self.iso_name = iso_name
        if iso_name.lower() == 'PARSEC'.lower():
            pass
        elif iso_name.lower() == 'MIST'.lower():
            if grids_type == 'u':
                raise ValueError("MIST isochrones could only use 'Mini' grids.")
            if METHOD == 'DIRECT':
                raise ValueError("MIST isochrones could only use the 'PARS' method.")
        else:
            raise ValueError("iso_name must be either 'PARSEC' or 'MIST'.")

        self.grids_path = grids_path
        self.grids_type = grids_type
        self.METHOD = METHOD
        self.flag_64 = flag_64
        self.interp_upper_limit = []
        self.interp_lower_limit = []

        if grids_type == 'Mini':
            if iso_name.lower() == 'PARSEC'.lower():
                interp_func, interp_u, interp_u_to_outputs, interp_u_bounds, interp_variables_list, interp_output_list, data \
                    = load_regularized_Mini_interpolator(grids_path, force_float64=flag_64)
                self.Model_interp = interp_func
                self.interp_u = interp_u
                self.interp_u_to_outputs = interp_u_to_outputs
                self.interp_u_bounds = interp_u_bounds
                self.interp_variables_list = interp_variables_list
                self.interp_output_list = interp_output_list
                self.data = data
                for grid_i in [i + '_grid' for i in self.interp_variables_list]:
                    self.interp_upper_limit.append(np.max(self.data[grid_i]))
                    self.interp_lower_limit.append(np.min(self.data[grid_i]))
            elif iso_name.lower() == 'MIST'.lower():
                interp_func, interp_variables_list, interp_output_list, data = load_regularized_MIST_interpolator(grids_path, force_float64=flag_64)
                self.Model_interp = interp_func
                self.interp_variables_list = interp_variables_list
                self.interp_output_list = interp_output_list
                self.data = data
                for grid_i in [i + '_grid' for i in self.interp_variables_list]:
                    self.interp_upper_limit.append(np.max(self.data[grid_i]))
                    self.interp_lower_limit.append(np.min(self.data[grid_i]))

        elif grids_type == 'u':
            interp_func, interp_u, interp_u_to_outputs, interp_u_bounds, interp_variables_list, interp_output_list, data \
                = load_regularized_u_interpolator(grids_path, force_float64=flag_64)
            self.Model_interp = interp_func
            self.interp_u = interp_u
            self.interp_u_to_outputs = interp_u_to_outputs
            self.interp_u_bounds = interp_u_bounds
            self.interp_variables_list = interp_variables_list
            self.interp_output_list = interp_output_list
            self.data = data
            for grid_i in [i + '_grid' for i in self.interp_variables_list]:
                if grid_i == 'Mini_grid':
                    self.interp_upper_limit.append(np.nanmax(self.data['grid_bounds'][:, :, :, 1]))
                    self.interp_lower_limit.append(np.nanmin(self.data['grid_bounds'][:, :, :, 0]))
                else:
                    self.interp_upper_limit.append(np.max(self.data[grid_i]))
                    self.interp_lower_limit.append(np.min(self.data[grid_i]))

        self.interp_upper_limit = np.array(self.interp_upper_limit)
        self.interp_lower_limit = np.array(self.interp_lower_limit)

        if METHOD == 'DIRECT':
            if 'Inclination' not in self.interp_variables_list:
                self.interp_variables_list.append('Inclination')
        elif METHOD == 'PARS':
            if 'Inclination' in self.interp_variables_list:
                self.interp_variables_list.remove('Inclination')
            if PARS_limb_darkening_path is None:
                raise Exception('PARS_limb_darkening_path must be defined.')
            if PARS_filters is None:
                raise Exception('PARS_filters must be defined.')
            self.PARS_limb_darkening_path = PARS_limb_darkening_path
            self.PARS_filters = PARS_filters
            PARS_limb_darkening_filelist = glob(PARS_limb_darkening_path)
            PARS_limb_darkening_filelist.sort()
            limb_darkening_MH_dic = {file: extract_MH(file) for file in PARS_limb_darkening_filelist if extract_MH(file) is not None}
            print("Loading limb darkening files from Kurucz's model.")
            MH_index = self.interp_variables_list.index('MH')
            PARS_MH_lower_limit = self.interp_lower_limit[MH_index] + PARS_MH_offset
            PARS_MH_upper_limit = self.interp_upper_limit[MH_index] + PARS_MH_offset
            made_limb_darkening_dict = make_limb_darkening_files(PARS_MH_lower_limit, PARS_MH_upper_limit, limb_darkening_MH_dic, PARS_filters, use_tqdm=True)
            self.made_limb_darkening_dict = made_limb_darkening_dict
            self.PARS_nz = PARS_nz
            self.PARS_MH_offset = PARS_MH_offset

        fixed_parameter_indices = []
        for interp_var in self.interp_variables_list:
            if interp_var not in self.sampling_variables_list:
                fixed_parameter_index = self.interp_variables_list.index(interp_var)
                fixed_parameter_indices.append(fixed_parameter_index)
        self.fixed_parameter_indices = fixed_parameter_indices



    def set_sampling(self,
                     n_walkers=30,
                     age_limit=None,
                     parallax_limit=None,
                     sigma_limit=None,
                     extinction_limit=None,

                     Vrot_prior=None,
                     mini_prior='default',
                     logAge_prior='default',
                     MH_prior='default',
                     omega_ini_prior='default',
                     inclination_prior='default',
                     parallax_prior='default',
                     extinction_prior='default',
                     ):
        """
        Set up the sampling configuration and priors for the target.

        Args:
            initial_type (str): How to initialize walkers ('random', 'better', or 'closer').
            n_walkers (int): Number of walkers for MCMC.
            parallax_limit (list): Parallax prior limits [min, max].
            sigma_limit (list): Sigma prior limits [min, max] or list of such for multiple sigmas.
            extinction_limit (list): Extinction prior limits [min, max].
            mini_prior (str/callable): Prior for mini.
            logAge_prior (str/callable): Prior for logAge.
            MH_prior (str/callable): Prior for metallicity.
            omega_ini_prior (str/callable): Prior for rotation parameter.
            inclination_prior (str/callable): Prior for inclination.
            parallax_prior (str/callable): Prior for parallax.
            extinction_prior (str/callable): Prior for extinction.

        Sets prior distributions and initializes the starting positions of the walkers for sampling.
        """
        if parallax_limit is None:
            parallax_limit = [self.parallax_obs - 5, self.parallax_obs + 5]
        if sigma_limit is None:
            sigma_limit = [1e-8, 0.1]
        if extinction_limit is None:
            extinction_limit = [1e-8, 1]

        self.age_limit = age_limit
        self.parallax_limit = parallax_limit
        self.sigma_limit = sigma_limit
        self.extinction_limit = extinction_limit

        if any(x is not None for x in self.true_values) and not all(x is not None for x in self.true_values):
            if self.known_omega_ini is not None and self.known_Inclination is not None:
                if np.isclose(self.known_omega_ini * np.sin(self.known_Inclination), 0):
                    Vrot_prior = lambda x : 0

        invalid_keys = []
        if mini_prior != 'default' and not callable(mini_prior):
            invalid_keys.append('mini_prior')
        if logAge_prior != 'default' and not callable(logAge_prior):
            invalid_keys.append('logAge_prior')
        if MH_prior != 'default' and not callable(MH_prior):
            invalid_keys.append('MH_prior')
        if omega_ini_prior != 'default' and not callable(omega_ini_prior):
            invalid_keys.append('omega_ini_prior')
        if inclination_prior != 'default' and not callable(inclination_prior):
            invalid_keys.append('inclination_prior')
        if parallax_prior != 'default' and not callable(parallax_prior):
            invalid_keys.append('parallax_prior')
        if extinction_prior != 'default' and not callable(extinction_prior):
            invalid_keys.append('extinction_prior')
        if Vrot_prior is not None and not callable(Vrot_prior):
            invalid_keys.append('Vrot_prior')

        self.Vrot_prior = Vrot_prior
        self.rough_mass = None

        if invalid_keys:
            message = (
                f"❌ The following priors are invalid: {invalid_keys}\n\n"
                f"✅ Each prior must be eithor:\n"
                f"    - A string: 'default', OR\n"
                f"    - A callable custom function (callable)"
            )
            raise ValueError(message)

        self.mini_prior = mini_prior
        self.logAge_prior = logAge_prior
        self.MH_prior = MH_prior
        self.omega_ini_prior = omega_ini_prior
        self.inclination_prior = inclination_prior
        self.parallax_prior = parallax_prior
        self.extinction_prior = extinction_prior

        sampling_lower_limit_original = np.zeros(len(self.sampling_variables_list))
        sampling_upper_limit_original = np.zeros(len(self.sampling_variables_list))
        for var in self.sampling_variables_list:
            if var == 'Inclination':
                sampling_lower_limit_original[self.sampling_variables_list.index(var)] = 0.0
                sampling_upper_limit_original[self.sampling_variables_list.index(var)] = np.deg2rad(90.0)
            elif var == 'parallax':
                sampling_lower_limit_original[self.sampling_variables_list.index(var)] = parallax_limit[0]
                sampling_upper_limit_original[self.sampling_variables_list.index(var)] = parallax_limit[1]
            elif var in self.interp_variables_list:
                samp_index = self.sampling_variables_list.index(var)
                interp_index = self.interp_variables_list.index(var)
                sampling_lower_limit_original[samp_index] = self.interp_lower_limit[interp_index]
                sampling_upper_limit_original[samp_index] = self.interp_upper_limit[interp_index]

        if self.sigma_adjustment_parameter in [1]:
            sampling_lower_limit_original[self.sampling_variables_list.index('sigma')] = sigma_limit[0]
            sampling_upper_limit_original[self.sampling_variables_list.index('sigma')] = sigma_limit[1]
        if self.sigma_adjustment_parameter in [3]:
            if len(sigma_limit) == 2:
                sigma_limit = [sigma_limit] * 3
            sampling_lower_limit_original[self.sampling_variables_list.index('sigma_gaia')] = sigma_limit[0][0]
            sampling_upper_limit_original[self.sampling_variables_list.index('sigma_gaia')] = sigma_limit[0][1]
            sampling_lower_limit_original[self.sampling_variables_list.index('sigma_tycho2')] = sigma_limit[1][0]
            sampling_upper_limit_original[self.sampling_variables_list.index('sigma_tycho2')] = sigma_limit[1][1]
            sampling_lower_limit_original[self.sampling_variables_list.index('sigma_2mass')] = sigma_limit[2][0]
            sampling_upper_limit_original[self.sampling_variables_list.index('sigma_2mass')] = sigma_limit[2][1]
        if self.extinction_parameter == True:
            sampling_lower_limit_original[self.sampling_variables_list.index('extinction')] = extinction_limit[0]
            sampling_upper_limit_original[self.sampling_variables_list.index('extinction')] = extinction_limit[1]

        if age_limit is not None:
            sampling_lower_limit_original[self.sampling_variables_list.index('logAge')] = age_limit[0]
            sampling_upper_limit_original[self.sampling_variables_list.index('logAge')] = age_limit[1]

        self.sampling_lower_limit = sampling_lower_limit_original
        self.sampling_upper_limit = sampling_upper_limit_original

        ndim = len(self.sampling_variables_list)
        value_differences = np.array(self.sampling_upper_limit) - np.array(self.sampling_lower_limit)

        initial_states = np.random.rand(n_walkers, ndim) * value_differences + np.array(self.sampling_lower_limit)

        self.n_walkers = n_walkers
        self.ndim = ndim
        self.initial_states = initial_states

        print('Sampling parameter space:')
        print(f"{'Parameter':<12} {'Min':>10} {'Max':>10}")
        print("-" * 34)

        for name, low, high in zip(
                self.sampling_variables_list,
                self.sampling_lower_limit,
                self.sampling_upper_limit
        ):
            low_str = f"{low:.4f}" if np.isfinite(low) else "nan"
            high_str = f"{high:.4f}" if np.isfinite(high) else "nan"
            print(f"{name:<12} {low_str:>10} {high_str:>10}")


    def save(self, filename):
        """
        Save the current object configuration to an npz file.

        Notes
        -----
        The following runtime objects are not saved and should be rebuilt
        when loading the object:

            Model_interp
            interp_u
            interp_u_to_outputs
            interp_u_bounds
            data

        User-defined prior functions are saved as source-code strings
        using inspect.getsource().
        """

        filename = Path(filename)
        filename.parent.mkdir(parents=True, exist_ok=True)

        skip_keys = {
            "Model_interp",
            "interp_u",
            "interp_u_to_outputs",
            "interp_u_bounds",
            "data",
        }

        prior_keys = {
            "Vrot_prior",
            "mini_prior",
            "logAge_prior",
            "MH_prior",
            "omega_ini_prior",
            "inclination_prior",
            "parallax_prior",
            "extinction_prior",
        }

        arrays_to_save = {}
        config = {}

        def to_jsonable(value):
            """Convert values to JSON-serializable objects."""
            if value is None:
                return None

            if isinstance(value, (str, int, float, bool)):
                return value

            if isinstance(value, (np.integer, np.floating, np.bool_)):
                return value.item()

            if isinstance(value, Path):
                return str(value)

            if isinstance(value, np.ndarray):
                return value.tolist()

            if isinstance(value, (list, tuple)):
                return [to_jsonable(v) for v in value]

            if isinstance(value, dict):
                return {str(k): to_jsonable(v) for k, v in value.items()}

            raise TypeError(
                f"Cannot convert object of type {type(value)} to JSON."
            )

        def serialize_prior(name, prior):
            """
            Serialize prior settings.

            Default priors such as 'default' or None are saved directly.
            User-defined callable priors are saved by source code.
            """
            if prior is None:
                return None

            if isinstance(prior, str):
                return prior

            if callable(prior):
                try:
                    source = inspect.getsource(prior)
                except OSError as e:
                    raise RuntimeError(
                        f"Could not get source code for prior `{name}`. "
                        "Please define it as a normal Python function."
                    ) from e

                return {
                    "type": "custom_source",
                    "name": prior.__name__,
                    "source": source,
                }

            raise TypeError(
                f"Prior `{name}` has unsupported type {type(prior)}. "
                "Expected None, str, or callable."
            )

        for key, value in self.__dict__.items():
            if key in skip_keys:
                continue

            if key in prior_keys:
                config[key] = serialize_prior(key, value)
                continue

            if isinstance(value, np.ndarray):
                # Numeric arrays can be safely saved directly in npz.
                # Object arrays, usually containing None, are saved in JSON.
                if value.dtype == object:
                    config[key] = to_jsonable(value)
                else:
                    arrays_to_save[key] = value
                continue

            if isinstance(value, (str, int, float, bool, type(None), list, tuple, dict, Path)):
                config[key] = to_jsonable(value)
                continue

            if isinstance(value, (np.integer, np.floating, np.bool_)):
                config[key] = to_jsonable(value)
                continue

            raise TypeError(
                f"Attribute `{key}` has unsupported type {type(value)}. "
                "Please either add it to `skip_keys` or define how to serialize it."
            )

        arrays_to_save["config_json"] = np.array(json.dumps(config, indent=2))

        np.savez(filename, **arrays_to_save)

        print(f"Saved object to: {os.path.abspath(filename)}")



    @classmethod
    def load(cls, filename):
        """
        Load an object from a saved npz file.

        Loading order:
        1. Read saved config.
        2. Change directory to saved current_path.
        3. Restore attributes needed by set_method().
        4. Call set_method() to rebuild runtime-only objects.
        5. Restore all saved attributes from file, overriding values set by set_method().
           Non-prior attributes are restored first, then prior functions are restored.
        """

        import os
        import json
        import textwrap
        import numpy as np
        from pathlib import Path
        from scipy.stats import norm

        from belfast.mcmc_utils import (
            select_Vrot_prior_by_rough_mass,
            log_prior_Vrot_dict,
            _log_prior_mixture_maxwell_lagged_truncated,
        )

        filename = Path(filename)
        data = np.load(filename, allow_pickle=False)

        if "config_json" not in data:
            raise KeyError("`config_json` was not found in the saved npz file.")

        config = json.loads(str(data["config_json"]))

        obj = cls.__new__(cls)

        prior_keys = {
            "Vrot_prior",
            "mini_prior",
            "logAge_prior",
            "MH_prior",
            "omega_ini_prior",
            "inclination_prior",
            "parallax_prior",
            "extinction_prior",
        }

        object_array_keys = {
            "true_values",
        }

        def _clean_source(source):
            """
            Clean source code saved by inspect.getsource().

            Handles two common cases:
            1. Source defined inside another function/class, with leading indentation.
            2. One-line lambda assignment accidentally ending with a comma.
            """

            source = textwrap.dedent(source).strip()

            lines = source.splitlines()

            # Example:
            # extinction_prior = lambda x: norm.logpdf(...),
            if len(lines) == 1 and "lambda" in lines[0] and lines[0].rstrip().endswith(","):
                lines[0] = lines[0].rstrip().rstrip(",")

            return "\n".join(lines)

        def restore_prior(name, value):
            """
            Restore prior.

            - None remains None.
            - Strings such as 'default' remain strings.
            - User-defined functions saved as source code are restored by exec().
            """

            if value is None:
                return None

            if isinstance(value, str):
                return value

            if isinstance(value, dict) and value.get("type") == "custom_source":
                source = _clean_source(value["source"])
                func_name = value["name"]

                # Special handling for Vrot prior.
                # This avoids problems with closures such as params inside _make_log_prior_Vrot().
                if name == "Vrot_prior":
                    rough_mass = getattr(obj, "rough_mass", None)

                    # Compatibility with your earlier attribute name.
                    if rough_mass is None:
                        rough_mass = getattr(obj, "rough_mini", None)

                    if rough_mass is None:
                        return None

                    _, func = select_Vrot_prior_by_rough_mass(
                        rough_mass,
                        log_prior_Vrot_dict,
                    )

                    if func is None:
                        return None

                    return func

                namespace = {
                    "np": np,
                    "numpy": np,
                    "norm": norm,

                    # For priors like:
                    # extinction_prior = lambda x: norm.logpdf(x, loc=extinction, scale=extinction_err)
                    "extinction": getattr(obj, "extinction", None),
                    "extinction_err": getattr(obj, "extinction_err", None),

                    # For possible custom priors using your internal function.
                    "_log_prior_mixture_maxwell_lagged_truncated": (
                        _log_prior_mixture_maxwell_lagged_truncated
                    ),
                }

                exec(source, namespace)

                if func_name in namespace:
                    func = namespace[func_name]
                elif name in namespace:
                    func = namespace[name]
                else:
                    raise RuntimeError(
                        f"Could not restore prior `{name}`. "
                        f"Neither `{func_name}` nor `{name}` was found after exec()."
                    )

                # Handle accidental trailing comma:
                # prior = lambda x: ...,  -> tuple with one function
                if isinstance(func, tuple):
                    if len(func) == 1 and callable(func[0]):
                        func = func[0]
                    else:
                        raise TypeError(
                            f"Restored prior `{name}` is a tuple, "
                            f"but not a one-function tuple. Got length {len(func)}."
                        )

                if not callable(func):
                    raise TypeError(
                        f"Restored prior `{name}` is not callable. Got {type(func)}."
                    )

                return func

            raise TypeError(
                f"Cannot restore prior `{name}` from type {type(value)}."
            )

        def restore_saved_attributes():
            """
            Restore all saved attributes.

            Important:
            Non-prior attributes are restored first, then prior functions.
            This ensures custom priors depending on saved attributes
            such as extinction and extinction_err can be reconstructed correctly.
            """

            # ------------------------------------------------------------
            # 1. Restore non-prior config attributes first.
            # ------------------------------------------------------------
            for key, value in config.items():
                if key in prior_keys:
                    continue

                if key in object_array_keys:
                    setattr(obj, key, np.array(value, dtype=object))
                else:
                    setattr(obj, key, value)

            # ------------------------------------------------------------
            # 2. Restore numeric arrays saved directly in npz.
            # ------------------------------------------------------------
            for key in data.files:
                if key == "config_json":
                    continue

                setattr(obj, key, data[key])

            # ------------------------------------------------------------
            # 3. Restore prior functions after all other attributes exist.
            # ------------------------------------------------------------
            for key, value in config.items():
                if key in prior_keys:
                    setattr(obj, key, restore_prior(key, value))

        # ------------------------------------------------------------
        # 1. Set current_path and change working directory first.
        # ------------------------------------------------------------
        if "current_path" in config and config["current_path"] is not None:
            obj.current_path = str(config["current_path"])
            os.chdir(obj.current_path)

        # ------------------------------------------------------------
        # 2. Some attributes may be needed by set_method().
        # ------------------------------------------------------------
        pre_set_method_keys = [
            "cluster",
            "sampling_variables_list",
        ]

        for key in pre_set_method_keys:
            if key in config:
                setattr(obj, key, config[key])

        # ------------------------------------------------------------
        # 3. Get set_method inputs from saved config.
        # ------------------------------------------------------------
        required_set_method_keys = [
            "grids_path",
            "grids_type",
            "METHOD",
        ]

        for key in required_set_method_keys:
            if key not in config:
                raise KeyError(
                    f"`{key}` is required to rebuild runtime objects with set_method(), "
                    "but it was not found in the saved config."
                )

        set_method_kwargs = {
            "grids_path": config["grids_path"],
            "grids_type": config["grids_type"],
            "METHOD": config["METHOD"],
            "iso_name": config.get("iso_name", "PARSEC"),
            "flag_64": config.get("flag_64", False),

            # PARS-related settings
            "PARS_limb_darkening_path": config.get("PARS_limb_darkening_path", None),
            "PARS_filters": config.get("PARS_filters", None),
            "PARS_nz": config.get("PARS_nz", 100),
            "PARS_MH_offset": config.get("PARS_MH_offset", -0.046),
        }

        # ------------------------------------------------------------
        # 4. Call set_method() to rebuild runtime-only objects.
        # ------------------------------------------------------------
        obj.set_method(**set_method_kwargs)

        # ------------------------------------------------------------
        # 5. Restore saved attributes again.
        #    Saved file values are the source of truth.
        # ------------------------------------------------------------
        restore_saved_attributes()

        obj.load_flag = str(filename)

        return obj