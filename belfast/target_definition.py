import os
import emcee
import numpy as np
from glob import glob
from pathlib import Path
import pandas as pd
from belfast.paths import DATA_DIR, ISOCHRONE_DIR, KLD_DIR, PARS_FILTER_DIR
from belfast.PARS_utils import extract_MH, make_limb_darkening_files
from belfast.target_utils import cal_abs_mag, cal_obs_mag, remove_extinction, classify_posterior_samples_safety
from belfast.mcmc_utils import select_Vrot_prior_by_rough_mass, log_prior_Vrot_dict, _log_prior_mixture_maxwell_lagged_truncated
from belfast.isochrone_utils import load_regularized_Mini_interpolator, load_regularized_u_interpolator, load_regularized_MIST_interpolator
from belfast.isochrone_utils import load_default_regularized_PARSECv2_Mini_interpolator, load_default_regularized_PARSECv2_u_interpolator, load_default_regularized_MIST_Mini_interpolator
# from belfast.isochrone_utils_new import load_regularized_Mini_interpolator#, load_regularized_u_interpolator, load_regularized_MIST_interpolator

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
        self.name = name.strip().replace(' ', '_')
        self.bands_list = bands_list
        self.obs_mags = obs_mags
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
        self.cluster = cluster.strip().replace(' ', '_')
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
        #     collect_variables_list = ['Mini', 'Age', 'MH', 'logTe', 'logL', 'logg', 'Req', 'Rpol', 'vtaneq', 'Inclination', 'omega_ini'] # Do not change the order of this list; Delete if needed
        if interp_variables_list is None:
            interp_variables_list = ['Mini', 'Age', 'MH', 'omega_ini'] # Do not change the order of this list; Delete if needed
        if sampling_variables_list is None:
            sampling_variables_list = ['Mini', 'Age', 'MH', 'omega_ini', 'Inclination', 'parallax'] # Do not change the order of this list; Delete if needed

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

        self.known_Mini = None
        self.known_Age = None
        self.known_MH = None
        self.known_omega_ini = None
        self.known_Inclination = None
        self.known_parallax = None
        self.true_values = np.array([None, None, None, None, None, None,], dtype=object)


    def known_parameter(
            self,
            true_values=None,
            known_Mini=None,
            known_Age=None,
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
           [Mini, Age, MH, omega_ini, Inclination, parallax].

        2. Provide individual known values:
           known_Mini, known_Age, known_MH, known_omega_ini,
           known_Inclination, and known_parallax.

        If neither `true_values` nor an individual value is provided for a
        parameter, that parameter is stored as None.

        Args:
            true_values (array-like, optional):
                True parameter values in the order
                [Mini, Age, MH, omega_ini, Inclination, parallax].
            known_Mini (float, optional): Known initial stellar mass.
            known_Age (float, optional): Known log(age).
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
                    "[Mini, Age, MH, omega_ini, Inclination, parallax]."
                )

            (
                self.known_Mini,
                self.known_Age,
                self.known_MH,
                self.known_omega_ini,
                self.known_Inclination,
                self.known_parallax,
            ) = true_values

            self.true_values = true_values

        else:
            self.known_Mini = known_Mini
            self.known_Age = known_Age
            self.known_MH = known_MH
            self.known_omega_ini = known_omega_ini
            self.known_Inclination = known_Inclination
            self.known_parallax = known_parallax

            self.true_values = np.array(
                [
                    known_Mini,
                    known_Age,
                    known_MH,
                    known_omega_ini,
                    known_Inclination,
                    known_parallax,
                ],
                dtype=object,
            )

    def set_method(
            self,
            grids_path=None,
            grids_type='Mini',
            METHOD='DIRECT',
            iso_name='PARSEC',
            label_list=None,
            correct_MIST_MH_zeropoint=False,
            MIST_MH_offset=0.018,
            flag_64=False,
            # PARS setting
            PARS_limb_darkening_path=KLD_DIR / 'i[pm]*k*.pck*',
            PARS_filters=PARS_FILTER_DIR / '*.dat',
            PARS_nz=100,
            PARS_MH_offset=-0.046,
    ):
        iso_name = iso_name.upper()
        METHOD = METHOD.upper()

        # Standardize grids_type
        grids_type_map = {'mini': 'Mini', 'u': 'u'}
        try:
            grids_type = grids_type_map[grids_type.lower()]
        except (AttributeError, KeyError):
            raise ValueError('grids_type must be "Mini" or "u".')

        # Basic checks
        if iso_name not in {'PARSEC', 'MIST'}:
            raise ValueError('iso_name must be "PARSEC" or "MIST".')

        if METHOD not in {'DIRECT', 'PARS'}:
            raise ValueError('METHOD must be "DIRECT" or "PARS".')

        if iso_name == 'MIST' and (grids_type != 'Mini' or METHOD != 'PARS'):
            raise ValueError(
                'MIST only supports grids_type="Mini" with METHOD="PARS".'
            )

        if label_list is None:
            label_list = [1]

        # DIRECT requires inclination grids; PARS does not.
        grid_type = 'with_inc' if METHOD == 'DIRECT' else 'without_inc'

        if isinstance(PARS_limb_darkening_path, os.PathLike):
            PARS_limb_darkening_path = str(PARS_limb_darkening_path)

        if isinstance(PARS_filters, os.PathLike):
            PARS_filters = str(PARS_filters)

        # Basic attributes
        self.cluster_name = f'_{self.cluster}' if self.cluster else ''
        self.iso_name = iso_name
        self.grids_type = grids_type
        self.METHOD = METHOD
        self.PARS_limb_darkening_path = PARS_limb_darkening_path
        self.PARS_filters = PARS_filters
        self.correct_MIST_MH_zeropoint = correct_MIST_MH_zeropoint
        self.MIST_MH_offset = MIST_MH_offset
        self.flag_64 = flag_64

        print('*' * 100)
        print(
            f'Using {iso_name} isochrones with '
            f'{grids_type} grids and {METHOD} method.'
        )
        print('*' * 100)

        # ============================================================
        # Load grids
        # ============================================================

        if iso_name == 'PARSEC':

            loader = (
                load_default_regularized_PARSECv2_Mini_interpolator
                if grids_type == 'Mini'
                else load_default_regularized_PARSECv2_u_interpolator
            )

            (
                self.Model_interp,
                self.interp_u,
                self.interp_u_to_outputs,
                self.interp_u_bounds,
                self.interp_variables_list,
                self.interp_output_list,
                self.data,
            ) = loader(
                npz_path=grids_path,
                label_list=label_list,
                grid_type=grid_type,
                force_float64=flag_64,
            )

            # If grids_path is explicitly given, grid_type is read from the file.
            # Check that it is compatible with METHOD.
            actual_grid_type = str(self.data['grid_type'])

            if actual_grid_type != grid_type:
                raise ValueError(
                    f'{METHOD} requires a "{grid_type}" grid, '
                    f'but the loaded file is "{actual_grid_type}".'
                )

            # Use the actual labels stored in the file.
            self.label_list = (
                self.data['label_list'].tolist()
                if 'label_list' in self.data.files
                else list(label_list)
            )

        else:  # MIST

            (
                self.Model_interp,
                self.interp_variables_list,
                self.interp_output_list,
                self.data,
            ) = load_default_regularized_MIST_Mini_interpolator(
                npz_path=grids_path,
                force_float64=flag_64,
            )

            self.interp_u = None
            self.interp_u_to_outputs = None
            self.interp_u_bounds = None
            self.label_list = None

        # Save the actual loaded path if available
        loaded_path = getattr(
            getattr(self.data, 'zip', None),
            'filename',
            None,
        )

        self.grids_path = (
            str(loaded_path or grids_path)
            if (loaded_path or grids_path)
            else None
        )

        # ============================================================
        # Interpolation limits
        # ============================================================

        lower_limits = []
        upper_limits = []

        for var in self.interp_variables_list:

            if grids_type == 'u' and var == 'Mini':
                bounds = self.data['grid_bounds']

                lower_limits.append(
                    np.nanmin(bounds[..., 0])
                )
                upper_limits.append(
                    np.nanmax(bounds[..., 1])
                )

            else:
                grid = self.data[f'{var}_grid']

                lower_limits.append(
                    np.nanmin(grid)
                )
                upper_limits.append(
                    np.nanmax(grid)
                )

        self.interp_lower_limit = np.asarray(lower_limits)
        self.interp_upper_limit = np.asarray(upper_limits)

        # ============================================================
        # PARS
        # ============================================================

        if METHOD == 'PARS':

            if PARS_limb_darkening_path is None:
                raise ValueError(
                    'PARS_limb_darkening_path must be defined.'
                )

            if PARS_filters is None:
                raise ValueError(
                    'PARS_filters must be defined.'
                )

            limb_darkening_MH_dic = {
                file: mh
                for file in sorted(glob(PARS_limb_darkening_path))
                if (mh := extract_MH(file)) is not None
            }

            print("Loading limb darkening files from Kurucz's model.")

            MH_index = self.interp_variables_list.index('MH')

            MH_lower = (
                    self.interp_lower_limit[MH_index]
                    + PARS_MH_offset
            )

            MH_upper = (
                    self.interp_upper_limit[MH_index]
                    + PARS_MH_offset
            )

            self.made_limb_darkening_dict = make_limb_darkening_files(
                MH_lower,
                MH_upper,
                limb_darkening_MH_dic,
                PARS_filters,
                use_tqdm=True,
            )

            self.PARS_nz = PARS_nz
            self.PARS_MH_offset = PARS_MH_offset

        # ============================================================
        # Fixed parameters
        # ============================================================

        self.fixed_parameter_indices = [
            i
            for i, var in enumerate(self.interp_variables_list)
            if var not in self.sampling_variables_list
        ]

    # def set_method(self,
    #                # general setting
    #                grids_path=None,
    #                grids_type='Mini',
    #                METHOD='DIRECT',
    #                iso_name='PARSEC',
    #                correct_MIST_MH_zeropoint=False,
    #                MIST_MH_offset=0.018,
    #                flag_64=False,
    #                 # PARS setting
    #                PARS_limb_darkening_path=DATA_DIR / 'Kurucz_limb_darkening' / 'i[pm]*k*.pck*',
    #                PARS_filters=DATA_DIR / 'PARS_filters' / '*.dat',
    #                PARS_nz=100,
    #                PARS_MH_offset=-0.046, # [M/H] zeropoint conversion from Caffau et. (2011) to Grevesse & Sauval (1998)
    #                # NN setting
    #                # NN_prepare_data=False,
    #                # NN_photo_err_scaler=40,
    #                ):
    #
    #     iso_name = iso_name.upper()
    #     METHOD = METHOD.upper()
    #
    #     if grids_path is None:
    #         if iso_name == 'PARSEC' and grids_type == 'Mini' and METHOD == 'DIRECT':
    #             grids_path = DATA_DIR / 'isochrones' / 'regular_PARSECv2_Mini_grids' / 'regularized_Mini_grid_1e6_2001e6yr_step10e6_linear_isochrones_MS_with_inc.npz'
    #         elif iso_name == 'PARSEC' and grids_type == 'Mini' and METHOD == 'PARS':
    #             grids_path = DATA_DIR / 'isochrones' / 'regular_PARSECv2_Mini_grids' / 'regularized_Mini_grid_1e6_2001e6yr_step10e6_linear_isochrones_MS_without_inc.npz'
    #         elif iso_name == 'PARSEC' and grids_type == 'u' and METHOD == 'DIRECT':
    #             grids_path = DATA_DIR / 'isochrones' / 'regular_PARSECv2_u_grids' / 'regularized_u_grid_1e6_2001e6yr_step10e6_linear_isochrones_MS_with_inc.npz'
    #         elif iso_name == 'PARSEC' and grids_type == 'u' and METHOD == 'PARS':
    #             grids_path = DATA_DIR / 'isochrones' / 'regular_PARSECv2_u_grids' / 'regularized_u_grid_1e6_2001e6yr_step10e6_linear_isochrones_MS_without_inc.npz'
    #         elif iso_name == 'MIST' and grids_type == 'Mini' and METHOD == 'PARS':
    #             grids_path = DATA_DIR / 'isochrones' / 'regular_MIST_Mini_grids' / 'regularized_MIST_Mini_grid.npz'
    #         else:
    #             raise ValueError('"iso_name" must be either "PARSEC" or "MIST".\n'
    #                              'If using PARSEC v2.0 isochrones, "grids_type" can be either "Mini" or "u" and "METHOD" can be either "DIRECT" or "PARS".\n'
    #                              'If using MIST v1.2 isochrones,  "grids_type" must be either "Mini" and "METHOD" must be "PARS".')
    #
    #     if isinstance(grids_path, os.PathLike):
    #         grids_path = str(grids_path)
    #     if isinstance(PARS_limb_darkening_path, os.PathLike):
    #         PARS_limb_darkening_path = str(PARS_limb_darkening_path)
    #     if isinstance(PARS_filters, os.PathLike):
    #         PARS_filters = str(PARS_filters)
    #
    #     if self.cluster != '':
    #         self.cluster_name = '_' + self.cluster
    #     else:
    #         self.cluster_name = self.cluster
    #
    #     self.iso_name = iso_name
    #     self.grids_path = grids_path
    #     self.grids_type = grids_type
    #     self.METHOD = METHOD
    #     self.PARS_limb_darkening_path = PARS_limb_darkening_path
    #     self.PARS_filters = PARS_filters
    #     self.correct_MIST_MH_zeropoint = correct_MIST_MH_zeropoint
    #     self.MIST_MH_offset = MIST_MH_offset
    #     self.flag_64 = flag_64
    #     self.interp_upper_limit = []
    #     self.interp_lower_limit = []
    #
    #     print('*' * 100)
    #     print(f'Using {iso_name} isochrones with {grids_type} grids and {METHOD} method.')
    #     print('*' * 100)
    #
    #     if grids_type == 'Mini':
    #         if iso_name.lower() == 'PARSEC'.lower():
    #             interp_func, interp_u, interp_u_to_outputs, interp_u_bounds, interp_variables_list, interp_output_list, data \
    #                 = load_regularized_Mini_interpolator(grids_path, force_float64=flag_64)
    #             self.Model_interp = interp_func
    #             self.interp_u = interp_u
    #             self.interp_u_to_outputs = interp_u_to_outputs
    #             self.interp_u_bounds = interp_u_bounds
    #             self.interp_variables_list = interp_variables_list
    #             self.interp_output_list = interp_output_list
    #             self.data = data
    #             for grid_i in [i + '_grid' for i in self.interp_variables_list]:
    #                 self.interp_upper_limit.append(np.max(self.data[grid_i]))
    #                 self.interp_lower_limit.append(np.min(self.data[grid_i]))
    #         elif iso_name.lower() == 'MIST'.lower():
    #             interp_func, interp_variables_list, interp_output_list, data = load_regularized_MIST_interpolator(grids_path, force_float64=flag_64)
    #             self.Model_interp = interp_func
    #             self.interp_variables_list = interp_variables_list
    #             self.interp_output_list = interp_output_list
    #             self.data = data
    #             for grid_i in [i + '_grid' for i in self.interp_variables_list]:
    #                 self.interp_upper_limit.append(np.max(self.data[grid_i]))
    #                 self.interp_lower_limit.append(np.min(self.data[grid_i]))
    #
    #     elif grids_type == 'u':
    #         interp_func, interp_u, interp_u_to_outputs, interp_u_bounds, interp_variables_list, interp_output_list, data \
    #             = load_regularized_u_interpolator(grids_path, force_float64=flag_64)
    #         self.Model_interp = interp_func
    #         self.interp_u = interp_u
    #         self.interp_u_to_outputs = interp_u_to_outputs
    #         self.interp_u_bounds = interp_u_bounds
    #         self.interp_variables_list = interp_variables_list
    #         self.interp_output_list = interp_output_list
    #         self.data = data
    #         for grid_i in [i + '_grid' for i in self.interp_variables_list]:
    #             if grid_i == 'Mini_grid':
    #                 self.interp_upper_limit.append(np.nanmax(self.data['grid_bounds'][:, :, :, 1]))
    #                 self.interp_lower_limit.append(np.nanmin(self.data['grid_bounds'][:, :, :, 0]))
    #             else:
    #                 self.interp_upper_limit.append(np.max(self.data[grid_i]))
    #                 self.interp_lower_limit.append(np.min(self.data[grid_i]))
    #
    #     self.interp_upper_limit = np.array(self.interp_upper_limit)
    #     self.interp_lower_limit = np.array(self.interp_lower_limit)
    #
    #     if METHOD == 'DIRECT':
    #         if 'Inclination' not in self.interp_variables_list:
    #             self.interp_variables_list.append('Inclination')
    #     elif METHOD == 'PARS':
    #         if 'Inclination' in self.interp_variables_list:
    #             self.interp_variables_list.remove('Inclination')
    #         if PARS_limb_darkening_path is None:
    #             raise Exception('PARS_limb_darkening_path must be defined.')
    #         if PARS_filters is None:
    #             raise Exception('PARS_filters must be defined.')
    #
    #         PARS_limb_darkening_filelist = glob(PARS_limb_darkening_path)
    #         PARS_limb_darkening_filelist.sort()
    #         limb_darkening_MH_dic = {file: extract_MH(file) for file in PARS_limb_darkening_filelist if extract_MH(file) is not None}
    #         print("Loading limb darkening files from Kurucz's model.")
    #         MH_index = self.interp_variables_list.index('MH')
    #         PARS_MH_lower_limit = self.interp_lower_limit[MH_index] + PARS_MH_offset
    #         PARS_MH_upper_limit = self.interp_upper_limit[MH_index] + PARS_MH_offset
    #         made_limb_darkening_dict = make_limb_darkening_files(PARS_MH_lower_limit, PARS_MH_upper_limit, limb_darkening_MH_dic, PARS_filters, use_tqdm=True)
    #         self.made_limb_darkening_dict = made_limb_darkening_dict
    #         self.PARS_nz = PARS_nz
    #         self.PARS_MH_offset = PARS_MH_offset
    #
    #     fixed_parameter_indices = []
    #     for interp_var in self.interp_variables_list:
    #         if interp_var not in self.sampling_variables_list:
    #             fixed_parameter_index = self.interp_variables_list.index(interp_var)
    #             fixed_parameter_indices.append(fixed_parameter_index)
    #     self.fixed_parameter_indices = fixed_parameter_indices



    def set_sampling(self,
                     n_walkers=30,
                     age_limit=None,
                     parallax_limit=None,
                     sigma_limit=None,
                     extinction_limit=None,

                     emcee_moves=None,
                     Vrot_prior=None,
                     mini_prior='default',
                     Age_prior='default',
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
            Age_prior (str/callable): Prior for Age.
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

        if emcee_moves is None:
            self.emcee_moves = [
                (emcee.moves.StretchMove(a=2.0),         0.95),
                (emcee.moves.KDEMove(),                  0.05),
            ]
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
        if Age_prior != 'default' and not callable(Age_prior):
            invalid_keys.append('Age_prior')
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
        self.Age_prior = Age_prior
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
            sampling_lower_limit_original[self.sampling_variables_list.index('Age')] = age_limit[0]
            sampling_upper_limit_original[self.sampling_variables_list.index('Age')] = age_limit[1]

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


    def check_risk(self, critical_boundary_path=None, return_flag=False):

        if critical_boundary_path is None:
            critical_boundary_path = DATA_DIR / 'critical_boundary.csv'
        critical_boundary_df = pd.read_csv(critical_boundary_path)

        path = self.posterior_path
        data = np.load(path, allow_pickle=True)

        result = classify_posterior_samples_safety(data, critical_boundary_df, batch_size=5000)

        if return_flag:
            return result


    def save(
            self,
            filename,
            verbose=True,
    ):
        """
        Save the current Target object configuration to an NPZ file.

        Notes
        -----
        The following runtime objects are not saved directly:

            Model_interp
            interp_u
            interp_u_to_outputs
            interp_u_bounds
            data
            made_limb_darkening_dict
            emcee_moves

        ``emcee_moves`` is converted into a JSON-compatible
        ``emcee_move_config`` dictionary before saving.

        User-defined prior functions are saved as source-code strings using
        ``inspect.getsource()``.
        """
        import inspect
        import json
        import os
        from pathlib import Path

        import emcee
        import numpy as np

        filename = Path(filename)

        filename.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        # ============================================================
        # Runtime-only attributes
        # ============================================================
        skip_keys = {
            "Model_interp",
            "interp_u",
            "interp_u_to_outputs",
            "interp_u_bounds",
            "data",
            "made_limb_darkening_dict",

            # emcee Move objects are not JSON serializable.
            "emcee_moves",

            # This is generated explicitly from self.emcee_moves below.
            # Skip any possibly stale existing attribute.
            "emcee_move_config",
        }

        # ============================================================
        # Prior attributes
        # ============================================================
        prior_keys = {
            "Vrot_prior",
            "mini_prior",
            "Age_prior",
            "MH_prior",
            "omega_ini_prior",
            "inclination_prior",
            "parallax_prior",
            "extinction_prior",
        }

        arrays_to_save = {}
        config = {}

        # ============================================================
        # Convert ordinary values to JSON-compatible objects
        # ============================================================
        def to_jsonable(value):
            """
            Convert a Python or NumPy value into a JSON-compatible value.
            """
            if value is None:
                return None

            if isinstance(
                    value,
                    (
                            str,
                            int,
                            float,
                            bool,
                    ),
            ):
                return value

            if isinstance(
                    value,
                    (
                            np.integer,
                            np.floating,
                            np.bool_,
                    ),
            ):
                return value.item()

            if isinstance(
                    value,
                    Path,
            ):
                return str(value)

            if isinstance(
                    value,
                    np.ndarray,
            ):
                return value.tolist()

            if isinstance(
                    value,
                    (list, tuple),
            ):
                return [
                    to_jsonable(item)
                    for item in value
                ]

            if isinstance(
                    value,
                    dict,
            ):
                return {
                    str(key): to_jsonable(item)
                    for key, item in value.items()
                }

            raise TypeError(
                "Cannot convert object of type "
                f"{type(value)} to JSON."
            )

        # ============================================================
        # Serialize priors
        # ============================================================
        def serialize_prior(
                name,
                prior,
        ):
            """
            Serialize a prior setting.

            Supported prior values
            ----------------------
            None
            str
            callable
            """
            if prior is None:
                return None

            if isinstance(
                    prior,
                    str,
            ):
                return prior

            if callable(prior):
                try:
                    source = inspect.getsource(
                        prior
                    )

                except (
                        OSError,
                        TypeError,
                ) as exc:
                    raise RuntimeError(
                        f"Could not get source code for prior `{name}`. "
                        "Please define the prior as a normal Python "
                        "function or a named lambda in a source file."
                    ) from exc

                return {
                    "type": "custom_source",
                    "name": getattr(
                        prior,
                        "__name__",
                        name,
                    ),
                    "source": source,
                }

            raise TypeError(
                f"Prior `{name}` has unsupported type "
                f"{type(prior)}. Expected None, str, or callable."
            )

        # ============================================================
        # Serialize emcee move objects
        # ============================================================
        def serialize_emcee_moves(
                emcee_moves,
        ):
            """
            Convert emcee move objects into a JSON-compatible dictionary.

            Parameters
            ----------
            emcee_moves : list, tuple, or None
                Example:

                [
                    (emcee.moves.StretchMove(a=2.0), 0.50),
                    (emcee.moves.DEMove(gamma0=0.4), 0.20),
                    (emcee.moves.DESnookerMove(), 0.15),
                    (emcee.moves.KDEMove(), 0.15),
                ]

            Returns
            -------
            dict or None
            """
            if emcee_moves is None:
                return None

            if not isinstance(
                    emcee_moves,
                    (list, tuple),
            ):
                raise TypeError(
                    "self.emcee_moves must be None or a list/tuple "
                    "of (move, fraction) pairs."
                )

            if len(emcee_moves) == 0:
                raise ValueError(
                    "self.emcee_moves cannot be empty."
                )

            move_config = {}
            total_fraction = 0.0

            for move_index, item in enumerate(
                    emcee_moves
            ):
                if (
                        not isinstance(
                            item,
                            (list, tuple),
                        )
                        or len(item) != 2
                ):
                    raise ValueError(
                        "Each self.emcee_moves entry must be "
                        "(move, fraction). "
                        f"Invalid entry at index {move_index}: "
                        f"{item!r}"
                    )

                move, fraction = item

                if not isinstance(
                        move,
                        emcee.moves.Move,
                ):
                    raise TypeError(
                        f"Entry {move_index} does not contain "
                        "an emcee Move object. "
                        f"Got {type(move).__name__}."
                    )

                if not isinstance(
                        fraction,
                        (
                                int,
                                float,
                                np.integer,
                                np.floating,
                        ),
                ):
                    raise TypeError(
                        f"Move fraction at index {move_index} "
                        "must be numeric."
                    )

                fraction = float(fraction)

                if (
                        not np.isfinite(fraction)
                        or fraction <= 0
                ):
                    raise ValueError(
                        f"Move fraction at index {move_index} "
                        "must be finite and greater than zero."
                    )

                total_fraction += fraction

                # ----------------------------------------------------
                # StretchMove
                # ----------------------------------------------------
                if isinstance(
                        move,
                        emcee.moves.StretchMove,
                ):
                    a = getattr(
                        move,
                        "a",
                        getattr(
                            move,
                            "_a",
                            None,
                        ),
                    )

                    if a is None:
                        raise AttributeError(
                            "Could not read parameter `a` "
                            "from StretchMove."
                        )

                    move_name = "StretchMove"

                    this_config = {
                        "a": float(a),
                        "fraction": fraction,
                    }

                # ----------------------------------------------------
                # DEMove
                # ----------------------------------------------------
                elif isinstance(
                        move,
                        emcee.moves.DEMove,
                ):
                    gamma0 = getattr(
                        move,
                        "gamma0",
                        getattr(
                            move,
                            "_gamma0",
                            None,
                        ),
                    )

                    move_name = "DEMove"

                    this_config = {
                        "gamma0": (
                            None
                            if gamma0 is None
                            else float(gamma0)
                        ),
                        "fraction": fraction,
                    }

                # ----------------------------------------------------
                # DESnookerMove
                # ----------------------------------------------------
                elif isinstance(
                        move,
                        emcee.moves.DESnookerMove,
                ):
                    move_name = "DESnookerMove"

                    this_config = {
                        "fraction": fraction,
                    }

                # ----------------------------------------------------
                # KDEMove
                # ----------------------------------------------------
                elif isinstance(
                        move,
                        emcee.moves.KDEMove,
                ):
                    move_name = "KDEMove"

                    this_config = {
                        "fraction": fraction,
                    }

                else:
                    raise ValueError(
                        "Unsupported emcee move type: "
                        f"{type(move).__name__}"
                    )

                # This dictionary format supports one instance
                # of each move type.
                if move_name in move_config:
                    raise ValueError(
                        "Duplicate emcee move type found: "
                        f"{move_name}. "
                        "The current serialization format supports "
                        "only one instance of each move type."
                    )

                move_config[
                    move_name
                ] = this_config

            if not np.isclose(
                    total_fraction,
                    1.0,
                    rtol=0.0,
                    atol=1e-8,
            ):
                raise ValueError(
                    "Move fractions must sum to 1. "
                    f"Got {total_fraction:.12g}."
                )

            return move_config

        # ============================================================
        # Save emcee move configuration explicitly
        # ============================================================
        config["emcee_move_config"] = (
            serialize_emcee_moves(
                getattr(
                    self,
                    "emcee_moves",
                    None,
                )
            )
        )

        # ============================================================
        # Save attributes
        # ============================================================
        for key, value in self.__dict__.items():
            if key in skip_keys:
                continue

            if key in prior_keys:
                config[key] = serialize_prior(
                    key,
                    value,
                )
                continue

            if isinstance(
                    value,
                    np.ndarray,
            ):
                # Numeric arrays are saved directly in NPZ.
                if value.dtype == object:
                    config[key] = to_jsonable(
                        value
                    )
                else:
                    arrays_to_save[key] = value

                continue

            if isinstance(
                    value,
                    (
                            str,
                            int,
                            float,
                            bool,
                            type(None),
                            list,
                            tuple,
                            dict,
                            Path,
                    ),
            ):
                config[key] = to_jsonable(
                    value
                )
                continue

            if isinstance(
                    value,
                    (
                            np.integer,
                            np.floating,
                            np.bool_,
                    ),
            ):
                config[key] = to_jsonable(
                    value
                )
                continue

            raise TypeError(
                f"Attribute `{key}` has unsupported type "
                f"{type(value)}. "
                "Add it to skip_keys or define how to serialize it."
            )

        # ============================================================
        # Save config and arrays
        # ============================================================
        arrays_to_save[
            "config_json"
        ] = np.array(
            json.dumps(
                config,
                indent=2,
            )
        )

        np.savez(
            filename,
            **arrays_to_save,
        )

        if verbose:
            print(
                "Saved object to: "
                f"{os.path.abspath(filename)}"
            )

    @classmethod
    def load(
            cls,
            filename,
            grids_path=None,
    ):
        """
        Load a Target object from an NPZ file.

        Parameters
        ----------
        filename : str or pathlib.Path
            Saved target NPZ file.

        grids_path : str or pathlib.Path, optional
            Fallback grid path for older target files that did not save
            ``grids_path``.

        Loading order
        -------------
        1. Read saved config.
        2. Change to the saved current working directory.
        3. Restore attributes needed by set_method().
        4. Call set_method() to rebuild interpolation objects.
        5. Restore saved ordinary attributes and arrays.
        6. Restore prior functions.
        7. Rebuild emcee move objects.
        """
        import json
        import os
        import textwrap
        from pathlib import Path

        import emcee
        import numpy as np
        from scipy.stats import norm

        filename = Path(filename)

        data = np.load(
            filename,
            allow_pickle=False,
        )

        if "config_json" not in data.files:
            raise KeyError(
                "`config_json` was not found in the saved NPZ file."
            )

        config = json.loads(
            str(data["config_json"])
        )

        obj = cls.__new__(
            cls
        )

        # ============================================================
        # Prior attributes
        # ============================================================
        prior_keys = {
            "Vrot_prior",
            "mini_prior",
            "Age_prior",
            "MH_prior",
            "omega_ini_prior",
            "inclination_prior",
            "parallax_prior",
            "extinction_prior",
        }

        object_array_keys = {
            "true_values",
        }

        # ============================================================
        # Clean saved source code
        # ============================================================
        def _clean_source(
                source,
        ):
            """
            Clean Python source saved by inspect.getsource().
            """
            source = textwrap.dedent(
                source
            ).strip()

            lines = source.splitlines()

            # Handle:
            #
            # prior = lambda x: ...,
            if (
                    len(lines) == 1
                    and "lambda" in lines[0]
                    and lines[0].rstrip().endswith(",")
            ):
                lines[0] = (
                    lines[0]
                    .rstrip()
                    .rstrip(",")
                )

            return "\n".join(
                lines
            )

        # ============================================================
        # Restore prior
        # ============================================================
        def restore_prior(
                name,
                value,
        ):
            """
            Restore one saved prior.
            """
            if value is None:
                return None

            if isinstance(
                    value,
                    str,
            ):
                return value

            if (
                    isinstance(
                        value,
                        dict,
                    )
                    and value.get("type")
                    == "custom_source"
            ):
                source = _clean_source(
                    value["source"]
                )

                func_name = value.get(
                    "name",
                    name,
                )

                # ----------------------------------------------------
                # Reconstruct the default Vrot prior from rough mass.
                # ----------------------------------------------------
                if name == "Vrot_prior":
                    rough_mass = getattr(
                        obj,
                        "rough_mass",
                        None,
                    )

                    # Compatibility with older attribute name.
                    if rough_mass is None:
                        rough_mass = getattr(
                            obj,
                            "rough_mini",
                            None,
                        )

                    if rough_mass is None:
                        return None

                    _, function = (
                        select_Vrot_prior_by_rough_mass(
                            rough_mass,
                            log_prior_Vrot_dict,
                        )
                    )

                    return function

                namespace = {
                    "np": np,
                    "numpy": np,
                    "norm": norm,

                    # Values sometimes used by extinction priors.
                    "extinction": getattr(
                        obj,
                        "extinction",
                        None,
                    ),
                    "extinction_err": getattr(
                        obj,
                        "extinction_err",
                        None,
                    ),

                    # Internal prior helper.
                    "_log_prior_mixture_maxwell_lagged_truncated": (
                        _log_prior_mixture_maxwell_lagged_truncated
                    ),
                }

                exec(
                    source,
                    namespace,
                )

                if func_name in namespace:
                    function = namespace[
                        func_name
                    ]

                elif name in namespace:
                    function = namespace[
                        name
                    ]

                else:
                    raise RuntimeError(
                        f"Could not restore prior `{name}`. "
                        f"Neither `{func_name}` nor `{name}` "
                        "was found after exec()."
                    )

                # Handle accidental:
                #
                # prior = lambda x: ...,
                if isinstance(
                        function,
                        tuple,
                ):
                    if (
                            len(function) == 1
                            and callable(function[0])
                    ):
                        function = function[0]

                    else:
                        raise TypeError(
                            f"Restored prior `{name}` is a tuple "
                            "but not a one-function tuple."
                        )

                if not callable(
                        function
                ):
                    raise TypeError(
                        f"Restored prior `{name}` is not callable. "
                        f"Got {type(function)}."
                    )

                return function

            raise TypeError(
                f"Cannot restore prior `{name}` "
                f"from type {type(value)}."
            )

        # ============================================================
        # Restore ordinary saved attributes
        # ============================================================
        def restore_saved_attributes():
            """
            Restore ordinary attributes, arrays, and actual priors.

            Importantly, ordinary dictionary entries such as
            ``emcee_move_config`` are never sent to restore_prior().
            """

            # --------------------------------------------------------
            # 1. Restore non-prior config attributes
            # --------------------------------------------------------
            for key, value in config.items():
                if key in prior_keys:
                    continue

                if key in object_array_keys:
                    setattr(
                        obj,
                        key,
                        np.array(
                            value,
                            dtype=object,
                        ),
                    )

                else:
                    setattr(
                        obj,
                        key,
                        value,
                    )

            # --------------------------------------------------------
            # 2. Restore numeric arrays stored directly in NPZ
            # --------------------------------------------------------
            for key in data.files:
                if key == "config_json":
                    continue

                setattr(
                    obj,
                    key,
                    data[key],
                )

            # --------------------------------------------------------
            # 3. Restore only explicitly listed prior attributes
            # --------------------------------------------------------
            for key in prior_keys:
                if key not in config:
                    continue

                setattr(
                    obj,
                    key,
                    restore_prior(
                        key,
                        config[key],
                    ),
                )

        # ============================================================
        # Rebuild emcee moves
        # ============================================================
        def build_emcee_moves(
                emcee_move_config,
        ):
            """
            Rebuild emcee move objects from a saved configuration.
            """
            if emcee_move_config is None:
                return None

            if not isinstance(
                    emcee_move_config,
                    dict,
            ):
                raise TypeError(
                    "emcee_move_config must be a dictionary or None."
                )

            emcee_moves = []
            total_fraction = 0.0

            for move_name, move_config in (
                    emcee_move_config.items()
            ):
                if not isinstance(
                        move_config,
                        dict,
                ):
                    raise TypeError(
                        f"Configuration for `{move_name}` "
                        "must be a dictionary."
                    )

                if "fraction" not in move_config:
                    raise KeyError(
                        f"`fraction` is missing for move "
                        f"`{move_name}`."
                    )

                fraction = float(
                    move_config["fraction"]
                )

                if (
                        not np.isfinite(fraction)
                        or fraction <= 0
                ):
                    raise ValueError(
                        f"Invalid fraction for `{move_name}`: "
                        f"{fraction}"
                    )

                total_fraction += fraction

                # ----------------------------------------------------
                # StretchMove
                # ----------------------------------------------------
                if move_name == "StretchMove":
                    if "a" not in move_config:
                        raise KeyError(
                            "`a` is missing for StretchMove."
                        )

                    move = emcee.moves.StretchMove(
                        a=float(
                            move_config["a"]
                        )
                    )

                # ----------------------------------------------------
                # DEMove
                # ----------------------------------------------------
                elif move_name == "DEMove":
                    gamma0 = move_config.get(
                        "gamma0",
                        None,
                    )

                    move = emcee.moves.DEMove(
                        gamma0=(
                            None
                            if gamma0 is None
                            else float(gamma0)
                        )
                    )

                # ----------------------------------------------------
                # DESnookerMove
                # ----------------------------------------------------
                elif move_name == "DESnookerMove":
                    move = (
                        emcee.moves.DESnookerMove()
                    )

                # ----------------------------------------------------
                # KDEMove
                # ----------------------------------------------------
                elif move_name == "KDEMove":
                    move = emcee.moves.KDEMove()

                else:
                    raise ValueError(
                        "Unknown emcee move type: "
                        f"{move_name}"
                    )

                emcee_moves.append(
                    (
                        move,
                        fraction,
                    )
                )

            if not np.isclose(
                    total_fraction,
                    1.0,
                    rtol=0.0,
                    atol=1e-8,
            ):
                raise ValueError(
                    "Saved move fractions must sum to 1. "
                    f"Got {total_fraction:.12g}."
                )

            return emcee_moves

        # ============================================================
        # 1. Restore current_path first
        # ============================================================
        saved_current_path = config.get(
            "current_path",
            None,
        )

        if saved_current_path is not None:
            obj.current_path = str(
                saved_current_path
            )

            if os.path.isdir(
                    obj.current_path
            ):
                os.chdir(
                    obj.current_path
                )

            else:
                raise FileNotFoundError(
                    "The saved current_path does not exist: "
                    f"{obj.current_path}"
                )

        # ============================================================
        # 2. Restore attributes needed before set_method()
        # ============================================================
        pre_set_method_keys = [
            "cluster",
            "sampling_variables_list",
        ]

        for key in pre_set_method_keys:
            if key in config:
                setattr(
                    obj,
                    key,
                    config[key],
                )

        # ============================================================
        # 3. Resolve set_method inputs
        # ============================================================
        saved_grids_path = config.get(
            "grids_path",
            None,
        )

        # Fallback for old target files.
        if saved_grids_path is None:
            saved_grids_path = grids_path

        if saved_grids_path is None:
            raise KeyError(
                "`grids_path` was not found in the saved target "
                "configuration. This is probably an older target file.\n"
                "Provide it explicitly:\n\n"
                "Target.load(filename, grids_path='path/to/grid.npz')"
            )

        required_set_method_keys = [
            "grids_type",
            "METHOD",
        ]

        for key in required_set_method_keys:
            if key not in config:
                raise KeyError(
                    f"`{key}` is required to rebuild runtime "
                    "objects with set_method(), but it was not "
                    "found in the saved config."
                )

        set_method_kwargs = {
            "grids_path": saved_grids_path,
            "grids_type": config["grids_type"],
            "METHOD": config["METHOD"],
            "iso_name": config.get(
                "iso_name",
                "PARSEC",
            ),
            "flag_64": config.get(
                "flag_64",
                False,
            ),

            # PARS settings
            "PARS_limb_darkening_path": config.get(
                "PARS_limb_darkening_path",
                None,
            ),
            "PARS_filters": config.get(
                "PARS_filters",
                None,
            ),
            "PARS_nz": config.get(
                "PARS_nz",
                100,
            ),
            "PARS_MH_offset": config.get(
                "PARS_MH_offset",
                -0.046,
            ),
        }

        # ============================================================
        # 4. Rebuild interpolation/runtime objects
        # ============================================================
        obj.set_method(
            **set_method_kwargs
        )

        # ============================================================
        # 5. Restore saved attributes and priors
        # ============================================================
        restore_saved_attributes()

        # Ensure fallback grid path is retained for old files.
        obj.grids_path = str(
            saved_grids_path
        )

        # ============================================================
        # 6. Rebuild emcee move objects
        # ============================================================
        obj.emcee_move_config = config.get(
            "emcee_move_config",
            None,
        )

        obj.emcee_moves = build_emcee_moves(
            obj.emcee_move_config
        )

        obj.load_flag = str(
            filename
        )

        return obj