# import necessary packages
import os
import pandas as pd
import warnings
import numpy as np
from tqdm import tqdm
import pyarrow
import pyarrow.parquet as pq
import re
from glob import glob
from itertools import product
# from lib.isochrone_utils import read_isochrones, PARSECv2_collect_data, MIST_collect_data, make_model
from lib.isochrone_utils import PARSECv2_collect_data, load_regularized_Mini_interpolator, load_regularized_u_interpolator, load_regularized_MIST_interpolator
from lib.target_utils import cal_abs_mag, cal_obs_mag, remove_extinction
from lib.PARS_utils import extract_MH, make_limb_darkening_files


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
        allowed_sigma_adjustment_parameter = {0, 1, 3}
        if sigma_adjustment_parameter not in allowed_sigma_adjustment_parameter:
            raise ValueError(f'"sigma_adjustment_parameter" must be one of {allowed_sigma_adjustment_parameter}, '
                             f'but got {sigma_adjustment_parameter}.')

        if sigma_adjustment_parameter == 3 and extinction_parameter == True:
            raise ValueError('Invalid combination: "sigma_adjustment_parameter=3" cannot be used with "extinction_parameter=True".\n'
                             'The degree of freedom will be ZERO!')

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

        self.cluster = cluster
        if extinction == 0:
            print('Assuming no extinction.')

        # self.collect_variables_list = collect_variables_list
        self.interp_variables_list = interp_variables_list
        self.sampling_variables_list = sampling_variables_list
        self.obs_mags_corrected, self.obs_err_mags_corrected = remove_extinction(self.obs_mags, self.obs_err_mags, self.bands_list,
                                                                                 extinction, extinction_err, extinction_type, print_flag=print_flag)
        self.abs_mags, self.abs_err_mags = cal_abs_mag(self.obs_mags_corrected, self.obs_err_mags_corrected, self.parallax_obs, self.parallax_obs_err,
                                                       max_sigma_adjustment_parameter=max_sigma_adjustment_parameter)


    def known_parameter(self, true_values=None, known_Mini=None, known_logAge=None, known_MH=None,
                        known_omega_ini=None, known_Inclination=None, known_parallax=None):
        """
        Specify known (fixed) parameters for the target.
        Note:
            For parameters [Mini, logAge, MH, omega_ini, Inclination]:
            - If a parameter is not included in `sampling_variables_list`, you must provide its known value here.
            - Otherwise, the method will raise an error.

        Args:
            known_Mini (float): Known stellar mini, if available.
            known_logAge (float): Known log(age), if available.
            known_MH (float): Known metallicity, if available.
            known_omega_ini (float): Known rotation parameter, if available.
            known_Inclination (float): Known inclination, if available.
            known_parallax (float): Known parallax, if available.

        Checks that all non-sampled variables are specified, and stores true parameter values.
        """

        if true_values is not None:
            self.true_values = true_values
            self.known_Mini = true_values[0]
            self.known_logAge = true_values[1]
            self.known_MH = true_values[2]
            self.known_omega_ini = true_values[3]
            self.known_Inclination = true_values[4]
            self.known_parallax = true_values[5]
        else:
            self.known_Mini = known_Mini
            self.known_logAge = known_logAge
            self.known_MH = known_MH
            self.known_omega_ini = known_omega_ini
            self.known_Inclination = known_Inclination
            self.known_parallax = known_parallax
            self.true_values = np.array([known_Mini, known_logAge, known_MH, known_omega_ini, known_Inclination, known_parallax])


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
                   PARS_MH_offset=-0.110,
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
            if PARS_limb_darkening_path is None:
                raise Exception('PARS_limb_darkening_path must be defined.')
            if PARS_filters is None:
                raise Exception('PARS_filters must be defined.')
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


    # def iso_grid_cut(
    #         self,
    #         iso_name=None,
    #         isochrone_path=None,
    #         isos_list=None,
    #         phot_list=None,
    #         print_file_list=False,
    #         use_tqdm=True,
    #         photo_err_scaler=None,
    #         age_type=None,
    #         skip=False,
    #         label_list=None,
    #         remove_young_and_old_flag=True,
    # ):
    #     """
    #     Read isochrone files and collect relevant data for the target.
    #
    #     Parameters
    #     ----------
    #     iso_name : str, optional
    #         Name of isochrone model. Supported: ['PARSECv2', 'MIST'].
    #         If None, will try to use self.iso_name.
    #     isochrone_path : str or list of str, optional
    #         For PARSECv2 only. Path pattern(s) to isochrone files.
    #     isos_list : str or list of str, optional
    #         For MIST only. Path pattern(s) to MIST isochrone files.
    #     phot_list : str or list of str, optional
    #         For MIST only. Path pattern(s) to MIST photometry files.
    #     print_file_list : bool
    #         Whether to print matched file lists.
    #     use_tqdm : bool
    #         Whether to use tqdm.
    #     photo_err_scaler : list or None
    #         Photometric error scaler. Default is [0.1] * 8.
    #     age_type : str, optional
    #         Age type for PARSECv2, e.g. 'log' or 'linear'.
    #         If None, will try to infer from isochrone_path.
    #     skip : bool
    #         For PARSECv2 only.
    #     label_list : list, optional
    #         For PARSECv2 only.
    #     remove_young_and_old_flag : bool
    #         For PARSECv2 only.
    #
    #     Notes
    #     -----
    #     - For PARSECv2, required input:
    #         isochrone_path
    #     - For MIST, required inputs:
    #         isos_list, phot_list
    #     """
    #
    #     iso_name = iso_name.strip()
    #     if iso_name.lower() == "parsec":
    #         iso_name = "PARSECv2"
    #     elif iso_name.lower() == "mist":
    #         iso_name = "MIST"
    #
    #     def _expand_file_list(path_or_list, arg_name):
    #         file_list = []
    #         if isinstance(path_or_list, str):
    #             file_list = glob(path_or_list)
    #         elif isinstance(path_or_list, list):
    #             for path_i in path_or_list:
    #                 file_list += glob(path_i)
    #         else:
    #             raise ValueError(f"'{arg_name}' must be a string or a list of strings.")
    #         file_list.sort()
    #         return file_list
    #
    #     # if photo_err_scaler is None:
    #     #     photo_err_scaler = [0.1] * 8
    #
    #     if iso_name is None:
    #         iso_name = getattr(self, "iso_name", None)
    #
    #     if iso_name is None:
    #         raise ValueError(
    #             "'iso_name' must be provided. Supported values are ['PARSECv2', 'MIST']."
    #         )
    #
    #     self.iso_name = iso_name
    #     self.use_tqdm = use_tqdm
    #     self.skip = skip
    #     self.label_list = label_list
    #     self.remove_young_and_old_flag = remove_young_and_old_flag
    #
    #     if iso_name == "PARSECv2":
    #         if isochrone_path is None:
    #             raise ValueError("For iso_name='PARSECv2', 'isochrone_path' must be provided.")
    #
    #         if age_type is None:
    #             if isinstance(isochrone_path, str):
    #                 path_for_check = isochrone_path
    #             elif isinstance(isochrone_path, list):
    #                 path_for_check = " ".join(isochrone_path)
    #             else:
    #                 raise ValueError("'isochrone_path' must be a string or a list of strings.")
    #
    #             if "linear" in path_for_check:
    #                 age_type = "linear"
    #             elif "log" in path_for_check:
    #                 age_type = "log"
    #             else:
    #                 raise ValueError(
    #                     "'age_type' must be given or shown in isochrone_path "
    #                     "(either 'linear' or 'log')."
    #                 )
    #
    #         isochrone_file_list = _expand_file_list(isochrone_path, "isochrone_path")
    #         self.isochrone_file_list = isochrone_file_list
    #
    #         self.age_type = age_type
    #
    #         if skip == True:
    #             return
    #
    #         if print_file_list:
    #             print("Isochrone files:\n", isochrone_file_list)
    #
    #         print("Reading isochrones and collecting data.")
    #
    #         df_table = PARSECv2_collect_data(
    #             isochrone_file_list,
    #             self.bands_list,
    #             self.abs_mags,
    #             self.abs_err_mags,
    #             use_tqdm=use_tqdm,
    #             label_list=self.label_list,
    #             remove_young_and_old_flag=self.remove_young_and_old_flag,
    #             photo_err_scaler=photo_err_scaler,
    #         )
    #
    #         self.df_table = df_table
    #
    #         print(f"Totally we got {len(df_table)} datapoints.")
    #         print(f'{"":<12} {"Min":<10} {"Max":<10}')
    #         print("-" * 34)
    #         for col in ["Mini", "logAge", "MH", "omega_ini", "Inclination"]:
    #             min_val = np.min(df_table[col].values)
    #             max_val = np.max(df_table[col].values)
    #             print(f"{col:<12} {min_val:<10.4f} {max_val:<10.4f}")
    #
    #     elif iso_name == "MIST":
    #         if isos_list is None:
    #             raise ValueError("For iso_name='MIST', 'isos_list' must be provided.")
    #         if phot_list is None:
    #             raise ValueError("For iso_name='MIST', 'phot_list' must be provided.")
    #
    #         isos_file_list = _expand_file_list(isos_list, "isos_list")
    #         phot_file_list = _expand_file_list(phot_list, "phot_list")
    #
    #         self.isos_file_list = isos_file_list
    #         self.phot_file_list = phot_file_list
    #
    #         if print_file_list:
    #             print("Isochrone files:\n", isos_file_list)
    #             print("Photometry files:\n", phot_file_list)
    #
    #         print("Reading isochrones and collecting data.")
    #
    #         df_table_original = MIST_collect_data(
    #             isos_file_list,
    #             phot_file_list,
    #             self.bands_list,
    #             self.abs_mags,
    #             self.abs_err_mags,
    #             use_tqdm=use_tqdm,
    #             photo_err_scaler=photo_err_scaler,
    #         )
    #
    #         df_table = pd.DataFrame()
    #         df_table[["Mini", "logAge", "MH", "omega_ini"]] = df_table_original[
    #             ["star_mini", "log10_isochrone_age_yr", "[Fe/H]", "surf_avg_v_div_v_crit"]
    #         ]
    #         df_table["logL"] = df_table_original["log_L"]
    #         df_table["Req"] = 10 ** df_table_original["log_R"]
    #         df_table["Rpol"] = 10 ** df_table_original["log_R"]
    #         df_table[
    #             ["J", "H", "Ks", "B_T", "V_T", "G", "G_BP", "G_RP"]
    #         ] = df_table_original[
    #             ["2MASS_J", "2MASS_H", "2MASS_Ks", "Tycho_B", "Tycho_V",
    #              "Gaia_G_EDR3", "Gaia_BP_EDR3", "Gaia_RP_EDR3"]
    #         ]
    #
    #         self.df_table = df_table
    #
    #         print(f"Totally we got {len(df_table)} datapoints.")
    #         print(f'{"":<12} {"Min":<10} {"Max":<10}')
    #         print("-" * 34)
    #         for col in ["Mini", "logAge", "MH", "omega_ini"]:
    #             min_val = np.min(df_table[col].values)
    #             max_val = np.max(df_table[col].values)
    #             print(f"{col:<12} {min_val:<10.4f} {max_val:<10.4f}")
    #
    #     else:
    #         raise ValueError(
    #             f"Unsupported iso_name: {iso_name}. Supported values are ['PARSECv2', 'MIST']."
    #         )

    # def PARSECv2_make_table(self, isochrone_path, print_isochrone_file_list=False, use_tqdm=True, photo_err_scaler=None,
    #                           age_type=None, skip=False, label_list=None, remove_young_and_old_flag=True):
    #     """
    #     Read isochrone files and collect relevant data for the target.
    #
    #     Args:
    #         isochrone_path (str or list): Path pattern to isochrone files.
    #         age_type (str): Type of age (e.g., 'log' or 'linear').
    #         print_isochrone_file_list (bool): Whether to print found isochrone files.
    #         photo_err_scaler (float): Sigma deviation for photometry selection.
    #         select_rotation (bool): Whether to select rotation (drop non-rotation rows).
    #         skip (bool): Skip collecting and collect data for neural network training later.
    #         label_list (list): List of labels to keep when removing young and old stars, (e.g., [0, 1, 2]).
    #         remove_young_and_old_flag (bool): Remove very young/old stars, keeping only labels in [1.0, 2.0, 3.0].
    #
    #     Reads isochrone tables, collects data, and stores as DataFrame for later modeling.
    #     """
    #
    #     # if photo_err_scaler is None:
    #     #     photo_err_scaler = [0.1] * 8
    #
    #     if age_type is None:
    #         if 'linear' in isochrone_path:
    #             age_type = 'linear'
    #         elif 'log' in isochrone_path:
    #             age_type = 'log'
    #         else:
    #             raise ValueError("'age_type' must be given or shown in isochrone_path (either 'linear' or 'log').")
    #
    #     isochrone_file_list=[]
    #     if isinstance(isochrone_path, str):
    #         isochrone_file_list = glob(isochrone_path)
    #     elif isinstance(isochrone_path, list):
    #         for path_i in isochrone_path:
    #             isochrone_file_list += glob(path_i)
    #     else:
    #         raise ValueError("'isochrone_path' must be a string or a list of strings.")
    #     isochrone_file_list.sort()
    #     self.isochrone_file_list = isochrone_file_list
    #     self.age_type = age_type
    #     self.skip = skip
    #     self.label_list = label_list
    #     self.remove_young_and_old_flag = remove_young_and_old_flag
    #     if print_isochrone_file_list:
    #         print('Isochrone files:\n', isochrone_file_list)
    #
    #     print('Reading isochrones and collecting data.')
    #     # df_table_list = []
    #
    #     df_table = PARSECv2_collect_data(isochrone_file_list, self.bands_list, self.abs_mags, self.abs_err_mags,
    #                                      use_tqdm=use_tqdm, label_list=self.label_list,
    #                                      remove_young_and_old_flag=self.remove_young_and_old_flag,
    #                                      photo_err_scaler=photo_err_scaler)
    #
    #     self.df_table = df_table
    #
    #     print(f'Totally we got {len(df_table)} datapoints.')
    #     print(f'{"":<12} {"Min":<10} {"Max":<10}')
    #     for col in ['Mini', 'logAge', 'MH', 'omega_ini', 'Inclination']:
    #         min_val = np.min(df_table[col].values)
    #         max_val = np.max(df_table[col].values)
    #         print(f'{col:<12} {min_val:<10.4f} {max_val:<10.4f}')



    # def MIST_make_table(self, isos_list, phot_list, print_file_list=False, use_tqdm=True, photo_err_scaler=None):
    #
    #     # if photo_err_scaler is None:
    #     #     photo_err_scaler = [0.1] * 8
    #
    #     isos_file_list=[]
    #     if isinstance(isos_list, str):
    #         isos_file_list = glob(isos_list)
    #     elif isinstance(isos_list, list):
    #         for path_i in isos_list:
    #             isos_file_list += glob(path_i)
    #     else:
    #         raise ValueError("'isochrone_path' must be a string or a list of strings.")
    #     isos_file_list.sort()
    #     self.isos_file_list = isos_file_list
    #
    #     phot_file_list=[]
    #     if isinstance(phot_list, str):
    #         phot_file_list = glob(phot_list)
    #     elif isinstance(phot_list, list):
    #         for path_i in phot_list:
    #             phot_file_list += glob(path_i)
    #     else:
    #         raise ValueError("'isochrone_path' must be a string or a list of strings.")
    #     phot_file_list.sort()
    #     self.phot_file_list = phot_file_list
    #
    #     if print_file_list:
    #         print('Isochrone files:\n', isos_file_list)
    #         print('Photometry files:\n', phot_file_list)
    #
    #     print('Reading isochrones and collecting data.')
    #     # df_table_list = []
    #
    #     df_table_original = MIST_collect_data(isos_file_list, phot_file_list, self.bands_list, self.abs_mags, self.abs_err_mags,
    #                                           use_tqdm=use_tqdm, photo_err_scaler=photo_err_scaler)
    #
    #     df_table = pd.DataFrame()
    #     df_table[['Mini', 'logAge', 'MH', 'omega_ini']] = df_table_original[['star_mini', 'log10_isochrone_age_yr', '[Fe/H]', 'surf_avg_v_div_v_crit']]
    #     df_table['logL'] = df_table_original['log_L']
    #     df_table['Req'] = 10 ** df_table_original['log_R']
    #     df_table['Rpol'] = 10 ** df_table_original['log_R']
    #     df_table[['J', 'H', 'Ks', 'B_T', 'V_T', 'G', 'G_BP', 'G_RP']] = df_table_original[['2MASS_J', '2MASS_H', '2MASS_Ks', 'Tycho_B', 'Tycho_V', 'Gaia_G_EDR3', 'Gaia_BP_EDR3', 'Gaia_RP_EDR3']]
    #
    #     self.df_table = df_table
    #
    #     print(f'Totally we got {len(df_table)} datapoints.')
    #     print(f'{"":<12} {"Min":<10} {"Max":<10}')
    #     for col in ['Mini', 'logAge', 'MH', 'omega_ini']:
    #         min_val = np.min(df_table[col].values)
    #         max_val = np.max(df_table[col].values)
    #         print(f'{col:<12} {min_val:<10.4f} {max_val:<10.4f}')

    #
    # def set_method(self,
    #                METHOD,
    #                use_tqdm = True,
    #
    #                # INTERP setting
    #                fewer_INTERP_cut_inclination = None,
    #                fewer_INTERP_cut_MH = 2,
    #                fewer_INTERP_cut_logAge = 1,
    #
    #                # REGULAR
    #                REGULAR_grids_folder_list=None,
    #                pattern="resampled_*.txt.gz",
    #                sep=r"\s+",
    #
    #                # PARS setting
    #                PARS_limb_darkening_path = None,
    #                PARS_filters = None,
    #                PARS_nz = 100,
    #                PARS_MH_offset = -0.110,
    #
    #                # NN setting
    #                NN_prepare_data = False,
    #                NN_photo_err_scaler = 40,
    #                ):
    #     """
    #     Set the modeling method and any method-specific settings for the target.
    #
    #     Args:
    #         METHOD (str): The method to use ('fewer-INTERP', 'PARS', or 'NN').
    #         fewer_INTERP_cut_inclination (list): Inclination cuts for interpolation.
    #         fewer_INTERP_cut_MH (int): Metallicity cut step for interpolation.
    #         fewer_INTERP_cut_logAge (int): logAge cut step for interpolation.
    #         PARS_limb_darkening_path (str): Path to limb darkening files for PARS.
    #         PARS_filters (str): Filters for PARS model to generate magnitudes.
    #         PARS_nz (int): number of z coordinates for PARS in the upper hemisphere (normally 100).
    #         PARS_MH_offset (float): Metallicity offset for PARS (default = -0.1).
    #         NN_prepare_data (bool): Wehether to prepare and save data for NN training.
    #         NN_photo_err_scaler (int): Sigma deviation for NN photometry.
    #         NN_select_rotation (bool): Whether to select rotation (drop non-rotation rows) for the data for training NN.
    #
    #     Selects and configures the method (interpolation, PARS, or neural network) for later inference.
    #     """
    #     self.METHOD = METHOD
    #     if self.cluster != '':
    #         self.cluster_name = '_' + self.cluster
    #     else:
    #         self.cluster_name = self.cluster
    #
    #     self.NN_photo_err_scaler = NN_photo_err_scaler
    #
    #     # INTERP method
    #     if METHOD == 'REGULAR':
    #         # self.METHOD = 'REGULAR'
    #         # self.REGULAR_grids_folder_list = REGULAR_grids_folder_list
    #         # if isinstance(self.REGULAR_grids_folder_list, str):
    #         #     self.df_for_interp = load_and_merge_resampled_files(REGULAR_grids_folder_list, pattern=pattern, sep=sep)
    #         # elif isinstance(self.REGULAR_grids_folder_list, list):
    #         #     df_list = []
    #         #     for folder_i in self.REGULAR_grids_folder_list:
    #         #         df_list.append(load_and_merge_resampled_files(folder_i, pattern=pattern, sep=sep))
    #         #     self.df_for_interp = pd.concat(df_list)
    #         # else:
    #         #     raise TypeError("'REGULAR_grids_folder_list' must be a list or str.")
    #
    #         self.METHOD = 'REGULAR'
    #         self.REGULAR_grids_folder_list = REGULAR_grids_folder_list
    #         if isinstance(self.REGULAR_grids_folder_list, str):
    #             self.df_for_interp = load_resampled_parquet(REGULAR_grids_folder_list)
    #         elif isinstance(self.REGULAR_grids_folder_list, list):
    #             df_list = []
    #             for folder_i in self.REGULAR_grids_folder_list:
    #                 df_list.append(load_resampled_parquet(folder_i))
    #             self.df_for_interp = pd.concat(df_list)
    #         else:
    #             raise TypeError("'REGULAR_grids_folder_list' must be a list or str.")
    #
    #     elif METHOD == 'fewer-INTERP':
    #         if fewer_INTERP_cut_inclination is None:
    #             fewer_INTERP_cut_inclination = [0.0, 5.0, 10.0, 15, 20.0, 30.0, 35.0, 45.0, 55.0, 65.0, 90.0]
    #
    #         df_table_cut_inclination = self.df_table[self.df_table["Inclination"].isin(fewer_INTERP_cut_inclination)]
    #         print('length of df_table_cut_inc: ', len(df_table_cut_inclination))
    #         MH_values = np.unique(df_table_cut_inclination['MH'])[::fewer_INTERP_cut_MH]
    #         df_table_cut_mh = df_table_cut_inclination[df_table_cut_inclination["MH"].isin(MH_values)]
    #         print('length of df_table_cut_mh: ', len(df_table_cut_mh))
    #         logAge_values = np.unique(df_table_cut_mh['logAge'])[::fewer_INTERP_cut_logAge]
    #         df_table_cut_logAge = df_table_cut_mh[df_table_cut_mh["logAge"].isin(logAge_values)]
    #         print('length of df_table_cut_logAge: ', len(df_table_cut_logAge))
    #         df_for_interp = df_table_cut_logAge
    #
    #         print(f'Totally we got {len(df_for_interp)} datapoints for interpolation.')
    #         print(f'{"":<12} {"Min":<10} {"Max":<10}')
    #         for col in ['Mini', 'logAge', 'MH', 'omega_ini', 'Inclination']:
    #             min_val = np.min(df_for_interp[col].values)
    #             max_val = np.max(df_for_interp[col].values)
    #             print(f'{col:<12} {min_val:<10.4f} {max_val:<10.4f}')
    #
    #         if len(df_for_interp) > 8000:
    #             warnings.warn(
    #                 'Too many datapoints (> 8000) for interpolation.\n'
    #                 '→ This may take ~20 min to build.\n'
    #                 '→ Sampling speed will drop to ~5 iterations/s.',
    #                 UserWarning
    #             )
    #         self.df_for_interp = df_for_interp
    #
    #     # PARS method
    #     elif METHOD == 'PARS':
    #         self.df_for_interp = self.df_table
    #         PARS_limb_darkening_filelist = glob(PARS_limb_darkening_path)
    #         PARS_limb_darkening_filelist.sort()
    #         limb_darkening_MH_dic = {file: extract_MH(file) for file in PARS_limb_darkening_filelist if extract_MH(file) is not None}
    #         print("Loading limb darkening files from Kurucz's model.")
    #         PARS_MH_lower_limit = self.df_table.min()['MH'] + PARS_MH_offset
    #         PARS_MH_upper_limit = self.df_table.max()['MH'] + PARS_MH_offset
    #         made_limb_darkening_dict = make_limb_darkening_files(PARS_MH_lower_limit, PARS_MH_upper_limit, limb_darkening_MH_dic, PARS_filters, use_tqdm=use_tqdm)
    #         self.made_limb_darkening_dict = made_limb_darkening_dict
    #         self.PARS_nz = PARS_nz
    #         self.PARS_MH_offset = PARS_MH_offset
    #
    #     # NN method
    #     # NOT collecting data for training a neural network
    #     elif METHOD == 'NN' and NN_prepare_data == False:
    #         if self.skip == False:
    #             self.df_for_interp = self.df_table
    #         print('You have chose not to save data for training a nerual network.')
    #         print('Please input your trained NN model by Target.set_NN() ')
    #     # collecting data for training a neural network
    #     elif METHOD == 'NN' and NN_prepare_data == True:
    #         if self.skip == False:
    #             self.df_for_interp = self.df_table
    #         print('Reading isochrones and collecting data for training a neural network.')
    #         # df_table_NN_list = []
    #
    #         # if use_tqdm == True:
    #         #     iterable = tqdm(self.isochrone_file_list)
    #         # else:
    #         #     iterable = self.isochrone_file_list
    #         # for iso_NN_i in iterable:
    #         #     # df_iso_NN_list = read_isochrones(iso_NN_i, use_tqdm=False)
    #         df_NN_table = PARSECv2_collect_data(self.isochrone_file_list, self.bands_list, self.abs_mags, self.abs_err_mags,
    #                                               use_tqdm=self.use_tqdm, label_list=self.label_list,
    #                                               remove_young_and_old_flag=self.remove_young_and_old_flag,
    #                                               photo_err_scaler=self.NN_photo_err_scaler)
    #             # df_table_NN_list.append(df_table_NN_i)
    #         # df_NN_table = pd.concat(df_table_NN_i, ignore_index=True)
    #         self.df_NN_table = df_NN_table
    #         # save DataFrame for training NN
    #         df_NN_target_col_a = self.bands_list + ['vsini_obs', 'parallax_obs']
    #         errs_list = [c + '_err' for c in self.bands_list]
    #         df_NN_target_col_b = errs_list + ['vsini_obs_err', 'parallax_obs_err']
    #         df_NN_target_col = [item for pair in zip(df_NN_target_col_a, df_NN_target_col_b) for item in pair]
    #         df_NN_target = pd.DataFrame(columns=df_NN_target_col)
    #         df_NN_target.loc[0, self.bands_list] = self.obs_mags
    #         df_NN_target.loc[0, errs_list] = self.obs_err_mags
    #         df_NN_target.loc[0, 'vsini_obs'] = self.vsini_obs
    #         df_NN_target.loc[0, 'vsini_obs_err'] = self.vsini_obs_err
    #         df_NN_target.loc[0, 'parallax_obs'] = self.parallax_obs
    #         df_NN_target.loc[0, 'parallax_obs_err'] = self.parallax_obs_err
    #         df_NN_target.loc[0, 'extinction'] = self.extinction
    #         df_NN_target.loc[0, 'extinction_err'] = self.extinction_err
    #         df_NN_target.loc[0, 'extinction_type'] = self.extinction_type
    #         numeric_cols = df_NN_target.columns.difference(['extinction_type'])  # 假设只有这一列是字符串
    #         df_NN_target[numeric_cols] = df_NN_target[numeric_cols].astype('float64')
    #         df_NN_target_float = df_NN_target.copy()
    #         df_NN_table_float = df_NN_table.astype('float64')
    #         schema_training = pyarrow.schema([pyarrow.field(col, pyarrow.float64()) for col in df_NN_table_float.columns])
    #         # schema_real = pyarrow.schema([pyarrow.field(col, pyarrow.float64()) for col in df_NN_target_float.columns])
    #         schema_real = pyarrow.schema([
    #             pyarrow.field(col, pyarrow.float64()) if pd.api.types.is_float_dtype(df_NN_target_float[col])
    #             else pyarrow.field(col, pyarrow.string()) if pd.api.types.is_string_dtype(df_NN_target_float[col])
    #             else pyarrow.field(col, pyarrow.int64()) if pd.api.types.is_integer_dtype(df_NN_target_float[col])
    #             else pyarrow.field(col, pyarrow.string())  # fallback
    #             for col in df_NN_target_float.columns
    #         ])
    #         table_training = pyarrow.Table.from_pandas(df_NN_table_float, schema=schema_training, preserve_index=False)
    #         table_real = pyarrow.Table.from_pandas(df_NN_target_float, schema=schema_real, preserve_index=False)
    #
    #         saved_NN_data_name = f'Neural_Network_Model/training_data/NN{self.cluster_name}_{self.age_type}_age_deviation{self.NN_photo_err_scaler}_' \
    #                              f'{int(len(self.bands_list))}bands_{self.name.replace(" ", "_")}.parquet'
    #         saved_NN_target_name = f'Neural_Network_Model/training_data/NN{self.cluster_name}_target_{self.name.replace(" ", "_")}.parquet'
    #         pq.write_table(table_training, saved_NN_data_name)
    #         pq.write_table(table_real, saved_NN_target_name)
    #         print(f'Your data for training a nerual network are saved at:\n {os.getcwd()+"/"+saved_NN_data_name} \n'
    #               f'The observed values of your target are saved at:\n {os.getcwd()+"/"+saved_NN_target_name} \n'
    #               'Please input your trained NN model by Target.set_NN() ')
    #         print('************************************************************************')
    #         print(f'Totally we got {len(df_NN_table)} datapoints for training a nerual network.')
    #         print(f'{"":<12} {"Min":<10} {"Max":<10}')
    #         for NN_col in ['Mini', 'logAge', 'MH', 'omega_ini', 'Inclination']:
    #             min_NN_val = np.min(df_NN_table[NN_col].values)
    #             max_NN_val = np.max(df_NN_table[NN_col].values)
    #             print(f'{NN_col:<12} {min_NN_val:<10.4f} {max_NN_val:<10.4f}')
    #     else:
    #         raise ValueError('METHOD must be any of "fewer-INTERP", "PARS", or "NN".')
    #
    #
    #
    #
    # def set_NN(self, NN_model_file=None, NN_model=None, NN_scalers=None, input_num=6):
    #     """
    #     Set or load the neural network model and scalers for inference.
    #
    #     Args:
    #         NN_model_file (str): File path to load NN model from.
    #         NN_model: Pre-loaded NN model object, if not giving NN_model_file
    #         NN_scalers: Pre-loaded NN scalers object, if not giving NN_model_file
    #         input_num (int): Number of inputs for the NN model.
    #
    #     Loads or assigns the NN model and scalers for use with the target.
    #     """
    #     if NN_model is None and NN_scalers is None:
    #         from Neural_Network_Model.mlp_regressor import MLPRegressor
    #         import torch
    #         if NN_model_file is None:
    #             NN_model_file = f"Neural_Network_Model/trained_model/NN_model_{self.name.replace(' ', '_')}{self.cluster_name}_{self.age_type}_age_deviation" \
    #                             f"{self.NN_photo_err_scaler}_input{int(input_num)}_output{int(len(self.bands_list))}.pt"
    #         checkpoint = torch.load(NN_model_file, map_location='cpu', weights_only=False)
    #
    #         self.NN_model = MLPRegressor(activation='swish', input_dim=6)
    #         self.NN_model.load_state_dict(checkpoint['model_state_dict'])
    #         self.NN_scalers = checkpoint['scalers']
    #     elif NN_model is not None and NN_scalers is not None:
    #         self.NN_model = NN_model
    #         self.NN_scalers = NN_scalers
    #     else:
    #         raise ValueError('You should give "NN_model" and "NN_scalers" at the same time.')
    #
    #
    # def build_interp_model(self, drop_columns=True, rescale=True):
    #     """
    #     Build the interpolation model for the target using the selected method.
    #
    #     Args:
    #         drop_columns (bool): Whether to drop duplicated rows by interp_variables_list.
    #
    #     Constructs the interpolation model and determines parameter limits for sampling.
    #     """
    #     print(f'You are using the {self.METHOD} method.')
    #     if self.METHOD in ['fewer-INTERP', 'REGULAR']:
    #         if 'Inclination' not in self.interp_variables_list:
    #             self.interp_variables_list.append('Inclination')
    #     elif self.METHOD in ['NN', 'PARS']:
    #         if 'Inclination' in self.interp_variables_list:
    #             self.interp_variables_list.remove('Inclination')
    #
    #     fixed_parameter_indices = []
    #     for interp_var in self.interp_variables_list:
    #         if interp_var not in self.sampling_variables_list:
    #             fixed_parameter_index = self.interp_variables_list.index(interp_var)
    #             fixed_parameter_indices.append(fixed_parameter_index)
    #     self.fixed_parameter_indices = fixed_parameter_indices
    #
    #
    #     self.Model_interp, self.df_sampling, self.interp_upper_limit, self.interp_lower_limit\
    #         = make_model(self.df_for_interp, self.interp_variables_list, self.bands_list, self.METHOD, drop_columns=drop_columns, rescale=rescale)


    def set_sampling(self,
                     nwalkers=30,
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
            nwalkers (int): Number of walkers for MCMC.
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
        self.rough_mini = None

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

        initial_states = np.random.rand(nwalkers, ndim) * value_differences + np.array(self.sampling_lower_limit)

        self.nwalkers = nwalkers
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
