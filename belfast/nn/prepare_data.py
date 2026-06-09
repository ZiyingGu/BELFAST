import re
import numpy as np
import pyarrow.parquet as pq


# # parallax_obs to parsec
# def cal_parsec(distance_mas):
#     # p is parallax_obs in mas
#     distance_pc = 1000/distance_mas # distance in pc
#     return distance_pc


# Absolute Magnitude and Uncertainty
# For collecting data from isochrones
# def cal_abs_mag(obs_mags, obs_err_mags, parallax_obs, parallax_obs_err):
#     # obs_mag is observed apparent magnitude
#     # distance_mas is the distance in mas
#     abs_mags = obs_mags - 5*np.log10(100/parallax_obs)
#     abs_err_mags = np.sqrt(obs_err_mags**2 + (5*parallax_obs_err/np.log(10)/parallax_obs)**2)
#     return abs_mags.flatten(), abs_err_mags.flatten()


# # For calculating likelihood
# # Observed Magnitude and Uncertainty
# def cal_obs_mag(abs_mags, parallax_obs):
#     # abs_mag is absolute magnitude
#     # distance_mas is the distance in mas
#     obs_mags = abs_mags + 5*np.log10(100/parallax_obs)
#     return obs_mags


#
# # A_lambda/E(G_BP-G_RP) from Wang & Chen 2019
# A_lambda_over_EBP_RP_dic = {
#     'G':    1.890,  'G_err':    0.015,
#     'G_BP': 2.429,  'G_BP_err': 0.015,
#     'G_RP': 1.429,  'G_RP_err': 0.015,
#     'B_T':  3.151,  'B_T_err':  0.027,
#     'V_T':  2.394,  'V_T_err':  0.018,
#     'J':    0.582,  'J_err':    0.011,
#     'H':    0.313,  'H_err':    0.014,
#     'Ks':   0.186,  'Ks_err':   0.009
# }
#
# # A_lambda/A_v from Wang & Chen 2019
# A_lambda_over_A_V_dic = {
#     'G':    0.789,  'G_err':    0.005,
#     'G_BP': 1.002,  'G_BP_err': 0.007,
#     'G_RP': 0.589,  'G_RP_err': 0.004,
#     'B_T':  1.317,  'B_T_err':  0.016,
#     'V_T':  1,      'V_T_err':  0,
#     'J':    0.243,  'J_err':    0.004,
#     'H':    0.131,  'H_err':    0.006,
#     'Ks':   0.078,  'Ks_err':   0.004
# }

#
# # 20250804
# def remove_extinction(obs_mags_no_cor, obs_err_mags_no_cor, bands_list, extinction, extinction_err, extinction_type,
#                       R_V=3.16, R_V_err=0.15, print_flag=True):
#     # (EB_V: B. J. Taylor 2006) and (R_V: Wang & Chen 2019)
#
#     if isinstance(obs_mags_no_cor, list):
#         obs_mags_no_cor = np.array(obs_mags_no_cor)
#     if isinstance(obs_err_mags_no_cor, list):
#         obs_err_mags_no_cor = np.array(obs_err_mags_no_cor)
#
#     correction = []
#     correction_err = []
#
#     if extinction_type in ['A_V', 'EB_V']:
#         if extinction_type == 'A_V':
#             A_V = extinction
#             A_V_err = extinction_err
#         elif extinction_type == 'EB_V':
#             A_V = R_V * extinction
#             A_V_err = np.sqrt((extinction * R_V_err) ** 2 + (extinction_err * R_V) ** 2)
#         for band_lambda in bands_list:
#             if band_lambda not in A_lambda_over_A_V_dic or band_lambda + '_err' not in A_lambda_over_A_V_dic:
#                 raise ValueError(f"Missing extinction coefficients for band: {band_lambda}")
#             A_lambda_over_A_V = A_lambda_over_A_V_dic[band_lambda]
#             A_lambda_over_A_V_err = A_lambda_over_A_V_dic[band_lambda + '_err']
#             correction_lambda = A_lambda_over_A_V * A_V
#             correction_err_lambda = np.sqrt((A_lambda_over_A_V * A_V_err) ** 2 + (A_V * A_lambda_over_A_V_err) ** 2)
#             if print_flag == True:
#                 print(f"A_{band_lambda}: {correction_lambda:.4f} ± {correction_err_lambda:.4f}")
#             correction.append(correction_lambda)
#             correction_err.append(correction_err_lambda)
#
#     elif extinction_type == 'EBP_RP':
#         EBP_RP = extinction
#         EBP_RP_err = extinction_err
#         for band_lambda in bands_list:
#             if band_lambda not in A_lambda_over_EBP_RP_dic or band_lambda + '_err' not in A_lambda_over_EBP_RP_dic:
#                 raise ValueError(f"Missing extinction coefficients for band: {band_lambda}")
#             A_lambda_over_EBP_RP = A_lambda_over_EBP_RP_dic[band_lambda]
#             A_lambda_over_EBP_RP_err = A_lambda_over_EBP_RP_dic[band_lambda + '_err']
#             correction_lambda = A_lambda_over_EBP_RP * EBP_RP
#             correction_err_lambda = np.sqrt((A_lambda_over_EBP_RP * EBP_RP_err) ** 2 + (EBP_RP * A_lambda_over_EBP_RP_err) ** 2)
#             if print_flag == True:
#                 print(f"A_{band_lambda}: {correction_lambda:.4f} ± {correction_err_lambda:.4f}")
#             correction.append(correction_lambda)
#             correction_err.append(correction_err_lambda)
#
#     obs_mags_removed = obs_mags_no_cor - np.array(correction)
#     obs_err_mags_removed = np.sqrt(np.array(obs_err_mags_no_cor) ** 2 + np.array(correction_err) ** 2)
#
#     return obs_mags_removed, obs_err_mags_removed
#
#
# def add_extinction(obs_mags_model, obs_err_mags_model, bands_list, extinction, extinction_err, extinction_type,
#                    R_V=3.16, R_V_err=0.15, print_flag=True):
#     # (EB_V: B. J. Taylor 2006) and (R_V: Wang & Chen 2019)
#
#     if isinstance(obs_mags_model, list):
#         obs_mags_model = np.array(obs_mags_model)
#     if isinstance(obs_err_mags_model, list):
#         obs_err_mags_model = np.array(obs_err_mags_model)
#
#     correction = []
#     correction_err = []
#
#     if extinction_type in ['A_V', 'EB_V']:
#         if extinction_type == 'A_V':
#             A_V = extinction
#             A_V_err = extinction_err
#         elif extinction_type == 'EB_V':
#             A_V = R_V * extinction
#             A_V_err = np.sqrt((extinction * R_V_err) ** 2 + (extinction_err * R_V) ** 2)
#         for band_lambda in bands_list:
#             if band_lambda not in A_lambda_over_A_V_dic or band_lambda + '_err' not in A_lambda_over_A_V_dic:
#                 raise ValueError(f"Missing extinction coefficients for band: {band_lambda}")
#             A_lambda_over_A_V = A_lambda_over_A_V_dic[band_lambda]
#             A_lambda_over_A_V_err = A_lambda_over_A_V_dic[band_lambda + '_err']
#             correction_lambda = A_lambda_over_A_V * A_V
#             correction_err_lambda = np.sqrt((A_lambda_over_A_V * A_V_err) ** 2 + (A_V * A_lambda_over_A_V_err) ** 2)
#             if print_flag == True:
#                 print(f"A_{band_lambda}: {correction_lambda:.4f} ± {correction_err_lambda:.4f}")
#             correction.append(correction_lambda)
#             correction_err.append(correction_err_lambda)
#
#     elif extinction_type == 'EBP_RP':
#         EBP_RP = extinction
#         EBP_RP_err = extinction_err
#         for band_lambda in bands_list:
#             if band_lambda not in A_lambda_over_EBP_RP_dic or band_lambda + '_err' not in A_lambda_over_EBP_RP_dic:
#                 raise ValueError(f"Missing extinction coefficients for band: {band_lambda}")
#             A_lambda_over_EBP_RP = A_lambda_over_EBP_RP_dic[band_lambda]
#             A_lambda_over_EBP_RP_err = A_lambda_over_EBP_RP_dic[band_lambda + '_err']
#             correction_lambda = A_lambda_over_EBP_RP * EBP_RP
#             correction_err_lambda = np.sqrt((A_lambda_over_EBP_RP * EBP_RP_err) ** 2 + (EBP_RP * A_lambda_over_EBP_RP_err) ** 2)
#             if print_flag == True:
#                 print(f"A_{band_lambda}: {correction_lambda:.4f} ± {correction_err_lambda:.4f}")
#             correction.append(correction_lambda)
#             correction_err.append(correction_err_lambda)
#
#     obs_mags_model_extincted = obs_mags_model + np.array(correction)
#     mags_ext_err_model_extincted = np.sqrt(np.array(obs_err_mags_model) ** 2 + np.array(correction_err) ** 2)
#
#     return obs_mags_model_extincted, mags_ext_err_model_extincted


def extract_info(file_training):
    """
    Parses a training filename to extract target information such as cluster name, age type, number of bands, and sigma deviation.
    The filename is expected to match the pattern:
    'NN_{cluster?}_{age_type}_age_deviation{sigma_deviation}_{bands_num}bands_{target_name}.parquet'

    Args:
        file_training (str): The filename of the training data parquet file.

    Returns:
        tuple: (target_name, cluster_name, age_type, bands_num, sigma_deviation)
    """
    pattern = r'^NN_(?:(\w+)_)?([a-z]+)_age_deviation(\d+)_([0-9]+)bands_(.+)\.parquet$'
    match = re.match(pattern, file_training)

    if match:
        cluster = match.group(1) or None  # 如果没有匹配到 cluster，就为 None
        age_type = match.group(2)
        sigma_deviation = int(match.group(3))
        bands_num = int(match.group(4))
        target_name = match.group(5).replace('_', ' ')
        
        print('cluster_name:', cluster)
        print('target_name:', target_name)
        print('age_type:', age_type)
        print('bands_num:', bands_num)
        print('sigma_deviation:', sigma_deviation)
    else:
        raise ValueError('Filename format does not match expected pattern.')

    if cluster is not None:
        cluster_name = '_' + cluster
    else:
        cluster_name = ''
    
    return target_name, cluster_name, age_type, bands_num, sigma_deviation


def read_training_data(file_training, file_target):
    """
    Reads parquet files for training and target data, extracts photometric magnitudes, errors, and other parameters, and returns them.

    Args:
        file_training (str): Filename for the training data parquet file.
        file_target (str): Filename for the target (observation) data parquet file.

    Returns:
        tuple: (table_training, table_real, obs_mags, obs_err_mags, bands_list,
                parallax_obs, parallax_obs_err, extinction, extinction_err, extinction_type)
            - table_training: pandas DataFrame of training data.
            - table_real: pandas DataFrame of target data.
            - obs_mags: numpy array of observed magnitudes.
            - obs_err_mags: numpy array of observed magnitude errors.
            - bands_list: list of photometric band names.
            - parallax_obs: observed parallax value.
            - parallax_obs_err: error in observed parallax.
            - extinction: extinction value.
            - extinction_err: error in extinction.
            - extinction_type: type of extinction (e.g., 'A_V', 'EB_V', 'EBP_RP').
    """

    # Reading the training data
    table_training = pq.read_table('Neural_Network_Model/training_data/' + file_training).to_pandas()
    print("Training data loaded. Shape:", table_training.shape)
    print("Columns of training data:")
    print(list(table_training.columns)[:10])
    print(list(table_training.columns)[10:])

    # Reading the target data
    table_real = pq.read_table('Neural_Network_Model/training_data/' + file_target).to_pandas()
    # print("Target values:")
    # print(table_real[list(table_real.columns)[:6]])
    # print(table_real[list(table_real.columns)[6:16]])
    # print(table_real[list(table_real.columns)[16:]])


    # # Show several begining lines of the training data
    # print(table_training.iloc[:5, :])
    
    bands_cols = list(table_real.columns)[:-7]
    bands_list = bands_cols[::2]
    bands_err_columns = bands_cols[1::2]
    # print('values_list:', values_list)
    # print('error_columns:', error_columns)
    # bands_list = values_list[:-4]
    # bands_err_columns = error_columns[:-4]
    print('*'*100)
    print('bands_list:', bands_list)
    print('bands_err_columns:', bands_err_columns)

    print('*'*100)
    for i in range(len(bands_list)):
        band = bands_list[i]
        err = bands_err_columns[i]
        print(f"Band {band}: {np.round(table_real[band].values[0], 4)} ± {np.round(table_real[err].values[0], 4)}")

    obs_mags = table_real[bands_list].values.flatten()
    obs_err_mags = table_real[bands_err_columns].values.flatten()
    vsini_obs = table_real['vsini_obs'].values[0]
    vsini_obs_err = table_real['vsini_obs_err'].values[0]
    parallax_obs = table_real['parallax_obs'].values[0]
    parallax_obs_err = table_real['parallax_obs_err'].values[0]
    extinction = table_real['extinction'].values[0]
    extinction_err = table_real['extinction_err'].values[0]
    extinction_type = table_real['extinction_type'].values[0] 

    print(f"vsini_obs: {np.round(vsini_obs, 4)} ± {np.round(vsini_obs_err, 4)} km/s")
    print(f"parallax_obs: {np.round(parallax_obs, 4)} ± {np.round(parallax_obs_err, 4)} mas")
    print(f"{extinction_type}: {np.round(extinction, 4)} ± {np.round(extinction_err, 4)}")

    return table_training, table_real, obs_mags, obs_err_mags, bands_list, parallax_obs, parallax_obs_err, extinction, extinction_err, extinction_type
