import sys
print(f"Current Python interpreter: {sys.executable}")

from belfast.target_definition import Target
# from belfast.target_utils import query_data
import belfast.mcmc_utils as mcmc
import numpy as np
import emcee
from scipy.stats import norm
# import re
import os
# from glob import glob
from datetime import datetime
print(f"Script started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

import pandas as pd
import time



df_cluster = pd.read_csv(f"../data/clusters/targets_for_sampling.csv")



# finished_paths = glob('../results/*PARS*')
# finished_stars = [p.split('/')[-1].split('_' + cluster)[0].replace('_', ' ')for p in finished_paths]



i_pleiades, i_praesepe, i_hyades = 0, 0, 0

for i in range(len(df_cluster)):
    df_i = df_cluster.iloc[i]

    cluster = df_i['Cluster']

    # if cluster != 'Pleiades':
    #     continue

    if cluster == 'Hyades':
        MH = 0.090 # [Fe/H] = 0.11 from Carrera & Pancino 2011, corrected the difference of zeropoint between Caffau+2011
        MH_err = 0.079
        # np.sqrt(0.01**2 + 0.03**2 + 0.04**2 + 0.06**2) = 0.079
        # errors from: statistical and systematic of Carrera & Pancino 2011; solar abundance from Grevesse+1996 and Caffau+2011
        if i_hyades == 0:
            test_save_path = f"../results/{cluster}_test_sampling"
            os.makedirs(test_save_path, exist_ok=True)
            print('###############################################################################')
            print(f'******************************* {cluster} *************************************')
            print('###############################################################################')
        i_hyades += 1

    elif cluster == 'Praesepe':
        MH = 0.140 # [Fe/H] = 0.16 from Carrera & Pancino 2011, corrected the difference of zeropoint between Caffau+2011
        MH_err = 0.093
        # np.sqrt(0.05**2 + 0.03**2 + 0.04**2 + 0.06**2) = 0.093
        # errors from: statistical and systematic of Carrera & Pancino 2011; solar abundance from Grevesse+1996 and Caffau+2011
        if i_praesepe == 0:
            test_save_path = f"../results/{cluster}_test_sampling"
            os.makedirs(test_save_path, exist_ok=True)
            print('###############################################################################')
            print(f'******************************* {cluster} *************************************')
            print('###############################################################################')
        i_praesepe += 1

    elif cluster == 'Pleiades':
        MH = 0.010 # [Fe/H] = 0.03 from Soderblom+2009, corrected the difference of zeropoint between Caffau+2011
        MH_err = 0.081
        # np.sqrt(0.03**2 + 0.02**2 + 0.04**2 + 0.06**2) = 0.081
        # errors from: statistical and systematic of Soderblom+2009; solar abundance from Grevesse+1996 and Caffau+2011 (0.04 is additional added to align with the Hyades and  Praesepe.)
        if i_pleiades == 0:
            test_save_path = f"../results/{cluster}_test_sampling"
            os.makedirs(test_save_path, exist_ok=True)
            print('###############################################################################')
            print(f'******************************* {cluster} *************************************')
            print('###############################################################################')
        i_pleiades += 1

    else:
        raise ValueError('"cluster" must be either "Hyades", or "Praesepe", or "Pleiades".')

    target_name = df_i['HIP ID']
    if isinstance(target_name, str):
        pass
    elif not np.isnan(target_name) and isinstance(target_name, (int, float)):
        target_name = 'HIP ' + str(int(target_name))
    elif np.isnan(target_name):
        target_name = df_i['HD ID']
        if isinstance(target_name, str):
            pass
        elif not np.isnan(target_name) and isinstance(target_name, (int, float)):
            target_name = 'HD ' + str(int(target_name))
        elif np.isnan(target_name):
            target_name = df_i['Other Name']
            if isinstance(target_name, str):
                pass
            elif isinstance(target_name, (int, float)):
                target_name = str(int(target_name))

    # if target_name in finished_stars:
    #     continue

    print(f'>>>>>>>>>> Now working on Target: {target_name}')

    extinction = df_i['A550']
    extinction_err = df_i['A550_err']
    extinction_type = 'A_V'

    bands_list = ['G', 'G_BP', 'G_RP', 'B_T', 'V_T', 'J', 'H', 'Ks']
    band_mags_list = [band + '_mag' for band in bands_list]
    errors_list = [band + '_err' for band in bands_list]
    obs_mags = df_i[band_mags_list].values
    obs_err_mags = df_i[errors_list].values
    obs_mags = np.asarray(obs_mags, dtype=float)
    obs_err_mags = np.asarray(obs_err_mags, dtype=float)
    vsini_obs = df_i['vsini']
    vsini_obs_err = df_i['vsini_err']
    parallax_obs = df_i['parallax']
    parallax_obs_err = df_i['parallax_err']


    target = Target(name=target_name,
                    bands_list=['G', 'G_BP', 'G_RP', 'B_T', 'V_T', 'J', 'H', 'Ks'],
                    obs_mags=obs_mags,
                    obs_err_mags=obs_err_mags,
                    vsini_obs=vsini_obs,
                    vsini_obs_err=vsini_obs_err,
                    parallax_obs=parallax_obs,
                    parallax_obs_err=parallax_obs_err,
                    sigma_adjustment_parameter=1,
                    extinction_parameter=True,
                    true_values=False,
                    )

    target.add_info(cluster=cluster,
                    sampling_variables_list=['Mini', 'logAge', 'MH', 'omega_ini', 'Inclination', 'parallax'],
                    extinction=extinction,
                    extinction_err=extinction_err,
                    extinction_type=extinction_type
                    )
    target.known_parameter(true_values=np.full(6, None))

    # using regular PARSEC v2.0 isochrones. For regularization, please see preparation_tutorial.ipynb

    if target_name in ['HIP 42485', 'HIP 42578']:
        target.set_method(
            grids_path='../data/isochrones/regular_u_grids/regularized_u_grid_1e6_2000e6yr_step10e6_linear_isochrones_MS_with_inc.npz',
            grids_type='u',
            METHOD='DIRECT',
        )
    else:
        target.set_method(
            grids_path='../data/isochrones/regular_Mini_grids/regularized_Mini_grid_1e6_2000e6yr_step10e6_linear_isochrones_preMS&MS_with_inc.npz',
            grids_type='Mini',
            METHOD='DIRECT',
        )

    if cluster != 'Pleiades':
        target.interp_lower_limit[1] = np.log10(100)+6

    target.set_sampling(n_walkers=30,
                        MH_prior = lambda x: norm.logpdf(x, loc=MH, scale=MH_err),
                        extinction_prior = lambda x: norm.logpdf(x, loc=extinction, scale=extinction_err),
                        )

    mcmc.target = target
    start_time = time.time()

    emcee_moves=[
        (emcee.moves.StretchMove(a=2.0),  0.50),
        (emcee.moves.DEMove(gamma0=0.4),  0.20),
        (emcee.moves.DESnookerMove(),     0.15),
        (emcee.moves.KDEMove(),           0.15),
    ]

    sampler_test = mcmc.run_test_sampling(use_tqdm=False, show_fig=False, n_steps_test=3000,
                                                      save_path=test_save_path,
                                                      emcee_moves=emcee_moves,
    )
    end_time = time.time()
    print(f"This test took {end_time - start_time:.4f} seconds.")

    sampler = mcmc.run_sampling(initial_steps = 10000, # the interval for checking convergence (e.g. 10000, 30000, 60000)
                                max_hours=6, # if reaching max_hours, stop sampling anyway.
                                n_processes=8, # the number of threads used for parallel sampling.
                                use_tqdm=False,
                                emcee_moves = emcee_moves,
                                )

    mcmc.save_chains_plot(sampler, show_fig=False, folder_suffix=f'G-Tomo A_V prior, MH prior {MH}±{MH_err}', save_samples=True)

    print('**********************************************************************************************************************')
