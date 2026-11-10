import sys
print(f"Current Python interpreter: {sys.executable}")

import os
import json
import time
import emcee
import numpy as np
import pandas as pd
from scipy.stats import norm
from datetime import datetime

import belfast.mcmc_utils as mcmc
from belfast.target_definition import Target
from belfast.paths import REPO_ROOT, RESULTS_DIR, DATA_DIR


print(f"Script started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

configuration = "mock_tests"
save_path = RESULTS_DIR / configuration
folder_i = 2

while save_path.exists():
    save_path = RESULTS_DIR / f"{configuration}_{folder_i}"
    folder_i += 1

test_save_path = save_path / "mock_test_quick_look"
os.makedirs(test_save_path, exist_ok=True)


df_mock = pd.read_csv(DATA_DIR / 'BELFAST_mock_Mini_Age_grid.csv')


for i, row in df_mock.iloc[::-1].iterrows():

    if row['interpolation_success'] == False:
        continue

    # ========================================================
    # Recover true parameters
    # ========================================================
    true_values = np.array(json.loads(row['true_values']),dtype=float)

    # ========================================================
    # Recover observational settings
    # ========================================================
    noise = True
    bands_list = json.loads(row['bands_list'])
    obs_err_mags = np.array(json.loads(row['obs_err_mags']),dtype=float)
    vsini_obs = float(row['vsini_obs'])
    vsini_obs_err = float(row['vsini_obs_err'])
    parallax_obs = float(row['parallax_obs'])
    parallax_obs_err = float(row['parallax_obs_err'])
    extinction = float(row['extinction'])
    extinction_err = float(row['extinction_err'])
    extinction_type = row['extinction_type']
    interpolation_method = row['interpolation_method']
    MH = float(true_values[2])
    MH_err = 0.05
    inc = np.round(np.rad2deg(true_values[4]), 2)
    inc_err = 6 # deg

    Mini = true_values[0]
    Age = true_values[1]
    if Mini==1.9 or Mini==1.7 or Mini==1.5:
        continue

    # ========================================================
    # Recover final mock observed magnitudes
    #
    # This is the magnitude after adding both extinction
    # and Gaussian observational noise.
    # ========================================================
    if noise == False:
        obs_mags = np.array(json.loads(row['obs_mags_add_extinction']),dtype=float)
        folder_suffix = f'G-Tomo A_V prior; MH prior={MH}±{MH_err}; inc prior={inc}±{inc_err}; no noise'
    else:
        obs_mags = np.array(json.loads(row['obs_mags']), dtype=float)
        folder_suffix = f'G-Tomo A_V prior; MH prior={MH}±{MH_err}; inc prior={inc}±{inc_err}'

    cluster = 'None'
    target_name = f'Mock_{np.round(true_values[0], 1)}Msun_{int(true_values[1])}Myr'

    print('###############################################################################')
    print('###############################################################################')
    print(f'>>>>>>>>>> Now working on Target: {target_name}')


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
                    sampling_variables_list=['Mini', 'Age', 'MH', 'omega_ini', 'Inclination', 'parallax'],
                    extinction=extinction,
                    extinction_err=extinction_err,
                    extinction_type=extinction_type
                    )
    target.known_parameter(true_values=true_values)

    if interpolation_method == 'u':
        grids_type = "u"
    else:
        grids_type = "Mini"

    target.set_method(regrid_type=grids_type, METHOD="DIRECT", label_list=[0, 1])


    target.set_sampling(n_walkers=30,
                        MH_prior = lambda x: norm.logpdf(x, loc=MH, scale=MH_err),
                        inclination_prior=lambda x: norm.logpdf(x, loc=np.deg2rad(inc), scale=np.deg2rad(inc_err)),
                        extinction_prior = lambda x: norm.logpdf(x, loc=extinction, scale=extinction_err),
                        )


    target.emcee_moves=[
        (emcee.moves.StretchMove(a=2.0),  0.50),
        (emcee.moves.DEMove(gamma0=0.4),  0.20),
        (emcee.moves.DESnookerMove(),     0.15),
        (emcee.moves.KDEMove(),           0.15),
    ]

    mcmc.target = target
    start_time = time.time()

    sampler_test, rough_mass = mcmc.run_test_sampling(
        show_progress=False,
        show_fig=False,
        n_steps_test=3000,
        save_path=test_save_path,
    )
    end_time = time.time()
    print(f"This test took {end_time - start_time:.4f} seconds.")

    sampler = mcmc.run_sampling(
        initial_steps = 10000, # the interval for checking convergence (e.g. 10000, 30000, 60000)
        max_hours=3, # if reaching max_hours, stop sampling anyway.
        n_processes=8, # the number of threads used for parallel sampling.
        show_progress=False,
        max_check_steps=100000,
    )

    mcmc.save_and_plot(
        sampler,
        plot_samples=20000,
        show_fig=False,
        save_path=save_path,
        folder_suffix=folder_suffix,
        save_samples=True
    )

    target.check_risk(ignore_out_of_range_mini=True)

    print('**********************************************************************************************************************')

print('###############################################################################')
print('Finished!')