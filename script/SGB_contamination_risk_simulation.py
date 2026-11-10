import sys
print(f"Current Python interpreter: {sys.executable}")

import time
import os
import emcee
import numpy as np
import pandas as pd
from scipy.stats import norm
from datetime import datetime
import matplotlib.pyplot as plt

import belfast.mcmc_utils as mcmc
from belfast.target_definition import Target
from belfast.target_utils import  add_extinction
from belfast.isochrone_utils import read_isochrones, PARSECv2_collect_data
from belfast.paths import REPO_ROOT, RESULTS_DIR, DATA_DIR, ISOCHRONE_DIR


print(f"Script started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

configuration = "SGB_contamination_risk_simulation"
save_path = RESULTS_DIR / configuration
folder_i = 2

while save_path.exists():
    save_path = RESULTS_DIR / f"{configuration}_{folder_i}"
    folder_i += 1

test_save_path = save_path / f"test_sampling"
test_save_path.mkdir(parents=True, exist_ok=True)

# Input PARSEC v2.0 isochrone with user-defined parameters (i.e., MH and rotation rate)
iso_i = PARSECv2_collect_data(
    ISOCHRONE_DIR /
    'PARSEC_v2/1e6_2001e6yr_linear_isochrones_rearranged_2mass_tycho2_gaia_dr3/'
    'omega_ini0.60_linear_age1e6to2001e6d10e6_Z0.02to0.02d0.0_2mass_tycho2_gaia_dr3.dat.gz',
    show_progress=False, label_list=[2])

early_iso_i = iso_i[(iso_i['Mini'] > 1.4) & (iso_i['Mini'] < 5)]

df_subset = early_iso_i.groupby('logAge', group_keys=False).sample(n=1)

df_subset_dropped = (
    df_subset
    .groupby(pd.cut(df_subset['Mini'], bins=np.arange(
        df_subset['Mini'].min(),
        df_subset['Mini'].max() + 0.05,
        0.05
    )), observed=True, group_keys=False)
    .sample(n=1, random_state=42)
    .reset_index(drop=True)
).copy()

df_subset_dropped.to_csv(save_path / 'test_SGB_parameters.csv', index=False)


# Plot a figure to show randomly selected SGB grid points as mock stars
plt.figure(figsize=[8, 8])
plt.scatter(early_iso_i['Mini'], early_iso_i['logAge'], s=10, c='tab:blue', label=f'All SGB grid points (n={len(early_iso_i)})')
plt.scatter(df_subset_dropped['Mini'], df_subset_dropped['logAge'], s=20, marker='x', c='tab:red', label=f'Randomly selected SGB grid points (n={len(df_subset_dropped)})')
plt.xlabel('Mini')
plt.ylabel('logAge')
plt.legend()
plt.savefig(save_path / 'Randomly selected SGB grid points.png', dpi=300)
plt.close()


for idx, row in df_subset_dropped.iterrows():
    df_i = row
    vsini_obs = df_i['vsini']
    vsini_obs_err = 5
    parallax_obs = 10
    parallax_obs_err = 0.2
    extinction = 0.01
    extinction_err = 0.01
    extinction_type = 'A_V'

    target_name = f'Mock star {idx}'
    cluster = ''
    true_list = ['Mini', 'logAge', 'MH', 'omega_ini', 'Inclination']
    true_values = df_i[true_list].values
    true_values[1] = np.round(10 ** (true_values[1] - 6))
    true_values = true_values.tolist() + [parallax_obs]

    bands_list = ['G', 'G_BP', 'G_RP', 'B_T', 'V_T', 'J', 'H', 'Ks']
    abs_mags_model = df_i[bands_list].values
    obs_mags_original = abs_mags_model + np.log10(1000 / parallax_obs) * 2.5
    obs_mags_model_err = np.zeros_like(obs_mags_original)
    obs_mags_add_extinction = add_extinction(
        obs_mags_original,
        obs_mags_model_err,
        bands_list,
        extinction,
        extinction_err,
        extinction_type,
        verbose=False
    )[0]
    obs_mags = obs_mags_add_extinction + np.random.normal(loc=0.0, scale=[0.02] * 8)
    obs_err_mags = np.array([0.002, 0.002, 0.002, 0.02, 0.02, 0.02, 0.02, 0.02])

    print('#'*50)
    print(f'############# Now Working on {target_name} #############')
    print('True values:', true_values)
    print('Observed values:', np.round(obs_mags, 3))


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

    target.set_method(regrid_type='u', METHOD="DIRECT", label_list=[0, 1])

    target.interp_lower_limit[1] = 6

    target.set_sampling(n_walkers=30,
                        MH_prior=lambda x: norm.logpdf(x, loc=true_values[2], scale=0.08),
                        # inclination_prior = lambda x: norm.logpdf(x, loc=np.deg2rad(30), scale=np.deg2rad(6)),
                        extinction_prior=lambda x: norm.logpdf(x, loc=extinction, scale=extinction_err),
                        )

    target.emcee_moves=[
        (emcee.moves.StretchMove(a=2.0),  0.50),
        (emcee.moves.DEMove(gamma0=0.4),  0.20),
        (emcee.moves.DESnookerMove(),     0.15),
        (emcee.moves.KDEMove(),           0.15),
    ]

    mcmc.target = target
    start_time = time.time()

    sampler_test, target = mcmc.run_test_sampling(
        n_steps_test=3000,
        n_processes=8,
        show_fig=False,
        save_path=test_save_path,
        show_progress=False,
    )

    mcmc.target = target
    sampler = mcmc.run_sampling(
        initial_steps = 10000, # the interval for checking convergence (e.g. 10000, 30000, 60000)
        max_hours=3, # hours; if reaching max_hours, stop sampling anyway.
        n_processes=8, # the number of threads used for parallel sampling.
        show_progress=False,
    )

    mcmc.save_and_plot(
        sampler,
        save_path=save_path,

        show_fig=False,
        plot_samples=20000,
        folder_suffix=f'G-Tomo A_V prior, MH prior {np.round(true_values[2], 2)}±{0.08};',
        save_samples=True,
    )

    res = target.check_risk(return_flag=True, ignore_out_of_range_mini=True, show_progress=False)

    p_risk = res['summary']['P_risky']
    print(f' ########## P_risky: {p_risk} ##########')

print('###############################################################################')
print('Finished!')