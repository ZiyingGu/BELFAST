import numpy as np
import pandas as pd
from scipy.special import logsumexp
import emcee


# ============================================================
# Basic helper
# ============================================================

def distance_modulus_from_parallax(parallax_mas):
    return 10.0 - 5.0 * np.log10(parallax_mas)


def gaussian_lnprior(x, mu, sigma):
    return -0.5 * ((x - mu) / sigma) ** 2


# ============================================================
# Isochrone evaluation
# ============================================================

def evaluate_nonrot_isochrone(
    interp_func,
    logAge,
    MH,
    Mini_grid=None,
    omega_ini=1e-8,
    Inclination=1e-8,
):
    if Mini_grid is None:
        Mini_grid = np.arange(1.0, 5.0 + 0.01, 0.01)

    x = np.column_stack([
        Mini_grid,
        np.full_like(Mini_grid, logAge, dtype=float),
        np.full_like(Mini_grid, MH, dtype=float),
        np.full_like(Mini_grid, omega_ini, dtype=float),
        np.full_like(Mini_grid, Inclination, dtype=float),
    ])

    y = interp_func(x)

    output_cols = [
        "Mass", "omega", "Req", "Rpol",
        "G", "G_BP", "G_RP",
        "B_T", "V_T",
        "J", "H", "Ks",
    ]

    return Mini_grid, pd.DataFrame(y, columns=output_cols)


# ============================================================
# One-star likelihood
# ============================================================

def lnlike_one_star_to_isochrone(
    row,
    iso_df,
    band_list,
    mag_col_suffix="_mag",
    err_col_suffix="_err",
    parallax_col="parallax",
    extinction_dict=None,
    Mini_prior=None,
):
    parallax = row[parallax_col]

    if not np.isfinite(parallax) or parallax <= 0:
        return -np.inf

    dm = distance_modulus_from_parallax(parallax)

    chi2 = np.zeros(len(iso_df), dtype=float)

    n_used = 0

    for band in band_list:
        mag_col = f"{band}{mag_col_suffix}"
        err_col = f"{band}{err_col_suffix}"

        if mag_col not in row.index or err_col not in row.index:
            continue

        obs_mag = row[mag_col]
        obs_err = row[err_col]

        if not np.isfinite(obs_mag) or not np.isfinite(obs_err) or obs_err <= 0:
            continue

        A_band = 0.0
        if extinction_dict is not None:
            A_band = extinction_dict.get(band, 0.0)

        model_app_mag = iso_df[band].values + dm + A_band

        valid = np.isfinite(model_app_mag)

        chi2_i = np.full(len(iso_df), np.inf)
        chi2_i[valid] = ((obs_mag - model_app_mag[valid]) / obs_err) ** 2

        chi2 += chi2_i
        n_used += 1

    if n_used == 0:
        return -np.inf

    lnlike_Mini = -0.5 * chi2

    if Mini_prior is not None:
        lnlike_Mini = lnlike_Mini + Mini_prior(iso_df["Mass"].values)

    return logsumexp(lnlike_Mini) - np.log(len(lnlike_Mini))


# ============================================================
# Cluster likelihood
# ============================================================

def lnlike_cluster(
    theta,
    df_cluster,
    interp_func,
    band_list,
    Mini_grid=None,
    age_prior=None,
    MH_prior=None,
    extinction_mode="none",
):
    logAge, MH = theta

    if not np.isfinite(logAge) or not np.isfinite(MH):
        return -np.inf

    lp = 0.0

    if age_prior is not None:
        lp += age_prior(logAge)

    if MH_prior is not None:
        lp += MH_prior(MH)

    if not np.isfinite(lp):
        return -np.inf

    Mini_grid, iso_df = evaluate_nonrot_isochrone(
        interp_func=interp_func,
        logAge=logAge,
        MH=MH,
        Mini_grid=Mini_grid,
        omega_ini=1e-8,
        Inclination=1e-8,
    )

    total_lnlike = 0.0

    for _, row in df_cluster.iterrows():

        extinction_dict = None

        if extinction_mode == "A550_same_for_all_bands":
            A550 = row.get("A550", 0.0)
            extinction_dict = {band: A550 for band in band_list}

        lnlike_i = lnlike_one_star_to_isochrone(
            row=row,
            iso_df=iso_df,
            band_list=band_list,
            parallax_col="parallax",
            extinction_dict=extinction_dict,
        )

        if not np.isfinite(lnlike_i):
            continue

        total_lnlike += lnlike_i

    return lp + total_lnlike


# ============================================================
# Sampling logAge and MH
# ============================================================

def run_cluster_age_MH_sampling(
    df_cluster,
    interp_func,
    band_list=("G", "G_BP", "G_RP"),
    Mini_grid=None,
    age_prior_mu=None,
    age_prior_sigma=None,
    MH_prior_mu=None,
    MH_prior_sigma=None,
    init_logAge=8.8,
    init_MH=0.0,
    n_walkers=32,
    n_steps=3000,
    burnin=500,
):
    if Mini_grid is None:
        Mini_grid = np.arange(1.0, 5.0 + 0.01, 0.01)

    age_prior = None
    if age_prior_mu is not None and age_prior_sigma is not None:
        age_prior = lambda x: gaussian_lnprior(x, age_prior_mu, age_prior_sigma)

    MH_prior = None
    if MH_prior_mu is not None and MH_prior_sigma is not None:
        MH_prior = lambda x: gaussian_lnprior(x, MH_prior_mu, MH_prior_sigma)

    def log_prob(theta):
        return lnlike_cluster(
            theta=theta,
            df_cluster=df_cluster,
            interp_func=interp_func,
            band_list=list(band_list),
            Mini_grid=Mini_grid,
            age_prior=age_prior,
            MH_prior=MH_prior,
            extinction_mode="none",
        )

    ndim = 2

    p0 = np.array([init_logAge, init_MH])
    p0 = p0 + 1e-3 * np.random.randn(n_walkers, ndim)

    sampler = emcee.EnsembleSampler(
        n_walkers,
        ndim,
        log_prob,
    )

    sampler.run_mcmc(p0, n_steps, progress=True)

    samples = sampler.get_chain(discard=burnin, flat=True)

    logAge_samples = samples[:, 0]
    MH_samples = samples[:, 1]
    age_samples_myr = 10 ** logAge_samples / 1e6

    age_q16, age_q50, age_q84 = np.percentile(age_samples_myr, [16, 50, 84])
    MH_q16, MH_q50, MH_q84 = np.percentile(MH_samples, [16, 50, 84])

    print(
        f"Age = {age_q50:.1f} "
        f"+{age_q84 - age_q50:.1f} "
        f"-{age_q50 - age_q16:.1f} Myr"
    )

    print(
        f"[M/H] = {MH_q50:.3f} "
        f"+{MH_q84 - MH_q50:.3f} "
        f"-{MH_q50 - MH_q16:.3f}"
    )

    return sampler, samples



import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Wrapper for interp_func
# ============================================================
def model_func_example(
    Mini,
    logAge,
    MH,
    omega_ini=1e-8,
    inclination=1e-8,
):
    x = np.array([
        Mini,
        logAge,
        MH,
        omega_ini,
        inclination,
    ], dtype=float)

    y = interp_func(x)

    return {
        "Mass": y[0],
        "omega": y[1],
        "Req": y[2],
        "Rpol": y[3],
        "G": y[4],
        "G_BP": y[5],
        "G_RP": y[6],
        "B_T": y[7],
        "V_T": y[8],
        "J": y[9],
        "H": y[10],
        "Ks": y[11],
    }


def evaluate_isochrone_grid(
    model_func,
    Mini_grid,
    logAge,
    MH,
    omega_ini=1e-8,
    inclination=1e-8,
):


    x = np.column_stack([
        Mini_grid,
        np.full_like(Mini_grid, logAge, dtype=float),
        np.full_like(Mini_grid, MH, dtype=float),
        np.full_like(Mini_grid, omega_ini, dtype=float),
        np.full_like(Mini_grid, inclination, dtype=float),
    ])

    # If your interp_func supports batch input, this is faster.
    y = interp_func(x)
    if logAge < 7:
        y=np.array([np.inf]*12)
    return {
        "Mass": y[:, 0],
        "omega": y[:, 1],
        "Req": y[:, 2],
        "Rpol": y[:, 3],
        "G": y[:, 4],
        "G_BP": y[:, 5],
        "G_RP": y[:, 6],
        "B_T": y[:, 7],
        "V_T": y[:, 8],
        "J": y[:, 9],
        "H": y[:, 10],
        "Ks": y[:, 11],
    }


# ============================================================
# Helper
# ============================================================
def abs_mag_from_parallax(mag, parallax_mas, A=0.0):
    return mag + 5.0 * np.log10(parallax_mas) - 10.0 - A


def abs_mag_err_from_parallax(mag_err, parallax_mas, parallax_err, A_err=0.0):
    return np.sqrt(
        mag_err**2
        + (5.0 / np.log(10.0) * parallax_err / parallax_mas)**2
        + A_err**2
    )


def posterior_summary(x):
    q16, q50, q84 = np.percentile(x, [16, 50, 84])
    return q16, q50, q84


def format_pm(q16, q50, q84, digits=1):
    return rf"{q50:.{digits}f}$^{{+{q84-q50:.{digits}f}}}_{{-{q50-q16:.{digits}f}}}$"


# ============================================================
# CMD plot function
# ============================================================
def plot_cluster_cmd_fit(
    df_cluster,
    posterior_samples,
    Mini_grid=None,
    fit_systems=("Gaia",),
    omega_ini=1e-8,
    inclination=1e-8,
    n_posterior_lines=100,
    random_seed=42,
    A_col="A550",
    A_err_col="A550_err",
    save_prefix=None,
    show=True,
):
    """
    posterior_samples:
        shape (N, 2)
        column 0 = logAge
        column 1 = MH
    """

    if Mini_grid is None:
        Mini_grid = np.arange(1.0, 5.0 + 0.01, 0.01)

    CMD_CONFIGS = {
        "Gaia": {
            "x_bands": ("G_BP", "G_RP"),
            "y_band": "G",
            "x_mag_cols": ("G_BP_mag", "G_RP_mag"),
            "x_err_cols": ("G_BP_err", "G_RP_err"),
            "y_mag_col": "G_mag",
            "y_err_col": "G_err",
            "xlabel": r"$M_{G_{\rm BP}} - M_{G_{\rm RP}}\;(\mathrm{mag})$",
            "ylabel": r"$M_G\;(\mathrm{mag})$",
        },
        "2MASS": {
            "x_bands": ("J", "Ks"),
            "y_band": "J",
            "x_mag_cols": ("J_mag", "Ks_mag"),
            "x_err_cols": ("J_err", "Ks_err"),
            "y_mag_col": "J_mag",
            "y_err_col": "J_err",
            "xlabel": r"$M_J - M_{K_s}\;(\mathrm{mag})$",
            "ylabel": r"$M_J\;(\mathrm{mag})$",
        },
        "Tycho2": {
            "x_bands": ("B_T", "V_T"),
            "y_band": "V_T",
            "x_mag_cols": ("B_T_mag", "V_T_mag"),
            "x_err_cols": ("B_T_err", "V_T_err"),
            "y_mag_col": "V_T_mag",
            "y_err_col": "V_T_err",
            "xlabel": r"$M_{B_T} - M_{V_T}\;(\mathrm{mag})$",
            "ylabel": r"$M_{V_T}\;(\mathrm{mag})$",
        },
    }

    posterior_samples = np.asarray(posterior_samples, dtype=float)

    logAge_samples = posterior_samples[:, 0]
    MH_samples = posterior_samples[:, 1]
    age_samples_myr = 10.0**logAge_samples / 1e6

    age_16, age_50, age_84 = posterior_summary(age_samples_myr)
    MH_16, MH_50, MH_84 = posterior_summary(MH_samples)

    age_label = format_pm(age_16, age_50, age_84, digits=1)
    MH_label = format_pm(MH_16, MH_50, MH_84, digits=3)

    median_label = (
        rf"median: age = {age_label} Myr, "
        rf"$[\mathrm{{M/H}}]$ = {MH_label}"
    )

    rng = np.random.default_rng(random_seed)
    n_draw = min(n_posterior_lines, len(posterior_samples))
    draw_idx = rng.choice(len(posterior_samples), size=n_draw, replace=False)

    figs = {}
    axes = {}

    for system in fit_systems:
        cfg = CMD_CONFIGS[system]

        required_cols = [
            cfg["x_mag_cols"][0],
            cfg["x_mag_cols"][1],
            cfg["x_err_cols"][0],
            cfg["x_err_cols"][1],
            cfg["y_mag_col"],
            cfg["y_err_col"],
            "parallax",
            "parallax_err",
        ]

        if A_col is not None and A_col in df_cluster.columns:
            required_cols.append(A_col)

        if A_err_col is not None and A_err_col in df_cluster.columns:
            required_cols.append(A_err_col)

        df = df_cluster.dropna(subset=required_cols).copy()

        if len(df) == 0:
            print(f"No valid stars for {system}.")
            continue

        A = df[A_col].values if A_col is not None and A_col in df.columns else 0.0
        A_err = df[A_err_col].values if A_err_col is not None and A_err_col in df.columns else 0.0

        y_obs = abs_mag_from_parallax(
            df[cfg["y_mag_col"]].values,
            df["parallax"].values,
            A=A,
        )

        y_err = abs_mag_err_from_parallax(
            df[cfg["y_err_col"]].values,
            df["parallax"].values,
            df["parallax_err"].values,
            A_err=A_err,
        )

        x1_obs = abs_mag_from_parallax(
            df[cfg["x_mag_cols"][0]].values,
            df["parallax"].values,
            A=A,
        )

        x2_obs = abs_mag_from_parallax(
            df[cfg["x_mag_cols"][1]].values,
            df["parallax"].values,
            A=A,
        )

        x1_err = abs_mag_err_from_parallax(
            df[cfg["x_err_cols"][0]].values,
            df["parallax"].values,
            df["parallax_err"].values,
            A_err=A_err,
        )

        x2_err = abs_mag_err_from_parallax(
            df[cfg["x_err_cols"][1]].values,
            df["parallax"].values,
            df["parallax_err"].values,
            A_err=A_err,
        )

        x_obs = x1_obs - x2_obs
        x_err = np.sqrt(x1_err**2 + x2_err**2)

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.errorbar(
            x_obs,
            y_obs,
            xerr=x_err,
            yerr=y_err,
            fmt="o",
            ms=4,
            color="black",
            ecolor="black",
            elinewidth=1,
            capsize=0,
            alpha=0.9,
            zorder=5,
            label="observed stars",
        )

        # Posterior isochrone draws
        for idx in draw_idx:
            logAge_i = logAge_samples[idx]
            MH_i = MH_samples[idx]

            iso_i = evaluate_isochrone_grid(
                model_func=model_func_example,
                Mini_grid=Mini_grid,
                logAge=logAge_i,
                MH=MH_i,
                omega_ini=omega_ini,
                inclination=inclination,
            )

            x_model_i = iso_i[cfg["x_bands"][0]] - iso_i[cfg["x_bands"][1]]
            y_model_i = iso_i[cfg["y_band"]]

            valid = np.isfinite(x_model_i) & np.isfinite(y_model_i)

            ax.plot(
                x_model_i[valid],
                y_model_i[valid],
                color="orange",
                alpha=0.08,
                linewidth=1.0,
                zorder=1,
            )

        # Median isochrone
        iso_med = evaluate_isochrone_grid(
            model_func=model_func_example,
            Mini_grid=Mini_grid,
            logAge=np.median(logAge_samples),
            MH=np.median(MH_samples),
            omega_ini=omega_ini,
            inclination=inclination,
        )

        x_model_med = iso_med[cfg["x_bands"][0]] - iso_med[cfg["x_bands"][1]]
        y_model_med = iso_med[cfg["y_band"]]

        valid = np.isfinite(x_model_med) & np.isfinite(y_model_med)

        ax.plot(
            x_model_med[valid],
            y_model_med[valid],
            color="black",
            linewidth=2.2,
            label=median_label,
            zorder=10,
        )

        ax.invert_yaxis()
        ax.grid(True, linestyle="--", alpha=0.4)

        ax.set_xlabel(cfg["xlabel"], fontsize=16)
        ax.set_ylabel(cfg["ylabel"], fontsize=16)
        ax.tick_params(axis="both", labelsize=13)
        ax.legend(fontsize=11, loc="best")

        fig.tight_layout()

        if save_prefix is not None:
            fig.savefig(
                f"{save_prefix}_{system}_CMD_fit.pdf",
                bbox_inches="tight",
            )

        figs[system] = fig
        axes[system] = ax

    if show:
        plt.show()
    else:
        plt.close("all")

    return figs, axes



import numpy as np
import matplotlib.pyplot as plt


MODEL_OUTPUTS = [
    "Mass", "omega", "Req", "Rpol",
    "G", "G_BP", "G_RP", "B_T", "V_T", "J", "H", "Ks"
]

def eval_isochrone(
    model_interp,
    Mini_grid,
    Age,
    MH=0.01,
    omega_ini=0.0,
    Inclination=0.0,
    interp_u=None,
):
    Mini_grid = np.asarray(Mini_grid, dtype=float)
    Age = float(Age)

    X = np.column_stack([
        Mini_grid,
        np.full_like(Mini_grid, Age, dtype=float),
        np.full_like(Mini_grid, MH, dtype=float),
        np.full_like(Mini_grid, omega_ini, dtype=float),
        np.full_like(Mini_grid, Inclination, dtype=float),
    ])

    Y = np.asarray(model_interp(X), dtype=float)

    if Y.ndim != 2:
        raise ValueError(
            "model_interp(X) must return a 2D array with shape "
            f"(N, {len(MODEL_OUTPUTS)}), but got shape {Y.shape}."
        )

    if Y.shape[0] != len(Mini_grid):
        raise ValueError(
            "The number of outputs from model_interp does not match "
            f"Mini_grid: {Y.shape[0]} != {len(Mini_grid)}."
        )

    if Y.shape[1] != len(MODEL_OUTPUTS):
        raise ValueError(
            f"model_interp(X) must return {len(MODEL_OUTPUTS)} outputs, "
            f"but got {Y.shape[1]}."
        )

    iso = {
        name: Y[:, i]
        for i, name in enumerate(MODEL_OUTPUTS)
    }

    # Optionally evaluate evolutionary phase u
    if interp_u is not None:
        u_raw = np.asarray(interp_u(X), dtype=float)

        # Accept either shape (N,) or (N, 1)
        if u_raw.shape == (len(Mini_grid),):
            u = u_raw

        elif u_raw.shape == (len(Mini_grid), 1):
            u = u_raw[:, 0]

        else:
            raise ValueError(
                "interp_u(X) must return an array with shape "
                f"({len(Mini_grid)},) or ({len(Mini_grid)}, 1), "
                f"but got shape {u_raw.shape}."
            )

        iso["u"] = u

    iso["Mini"] = Mini_grid
    iso["Age"] = Age
    iso["logAge"] = np.log10(Age)
    iso["age_Myr"] = Age / 1e6
    iso["MH"] = MH
    iso["omega_ini"] = omega_ini
    iso["Inclination"] = Inclination

    return iso



def get_cmd_xy(iso, color="G_BP-G_RP", mag="G"):
    b1, b2 = color.split("-")
    x = iso[b1] - iso[b2]
    y = iso[mag]
    return x, y


def _make_age_color_dict(ages_sorted, age_colors=None):
    if age_colors is None:
        age_colors = [
            "tab:blue", "tab:orange", "tab:red", "black",
            "tab:purple", "tab:brown", "tab:green",
        ]

    return {
        age: age_colors[i % len(age_colors)]
        for i, age in enumerate(ages_sorted)
    }


def _alpha_from_inclination(
    inc,
    inc_min=0.0,
    inc_max=1.57,
    alpha_min=0.35,
    alpha_max=1.0,
):
    t = (inc - inc_min) / (inc_max - inc_min)
    t = np.clip(t, 0.0, 1.0)
    return alpha_min + t * (alpha_max - alpha_min)


def _interp_point_at_mass(
    iso,
    x,
    y,
    mass_value,
    mass_key="Mini",
    selection_mask=None,
):
    mass_arr = np.asarray(iso[mass_key], dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if not (
        mass_arr.shape == x.shape == y.shape
    ):
        raise ValueError(
            "mass, x, and y must have the same shape, but got "
            f"{mass_arr.shape}, {x.shape}, and {y.shape}."
        )

    valid = (
        np.isfinite(mass_arr)
        & np.isfinite(x)
        & np.isfinite(y)
    )

    if selection_mask is not None:
        selection_mask = np.asarray(selection_mask, dtype=bool)

        if selection_mask.shape != mass_arr.shape:
            raise ValueError(
                "selection_mask must have the same shape as mass_arr, "
                f"but got {selection_mask.shape} and {mass_arr.shape}."
            )

        valid &= selection_mask

    mass_arr = mass_arr[valid]
    x_arr = x[valid]
    y_arr = y[valid]

    if len(mass_arr) < 2:
        return None

    order = np.argsort(mass_arr)
    mass_arr = mass_arr[order]
    x_arr = x_arr[order]
    y_arr = y_arr[order]

    # Remove duplicate mass values, since np.interp expects increasing x
    mass_arr, unique_indices = np.unique(
        mass_arr,
        return_index=True,
    )
    x_arr = x_arr[unique_indices]
    y_arr = y_arr[unique_indices]

    if len(mass_arr) < 2:
        return None

    if mass_value < mass_arr[0] or mass_value > mass_arr[-1]:
        return None

    x_m = np.interp(mass_value, mass_arr, x_arr)
    y_m = np.interp(mass_value, mass_arr, y_arr)

    return x_m, y_m


def plot_cmd_isochrones(
    interp_func,
    Mini_grid=np.arange(1.0, 5.0 + 0.01, 0.01),
    ages_Myr=(100, 180, 345, 700),
    MH=0.01,
    omega_ini_list=(0.0, 0.8),
    inclination_list=(0.0, 0.524, 1.047, 1.57),
    color="G_BP-G_RP",
    mag="G",
    obs=None,
    obs_err=None,
    title=None,
    figsize=(7, 8),
    age_colors=("tab:blue", "tab:orange", "tab:red", "black"),
    rot_alpha_range=(0.35, 1.0),
    nonrot_alpha=0.75,
    rot_lw=2.2,
    nonrot_lw=1.8,
    inc_alpha_min=0.0,
    inc_alpha_max=1.57,
    grid_alpha=0.25,
    legend_on=True,
    legend_fontsize=8,
    xlim=None,
    ylim=None,

    # evolutionary-phase filtering
    interp_u=None,
    u_range=None,

    # mass marker settings
    mass_marks=(1.5, 2.0, 2.5, 3.0),
    mass_key="Mini",
    mass_marker_cycle=("s", "^", "v", "x", "o"),
    mass_marker_size=55,
    mass_text=True,
    mass_text_iso_index=-1,
    mass_text_dx=0.025,
    mass_text_dy=0.035,
    mass_text_fontsize=9,

    save_name=None,
):
    # interp_u and u_range must be supplied together
    if (interp_u is None) != (u_range is None):
        raise ValueError(
            "interp_u and u_range must either both be provided "
            "or both be None."
        )

    if u_range is not None:
        u_range_arr = np.asarray(u_range, dtype=float)

        if u_range_arr.shape != (2,):
            raise ValueError(
                "u_range must contain exactly two values: "
                "(u_min, u_max)."
            )

        u_min, u_max = u_range_arr

        if not np.isfinite(u_min) or not np.isfinite(u_max):
            raise ValueError("Both values in u_range must be finite.")

        if u_min > u_max:
            raise ValueError(
                f"u_range must satisfy u_min <= u_max, "
                f"but got ({u_min}, {u_max})."
            )
    else:
        u_min = None
        u_max = None

    fig, ax = plt.subplots(figsize=figsize)

    ages_Myr = np.asarray(ages_Myr, dtype=float)
    ages_sorted = np.sort(ages_Myr)

    age_to_base_color = _make_age_color_dict(
        ages_sorted,
        age_colors=list(age_colors),
    )

    alpha_min, alpha_max = rot_alpha_range

    plotted_isochrones = []

    for age_Myr in ages_sorted:
        base_color = age_to_base_color[age_Myr]

        for omega_ini in omega_ini_list:

            if np.isclose(omega_ini, 0.0):
                inc_list_this = [0.0]
                linestyle = "--"
                lw = nonrot_lw
            else:
                inc_list_this = inclination_list
                linestyle = "-"
                lw = rot_lw

            for inc in inc_list_this:
                iso = eval_isochrone(
                    model_interp=interp_func,
                    Mini_grid=Mini_grid,
                    Age=age_Myr,
                    MH=MH,
                    omega_ini=omega_ini,
                    Inclination=inc,
                    interp_u=interp_u,
                )

                x, y = get_cmd_xy(
                    iso,
                    color=color,
                    mag=mag,
                )

                x = np.asarray(x, dtype=float)
                y = np.asarray(y, dtype=float)

                plot_mask = (
                    np.isfinite(x)
                    & np.isfinite(y)
                )

                # Restrict the curve to the requested evolutionary phase
                if u_range is not None:
                    u = np.asarray(iso["u"], dtype=float)

                    plot_mask &= (
                        np.isfinite(u)
                        & (u >= u_min)
                        & (u <= u_max)
                    )

                # Skip this isochrone when no point lies inside u_range
                if not np.any(plot_mask):
                    continue

                if np.isclose(omega_ini, 0.0):
                    plot_alpha = nonrot_alpha
                    label = f"{age_Myr:.0f} Myr, non-rot"
                else:
                    plot_alpha = _alpha_from_inclination(
                        inc,
                        inc_min=inc_alpha_min,
                        inc_max=inc_alpha_max,
                        alpha_min=alpha_min,
                        alpha_max=alpha_max,
                    )

                    label = (
                        f"{age_Myr:.0f} Myr, "
                        f"ωini={omega_ini:.2f}, "
                        f"i={inc:.3f}, "
                        f"α={plot_alpha:.2f}"
                    )

                # NaN outside the selected region prevents Matplotlib
                # from connecting separate parts of the curve.
                x_plot = np.where(plot_mask, x, np.nan)
                y_plot = np.where(plot_mask, y, np.nan)

                ax.plot(
                    x_plot,
                    y_plot,
                    color=base_color,
                    lw=lw,
                    ls=linestyle,
                    alpha=plot_alpha,
                    label=label,
                )

                mass_points = []

                if mass_marks is not None:
                    for j, m in enumerate(mass_marks):
                        point = _interp_point_at_mass(
                            iso=iso,
                            x=x,
                            y=y,
                            mass_value=m,
                            mass_key=mass_key,
                            selection_mask=plot_mask,
                        )

                        if point is None:
                            continue

                        x_m, y_m = point
                        marker = mass_marker_cycle[
                            j % len(mass_marker_cycle)
                        ]

                        ax.scatter(
                            x_m,
                            y_m,
                            s=mass_marker_size,
                            marker=marker,
                            color=base_color,
                            alpha=plot_alpha,
                            zorder=5,
                        )

                        mass_points.append(
                            (m, x_m, y_m, marker)
                        )

                plotted_isochrones.append({
                    "age_Myr": age_Myr,
                    "omega_ini": omega_ini,
                    "Inclination": inc,
                    "x": x,
                    "y": y,
                    "u": iso.get("u"),
                    "plot_mask": plot_mask,
                    "iso": iso,
                    "mass_points": mass_points,
                    "color": base_color,
                })

    if (
        mass_text
        and mass_marks is not None
        and len(plotted_isochrones) > 0
    ):
        try:
            ref_iso = plotted_isochrones[mass_text_iso_index]
        except IndexError as exc:
            raise IndexError(
                f"mass_text_iso_index={mass_text_iso_index} is invalid. "
                f"Only {len(plotted_isochrones)} isochrones were plotted."
            ) from exc

        for j, item in enumerate(ref_iso["mass_points"]):
            m, x_m, y_m, marker = item

            ax.text(
                x_m + mass_text_dx,
                y_m
                + mass_text_dy
                * (
                    j
                    - 0.5
                    * len(ref_iso["mass_points"])
                ),
                f"{m:.2f} $M_\\odot$",
                color=ref_iso["color"],
                fontsize=mass_text_fontsize,
                va="center",
                ha="right",
            )

    if obs is not None:
        c1, c2 = color.split("-")
        obs_x = obs[c1] - obs[c2]
        obs_y = obs[mag]

        if obs_err is not None:
            obs_xerr = np.sqrt(
                obs_err[c1] ** 2
                + obs_err[c2] ** 2
            )
            obs_yerr = obs_err[mag]
        else:
            obs_xerr = None
            obs_yerr = None

        ax.errorbar(
            obs_x,
            obs_y,
            xerr=obs_xerr,
            yerr=obs_yerr,
            fmt="*",
            ms=14,
            capsize=3,
            color="black",
            label="Observed target",
            zorder=10,
        )

    METHOD = "DIRECT"
    cluster = "hyades"

    from belfast.paths import REPO_ROOT, RESULTS_DIR, DATA_DIR
    df_post = pd.read_csv(DATA_DIR / f'clusters/praesepe_posterior_{METHOD}_summary.csv')


    c = df_post["age"]
    color_label = "Age (Myr)"

    sc = ax.scatter(
        df_post["MBP_MRP"],
        df_post["MG"],
        c=c,
        cmap="viridis",
        s=30,
        alpha=0.8,
        edgecolors="none",
    )

    x_line_hyades = 0.47

    ax.axvline(
        x=x_line_hyades,
        linestyle=":",
        color="green",
        linewidth=2,
        label=rf"$M_{{G_{{\mathrm{{BP}}}}}} - M_{{G_{{\mathrm{{RP}}}}}} = {x_line_hyades:.2f}$",
    )

    ax.grid(True, linestyle="--", alpha=0.6)

    ax.set_xlabel(
        r"$M_{G_\mathit{BP}} - M_{G_\mathit{RP}} \;(\mathrm{mag})$",
        fontsize=18,
    )

    ax.set_ylabel(
        r"$M_{\mathit{G}} \;(\mathrm{mag})$",
        fontsize=18,
    )

    ax.tick_params(axis="both", labelsize=14)

    cbar = plt.colorbar(sc, ax=ax)
    cbar.set_label(color_label, fontsize=18)
    cbar.ax.tick_params(labelsize=15)

    ax.invert_yaxis()
    ax.set_xlabel(color)
    ax.set_ylabel(mag)

    xmin, xmax = ax.get_xlim()
    xpad = 0.15 * (xmax - xmin)
    ax.set_xlim(xmin - xpad, xmax)

    if title is None:
        title = f"CMD isochrones, [M/H]={MH}"

        if u_range is not None:
            title += f", {u_min:g} ≤ u ≤ {u_max:g}"

    ax.set_title(title)

    if legend_on:
        ax.legend(fontsize=legend_fontsize)

    ax.grid(alpha=grid_alpha)

    if xlim is not None:
        ax.set_xlim(xlim)

    if ylim is not None:
        ax.set_ylim(ylim)

    fig.tight_layout()

    if save_name is not None:
        fig.savefig(
            save_name,
            bbox_inches="tight",
        )

    return fig, ax



import numpy as np
import matplotlib.pyplot as plt


EVOL_INPUTS = ["Mini", "logAge", "MH", "omega_ini"]
EVOL_OUTPUTS = ["Mass", "omega", "logL", "Req", "Rpol"]


def evaluate_evolution_track(
    interp_func,
    Mini,
    age_Myr_grid,
    MH=0.0,
    omega_ini=0.0,
):
    """
    Evaluate one evolutionary track at fixed Mini, MH, and omega_ini.

    Input order:
        ['Mini', 'logAge', 'MH', 'omega_ini']

    Output order:
        ['Mass', 'omega', 'logL', 'Req', 'Rpol']

    logAge is log10(age / yr).
    """

    age_Myr_grid = np.asarray(age_Myr_grid, dtype=float)
    # logAge_grid = np.log10(age_Myr_grid) + 6.0

    X = np.column_stack([
        np.full_like(age_Myr_grid, Mini, dtype=float),
        age_Myr_grid,
        np.full_like(age_Myr_grid, MH, dtype=float),
        np.full_like(age_Myr_grid, omega_ini, dtype=float),
    ])

    Y = interp_func(X)

    track = {}
    for i, name in enumerate(EVOL_OUTPUTS):
        track[name] = Y[:, i]

    track["Mini"] = np.full_like(age_Myr_grid, Mini, dtype=float)
    track["age_Myr"] = age_Myr_grid
    # track["logAge"] = age_Myr_grid
    track["MH"] = np.full_like(age_Myr_grid, MH, dtype=float)
    track["omega_ini"] = np.full_like(age_Myr_grid, omega_ini, dtype=float)

    return track


def _to_list(x):
    if np.isscalar(x):
        return [x]
    return list(x)


def _make_mass_color_dict(mass_list, mass_colors=None):
    if mass_colors is None:
        mass_colors = [
            "tab:blue",
            "tab:orange",
            "tab:red",
            "black",
            "tab:purple",
            "tab:brown",
            "tab:green",
        ]

    return {
        mass: mass_colors[i % len(mass_colors)]
        for i, mass in enumerate(mass_list)
    }


def _make_metallicity_linestyle_dict(MH_list, metallicity_linestyles=None):
    if metallicity_linestyles is None:
        metallicity_linestyles = [
            "-",
            "--",
            ":",
            "-.",
            (0, (5, 1)),
            (0, (3, 1, 1, 1)),
        ]

    return {
        MH: metallicity_linestyles[i % len(metallicity_linestyles)]
        for i, MH in enumerate(MH_list)
    }


def _alpha_from_omega(
    omega_ini,
    omega_min=0.0,
    omega_max=1.0,
    alpha_min=0.25,
    alpha_max=1.0,
):
    """
    Larger omega_ini -> darker / less transparent.
    omega_ini = omega_min gives alpha_min.
    omega_ini = omega_max gives alpha_max.
    """

    t = (omega_ini - omega_min) / (omega_max - omega_min)
    t = np.clip(t, 0.0, 1.0)

    return alpha_min + t * (alpha_max - alpha_min)


def plot_evolution_L_R_tracks(
    interp_func,
    Mini_list=(1.5, 2.0, 2.5),
    age_Myr_grid=np.linspace(10, 2000, 5000),
    MH=0.0,
    omega_ini_list=(0.0, 0.2, 0.4, 0.6, 0.8),
    mass_colors=None,
    metallicity_linestyles=None,
    omega_alpha_range=(0.25, 1.0),
    omega_alpha_min=0.0,
    omega_alpha_max=1.0,
    nonrot_alpha=0.5,
    nonrot_omega_value=0.0,
    figsize=(9, 5.5),
    lw=2.0,
    nonrot_lw=1.8,
    grid_alpha=0.25,
    legend_fontsize=8,
    title=None,
    xlim=None,
    ylim=None,
    save_name=None,
):
    """
    Plot three horizontal evolutionary panels:
        x-axis: age in Myr
        y-axis: luminosity, Req, Rpol

    Mini can be a list.
    MH can be either a scalar or a list.
    omega_ini_list can be a list.

    Same Mini uses the same color.
    Different metallicities use different linestyles.
    Larger omega_ini gives larger alpha.
    Non-rotating models use lower alpha by default.
    """

    Mini_list = _to_list(Mini_list)
    MH_list = _to_list(MH)
    omega_ini_list = _to_list(omega_ini_list)

    mass_to_color = _make_mass_color_dict(
        Mini_list,
        mass_colors=mass_colors,
    )

    MH_to_linestyle = _make_metallicity_linestyle_dict(
        MH_list,
        metallicity_linestyles=metallicity_linestyles,
    )

    fig, axes = plt.subplots(
        3,
        1,
        figsize=figsize,
        sharex=True,
        gridspec_kw={"hspace": 0.08},
    )

    y_info = [
        ("logL", r"$L$"),
        ("Req", r"$R_{\rm eq}$"),
        ("Rpol", r"$R_{\rm pol}$"),
    ]

    for Mini in Mini_list:
        base_color = mass_to_color[Mini]

        for MH_value in MH_list:
            base_linestyle = MH_to_linestyle[MH_value]

            for omega_ini in omega_ini_list:

                is_nonrot = np.isclose(omega_ini, nonrot_omega_value)

                if is_nonrot:
                    line_alpha = nonrot_alpha
                    line_lw = nonrot_lw
                else:
                    line_alpha = _alpha_from_omega(
                        omega_ini,
                        omega_min=omega_alpha_min,
                        omega_max=omega_alpha_max,
                        alpha_min=omega_alpha_range[0],
                        alpha_max=omega_alpha_range[1],
                    )
                    line_lw = lw

                track = evaluate_evolution_track(
                    interp_func=interp_func,
                    Mini=Mini,
                    age_Myr_grid=age_Myr_grid,
                    MH=MH_value,
                    omega_ini=omega_ini,
                )

                label = (
                    rf"$M_{{\rm ini}}={Mini:.2f}\,M_\odot$, "
                    rf"$[\mathrm{{M/H}}]={MH_value:.2f}$, "
                    rf"$\omega_{{\rm ini}}={omega_ini:.2f}$"
                )

                for ax, (key, ylabel) in zip(axes, y_info):
                    y = track[key]
                    valid = np.isfinite(track["age_Myr"]) & np.isfinite(y)

                    plot_x = track["age_Myr"][valid]
                    plot_y = y[valid]

                    if key == "logL":
                        plot_y = 10**plot_y

                    ax.plot(
                        plot_x,
                        plot_y,
                        color=base_color,
                        alpha=line_alpha,
                        lw=line_lw,
                        ls=base_linestyle,
                        label=label if key == "logL" else None,
                    )

                    ax.set_ylabel(ylabel)
                    ax.grid(alpha=grid_alpha)

    axes[-1].set_xlabel("Age [Myr]")

    if title is None:
        if len(MH_list) == 1:
            title = rf"Evolution tracks, $[\mathrm{{M/H}}] = {MH_list[0]:.3f}$"
        else:
            title = r"Evolution tracks for multiple metallicities"

    axes[0].set_title(title)

    axes[0].legend(
        fontsize=legend_fontsize,
        ncol=2,
        frameon=True,
    )

    if xlim is not None:
        axes[-1].set_xlim(xlim)

    if ylim is not None:
        for ax in axes:
            ax.set_ylim(ylim)

    plt.tight_layout()

    if save_name is not None:
        plt.savefig(save_name, bbox_inches="tight")

    return fig, axes


