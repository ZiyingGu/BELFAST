from lib.target_utils import query_data, cal_obs_mag, cal_abs_mag, remove_extinction
from lib.isochrone_utils import rearrange_isochrone, read_isochrones, check_rearranged_isochrones

import re
from glob import glob
from scipy.stats import gaussian_kde
from scipy.special import logsumexp


import numpy as np
import pandas as pd

import os

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import matplotlib.colors as mcolors

# === KDE 估计 PDF ===
def kde_pdf(samples, grid):
    samples = np.asarray(samples).ravel()  # flatten 成 1D
    if len(samples) < 5:
        # 太少的样本时直接用 delta-like 分布近似，避免 KDE 崩溃
        pdf = np.exp(-0.5 * ((grid - np.median(samples)) / np.std(samples or [1])) ** 2)
        pdf = np.maximum(pdf, 1e-300)
        pdf /= np.trapezoid(pdf, grid)
        return pdf

    kde = gaussian_kde(samples.T)  # 自动转置安全
    pdf = kde(grid)
    pdf = np.maximum(pdf, 1e-300)
    pdf /= np.trapezoid(pdf, grid)
    return pdf


# === Flat prior ===
def prior_tau(grid):
    return np.ones_like(grid)

# === ACT 计算 ===
def compute_act(samples_2d):
    """
    Compute integrated autocorrelation time (ACT) for 2D samples: (n_walkers, n_steps)
    """
    import emcee
    try:
        act = emcee.autocorr.integrated_time(samples_2d, quiet=True, tol=0)
        # emcee 返回 shape=(n_walkers,)，我们只取最大值
        act_val = np.nanmax(act)
    except Exception:
        act_val = np.nan
    return act_val

import numpy as np
from scipy.special import logsumexp


import numpy as np
from scipy.special import logsumexp


def combine_posteriors_age_unflat(
    chains_list,
    weight=False,
    n_grid=800,
    prior_fn=None,
):
    """
    Combine posterior samples of age from unflattened MCMC chains.

    Parameters
    ----------
    chains_list : list of ndarray
        Each element has shape (n_steps, n_walkers) or (n_walkers, n_steps).
    weight : False or array-like
        If False, all stars are equally weighted (w_i = 1).
        Otherwise, weight should be an array of membership probabilities
        with length equal to len(chains_list).
    n_grid : int
        Number of grid points for age.
    prior_fn : callable or None
        Prior function pi(tau). If None, a flat prior is assumed.

    Returns
    -------
    grid : ndarray
        Age grid.
    P : ndarray
        Normalized cluster-level posterior.
    """

    N_star = len(chains_list)

    # === 权重处理 ===
    if weight is False:
        weight_array = np.ones(N_star)
    else:
        weight_array = np.asarray(weight, dtype=float)
        if weight_array.shape[0] != N_star:
            raise ValueError(
                "Length of weight must match number of chains"
            )

    sample_list = []

    # === 逐星处理 MCMC chains ===
    for chain in chains_list:
        # 允许两种 shape 排列
        if chain.shape[0] < chain.shape[1]:
            ages = chain.T  # (n_walkers, n_steps)
        else:
            ages = chain

        n_walkers, n_steps = ages.shape

        # compute ACT on last 50% and last 30%
        half_idx = int(n_steps * 0.5)
        third_idx = int(n_steps * 0.7)

        act_half = compute_act(ages[:, half_idx:])
        act_third = compute_act(ages[:, third_idx:])

        # 取较大的 ACT ×10 作为 burn-in
        act_max = np.nanmax([act_half, act_third])
        burnin = int(min(n_steps, act_max * 10))

        if np.isnan(burnin) or burnin >= n_steps:
            burnin = n_steps // 2  # fallback

        # 提取 burn-in 后样本
        post_burnin = ages[:, burnin:].reshape(-1)
        sample_list.append(post_burnin)

    # === KDE + grid ===
    pooled = np.concatenate(sample_list)
    grid = np.linspace(pooled.min(), pooled.max(), n_grid)

    pdfs = [kde_pdf(s, grid) for s in sample_list]
    log_pdfs = [np.log(np.maximum(p, 1e-300)) for p in pdfs]

    # === 核心：加权 log-posterior ===
    logP = np.zeros_like(grid)

    for w, logp in zip(weight_array, log_pdfs):
        logP += w * logp

    # === prior 修正（None => flat prior） ===
    if prior_fn is not None:
        pi = np.maximum(prior_fn(grid), 1e-300)
        logP -= (np.sum(weight_array) - 1.0) * np.log(pi)

    # === 归一化（log-space） ===
    logP -= logsumexp(logP)
    P = np.exp(logP)

    # 数值积分归一化（稳妥）
    P /= np.trapezoid(P, grid)

    return grid, P




# ---------- Step 1. 读取 SIMBAD-HIP 对照表 ----------

# df_targets = df_targets.dropna(subset=["HIP_ID"]).copy()
# df_targets["HIP_ID"] = df_targets["HIP_ID"].str.strip()

# ---------- Step 2. 准备 posterior 路径 ----------



# ---------- Step 3. 定义 ACT 计算 ----------
def compute_act(samples_2d):
    import emcee
    try:
        act = emcee.autocorr.integrated_time(samples_2d, quiet=True, tol=0)
        act_val = np.nanmax(act)
    except Exception:
        act_val = np.nan
    return act_val

# ---------- Step 4. 定义后验提取函数 ----------
def posterior_summary(samples_2d):
    """Return median, upper(84%) and lower(16%) percentiles."""
    samples_flat = samples_2d.reshape(-1)
    median = np.percentile(samples_flat, 50)
    p84 = np.percentile(samples_flat, 84)
    p16 = np.percentile(samples_flat, 16)
    upper = p84 - median
    lower = median - p16
    return median, upper, lower



def summarize_cluster_age(
    cluster: str,
    samples_glob: str,
    df_targets: pd.DataFrame,
    save_name: str = None,
    print_flag: bool = True,
    sigma_index = 6
):
    """
    Summarize posterior estimates (mass, age, sigma) for a stellar cluster
    from saved MCMC posterior samples, and merge the results with target metadata.

    Parameters
    ----------
    cluster : str
        Cluster name used to identify relevant posterior files
        (e.g., "Pleiades", "Praesepe").
    samples_glob : str
        Glob pattern pointing to posterior `.npz` files.
    df_targets : pandas.DataFrame
        Target table containing at least HIP ID / HD ID / Other Name.
    print_flag : bool, optional
        If True, print the head of the merged DataFrame.

    Returns
    -------
    pandas.DataFrame
        Target table merged with posterior summary statistics.
    """

    records = []
    cluster_key = f"_{cluster}"

    # ------------------------------------------------------------------
    # Step 1. Iterate over posterior files and extract summaries
    # ------------------------------------------------------------------
    for path in glob(samples_glob):
        basename = os.path.basename(path)

        # Skip unrelated files early
        if "posterior" not in basename or cluster_key not in basename:
            continue

        # Extract object name from filename:
        # posterior_<NAME>_<CLUSTER>.npz  ->  <NAME>
        name_start = basename.index("posterior_") + len("posterior_")
        name_end = basename.index(cluster_key)
        posterior_name = basename[name_start:name_end].replace("_", " ").strip()

        data = np.load(path, allow_pickle=True)
        samples = data["samples"]  # shape: (n_steps, n_walkers, n_params)

        # Reorder to (n_walkers, n_steps) for time-series analysis
        # Parameter indices are assumed to be fixed by the sampling code
        mass_chain = samples[:, :, 0].T
        logage_chain = samples[:, :, 1].T
        sigma_chain = samples[:, :, sigma_index].T


        n_steps = logage_chain.shape[1]

        # ------------------------------------------------------------------
        # Step 2. Estimate burn-in using autocorrelation time (ACT)
        # ------------------------------------------------------------------
        half_idx = int(0.5 * n_steps)
        third_idx = int(0.7 * n_steps)

        act_half = compute_act(logage_chain[:, half_idx:])
        act_third = compute_act(logage_chain[:, third_idx:])
        act_max = np.nanmax([act_half, act_third])

        burnin = int(min(n_steps, 10 * act_max))
        if not np.isfinite(burnin) or burnin >= n_steps:
            burnin = n_steps // 2

        # ------------------------------------------------------------------
        # Step 3. Extract post–burn-in samples
        # ------------------------------------------------------------------
        age_post = 10 ** (logage_chain[:, burnin:] - 6.0)  # convert to Myr
        sigma_post = sigma_chain[:, burnin:]
        mass_post = mass_chain[:, burnin:]

        # ------------------------------------------------------------------
        # Step 4. Posterior summaries (median ± uncertainties)
        # ------------------------------------------------------------------
        age_med, age_up, age_low = posterior_summary(age_post)
        sig_med, sig_up, sig_low = posterior_summary(sigma_post)
        mass_med, mass_up, mass_low = posterior_summary(mass_post)

        records.append(
            {
                "posterior_name": posterior_name,
                "posterior_path": os.path.relpath(path),
                "mass": mass_med,
                "mass_upper_uncertainty": mass_up,
                "mass_lower_uncertainty": mass_low,
                "age": age_med,
                "age_upper_uncertainty": age_up,
                "age_lower_uncertainty": age_low,
                "sigma": sig_med,
                "sigma_upper_uncertainty": sig_up,
                "sigma_lower_uncertainty": sig_low,
            }
        )

    df_post = pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Step 5. Construct posterior_name for target table
    # ------------------------------------------------------------------
    def build_posterior_name(row):
        if not pd.isna(row.get("HIP ID")):
            return f"HIP {int(row['HIP ID'])}"
        if not pd.isna(row.get("HD ID")):
            return f"HD {int(row['HD ID'])}"
        return re.sub(r"\s+", " ", str(row.get("Other Name", ""))).strip()

    df_targets = df_targets.copy()
    df_targets["posterior_name"] = df_targets.apply(build_posterior_name, axis=1)

    # ------------------------------------------------------------------
    # Step 6. Merge posterior summaries with target metadata
    # ------------------------------------------------------------------
    df_all = pd.merge(df_targets, df_post, on="posterior_name", how="left")

    if save_name is not None:
        df_all.to_csv(save_name, index=False)

    if print_flag:
        print(df_all.head())

    return df_all



def plot_mass_age(
    cluster,
    df_summary_posterior,
    method,
    weight=False,
    title=None,
    save_name=None,
    age_xlim=None,
):
    """
    Plot the mass–age relation of cluster members (left panel) together with
    the combined cluster age posterior (right panel, rotated), sharing the
    same age (y) axis.
    """

    # ============================================================
    # Configuration (visual style)
    # ============================================================
    sigma_color = "orange"
    sigma_alpha = 0.2

    # ---- Font sizes (adjust here) ----
    label_fs = 14       # axis label font size
    tick_fs = 12        # tick label font size
    legend_fs = 12      # legend font size

    if title != False:
        if title is None:
            title = f"Mass-Age relation of the {cluster}"

    # ============================================================
    # Step 1. Load individual stellar age posteriors
    # ============================================================
    post_paths = [
        p for p in df_summary_posterior["posterior_path"].values
        if not pd.isna(p)
    ]

    if weight == False:
        weight_array = np.ones_like(df_summary_posterior["age"].values)
    else:
        weight_array = df_summary_posterior["BEST_HYP_MAX_PROB"].values

    star_age_samples = []
    for path in post_paths:
        data = np.load(path, allow_pickle=True)
        samples = data["samples"]
        logage = samples[:, :, 1]
        star_age_samples.append(10 ** (logage - 6.0))

    # ============================================================
    # Step 2. Combine cluster age posterior
    # ============================================================
    tau_grid, P_cluster = combine_posteriors_age_unflat(
        star_age_samples, weight_array
    )

    cdf = np.cumsum(P_cluster)
    cdf /= cdf[-1]

    low = tau_grid[np.searchsorted(cdf, 0.16)]
    med = tau_grid[np.searchsorted(cdf, 0.50)]
    high = tau_grid[np.searchsorted(cdf, 0.84)]

    err_low = med - low
    err_high = high - med

    # ============================================================
    # Step 3. Determine y-axis limits
    # ============================================================
    y_min = np.min(
        df_summary_posterior["age"]
        - df_summary_posterior["age_lower_uncertainty"]
    )
    y_max = np.max(
        df_summary_posterior["age"]
        + df_summary_posterior["age_upper_uncertainty"]
    )

    y_pad = 0.05 * (y_max - y_min)
    y_min -= y_pad
    y_max += y_pad

    # ============================================================
    # Step 4. Create figure
    # ============================================================
    fig, (ax_left, ax_right) = plt.subplots(
        1, 2,
        figsize=(8, 4),
        sharey=True,
        gridspec_kw={
            "width_ratios": [3, 1],
            "wspace": 0.0,
        },
    )

    # ============================================================
    # Left panel
    # ============================================================
    ax_left.axhspan(
        low, high,
        facecolor=sigma_color,
        alpha=sigma_alpha,
        edgecolor="none",
        zorder=0,
    )

    ax_left.errorbar(
        x=df_summary_posterior["mass"],
        y=df_summary_posterior["age"],
        xerr=df_summary_posterior[
            ["mass_lower_uncertainty", "mass_upper_uncertainty"]
        ].values.T,
        yerr=df_summary_posterior[
            ["age_lower_uncertainty", "age_upper_uncertainty"]
        ].values.T,
        fmt="o",
        zorder=2,
    )

    ax_left.set_xlabel(r"Mass ($M_\odot$)", fontsize=label_fs)
    ax_left.set_ylabel("Age (Myr)", fontsize=label_fs)

    if title != False:
        ax_left.set_title(title)  # Title font unchanged

    ax_left.set_ylim(y_min, y_max)
    ax_left.tick_params(axis="both", labelsize=tick_fs)

    legend_patch = Patch(
        facecolor=sigma_color,
        alpha=sigma_alpha,
        edgecolor="none",
        label=(
            rf"$\rm Cluster\ Age\ ({method}):\ "
            rf"{med:.0f}^{{+{err_high:.0f}}}_{{-{err_low:.0f}}}\,Myr$"
        ),
    )
    ax_left.legend(
        handles=[legend_patch],
        frameon=False,
        fontsize=legend_fs,
    )

    # ============================================================
    # Right panel
    # ============================================================
    ax_right.plot(P_cluster, tau_grid, color="black", lw=2)

    ax_right.fill_betweenx(
        tau_grid,
        0.0,
        P_cluster,
        color="gray",
        alpha=0.3,
    )

    ax_right.axhspan(
        low, high,
        facecolor=sigma_color,
        alpha=sigma_alpha,
        edgecolor="none",
        zorder=1,
    )

    ax_right.set_xlabel("Probability density", fontsize=label_fs)
    if title == False:
        ax_right.set_title(None)
    else:
        ax_right.set_title("Cluster Age posterior")  # Title unchanged
    ax_right.tick_params(axis="both", labelsize=tick_fs)

    if age_xlim is not None:
        ax_right.set_xlim(age_xlim)

    ax_right.tick_params(labelleft=False)

    # ============================================================
    # Final layout
    # ============================================================
    plt.tight_layout()

    if save_name is None:
        plt.savefig(title + ".pdf", bbox_inches="tight")
    else:
        plt.savefig(save_name, bbox_inches="tight")

    plt.show()



def plot_mass_age_rot(
    cluster,
    df_summary_posterior,
    method,
    weight=False,
    title=None,
    save_name=None,
    age_xlim=None,
    legend_loc='upper left',
):
    """
    Two-column friendly layout (rotated):
    Top: Mass–Age relation with Age on x-axis and Mass on y-axis.
    Bottom: Cluster age posterior with Age on x-axis.
    Two panels share the same x-axis and are fully attached (no gap).
    """

    # ============================================================
    # Configuration (visual style)
    # ============================================================
    sigma_color = "orange"
    sigma_alpha = 0.2

    # ---- Font sizes (adjust here) ----
    label_fs = 14
    tick_fs = 12
    legend_fs = 12

    if title != False:
        if title is None:
            title = f"Mass-Age relation of the {cluster}"

    # ============================================================
    # Step 1. Load individual stellar age posteriors
    # ============================================================
    post_paths = [
        p for p in df_summary_posterior["posterior_path"].values
        if not pd.isna(p)
    ]

    if weight == False:
        weight_array = np.ones_like(df_summary_posterior["age"].values)
    else:
        weight_array = df_summary_posterior["BEST_HYP_MAX_PROB"].values

    star_age_samples = []
    for path in post_paths:
        data = np.load(path, allow_pickle=True)
        samples = data["samples"]
        logage = samples[:, :, 1]
        star_age_samples.append(10 ** (logage - 6.0))  # Myr

    # ============================================================
    # Step 2. Combine cluster age posterior
    # ============================================================
    tau_grid, P_cluster = combine_posteriors_age_unflat(star_age_samples, weight_array)

    cdf = np.cumsum(P_cluster)
    cdf /= cdf[-1]

    low = tau_grid[np.searchsorted(cdf, 0.16)]
    med = tau_grid[np.searchsorted(cdf, 0.50)]
    high = tau_grid[np.searchsorted(cdf, 0.84)]

    err_low = med - low
    err_high = high - med

    # ============================================================
    # Step 3. Determine age x-limits from stellar age error bars
    # ============================================================
    x_min = np.min(
        df_summary_posterior["age"] - df_summary_posterior["age_lower_uncertainty"]
    )
    x_max = np.max(
        df_summary_posterior["age"] + df_summary_posterior["age_upper_uncertainty"]
    )

    x_pad = 0.05 * (x_max - x_min)
    x_min -= x_pad
    x_max += x_pad

    # ============================================================
    # Step 4. Create figure (share x, no gap)
    # ============================================================
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1,
        figsize=(6, 8),  # keep width; increase height if needed
        sharex=True,
        gridspec_kw={
            "height_ratios": [3, 1],
            "hspace": 0.0,   # 🔥 fully attached
        },
    )

    # ============================================================
    # Top panel: Mass–Age relation (Age on x, Mass on y)
    # ============================================================
    ax_top.axvspan(
        low, high,
        facecolor=sigma_color,
        alpha=sigma_alpha,
        edgecolor="none",
        zorder=0,
    )

    ax_top.errorbar(
        x=df_summary_posterior["age"],
        y=df_summary_posterior["mass"],
        xerr=df_summary_posterior[
            ["age_lower_uncertainty", "age_upper_uncertainty"]
        ].values.T,
        yerr=df_summary_posterior[
            ["mass_lower_uncertainty", "mass_upper_uncertainty"]
        ].values.T,
        fmt="o",
        zorder=2,
    )

    ax_top.set_ylabel(r"Mass ($M_\odot$)", fontsize=label_fs)

    if title != False:
        ax_top.set_title(title)  # title font unchanged

    ax_top.tick_params(axis="both", labelsize=tick_fs)
    ax_top.tick_params(labelbottom=False)  # hide x tick labels on top

    legend_patch = Patch(
        facecolor=sigma_color,
        alpha=sigma_alpha,
        edgecolor="none",
        label=(
            rf"Cluster Age" + "\n" + rf" ({method}):" + "\n"
            rf"${med:.0f}^{{+{err_high:.0f}}}_{{-{err_low:.0f}}}\,\mathrm{{Myr}}$"
        ),
    )

    # ============================================================
    # Bottom panel: Cluster age posterior (Age on x)
    # ============================================================
    ax_bot.plot(tau_grid, P_cluster, color="black", lw=2)

    ax_bot.fill_between(
        tau_grid,
        0.0,
        P_cluster,
        color="gray",
        alpha=0.3,
    )

    ax_bot.axvspan(
        low, high,
        facecolor=sigma_color,
        alpha=sigma_alpha,
        edgecolor="none",
        zorder=1,
    )

    ax_bot.set_xlabel("Age (Myr)", fontsize=label_fs)
    ax_bot.set_ylabel("Probability density", fontsize=label_fs)
    ax_bot.tick_params(axis="both", labelsize=tick_fs)
    ax_bot.set_title("")
    ax_bot.legend(
        handles=[legend_patch],
        frameon=False,
        fontsize=legend_fs,
        loc=legend_loc,
    )

    # x-range: user override > auto
    if age_xlim is not None:
        ax_bot.set_xlim(age_xlim)
    else:
        ax_bot.set_xlim(x_min, x_max)

    # ============================================================
    # Final layout and output (avoid tight_layout)
    # ============================================================
    plt.subplots_adjust(hspace=0)

    if save_name is None:
        outname = (title if title else f"Mass-Age relation of the {cluster}") + ".pdf"
        plt.savefig(outname, bbox_inches="tight")
    else:
        plt.savefig(save_name, bbox_inches="tight")

    plt.show()





