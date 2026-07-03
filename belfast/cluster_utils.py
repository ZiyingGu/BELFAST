from belfast.target_utils import query_data, cal_obs_mag, cal_abs_mag, remove_extinction
from belfast.isochrone_utils import rearrange_isochrones, read_isochrones, check_rearranged_isochrones

import re
from glob import glob
from scipy.stats import gaussian_kde
from scipy.special import logsumexp
import numpy as np
from math import erf, sqrt

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
def posterior_summary(samples_2d, sigma=1):
    """
    Return median, upper and lower uncertainties.

    sigma=1 gives the central 1-sigma interval.
    sigma=2 gives the central 2-sigma interval.
    """
    samples_flat = samples_2d.reshape(-1)

    median = np.percentile(samples_flat, 50)

    # Gaussian-equivalent central interval
    cdf_upper = 0.5 * (1 + erf(sigma / sqrt(2)))
    cdf_lower = 1 - cdf_upper

    p_upper = np.percentile(samples_flat, 100 * cdf_upper)
    p_lower = np.percentile(samples_flat, 100 * cdf_lower)

    upper = p_upper - median
    lower = median - p_lower

    return median, upper, lower


def summarize_cluster_age(
    cluster: str,
    samples_glob: str,
    df_targets: pd.DataFrame,
    save_name: str = None,
    print_flag: bool = True,
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
        blobs = data["blobs"]

        # Reorder to (n_walkers, n_steps) for time-series analysis
        # Parameter indices are assumed to be fixed by the sampling code
        mass_index = data['output_list'].tolist().index('Mass')
        mass_chain = blobs[:, :, mass_index].T
        logage_index = data['param_names'].tolist().index('logAge')
        logage_chain = samples[:, :, logage_index].T
        u_index = data['output_list'].tolist().index('u')
        u_chain = blobs[:, :, u_index].T
        sigma_index = data['param_names'].tolist().index('sigma')
        sigma_chain = samples[:, :, sigma_index].T


        # n_steps = logage_chain.shape[1]

        # ------------------------------------------------------------------
        # Step 2. Estimate burn-in using autocorrelation time (ACT)
        # ------------------------------------------------------------------
        # half_idx = int(0.5 * n_steps)
        # third_idx = int(0.7 * n_steps)
        #
        # act_half = compute_act(logage_chain[:, half_idx:])
        # act_third = compute_act(logage_chain[:, third_idx:])
        # act_max = np.nanmax([act_half, act_third])
        #
        # burnin = int(min(n_steps, 10 * act_max))
        # if not np.isfinite(burnin) or burnin >= n_steps:
        #     burnin = n_steps // 2
        burnin = int(data['n_discard'])

        # ------------------------------------------------------------------
        # Step 3. Extract post–burn-in samples
        # ------------------------------------------------------------------
        mass_post = mass_chain[:, burnin:]
        age_post = 10 ** (logage_chain[:, burnin:] - 6.0)  # convert to Myr
        u_post = u_chain[:, burnin:]
        sigma_post = sigma_chain[:, burnin:]

        # ------------------------------------------------------------------
        # Step 4. Posterior summaries (median ± uncertainties)
        # ------------------------------------------------------------------
        mass_med, mass_up, mass_low = posterior_summary(mass_post, sigma=1)
        age_med, age_up, age_low = posterior_summary(age_post, sigma=1)
        u_med, u_up, u_low = posterior_summary(u_post, sigma=2)
        sig_med, sig_up, sig_low = posterior_summary(sigma_post, sigma=1)

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
                "u": u_med,
                "u_upper_uncertainty": u_up,
                "u_lower_uncertainty": u_low,
                "sigma": sig_med,
                "sigma_upper_uncertainty": sig_up,
                "sigma_lower_uncertainty": sig_low,
            }
        )

    df_post = pd.DataFrame(records)

    # ------------------------------------------------------------------
    # Step 5. Construct posterior_name for target table
    # ------------------------------------------------------------------
    def get_col_value(row, target_name):
        """
        兼容 'HD ID' / 'HD_ID' / 'HD  ID' 这种列名差异
        """

        def norm(s):
            return re.sub(r"[\s_]+", " ", str(s)).strip().lower()

        target = norm(target_name)

        for col in row.index:
            if norm(col) == target:
                return row[col]

        return np.nan

    def build_posterior_name(row):
        hip_id = get_col_value(row, "HIP ID")
        hd_id = get_col_value(row, "HD ID")

        if not pd.isna(hip_id):
            return f"HIP {int(float(hip_id))}"

        if not pd.isna(hd_id):
            return f"HD {int(float(hd_id))}"

        other_name = get_col_value(row, "Other Name")
        return re.sub(r"\s+", " ", str(other_name)).strip()

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
        weight_array = df_summary_posterior[weight].values

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



def plot_mass_age_rot90(
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
        weight_array = df_summary_posterior[weight].values

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



from pathlib import Path
from urllib.request import urlopen
from urllib.parse import quote

import numpy as np
import pandas as pd
from astroquery.vizier import Vizier


def safe_catalog_name(catalog_id):
    return (
        str(catalog_id)
        .strip()
        .replace("/", "_")
        .replace(" ", "_")
        .replace(":", "_")
    )


def value_to_clean_str(x):
    """
    Convert Astropy / numpy scalar to clean string.
    Mainly used for Gaia source IDs.
    """
    if np.ma.is_masked(x):
        return ""

    if x is None:
        return ""

    if isinstance(x, (bytes, bytearray)):
        return x.decode("utf-8", errors="ignore").strip()

    try:
        if hasattr(x, "item"):
            x = x.item()
    except Exception:
        pass

    if isinstance(x, str):
        s = x.strip()
        return "" if s.lower() in ["nan", "none", "--", ""] else s

    if isinstance(x, (int, np.integer)):
        return str(int(x))

    if isinstance(x, (float, np.floating)):
        if not np.isfinite(x):
            return ""
        if float(x).is_integer():
            return f"{x:.0f}"
        return str(x)

    s = str(x).strip()
    return "" if s.lower() in ["nan", "none", "--", ""] else s


def find_source_column(tab, preferred_source_cols=None):
    """
    Find the source-ID column in a VizieR table.

    Priority examples:
        SourceCP, Source, ID, GaiaDR2, GaiaDR3, source_id, SourceID
    """
    if preferred_source_cols is None:
        preferred_source_cols = [
            "SourceCP",
            "Source",
            "source",
            "source_id",
            "SourceID",
            "GaiaDR2",
            "GaiaDR3",
            "GaiaDR2Source",
            "GaiaDR3Source",
            "Gaia",
            "ID",
            "id",
        ]

    colnames = list(tab.colnames)
    lower_to_real = {c.lower(): c for c in colnames}

    # exact priority match
    for c in preferred_source_cols:
        if c in colnames:
            return c
        if c.lower() in lower_to_real:
            return lower_to_real[c.lower()]

    # fallback: fuzzy match
    keywords = ["sourcecp", "source", "sourceid", "source_id", "gaiadr2", "gaiadr3", "gaia", "id"]

    for key in keywords:
        for c in colnames:
            if key in c.lower().replace("-", "").replace("_", ""):
                return c

    return None


def download_vizier_catalog_with_target_name(
    catalog_id,
    output_dir=None,
    row_limit=-1,
    target_prefix="Gaia DR2 ",
    target_col_name="target_name",
    preferred_source_cols=None,
    save_readme=True,
    verbose=True,
):
    """
    Download VizieR catalog ReadMe and all tables.

    For each table, add a first column:

        target_name = target_prefix + str(source_id)

    where source_id is taken from SourceCP / Source / ID / etc.

    Example
    -------
    res = download_vizier_catalog_with_target_name("J/A+A/623/A35")
    """

    catalog_id = str(catalog_id).strip()
    safe_name = safe_catalog_name(catalog_id)

    if output_dir is None:
        output_dir = Path(f"./vizier_{safe_name}")
    else:
        output_dir = Path(output_dir)

    output_dir = Path(str(output_dir).replace(" ", "_"))
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # 1. Save ReadMe
    # ------------------------------------------------------------
    readme_path = output_dir / f"{safe_name}_readme.txt"

    if save_readme:
        readme_url = "https://cdsarc.cds.unistra.fr/ftp/" + quote(catalog_id, safe="/+") + "/ReadMe"

        try:
            with urlopen(readme_url, timeout=60) as response:
                readme_text = response.read().decode("utf-8", errors="replace")

            readme_path.write_text(readme_text, encoding="utf-8")

            if verbose:
                print(f"[saved ReadMe] {readme_path}")

        except Exception as e:
            if verbose:
                print("[warning] Failed to download ReadMe")
                print(f"          {readme_url}")
                print(f"          {type(e).__name__}: {e}")

    # ------------------------------------------------------------
    # 2. Download all tables
    # ------------------------------------------------------------
    viz = Vizier(columns=["**"], row_limit=row_limit)
    tables = viz.get_catalogs(catalog_id)

    table_summary_rows = []
    saved_table_paths = []

    if verbose:
        print(f"[catalog] {catalog_id}")
        print(f"[tables] {list(tables.keys())}")

    for table_index, table_name in enumerate(tables.keys(), start=1):
        tab = tables[table_name]

        source_col = find_source_column(
            tab,
            preferred_source_cols=preferred_source_cols,
        )

        df = tab.to_pandas()

        # 如果原本已经有 target_name，先删掉，避免 insert 报错
        if target_col_name in df.columns:
            df = df.drop(columns=[target_col_name])

        if source_col is None:
            target_names = [""] * len(tab)
            if verbose:
                print(f"[warning] table{table_index}: no SourceCP/Source/ID-like column found.")
        else:
            target_names = [
                target_prefix + value_to_clean_str(v)
                for v in tab[source_col]
            ]

        # 插入为第一列
        df.insert(0, target_col_name, target_names)

        table_csv_path = output_dir / f"{safe_name}_table{table_index}.csv"
        df.to_csv(table_csv_path, index=False)

        saved_table_paths.append(table_csv_path)

        table_summary_rows.append({
            "table_index": table_index,
            "vizier_table_name": table_name,
            "saved_csv": str(table_csv_path),
            "source_column_used": source_col if source_col is not None else "",
            "target_prefix": target_prefix,
            "n_rows_downloaded": len(df),
            "n_columns_saved": len(df.columns),
            "columns": ", ".join(df.columns),
            "description": tab.meta.get("description", ""),
        })

        if verbose:
            print(f"[saved table{table_index}] {table_name}")
            print(f"              source column : {source_col}")
            print(f"              csv           : {table_csv_path}")
            print(f"              rows={len(df)}, cols={len(df.columns)}")

    # ------------------------------------------------------------
    # 3. Save summary
    # ------------------------------------------------------------
    table_summary_path = output_dir / f"{safe_name}_table_summary.csv"
    df_table_summary = pd.DataFrame(table_summary_rows)
    df_table_summary.to_csv(table_summary_path, index=False)

    if verbose:
        print(f"[saved summary] {table_summary_path}")

    return {
        "catalog_id": catalog_id,
        "safe_name": safe_name,
        "output_dir": output_dir,
        "readme_path": readme_path,
        "table_paths": saved_table_paths,
        "table_summary_path": table_summary_path,
        "table_summary": df_table_summary,
        "tables": tables,
    }



from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import numpy as np
import time

# from belfast.target_utils import query_data


BANYAN_COLS = [
    "ra", "dec",
    "pmra", "pmdec",
    "pmra_error", "pmdec_error",
    "parallax", "parallax_error",
    "rv", "rv_error",
]


def _clean_target_name(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if s.lower() in ["", "nan", "none", "--"]:
        return ""
    return s


def _query_one_target_safe_with_banyan(
    target_name,
    index=None,
    max_retries=3,
    query_verbose=False,
):
    """
    Query one target with query_data().
    Always returns a dict.
    """

    target_name = _clean_target_name(target_name)

    out_base = {
        "_input_index": index,
        "query_input_name": target_name,
        "query_failed": True,
        "query_error": "",
    }

    if target_name == "":
        out_base["query_error"] = "Empty target name"
        for c in BANYAN_COLS:
            out_base[c] = np.nan
        return out_base

    try:
        df_q = query_data(
            target_name,
            max_retries=max_retries,
            return_df=True,
            return_banyan_cols=True,
            verbose=query_verbose,
        )

        if df_q is None or len(df_q) == 0:
            raise RuntimeError("query_data returned empty DataFrame.")

        out = df_q.iloc[0].to_dict()

        out["_input_index"] = index
        out["query_input_name"] = target_name
        out["query_failed"] = False
        out["query_error"] = ""

        # 保证 BANYAN_COLS 一定存在；缺的填 NaN
        for c in BANYAN_COLS:
            if c not in out:
                out[c] = np.nan

        return out

    except Exception as e:
        out_base["query_error"] = f"{type(e).__name__}: {e}"
        for c in BANYAN_COLS:
            out_base[c] = np.nan
        return out_base


def query_data_many_with_banyan(
    target_names,
    max_retries=3,
    n_workers=4,
    query_verbose=False,
    verbose=True,
):
    """
    Query many targets in parallel.
    Return one DataFrame.
    """

    target_names = list(target_names)
    n_targets = len(target_names)

    if n_targets == 0:
        return pd.DataFrame()

    if verbose:
        print(f"[n targets] {n_targets}")
        print(f"[n workers] {n_workers}")

    results = [None] * n_targets
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        future_to_i = {}

        for i, target_name in enumerate(target_names):
            future = executor.submit(
                _query_one_target_safe_with_banyan,
                target_name,
                i,
                max_retries,
                query_verbose,
            )
            future_to_i[future] = i

        for n_done, future in enumerate(as_completed(future_to_i), start=1):
            i = future_to_i[future]

            try:
                result = future.result()
            except Exception as e:
                result = {
                    "_input_index": i,
                    "query_input_name": _clean_target_name(target_names[i]),
                    "query_failed": True,
                    "query_error": f"{type(e).__name__}: {e}",
                }
                for c in BANYAN_COLS:
                    result[c] = np.nan

            results[i] = result

            if verbose:
                status = "FAILED" if result.get("query_failed", True) else "OK"
                print(f"[{n_done}/{n_targets}] {status}: {result.get('query_input_name', '')}")

    df_query = pd.DataFrame(results)

    # 保证顺序和输入一致
    if "_input_index" in df_query.columns:
        df_query = df_query.sort_values("_input_index").reset_index(drop=True)
        df_query = df_query.drop(columns=["_input_index"])

    # 保证 BANYAN_COLS 都存在
    for c in BANYAN_COLS:
        if c not in df_query.columns:
            df_query[c] = np.nan

    if verbose:
        elapsed = time.time() - start_time
        n_failed = int(df_query["query_failed"].sum()) if "query_failed" in df_query.columns else 0
        print("\nDone.")
        print(f"[failed] {n_failed}/{n_targets}")
        print(f"[elapsed] {elapsed:.1f} sec")

    return df_query


def append_query_data_and_banyan_cols_to_table(
    input_table,
    output_table,
    target_col=None,
    max_retries=3,
    n_workers=4,
    query_verbose=False,
    verbose=True,
    read_original_as_str=True,
    drop_existing_banyan_cols=True,
):
    """
    Read a CSV table.
    First column or target_col is target_name.
    Query data with query_data(..., return_banyan_cols=True).
    Save:

        original df
        + query_data columns
        + BANYAN_COLS

    Parameters
    ----------
    input_table : str or Path
        Input CSV path.

    output_table : str or Path
        User-specified output CSV path.

    target_col : str or None
        Target-name column. If None, use first column.

    drop_existing_banyan_cols : bool
        If True, remove old BANYAN_COLS from original df before appending new ones.
    """

    input_table = Path(input_table)
    output_table = Path(output_table)

    if read_original_as_str:
        df_in = pd.read_csv(input_table, dtype=str)
    else:
        df_in = pd.read_csv(input_table)

    if len(df_in.columns) == 0:
        raise ValueError("Input table has no columns.")

    if target_col is None:
        target_col = df_in.columns[0]

    if target_col not in df_in.columns:
        raise ValueError(f"target_col={target_col} not found in input table.")

    target_names = df_in[target_col].tolist()

    if verbose:
        print(f"[input] {input_table}")
        print(f"[output] {output_table}")
        print(f"[target column] {target_col}")
        print(f"[n rows] {len(df_in)}")

    df_query = query_data_many_with_banyan(
        target_names,
        max_retries=max_retries,
        n_workers=n_workers,
        query_verbose=query_verbose,
        verbose=verbose,
    )

    # Rename Gaia DR3 RUWE to avoid conflict with existing DR2 RUWE
    if "RUWE" in df_query.columns:
        df_query = df_query.rename(columns={"RUWE": "RUWE_DR3"})

    # Move RUWE_DR3 right after P_binary
    if "P_binary" in df_query.columns and "RUWE_DR3" in df_query.columns:
        cols = list(df_query.columns)
        cols.remove("RUWE_DR3")
        insert_pos = cols.index("P_binary") + 1
        cols.insert(insert_pos, "RUWE_DR3")
        df_query = df_query[cols]

    # ------------------------------------------------------------
    # 1. 原始表：可选删除旧 BANYAN_COLS，避免重复
    # ------------------------------------------------------------
    df_in_clean = df_in.copy()

    if drop_existing_banyan_cols:
        old_banyan_cols = [c for c in BANYAN_COLS if c in df_in_clean.columns]

        if old_banyan_cols and verbose:
            print("[drop old BANYAN columns from input table]")
            print(old_banyan_cols)

        df_in_clean = df_in_clean.drop(columns=old_banyan_cols)

    # ------------------------------------------------------------
    # 2. 从 query 结果里分离 BANYAN_COLS
    #    注意顺序固定：rv 在 parallax_error 后面
    # ------------------------------------------------------------
    for c in BANYAN_COLS:
        if c not in df_query.columns:
            df_query[c] = np.nan

    df_banyan = df_query[BANYAN_COLS].copy()

    # ------------------------------------------------------------
    # 3. 普通 query_data columns
    #    去掉：
    #      - BANYAN_COLS，因为要放最后
    #      - 和原始表重复的列，避免重复
    # ------------------------------------------------------------
    df_query_rest = df_query.drop(columns=BANYAN_COLS)

    duplicated_cols = [c for c in df_query_rest.columns if c in df_in_clean.columns]

    if duplicated_cols and verbose:
        print("[drop duplicated query columns]")
        print(duplicated_cols)

    df_query_rest = df_query_rest.drop(columns=duplicated_cols)

    # ------------------------------------------------------------
    # 4. 拼接顺序：
    #    original df + query_data columns + banyan cols
    # ------------------------------------------------------------
    df_out = pd.concat(
        [
            df_in_clean.reset_index(drop=True),
            df_query_rest.reset_index(drop=True),
            df_banyan.reset_index(drop=True),
        ],
        axis=1,
    )

    output_table.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(output_table, index=False)

    if verbose:
        n_failed = int(df_out["query_failed"].sum()) if "query_failed" in df_out.columns else 0
        print(f"\n[saved] {output_table}")
        print(f"[failed] {n_failed}/{len(df_out)}")

    return df_out



import numpy as np
import pandas as pd
import re


def parallax_to_distance(parallax_mas, parallax_err_mas):
    if not np.isfinite(parallax_mas) or parallax_mas <= 0:
        return np.nan, np.nan
    d = 1000.0 / parallax_mas
    d_err = 1000.0 * parallax_err_mas / parallax_mas**2 if np.isfinite(parallax_err_mas) else np.nan
    return d, d_err


def interp_nan_safe(x, y, x_new):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)

    if good.sum() < 2 or not np.isfinite(x_new):
        return np.nan

    x = x[good]
    y = y[good]
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    if x_new < x[0] or x_new > x[-1]:
        return np.nan

    return np.interp(x_new, x, y)


def split_gtomo_output_by_input_targets(input_csv, output_csv):
    """
    input_csv:
        one column named 'targets'

    output_csv:
        G-Tomo 1D output with columns:
        0_x, 0_y, 0_erry, 0_yc, 0_erryc, ...
    """

    df_input = pd.read_csv(input_csv)
    df_output = pd.read_csv(output_csv)

    if "targets" not in df_input.columns:
        raise ValueError("input_csv must contain a column named 'targets'.")

    target_names = df_input["targets"].astype(str).str.strip().tolist()
    profile_dict = {}

    for i, target_name in enumerate(target_names):
        cols = {
            "distance": f"{i}_x",
            "d_extin": f"{i}_y",
            "d_extin_err": f"{i}_erry",
            "c_extin": f"{i}_yc",
            "c_extin_err": f"{i}_erryc",
        }

        missing = [v for v in cols.values() if v not in df_output.columns]
        if missing:
            raise ValueError(f"Missing G-Tomo columns for target {target_name}: {missing}")

        profile_dict[target_name] = (
            df_output[list(cols.values())]
            .rename(columns={v: k for k, v in cols.items()})
            .copy()
        )

    return profile_dict
def add_gtomo_from_input_output(
    df_targets_path,
    input_csv,
    output_csv,
    target_col="target_name",
    parallax_col="parallax",
    parallax_err_col="parallax_err",
    A_col="A550",
    A_err_col="A550_err",
    A_floor=None,
    insert_after="Ks_err",
    g_mag_col="G_mag",
    g_err_col="G_err",
    bp_mag_col="G_BP_mag",
    bp_err_col="G_BP_err",
    rp_mag_col="G_RP_mag",
    rp_err_col="G_RP_err",
    mag_col="MG",
    color_col="MBP_MRP",
    extinction_type="A_V",
):
    """
    Add G-Tomo cumulative extinction to df_targets_path.

    Also calculate extinction-corrected absolute Gaia quantities:

        MG      = M_G
        MBP_MRP = M_G_BP - M_G_RP

    Steps:
        1. Interpolate G-Tomo cumulative extinction.
        2. Remove extinction from G, G_BP, G_RP.
        3. Convert dereddened apparent magnitudes to absolute magnitudes.
        4. Compute MG and MBP_MRP.
    """

    def _safe_float(x):
        try:
            if x is None:
                return np.nan
            s = str(x).strip()
            if s.lower() in ["", "nan", "none", "--"]:
                return np.nan
            return float(s)
        except Exception:
            return np.nan

    profile_dict = split_gtomo_output_by_input_targets(input_csv, output_csv)

    df_targets = pd.read_csv(df_targets_path)

    # 防止重复运行时出现重复列
    gtomo_cols = [
        "distance_pc_from_parallax",
        "distance_pc_err_from_parallax",
        A_col,
        f"{A_col}_map_err",
        f"{A_col}_dist_err",
        f"{A_col}_formal_err",
        A_err_col,
    ]

    old_cols = [c for c in gtomo_cols + [mag_col, color_col] if c in df_targets.columns]
    if old_cols:
        df_targets = df_targets.drop(columns=old_cols)

    needed_cols = [
        target_col,
        parallax_col,
        parallax_err_col,
        g_mag_col,
        g_err_col,
        bp_mag_col,
        bp_err_col,
        rp_mag_col,
        rp_err_col,
    ]
    missing_cols = [c for c in needed_cols if c not in df_targets.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    rows = []

    for _, row in df_targets.iterrows():
        target_name = str(row[target_col]).strip()

        if target_name not in profile_dict:
            raise KeyError(f"{target_name} not found in G-Tomo input target list.")

        prof = profile_dict[target_name]

        parallax_obs = _safe_float(row[parallax_col])
        parallax_obs_err = _safe_float(row[parallax_err_col])

        distance, distance_err = parallax_to_distance(
            parallax_obs,
            parallax_obs_err,
        )

        x = prof["distance"].to_numpy(dtype=float)
        A_cum = prof["c_extin"].to_numpy(dtype=float)
        A_cum_err = prof["c_extin_err"].to_numpy(dtype=float)

        A_center = interp_nan_safe(x, A_cum, distance)
        A_map_err = interp_nan_safe(x, A_cum_err, distance)

        if np.isfinite(distance) and np.isfinite(distance_err):
            A_minus = interp_nan_safe(x, A_cum, max(distance - distance_err, 0.0))
            A_plus = interp_nan_safe(x, A_cum, distance + distance_err)

            delta_list = []
            if np.isfinite(A_minus) and np.isfinite(A_center):
                delta_list.append(abs(A_minus - A_center))
            if np.isfinite(A_plus) and np.isfinite(A_center):
                delta_list.append(abs(A_plus - A_center))

            A_dist_err = max(delta_list) if delta_list else np.nan
        else:
            A_dist_err = np.nan

        err_terms = [v for v in [A_map_err, A_dist_err] if np.isfinite(v)]
        A_formal_err = np.sqrt(np.sum(np.asarray(err_terms) ** 2)) if err_terms else np.nan

        if A_floor is not None:
            A_used_err = max(A_formal_err, A_floor) if np.isfinite(A_formal_err) else A_floor
        else:
            A_used_err = A_formal_err

        # ------------------------------------------------------------
        # Calculate extinction-corrected absolute MG and MBP_MRP
        # ------------------------------------------------------------
        g_mag = _safe_float(row[g_mag_col])
        g_err = _safe_float(row[g_err_col])

        bp_mag = _safe_float(row[bp_mag_col])
        bp_err = _safe_float(row[bp_err_col])

        rp_mag = _safe_float(row[rp_mag_col])
        rp_err = _safe_float(row[rp_err_col])

        MG = np.nan
        MBP_MRP = np.nan

        common_ok = (
            np.isfinite(parallax_obs)
            and np.isfinite(parallax_obs_err)
            and parallax_obs > 0
            and np.isfinite(A_center)
            and np.isfinite(A_used_err)
        )

        if common_ok:
            obs_mags = np.array([g_mag, bp_mag, rp_mag], dtype=float)
            obs_err_mags = np.array([g_err, bp_err, rp_err], dtype=float)
            bands_list = ["G", "G_BP", "G_RP"]

            if np.isfinite(obs_mags).all() and np.isfinite(obs_err_mags).all():
                obs_mags_dered, obs_err_mags_dered = remove_extinction(
                    obs_mags,
                    obs_err_mags,
                    bands_list,
                    extinction=A_center,
                    extinction_err=A_used_err,
                    extinction_type=extinction_type,
                    print_flag=False,
                )

                abs_mags_dered, abs_err_mags_dered = cal_abs_mag(
                    obs_mags_dered,
                    obs_err_mags_dered,
                    parallax_obs,
                    parallax_obs_err,
                )

                MG = abs_mags_dered[0]
                MBP_MRP = abs_mags_dered[1] - abs_mags_dered[2]

        rows.append({
            "distance_pc_from_parallax": distance,
            "distance_pc_err_from_parallax": distance_err,
            A_col: A_center,
            f"{A_col}_map_err": A_map_err,
            f"{A_col}_dist_err": A_dist_err,
            f"{A_col}_formal_err": A_formal_err,
            A_err_col: A_used_err,
            mag_col: MG,
            color_col: MBP_MRP,
        })

    df_add = pd.DataFrame(rows, index=df_targets.index)
    df_out = pd.concat([df_targets.copy(), df_add], axis=1)

    move_cols = [
        "distance_pc_from_parallax",
        "distance_pc_err_from_parallax",
        A_col,
        f"{A_col}_map_err",
        f"{A_col}_dist_err",
        f"{A_col}_formal_err",
        A_err_col,
    ]

    # 先排除 G-Tomo 列、MG、MBP_MRP
    cols = [c for c in df_out.columns if c not in move_cols + [mag_col, color_col]]

    insert_pos = cols.index(insert_after) + 1 if insert_after in cols else len(cols)

    for j, c in enumerate(move_cols):
        cols.insert(insert_pos + j, c)

    # 最后放 MG, MBP_MRP，保证 MG 在 color 前面
    cols.append(mag_col)
    cols.append(color_col)

    return df_out[cols]
