import re
import numpy as np
from glob import glob

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

from pa.lib import star as st
from astropy import units as u

from belfast.config import GM_SUN, R_SUN
from belfast.paths import KLD_DIR, PARS_FILTER_DIR
from belfast.PARS_utils import extract_MH, make_limb_darkening_files


# default plotting configurations
OUTPUTS_with_inc = [
    "Mass", "omega", "logL", "logTe", "Req", "Rpol",
    "G", "G_BP", "G_RP", "B_T", "V_T", "J", "H", "Ks",
]

OUTPUTS_without_inc = [
    "Mass", "omega", "logL", "logTe", "Req", "Rpol",
]

LINESTYLES = ["-", ":", "-.", (0, (5, 1)), (0, (3, 1, 1, 1))]

CMD = {
    "2mass": (("J", "Ks"), "J", r"$J-K_s$", r"$M_J$", "2MASS"),
    "gaiadr3": (("G_BP", "G_RP"), "G", r"$G_{\rm BP}-G_{\rm RP}$", r"$M_G$", "Gaia DR3"),
    "tycho2": (("B_T", "V_T"), "V_T", r"$B_T-V_T$", r"$M_{V_T}$", "Tycho-2"),
}

TAB_COLORS = [
    "tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
    "tab:brown", "tab:pink", "tab:gray", "tab:olive", "tab:cyan",
]


def create_interp_func_PARS(
    interp_func,
    PARS_limb_darkening_path=KLD_DIR / "i[pm]*k*.pck*",
    PARS_filters=PARS_FILTER_DIR / "*.dat",
    PARS_nz=100,
    PARS_MH_offset=-0.046,
    show_progress=True,
):
    """
    Create a PARS interpolator returning stellar parameters and photometry.

    Args:
        interp_func: Interpolator returning Mass, omega, logL, logTe, Req, and Rpol.
        PARS_limb_darkening_path (str or Path): Pattern for limb-darkening model files.
        PARS_filters (str or Path): Pattern for PARS filter files.
        PARS_nz (int): Number of points used in the stellar integration.
        PARS_MH_offset (float): Metallicity offset applied to the input [M/H].
        show_progress (bool): Whether to display progress during model preparation.

    Returns:
        Callable: An interpolator accepting parameters ordered as Mini, Age, MH, omega_ini,
            and Inclination. Returns 14 outputs ordered according to OUTPUTS_with_inc.
            Failed evaluations are represented by NaN.
    """
    output_bands = OUTPUTS_with_inc[6:]
    distance_cm = 10 * u.pc.to(u.cm)

    limb_darkening_MH_dic = {
        file: mh for file in sorted(glob(str(PARS_limb_darkening_path)))
        if (mh := extract_MH(file)) is not None
    }
    if not limb_darkening_MH_dic:
        raise RuntimeError(f"No limb-darkening files found: {PARS_limb_darkening_path}")

    made_limb_darkening_dict = make_limb_darkening_files(
        limb_darkening_MH_dic, str(PARS_filters), show_progress=show_progress
    )
    if not made_limb_darkening_dict:
        raise RuntimeError("Failed to load PARS limb-darkening models.")

    sorted_MH = np.array(sorted(made_limb_darkening_dict), dtype=float)
    reference_model = made_limb_darkening_dict[sorted_MH[0]]
    missing_bands = [band for band in output_bands if band not in reference_model.bands]
    if missing_bands:
        raise ValueError(f"Missing required bands in PARS models: {missing_bands}")

    output_rearrange_index = [reference_model.bands.index(band) for band in output_bands]

    def calculate_PARS_mags(model_output, inclination, limb_darkening_model):
        Mass, omega, logL, logTe, Req, Rpol = model_output
        omega_angular = omega * np.sqrt(GM_SUN * Mass / (R_SUN * 1.5 * Rpol) ** 3)
        omega_keplerian = np.sqrt(GM_SUN * Mass / (R_SUN * Req) ** 3)
        omega_k = omega_angular / omega_keplerian

        star = st.Star(
            omega_k, 10**logL, Mass, Req, distance_cm, PARS_nz, limb_darkening_model
        )
        original_mags = np.asarray(star.integrate(float(inclination)), dtype=float)
        return original_mags[output_rearrange_index]

    def interp_func_PARS(params):
        params = np.asarray(params, dtype=float)
        single_input = params.ndim == 1

        if single_input:
            if params.shape != (5,):
                raise ValueError("A single input must have shape (5,).")
            params = params[np.newaxis, :]
        elif params.ndim != 2 or params.shape[1] != 5:
            raise ValueError("Input must have shape (5,) or (n, 5).")

        results = np.full((len(params), len(OUTPUTS_with_inc)), np.nan)
        valid_inputs = np.all(np.isfinite(params), axis=1)
        valid_indices = np.flatnonzero(valid_inputs)
        if valid_indices.size == 0:
            return results[0] if single_input else results

        model_outputs = np.full((len(params), 6), np.nan)
        try:
            batch_outputs = np.asarray(interp_func(params[valid_indices, :4]), dtype=float)
            if batch_outputs.shape == (6,) and len(valid_indices) == 1:
                batch_outputs = batch_outputs[np.newaxis, :]
            if batch_outputs.shape == (len(valid_indices), 6):
                model_outputs[valid_indices] = batch_outputs
            else:
                raise ValueError("Unexpected output shape from interp_func.")
        except Exception:
            for i in valid_indices:
                try:
                    output = np.asarray(interp_func(params[i, :4]), dtype=float).reshape(-1)
                    if output.shape == (6,):
                        model_outputs[i] = output
                except Exception:
                    continue

        results[:, :6] = model_outputs

        for i in valid_indices:
            model_output = model_outputs[i]
            if not np.all(np.isfinite(model_output)):
                continue

            try:
                mh_pars = params[i, 2] + PARS_MH_offset
                inclination = params[i, 4]
                exact_idx = np.flatnonzero(
                    np.isclose(sorted_MH, mh_pars, rtol=0, atol=1e-10)
                )

                if exact_idx.size:
                    model = made_limb_darkening_dict[sorted_MH[exact_idx[0]]]
                    mags = calculate_PARS_mags(model_output, inclination, model)
                else:
                    upper_idx = np.searchsorted(sorted_MH, mh_pars)
                    if upper_idx == 0 or upper_idx == len(sorted_MH):
                        continue

                    mh_lower, mh_upper = sorted_MH[upper_idx - 1:upper_idx + 1]
                    model_lower = made_limb_darkening_dict[mh_lower]
                    model_upper = made_limb_darkening_dict[mh_upper]
                    mags_lower = calculate_PARS_mags(model_output, inclination, model_lower)
                    mags_upper = calculate_PARS_mags(model_output, inclination, model_upper)
                    weight = (mh_pars - mh_lower) / (mh_upper - mh_lower)
                    mags = mags_lower + weight * (mags_upper - mags_lower)

                if mags.shape == (8,) and np.all(np.isfinite(mags)):
                    results[i, 6:] = mags
            except Exception:
                continue

        return results[0] if single_input else results

    return interp_func_PARS




# helper functions
def _vals(x):
    """Convert input to sorted, unique float values."""
    vals = [float(x)] if np.isscalar(x) else [float(v) for v in x]
    return sorted(dict.fromkeys(vals))


def _colors(values, colors=None):
    """Assign colors to values, cycling through the available colors."""
    values = list(values)
    if colors is None:
        colors = TAB_COLORS
    return {value: colors[i % len(colors)] for i, value in enumerate(values)}


def _strengths(vals, limits=(0.4, 1.0)):
    """Assign evenly spaced strengths to values within the given limits."""
    s = [1.0] if len(vals) == 1 else np.linspace(*limits, len(vals))
    return dict(zip(vals, map(float, s)))


def _shade(color, strength):
    """Blend a color with white according to the given strength."""
    rgb = np.asarray(mcolors.to_rgb(color))
    return tuple(1 - strength * (1 - rgb))


def _styles(vals, zero_dashed=False):
    """Assign line styles to values, optionally using dashed lines for zero."""
    out, j = {}, 0
    for v in vals:
        if zero_dashed and np.isclose(v, 0):
            out[v] = "--"
        else:
            out[v] = LINESTYLES[j % len(LINESTYLES)]
            j += 1
    return out


def _inc_alpha(inc, limits=(0.35, 1.0)):
    """Map inclination in radians to an alpha value within the given limits."""
    t = np.clip(float(inc) / (np.pi / 2), 0, 1)
    return limits[0] + t * (limits[1] - limits[0])


def _evaluate(interp_func, Mini, Age, MH, omega_ini, Inclination=0.0, grid_type='without_inc'):
    """Broadcast model coordinates and return named outputs."""
    if grid_type == 'without_inc':
        OUTPUTS = OUTPUTS_without_inc
        x = np.broadcast_arrays(*map(np.asarray, (Mini, Age, MH, omega_ini)))
    elif grid_type == 'with_inc':
        OUTPUTS = OUTPUTS_with_inc
        x = np.broadcast_arrays(*map(np.asarray, (Mini, Age, MH, omega_ini, Inclination)))
    else:
        raise ValueError("grid_type must be 'without_inc' or 'with_inc'.")
    shape = x[0].shape
    X = np.column_stack([np.asarray(v, float).ravel() for v in x])
    Y = np.asarray(interp_func(X), float)

    if Y.ndim == 1 and len(X) == 1:
        Y = Y[None, :]
    if Y.shape != (len(X), len(OUTPUTS)):
        raise ValueError(
            f"interp_func must return (N, {len(OUTPUTS)}), got {Y.shape}. "
            "Use a PARSEC v2 with_inc Mini/u interpolator."
        )
    return {name: Y[:, i].reshape(shape) for i, name in enumerate(OUTPUTS)}


def _cmd_config(band):
    """Return the CMD configuration for a supported photometric band."""
    key = re.sub(r"[^a-z0-9]", "", str(band).lower())
    key = {"gaia": "gaiadr3", "tycho": "tycho2"}.get(key, key)
    if key not in CMD:
        raise ValueError("band must be '2MASS', 'Gaia DR3', or 'Tycho-2'.")
    return CMD[key]


def _save(fig, save_name):
    """Save the figure if a filename is provided."""
    if save_name is not None:
        fig.savefig(save_name, bbox_inches="tight")


# ============================================================
# CMD
# ============================================================
def plot_cmd_isochrones(
    interp_func, Age, band, Mini=None, MH=0.0, omega_ini=0.6,
    inclination=(0.0, np.pi / 2), colors=None, grid_type="with_inc",
    inclination_alpha=(0.35, 1.0), figsize=(7, 7), lw=2.0,
    xlim=None, ylim=None, title=None, legend=True, save_name=None,
    ax=None, legend_loc='best', legend_bbox=None
):
    """Plot CMD isochrones on a new or existing axis.

    Age controls color, inclination controls transparency, and MH or omega_ini
    controls line style when multiple values are provided.

    Args:
        interp_func: Interpolator that evaluates stellar models.
        Age: Stellar ages in Myr. A scalar or sequence of values.
        band: Photometric system, such as '2MASS', 'Gaia DR3', or 'Tycho-2'.
        Mini: Initial stellar masses. Defaults to 1001 points from 1 to 10 solar masses.
        MH: Metallicity values. A scalar or sequence of values.
        omega_ini: Initial rotation rates. A scalar or sequence of values.
        inclination: Inclinations in radians.
        colors: Optional colors assigned to the age values.
        inclination_alpha: Minimum and maximum alpha values for inclination scaling.
        figsize: Figure size used when creating a new figure.
        lw: Line width of the isochrones and legend handles.
        xlim: Optional x-axis limits.
        ylim: Optional y-axis limits.
        title: Plot title. Defaults to the photometric system and fixed parameters.
        legend: Whether to display the legend.
        save_name: Optional path for saving the figure.
        ax: Existing Matplotlib axis. If None, a new figure and axis are created.

    Returns:
        tuple: The figure and axis containing the isochrones.
    """
    Mini = np.linspace(1, 10, 1001) if Mini is None else np.asarray(Mini, float)
    ages, mhs, omegas, incs = map(_vals, (Age, MH, omega_ini, inclination))
    if len(mhs) > 1 and len(omegas) > 1:
        raise ValueError("For CMD, MH and omega_ini cannot both be lists.")

    (b1, b2), mag, xlabel, ylabel, system = _cmd_config(band)
    age_color = _colors(ages, colors)
    mh_style, om_style = _styles(mhs), _styles(omegas, zero_dashed=True)
    style_by = "MH" if len(mhs) > 1 else "omega" if len(omegas) > 1 else None

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    else:
        fig = ax.figure

    for age in ages:
        for mh in mhs:
            for om in omegas:
                inc_now = [0.0] if np.isclose(om, 0) else incs
                ls = "--" if np.isclose(om, 0) else (
                    mh_style[mh] if style_by == "MH" else
                    om_style[om] if style_by == "omega" else "-"
                )
                for inc in inc_now:
                    iso = _evaluate(interp_func, Mini, age, mh, om, inc, grid_type=grid_type)
                    x, y = iso[b1] - iso[b2], iso[mag]
                    ok = np.isfinite(x) & np.isfinite(y)
                    ax.plot(
                        np.where(ok, x, np.nan), np.where(ok, y, np.nan),
                        color=age_color[age], alpha=_inc_alpha(inc, inclination_alpha),
                        ls=ls, lw=lw,
                    )

    handles = [
        Line2D([0], [0], color=age_color[a], lw=lw, label=f"Age={a:g} Myr")
        for a in ages
    ]
    if style_by == "MH":
        handles += [
            Line2D([0], [0], color="black", ls=mh_style[m], lw=lw, label=rf"[M/H]={m:g}")
            for m in mhs
        ]
    elif style_by == "omega":
        handles += [
            Line2D([0], [0], color="black", ls=om_style[o], lw=lw, label=rf"$\omega_{{ini}}={o:g}$")
            for o in omegas
        ]
    if len(incs) > 1 and not all(np.isclose(o, 0) for o in omegas):
        handles += [
            Line2D(
                [0], [0], color="black", lw=lw,
                alpha=_inc_alpha(i, inclination_alpha),
                label=rf"$i={np.degrees(i):g}^\circ$",
            )
            for i in incs
        ]

    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.invert_yaxis()
    ax.grid(alpha=0.25)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    if title is None:
        fixed = []
        if len(mhs) == 1:
            fixed.append(rf"[M/H]={mhs[0]:g}")
        if len(omegas) == 1:
            fixed.append(rf"$\omega_{{ini}}={omegas[0]:g}$")
        title = f"{system} CMD" + (", " + ", ".join(fixed) if fixed else "")
    ax.set_title(title)

    if legend:
        extra_handles, extra_labels = ax.get_legend_handles_labels()
        iso_labels = {handle.get_label() for handle in handles}
        extra_handles = [
            handle for handle, label in zip(extra_handles, extra_labels)
            if label not in iso_labels
        ]
        ax.legend(handles=extra_handles + handles, fontsize=8, loc=legend_loc, bbox_to_anchor=legend_bbox,)

    _save(fig, save_name)
    return fig, ax

# ============================================================
# HRD
# ============================================================
def plot_hrd_isochrones(
    interp_func, Age, Mini=None, MH=0.0, omega_ini=0.6,
    colors=None, mh_shade_range=(0.4, 1.0), figsize=(7, 7), lw=2.0,
    xlim=None, ylim=None, title=None, legend=True, save_name=None, ax=None,
    return_handles=False, legend_loc='best', legend_bbox=None
):
    """Plot HR diagram isochrones on a new or existing axis.

    Age controls color, MH controls shade, and omega_ini controls line style.

    Args:
        interp_func: Interpolator that evaluates stellar models.
        Age: Stellar ages in Myr. A scalar or sequence of values.
        Mini: Initial stellar masses. Defaults to 1001 points from 1 to 10 solar masses.
        MH: Metallicity values. A scalar or sequence of values.
        omega_ini: Initial rotation rates. A scalar or sequence of values.
        colors: Optional colors assigned to the age values.
        mh_shade_range: Minimum and maximum shade strengths for metallicity.
        figsize: Figure size used when creating a new figure.
        lw: Line width of the isochrones and legend handles.
        xlim: Optional x-axis limits.
        ylim: Optional y-axis limits.
        title: Plot title. Defaults to the HR diagram and fixed parameters.
        legend: Whether to display the legend.
        save_name: Optional path for saving the figure.
        ax: Existing Matplotlib axis. If None, a new figure and axis are created.
        return_handles: Whether to return the legend handles and labels.

    Returns:
        tuple: The figure and axis, optionally followed by legend handles and labels.
    """
    Mini = np.linspace(1, 10, 1001) if Mini is None else np.asarray(Mini, float)
    ages, mhs, omegas = map(_vals, (Age, MH, omega_ini))
    age_color = _colors(ages, colors)
    mh_strength = _strengths(mhs, mh_shade_range)
    om_style = _styles(omegas, zero_dashed=True)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    else:
        fig = ax.figure

    for age in ages:
        for mh in mhs:
            color = _shade(age_color[age], mh_strength[mh])
            for om in omegas:
                iso = _evaluate(interp_func, Mini, age, mh, om)
                x, y = iso["logTe"], iso["logL"]
                ok = np.isfinite(x) & np.isfinite(y)
                ax.plot(
                    np.where(ok, x, np.nan), np.where(ok, y, np.nan),
                    color=color, ls=om_style[om], lw=lw,
                )

    handles = [
        Line2D([0], [0], color=age_color[a], lw=lw, label=f"Age={a:g} Myr")
        for a in ages
    ]
    if len(mhs) > 1:
        handles += [
            Line2D(
                [0], [0], color=_shade("black", mh_strength[m]), lw=lw,
                label=rf"[M/H]={m:g}",
            )
            for m in mhs
        ]
    if len(omegas) > 1 or np.isclose(omegas[0], 0):
        handles += [
            Line2D(
                [0], [0], color="black", ls=om_style[o], lw=lw,
                label=rf"$\omega_{{ini}}={o:g}$",
            )
            for o in omegas
        ]

    ax.set_xlabel(r"$\log T_{\rm eff}$")
    ax.set_ylabel(r"$\log(L/L_\odot)$")
    ax.invert_xaxis()
    ax.grid(alpha=0.25)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    if title is None:
        fixed = []
        if len(mhs) == 1:
            fixed.append(rf"[M/H]={mhs[0]:g}")
        if len(omegas) == 1:
            fixed.append(rf"$\omega_{{ini}}={omegas[0]:g}$")
        title = "HR diagram" + (", " + ", ".join(fixed) if fixed else "")
    ax.set_title(title)

    if legend:
        extra_handles, extra_labels = ax.get_legend_handles_labels()
        iso_labels = {handle.get_label() for handle in handles}
        extra_handles = [
            handle for handle, label in zip(extra_handles, extra_labels)
            if label not in iso_labels
        ]
        ax.legend(handles=extra_handles + handles, fontsize=8, loc=legend_loc, bbox_to_anchor=legend_bbox,)

    _save(fig, save_name)
    if return_handles:
        return fig, ax, handles, [handle.get_label() for handle in handles]
    return fig, ax


# ============================================================
# Isomass tracks
# ============================================================
def plot_isomass_tracks(
    interp_func, Mini, MH=0.0, omega_ini=0.6, age_range=(1.0, 2001.0),
    n_age=1001, colors=None, mh_shade_range=(0.4, 1.0),
    figsize=(12, 8), lw=1.8, title=None, legend=True, save_name=None,
    axes=None, extra_label=None, legend_loc='best', legend_bbox=None
):
    """Plot evolutionary tracks of fixed initial masses against stellar age.

    Initial mass controls color, MH controls shade, and omega_ini controls line style.
    Additional curves can be drawn on selected existing panels and labeled separately.

    Args:
        interp_func: Interpolator that evaluates stellar models.
        Mini: Initial stellar masses in solar masses. A scalar or sequence of values.
        MH: Metallicity values. A scalar or sequence of values.
        omega_ini: Initial rotation rates. A scalar or sequence of values.
        age_range: Minimum and maximum ages in Myr.
        n_age: Number of age points used to evaluate each track.
        colors: Optional colors assigned to the initial masses.
        mh_shade_range: Minimum and maximum shade strengths for metallicity.
        figsize: Figure size used when creating new axes.
        lw: Line width of the evolutionary tracks and legend handles.
        title: Plot title. Defaults to the track description and fixed parameters.
        legend: Whether to display the legend.
        save_name: Optional path for saving the figure.
        axes: Optional sequence of four existing Matplotlib axes.
        extra_label: Optional label for distinguishing additional curves in their panel legends.

    Returns:
        tuple: The figure and array of four axes.
    """
    masses, mhs, omegas = map(_vals, (Mini, MH, omega_ini))
    if len(age_range) != 2 or age_range[0] <= 0 or age_range[1] <= age_range[0]:
        raise ValueError("age_range must be (age_min, age_max) with 0 < min < max.")
    if int(n_age) < 2:
        raise ValueError("n_age must be >= 2.")

    age = np.linspace(float(age_range[0]), float(age_range[1]), int(n_age))
    mass_color = _colors(masses, colors)
    mh_strength = _strengths(mhs, mh_shade_range)
    om_style = _styles(omegas, zero_dashed=True)

    if axes is None:
        fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True, constrained_layout=True)
    else:
        axes = np.asarray(axes, dtype=object)
        if axes.shape != (4,):
            raise ValueError("axes must contain exactly four Matplotlib axes.")
        fig = axes[0].figure
        if any(ax.figure is not fig for ax in axes):
            raise ValueError("All axes must belong to the same figure.")

    panels = [
        ("logTe", r"$\log (T_{\rm eff}/K)$"),
        ("logL", r"$\log(L/L_\odot)$"),
        ("Req", r"$R_{\rm eq}$"),
        ("Rpol", r"$R_{\rm pol}$"),
    ]

    for mass in masses:
        for mh in mhs:
            color = _shade(mass_color[mass], mh_strength[mh])
            for om in omegas:
                t = _evaluate(interp_func, mass, age, mh, om)
                ys = {"logTe": t["logTe"], "logL": t["logL"], "Req": t["Req"], "Rpol": t["Rpol"]}
                for ax, (key, ylabel) in zip(axes, panels):
                    y = ys[key]
                    ok = np.isfinite(y)
                    ax.plot(
                        np.where(ok, age, np.nan), np.where(ok, y, np.nan),
                        color=color, ls=om_style[om], lw=lw,
                    )
                    ax.set_ylabel(ylabel)
                    ax.grid(alpha=0.25)

    handles = [
        Line2D([0], [0], color=mass_color[m], lw=lw, label=rf"$M_{{ini}}={m:g}\,M_\odot$")
        for m in masses
    ]
    if len(mhs) > 1:
        handles += [
            Line2D([0], [0], color=_shade("black", mh_strength[m]), lw=lw, label=rf"[M/H]={m:g}")
            for m in mhs
        ]
    if len(omegas) > 1 or np.isclose(omegas[0], 0):
        handles += [
            Line2D([0], [0], color="black", ls=om_style[o], lw=lw, label=rf"$\omega_{{ini}}={o:g}$")
            for o in omegas
        ]

    axes[-1].set_xlabel("Age [Myr]")
    # axes[-1].set_xlim(age_range)
    if title is None:
        fixed = []
        if len(mhs) == 1:
            fixed.append(rf"[M/H]={mhs[0]:g}")
        if len(omegas) == 1:
            fixed.append(rf"$\omega_{{ini}}={omegas[0]:g}$")
        title = "Isomass evolutionary tracks" + (", " + ", ".join(fixed) if fixed else "")
    if axes is not None:
        axes[0].set_title(title)

    if legend:
        extra_handles, extra_labels = axes[0].get_legend_handles_labels()
        track_labels = {handle.get_label() for handle in handles}
        extra_handles = [
            handle for handle, label in zip(extra_handles, extra_labels)
            if label not in track_labels
        ]
        axes[0].legend(handles=extra_handles + handles, fontsize=8, ncol=2, loc=legend_loc, bbox_to_anchor=legend_bbox,)

        for ax in axes[1:]:
            extra_handles, extra_labels = ax.get_legend_handles_labels()
            if extra_handles:
                ax.legend(handles=extra_handles, labels=extra_labels, fontsize=8)

    _save(fig, save_name)
    return fig, axes

