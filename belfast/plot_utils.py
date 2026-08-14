import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D

OUTPUTS = [
    "Mass", "omega", "logL", "logTe", "Req", "Rpol",
    "G", "G_BP", "G_RP", "B_T", "V_T", "J", "H", "Ks",
]
LINESTYLES = ["-", ":", "-.", (0, (5, 1)), (0, (3, 1, 1, 1))]
CMD = {
    "2mass": (("J", "Ks"), "J", r"$J-K_s$", r"$M_J$", "2MASS"),
    "gaiadr3": (("G_BP", "G_RP"), "G", r"$G_{\rm BP}-G_{\rm RP}$", r"$M_G$", "Gaia DR3"),
    "tycho2": (("B_T", "V_T"), "V_T", r"$B_T-V_T$", r"$M_{V_T}$", "Tycho-2"),
}


def _vals(x):
    vals = [float(x)] if np.isscalar(x) else [float(v) for v in x]
    return sorted(dict.fromkeys(vals))


TAB_COLORS = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
]


def _colors(values, colors=None):
    values = list(values)

    if colors is None:
        colors = TAB_COLORS

    return {
        value: colors[i % len(colors)]
        for i, value in enumerate(values)
    }


def _strengths(vals, limits=(0.4, 1.0)):
    s = [1.0] if len(vals) == 1 else np.linspace(*limits, len(vals))
    return dict(zip(vals, map(float, s)))


def _shade(color, strength):
    rgb = np.asarray(mcolors.to_rgb(color))
    return tuple(1 - strength * (1 - rgb))


def _styles(vals, zero_dashed=False):
    out, j = {}, 0
    for v in vals:
        if zero_dashed and np.isclose(v, 0):
            out[v] = "--"
        else:
            out[v] = LINESTYLES[j % len(LINESTYLES)]
            j += 1
    return out


def _inc_alpha(inc, limits=(0.35, 1.0)):
    t = np.clip(float(inc) / (np.pi / 2), 0, 1)
    return limits[0] + t * (limits[1] - limits[0])


def _evaluate(interp_func, Mini, Age, MH, omega_ini, Inclination=0.0):
    """Broadcast model coordinates and return named outputs."""
    x = np.broadcast_arrays(*map(np.asarray, (Mini, Age, MH, omega_ini, Inclination)))
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
    key = re.sub(r"[^a-z0-9]", "", str(band).lower())
    key = {"gaia": "gaiadr3", "tycho": "tycho2"}.get(key, key)
    if key not in CMD:
        raise ValueError("band must be '2MASS', 'Gaia DR3', or 'Tycho-2'.")
    return CMD[key]


def _save(fig, save_name):
    if save_name is not None:
        fig.savefig(save_name, bbox_inches="tight")


# ============================================================
# CMD
# ============================================================
def plot_cmd_isochrones(
    interp_func, Age, band, Mini=None, MH=0.0, omega_ini=0.6,
    inclination=(0.0, np.pi / 2), colors=None,
    inclination_alpha=(0.35, 1.0), figsize=(7, 7), lw=2.0,
    xlim=None, ylim=None, title=None, legend=True, save_name=None,
):
    """Age=color, inclination=alpha, MH or omega_ini=linestyle."""
    Mini = np.linspace(1, 10, 1001) if Mini is None else np.asarray(Mini, float)
    ages, mhs, omegas, incs = map(_vals, (Age, MH, omega_ini, inclination))
    if len(mhs) > 1 and len(omegas) > 1:
        raise ValueError("For CMD, MH and omega_ini cannot both be lists.")

    (b1, b2), mag, xlabel, ylabel, system = _cmd_config(band)
    age_color = _colors(ages, colors)
    mh_style, om_style = _styles(mhs), _styles(omegas, zero_dashed=True)
    style_by = "MH" if len(mhs) > 1 else "omega" if len(omegas) > 1 else None
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    for age in ages:
        for mh in mhs:
            for om in omegas:
                inc_now = [0.0] if np.isclose(om, 0) else incs
                ls = "--" if np.isclose(om, 0) else (
                    mh_style[mh] if style_by == "MH" else
                    om_style[om] if style_by == "omega" else "-"
                )
                for inc in inc_now:
                    iso = _evaluate(interp_func, Mini, age, mh, om, inc)
                    x, y = iso[b1] - iso[b2], iso[mag]
                    ok = np.isfinite(x) & np.isfinite(y)
                    ax.plot(np.where(ok, x, np.nan), np.where(ok, y, np.nan),
                            color=age_color[age], alpha=_inc_alpha(inc, inclination_alpha),
                            ls=ls, lw=lw)

    handles = [Line2D([0], [0], color=age_color[a], lw=lw, label=f"Age={a:g} Myr") for a in ages]
    if style_by == "MH":
        handles += [Line2D([0], [0], color="black", ls=mh_style[m], lw=lw,
                           label=rf"[M/H]={m:g}") for m in mhs]
    elif style_by == "omega":
        handles += [Line2D([0], [0], color="black", ls=om_style[o], lw=lw,
                           label=rf"$\omega_{{ini}}={o:g}$") for o in omegas]
    if len(incs) > 1 and not all(np.isclose(o, 0) for o in omegas):
        handles += [Line2D([0], [0], color="black", lw=lw, alpha=_inc_alpha(i, inclination_alpha),
                           label=rf"$i={np.degrees(i):g}^\circ$") for i in incs]

    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.invert_yaxis()
    ax.grid(alpha=0.25)
    if xlim is not None: ax.set_xlim(xlim)
    if ylim is not None: ax.set_ylim(ylim)
    if title is None:
        fixed = []
        if len(mhs) == 1: fixed.append(rf"[M/H]={mhs[0]:g}")
        if len(omegas) == 1: fixed.append(rf"$\omega_{{ini}}={omegas[0]:g}$")
        title = f"{system} CMD" + (", " + ", ".join(fixed) if fixed else "")
    ax.set_title(title)
    if legend: ax.legend(handles=handles, fontsize=8)
    _save(fig, save_name)
    return fig, ax


# ============================================================
# HRD
# ============================================================
def plot_hrd_isochrones(
    interp_func, Age, Mini=None, MH=0.0, omega_ini=0.6,
    colors=None, mh_shade_range=(0.4, 1.0), figsize=(7, 7), lw=2.0,
    xlim=None, ylim=None, title=None, legend=True, save_name=None,
):
    """Age=color, MH=light/dark shade, omega_ini=linestyle."""
    Mini = np.linspace(1, 10, 1001) if Mini is None else np.asarray(Mini, float)
    ages, mhs, omegas = map(_vals, (Age, MH, omega_ini))
    age_color = _colors(ages, colors)
    mh_strength = _strengths(mhs, mh_shade_range)
    om_style = _styles(omegas, zero_dashed=True)
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)

    for age in ages:
        for mh in mhs:
            color = _shade(age_color[age], mh_strength[mh])
            for om in omegas:
                iso = _evaluate(interp_func, Mini, age, mh, om, 0.0)
                x, y = iso["logTe"], iso["logL"]
                ok = np.isfinite(x) & np.isfinite(y)
                ax.plot(np.where(ok, x, np.nan), np.where(ok, y, np.nan),
                        color=color, ls=om_style[om], lw=lw)

    handles = [Line2D([0], [0], color=age_color[a], lw=lw, label=f"Age={a:g} Myr") for a in ages]
    if len(mhs) > 1:
        handles += [Line2D([0], [0], color=_shade("black", mh_strength[m]), lw=lw,
                           label=rf"[M/H]={m:g}") for m in mhs]
    if len(omegas) > 1 or np.isclose(omegas[0], 0):
        handles += [Line2D([0], [0], color="black", ls=om_style[o], lw=lw,
                           label=rf"$\omega_{{ini}}={o:g}$") for o in omegas]

    ax.set_xlabel(r"$\log T_{\rm eff}$")
    ax.set_ylabel(r"$\log(L/L_\odot)$")
    ax.invert_xaxis()
    ax.grid(alpha=0.25)
    if xlim is not None: ax.set_xlim(xlim)
    if ylim is not None: ax.set_ylim(ylim)
    if title is None:
        fixed = []
        if len(mhs) == 1: fixed.append(rf"[M/H]={mhs[0]:g}")
        if len(omegas) == 1: fixed.append(rf"$\omega_{{ini}}={omegas[0]:g}$")
        title = "HR diagram" + (", " + ", ".join(fixed) if fixed else "")
    ax.set_title(title)
    if legend: ax.legend(handles=handles, fontsize=8)
    _save(fig, save_name)
    return fig, ax


# ============================================================
# Isomass tracks
# ============================================================
def plot_isomass_tracks(
    interp_func, Mini, MH=0.0, omega_ini=0.6, age_range=(1.0, 2001.0),
    n_age=1001, colors=None, mh_shade_range=(0.4, 1.0),
    figsize=(12, 8), lw=1.8, title=None, legend=True, save_name=None,
):
    """4x1 panels: Teff, logL, Req, Rpol versus Age [Myr]."""
    masses, mhs, omegas = map(_vals, (Mini, MH, omega_ini))
    if len(age_range) != 2 or age_range[0] <= 0 or age_range[1] <= age_range[0]:
        raise ValueError("age_range must be (age_min, age_max) with 0 < min < max.")
    if int(n_age) < 2:
        raise ValueError("n_age must be >= 2.")

    age = np.linspace(float(age_range[0]), float(age_range[1]), int(n_age))
    mass_color = _colors(masses, colors)
    mh_strength = _strengths(mhs, mh_shade_range)
    om_style = _styles(omegas, zero_dashed=True)
    fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True, constrained_layout=True)
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
                t = _evaluate(interp_func, mass, age, mh, om, 0.0)
                ys = {"logTe": t["logTe"], "logL": t["logL"], "Req": t["Req"], "Rpol": t["Rpol"]}
                for ax, (key, ylabel) in zip(axes, panels):
                    y = ys[key]
                    ok = np.isfinite(y)
                    ax.plot(np.where(ok, age, np.nan), np.where(ok, y, np.nan),
                            color=color, ls=om_style[om], lw=lw)
                    ax.set_ylabel(ylabel)
                    ax.grid(alpha=0.25)

    handles = [Line2D([0], [0], color=mass_color[m], lw=lw,
                      label=rf"$M_{{ini}}={m:g}\,M_\odot$") for m in masses]
    if len(mhs) > 1:
        handles += [Line2D([0], [0], color=_shade("black", mh_strength[m]), lw=lw,
                           label=rf"[M/H]={m:g}") for m in mhs]
    if len(omegas) > 1 or np.isclose(omegas[0], 0):
        handles += [Line2D([0], [0], color="black", ls=om_style[o], lw=lw,
                           label=rf"$\omega_{{ini}}={o:g}$") for o in omegas]

    axes[-1].set_xlabel("Age [Myr]")
    axes[-1].set_xlim(age_range)
    if title is None:
        fixed = []
        if len(mhs) == 1: fixed.append(rf"[M/H]={mhs[0]:g}")
        if len(omegas) == 1: fixed.append(rf"$\omega_{{ini}}={omegas[0]:g}$")
        title = "Isomass evolutionary tracks" + (", " + ", ".join(fixed) if fixed else "")
    axes[0].set_title(title)
    if legend: axes[0].legend(handles=handles, fontsize=8, ncol=2)
    _save(fig, save_name)
    return fig, axes
