# -*- coding: utf-8 -*-
"""
plot_dist.py — generate all distribution diagnostics from the reduced inputs
written by create_plotting_inputs.py.

This file never touches the full 6D distributions.  It reads the small,
already-reduced arrays from the Excel file and plots them, so you can re-style
or re-select plots (e.g. pick different t) without rerunning the model.

Reduced sheets (see create_plotting_inputs.py):
  masses   : scenario, k, t, mc, mnc, mr
  defaults : scenario, k, t, def_c, def_nc
  cond     : scenario, loc, var, k, e, t, i, value   (owner marginals over l/m/h)
  renter   : scenario, k, e, t, i, value             (renter marginal over x)
  ltv      : scenario, loc, k, t, i, value           (LTV density, e aggregated)
  grids    : vTime, vM_sim, vH, vL_sim, vX_sim
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

K_LABEL = {0: "Realist", 1: "Sceptic"}
_VAR_GRID = {"l": "vL_sim", "m": "vM_sim", "h": "vH"}
_LOC_NAME = {"C": "Coastal", "NC": "Non-coastal"}

# where every figure is saved (as .png and .eps); set SAVE_FIGS = False to only show them
OUT_DIR = r"C:\Users\TPRINS\OneDrive - UvA\Documenten\Python files\New coding round July 2026\Plaatjes"
SAVE_FIGS = True
# .eps has no transparency: slightly transparent lines are drawn opaque; silence that warning
logging.getLogger("matplotlib.backends.backend_ps").setLevel(logging.ERROR)

# income level 1 -> 5: one blue hue, light -> dark (same as the welfare plots)
INCOME_COLORS = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]


def _save(fig, name):
    """Save fig to OUT_DIR as name.png and name.eps."""
    if not SAVE_FIGS:
        return
    os.makedirs(OUT_DIR, exist_ok=True)
    for ext in ("png", "eps"):
        fig.savefig(os.path.join(OUT_DIR, f"{name}.{ext}"), dpi=300, bbox_inches="tight")


# ---------- small numeric helpers (unchanged in spirit) ----------
def _norm(w):
    s = w.sum()
    return w / s if s > 0 else w


def _wmean(grid, w):
    s = w.sum()
    return np.dot(grid, w) / s if s > 0 else np.nan


def _wquant(grid, w, q):
    s = w.sum()
    if s <= 0:
        return np.nan
    c = np.cumsum(w) / s
    return np.interp(q, c, grid)


def _safe_div(a, b):
    """Elementwise a/b, returning NaN where b == 0 (zero-mass points)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    return np.divide(a, b, out=np.full_like(a, np.nan), where=b != 0)


# ---------- load reduced inputs from Excel ----------
def load_inputs(path="plotting_inputs.xlsx"):
    """Read all sheets into a dict of DataFrames + a grids dict."""
    sheets = pd.read_excel(path, sheet_name=["masses", "defaults", "cond",
                                             "renter", "ltv", "grids"])
    g = sheets.pop("grids")
    grids = {c: g[c].dropna().to_numpy() for c in g.columns}
    return sheets, grids


def _pivot_TN(df):
    """Long (t, i, value) rows -> dense (T, n) array, t and i sorted ascending."""
    m = df.pivot_table(index="t", columns="i", values="value", aggfunc="sum")
    m = m.sort_index().sort_index(axis=1)
    return m.to_numpy()


# ---------- 1) tenure shares by belief type ----------
def plot_tenure_shares(masses, tag, vTime=None):
    d = masses[masses["scenario"] == tag]
    ks = sorted(d["k"].unique()); K = len(ks)
    piv = d.pivot_table(index="t", columns="k", values=["mc", "mnc", "mr"])
    t = piv.index.to_numpy() if vTime is None else vTime[:len(piv)]
    mc = piv["mc"].to_numpy(); mnc = piv["mnc"].to_numpy(); mr = piv["mr"].to_numpy()
    tot = mc + mnc + mr

    fig, ax = plt.subplots(1, K, figsize=(5.5 * K, 4), squeeze=False); ax = ax[0]
    for j, k in enumerate(ks):
        ax[j].stackplot(t, _safe_div(mc[:, j], tot[:, j]),
                        _safe_div(mnc[:, j], tot[:, j]),
                        _safe_div(mr[:, j], tot[:, j]),
                        labels=["Coastal own", "Non-coastal own", "Renter"],
                        alpha=.85)
        ax[j].set_title(K_LABEL.get(k, f"k={k}"))
        ax[j].set_xlabel("t"); ax[j].set_ylim(0, 1)
    ax[0].set_ylabel("Population share"); ax[0].legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    _save(fig, f"tenure_shares_{tag}")
    return ax


def plot_coastal_share_lines(masses, tag, vTime=None):
    d = masses[masses["scenario"] == tag]
    ks = sorted(d["k"].unique())
    piv = d.pivot_table(index="t", columns="k", values=["mc", "mnc", "mr"])
    t = piv.index.to_numpy() if vTime is None else vTime[:len(piv)]
    mc = piv["mc"].to_numpy(); mnc = piv["mnc"].to_numpy(); mr = piv["mr"].to_numpy()

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for j, k in enumerate(ks):
        ax[0].plot(t, _safe_div(mc[:, j], mc[:, j] + mnc[:, j]), label=K_LABEL.get(k, k))
        ax[1].plot(t, _safe_div(mc[:, j] + mnc[:, j], mc[:, j] + mnc[:, j] + mr[:, j]),
                   label=K_LABEL.get(k, k))
    ax[0].set_title("Coastal share of owners"); ax[1].set_title("Homeownership rate")
    for a in ax: a.set_xlabel("t"); a.legend()
    plt.tight_layout()
    _save(fig, f"coastal_share_lines_{tag}")
    return ax


# ---------- 2) conditional distributions over l, m, h (owners) and x (renters) ----------
def _cond_marg(cond, tag, loc, var, k, e):
    """Reduced owner marginal -> (T, n) for one (scenario, loc, var, k, e)."""
    sel = cond[(cond["scenario"] == tag) & (cond["location"] == loc) &
               (cond["var"] == var) & (cond["k"] == k) & (cond["e"] == e)]
    return _pivot_TN(sel)


def plot_conditional_dists(cond, grids, tag, loc, k, e_list=None,
                           t_list=(0, -1)):
    d = cond[(cond["scenario"] == tag) & (cond["location"] == loc)]
    E = int(d["e"].max()) + 1
    e_list = list(range(E)) if e_list is None else e_list

    specs = [("l", grids["vL_sim"], "LTV $l$"),
             ("m", grids["vM_sim"], "Savings $m$"),
             ("h", grids["vH"], "House size $h$")]
    fig, ax = plt.subplots(3, len(e_list), figsize=(3.4 * len(e_list), 8),
                           squeeze=False)
    for r, (var, grid, name) in enumerate(specs):
        for c, e in enumerate(e_list):
            marg = _cond_marg(cond, tag, loc, var, k, e)     # (T, n)
            T = marg.shape[0]; tl = [t % T for t in t_list]
            for t in tl:
                ax[r, c].plot(grid[:marg.shape[1]], _norm(marg[t]), label=f"t={t}")
            if r == 0: ax[r, c].set_title(f"e={e}")
            if c == 0: ax[r, c].set_ylabel(name)
    ax[0, 0].legend(fontsize=8)
    fig.suptitle(f"{_LOC_NAME[loc]} owners — {K_LABEL.get(k, k)} ({tag})")
    plt.tight_layout()
    _save(fig, f"cond_dists_{tag}_{loc}_{K_LABEL[k].lower()}")
    return ax


def plot_renter_savings(renter, grids, tag, k_list=None, e_list=None,
                        t_list=(0, -1)):
    d = renter[renter["scenario"] == tag]
    vX = grids["vX_sim"]
    ks = sorted(d["k"].unique()); es = sorted(d["e"].unique())
    k_list = ks if k_list is None else k_list
    e_list = es if e_list is None else e_list

    fig, ax = plt.subplots(len(k_list), len(e_list),
                           figsize=(3.4 * len(e_list), 3.2 * len(k_list)),
                           squeeze=False)
    for r, k in enumerate(k_list):
        for c, e in enumerate(e_list):
            marg = _pivot_TN(d[(d["k"] == k) & (d["e"] == e)])      # (T, nx)
            T = marg.shape[0]; tl = [t % T for t in t_list]
            for t in tl:
                ax[r, c].plot(vX[:marg.shape[1]], _norm(marg[t]), label=f"t={t}")
            if r == 0: ax[r, c].set_title(f"e={e}")
            if c == 0: ax[r, c].set_ylabel(K_LABEL.get(k, k))
    ax[0, 0].legend(fontsize=8)
    fig.suptitle(f"Renter savings $x$ ({tag})")
    plt.tight_layout()
    _save(fig, f"renter_savings_{tag}")
    return ax


# ---------- 3) moment paths ----------
def plot_moment_paths(cond, grids, tag, loc, var="l", vTime=None):
    """Plot mean paths by income group, with a publication-style layout."""
    grid = grids[_VAR_GRID[var]]
    d = cond[(cond["scenario"] == tag) &
             (cond["location"] == loc) &
             (cond["var"] == var)]

    ks = sorted(d["k"].unique())
    es = sorted(d["e"].unique())
    E = len(es)
    T = int(d["t"].max()) + 1

    # Calendar years: model period 0 corresponds to 1998,
    # with each subsequent period representing two years.
    t = 1998 + 2 * np.arange(T)

    var_name = {
        "l": "LTV ratio",
        "m": "savings",
        "h": "house size",
    }.get(var, var)

    # Fixed y-axis ranges for comparability across all plots.
    y_limits = {
        "l": (0, 0.9),  # LTV ratio
        "m": (0, 3.0),  # Savings
        "h": (0, 5.0),  # House size
    }

    axes = []

    for k in ks:
        belief = K_LABEL.get(k, f"k={k}")

        fig, ax = plt.subplots(
            figsize=(7.0, 4.6),
            constrained_layout=True
        )

        # One line per income level, light (1) -> dark (5), as in the welfare plots.
        for idx, e in enumerate(es):
            marg = _pivot_TN(
                d[(d["k"] == k) & (d["e"] == e)]
            )

            g = grid[:marg.shape[1]]

            mu = np.array([
                _wmean(g, marg[i])
                for i in range(marg.shape[0])
            ])

            ax.plot(t[:len(mu)], mu, color=INCOME_COLORS[idx], lw=2.2, label=str(idx + 1))

        # Title and axis labels.
        ax.set_title(
            f"{belief} owners "
            f"{'flood-exposed' if loc == 'C' else 'non-flood-exposed'} housing",
            loc="left",
            fontsize=18,
            fontweight="semibold",
            pad=10,
        )

        ax.set_xlabel("Year", fontsize=18)
        ax.set_ylabel(f"Mean {var_name}", fontsize=18)

        # Fixed y-axis range depending on variable.
        if var in y_limits:
            ax.set_ylim(*y_limits[var])

        # Calendar-year x-axis.
        # The data begin in 1998 and remain at two-year intervals,
        # but only selected years are labelled.
        ax.set_xlim(t[0], t[-1])
        ax.set_xticks([
            2000,
            2020,
            2040,
            2060,
            2080,
            2100,
        ])

        # Quiet editorial styling:
        # horizontal guides only, no box around axes.
        ax.grid(
            axis="y",
            which="major",
            color="#D9D9D9",
            lw=0.7,
            alpha=0.8,
        )

        ax.set_axisbelow(True)

        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#888888")

        ax.tick_params(
            axis="both",
            labelsize=16,
            colors="#444444",
        )

        ax.margins(x=0.01)

        ax.legend(title="Income level", frameon=False, fontsize=15, title_fontsize=15,
                  ncol=E, loc="upper center", bbox_to_anchor=(0.5, -0.2))  # below the axes

        _save(fig, f"mean_{var_name.replace(' ', '_')}_{tag}_{loc}_{belief.lower()}")
        axes.append(ax)

    return axes


# ---------- 4) heatmap: LTV distribution over time ----------
def plot_ltv_heatmap(ltv, grids, tag, loc, k, vTime=None):
    vL = grids["vL_sim"]
    d = ltv[(ltv["scenario"] == tag) & (ltv["location"] == loc) & (ltv["k"] == k)]
    w = _pivot_TN(d)                                          # (T, l)
    w = np.array([_norm(row) for row in w])
    T = w.shape[0]; t = np.arange(T) if vTime is None else vTime[:T]

    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.pcolormesh(t, vL[:w.shape[1]], w.T, shading="auto", cmap="magma")
    fig.colorbar(im, ax=ax, label="density")
    ax.set_xlabel("t"); ax.set_ylabel("LTV $l$")
    ax.set_title(f"{_LOC_NAME[loc]} — {K_LABEL.get(k, k)} ({tag})")
    plt.tight_layout()
    _save(fig, f"ltv_heatmap_{tag}_{loc}_{K_LABEL[k].lower()}")
    return ax


# ---------- 5) default rates by belief type ----------
def plot_default_rates(masses, defaults, tag, vTime=None):
    """
    Plot default-rate paths separately for coastal and non-coastal owners,
    using a publication-style layout.
    """
    dm = masses[masses["scenario"] == tag]
    dd = defaults[defaults["scenario"] == tag]

    ks = sorted(dd["k"].unique())

    # Pivot masses and defaults into (T, K) arrays.
    m = dm.pivot_table(
        index="t",
        columns="k",
        values=["mc", "mnc"]
    )

    de = dd.pivot_table(
        index="t",
        columns="k",
        values=["def_c", "def_nc"]
    )

    T = len(de)

    # Calendar years: model period 0 = 1998,
    # with two-year spacing between periods.
    t = 1998 + 2 * np.arange(T)

    mc = m["mc"].to_numpy()[:T]
    mnc = m["mnc"].to_numpy()[:T]

    dc = de["def_c"].to_numpy()
    dnc = de["def_nc"].to_numpy()

    # Restrained, colour-blind-friendly colours.
    palette = ["tab:blue", "tab:orange", "tab:green", "tab:red"]   # matplotlib defaults

    # Location-specific inputs.
    specs = [
        {
            "name": "flood-exposed",
            "defaults": dc,
            "mass": mc,
        },
        {
            "name": "non-flood-exposed",
            "defaults": dnc,
            "mass": mnc,
        },
    ]

    axes = []

    for spec in specs:

        fig, ax = plt.subplots(
            figsize=(7.0, 4.6),
            constrained_layout=True
        )

        for j, k in enumerate(ks):

            rate = _safe_div(
                spec["defaults"][:, j],
                spec["mass"][:, j]
            )

            ax.plot(
                t,
                rate,
                label=K_LABEL.get(k, f"k={k}"),
                color=palette[j % len(palette)],
                lw=2.0,
                alpha=0.95,
            )

        # Title and axis labels.
        ax.set_title(
            f"Owners {spec['name']} housing",
            loc="left",
            fontsize=11,
            fontweight="semibold",
            pad=10,
        )

        ax.set_xlabel("Year")
        ax.set_ylabel("Default rate")

        # Calendar-year x-axis:
        # observations remain at two-year intervals, but only selected
        # years are labelled.
        ax.set_xlim(t[0], t[-1])
        ax.set_xticks([
            2000,
            2020,
            2040,
            2060,
            2080,
            2100,
        ])

        # Default rates should naturally start at zero.
        ax.set_ylim(bottom=0)

        # Quiet editorial styling.
        ax.grid(
            axis="y",
            which="major",
            color="#D9D9D9",
            lw=0.7,
            alpha=0.8,
        )

        ax.set_axisbelow(True)

        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#888888")

        ax.tick_params(
            axis="both",
            labelsize=9,
            colors="#444444",
        )

        ax.margins(x=0.01)

        # Legend outside plotting region.
        ax.legend(
            title="Belief type",
            loc="upper center",
            bbox_to_anchor=(0.5, -0.16),
            ncol=len(ks),
            fontsize=8.5,
            title_fontsize=8.5,
            frameon=False,
            handlelength=2.6,
            columnspacing=1.8,
        )

        _save(fig, f"default_rates_{tag}_{spec['name'].replace('-', '_')}")
        axes.append(ax)

    return axes


# ---------- 6) table: owner distribution over house size h ----------
def print_h_table(cond, grids, tag, t=0, k=None):
    """Owner mass on each h grid point (C, NC, total), as fractions of all owners at period t."""
    d = cond[(cond["scenario"] == tag) & (cond["var"] == "h")]
    if k is not None:
        d = d[d["k"] == k]                                    # one belief type; default sums over k
    t = t % (int(d["t"].max()) + 1)                           # allow t=-1
    tab = d[d["t"] == t].pivot_table(index="i", columns="location",
                                     values="value", aggfunc="sum")[["C", "NC"]]
    tab["Total"] = tab["C"] + tab["NC"]
    tab = tab / tab["Total"].sum()                            # fractions of all owners
    tab.index = grids["vH"][:len(tab)]; tab.index.name = "h"
    who = "all beliefs" if k is None else K_LABEL.get(k, k)
    print(f"\nOwners over h — {tag}, t={t}, {who}")
    print(pd.concat([tab, tab.sum().to_frame("Sum").T]).round(4).to_string())
    return tab

# ---------- 7) coastal share of owners by belief type ----------
def plot_coastal_owner_share(masses, tag="HE"):
    """Share of owners living in flood-exposed (coastal) housing, by belief type."""
    d = masses[masses["scenario"] == tag]
    piv = d.pivot_table(index="t", columns="k", values=["mc", "mnc"])
    t = 1998 + 2 * piv.index.to_numpy()                      # calendar years

    fig, ax = plt.subplots(figsize=(7.0, 4.6), constrained_layout=True)
    for k in sorted(d["k"].unique()):
        mc, mnc = piv["mc"][k].to_numpy(), piv["mnc"][k].to_numpy()
        ax.plot(t, _safe_div(mc, mc + mnc), lw=2.2, label=K_LABEL.get(k, f"k={k}"))

    ax.set_title("Flood-exposed share of homeowners", loc="left",
                 fontsize=18, fontweight="semibold", pad=10)
    ax.set_xlabel("Year", fontsize=18)
    ax.set_ylabel("Share of owners", fontsize=18)
    ax.set_ylim(0, 1.02)                                     # keep lines at 1 fully visible
    ax.set_xlim(t[0], t[-1])
    ax.set_xticks([2000, 2020, 2040, 2060, 2080, 2100])
    ax.grid(axis="y", color="#D9D9D9", lw=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#888888")
    ax.tick_params(axis="both", labelsize=16, colors="#444444")
    ax.legend(frameon=False, fontsize=16, loc="lower left")
    _save(fig, f"coastal_owner_share_{tag}")
    return ax


# ---------- 8) coastal ownership, split by LTV ----------
def plot_coastal_ltv_split(masses, ltv, grids, ltv_max=0.8):
    """Share of all households of a type owning coastal housing, stacked into
    LTV <= ltv_max (dark) and LTV > ltv_max (light); the top edge is P(coastal owner).
    One figure per (scenario, belief type), all on the same y-axis."""
    low = grids["vL_sim"] <= ltv_max
    panels = [("HE", 0, "Realist owners", "HE_realist"),
          ("HE", 1, "Sceptic owners", "HE_sceptic"),
          ("RE", 0, "", "RE")]

    axes = []
    for tag, k, title, fname in panels:
        m = masses[(masses["scenario"] == tag) & (masses["k"] == k)].sort_values("t")
        tot = (m["mc"] + m["mnc"] + m["mr"]).to_numpy()          # all households of type k
        w = _pivot_TN(ltv[(ltv["scenario"] == tag) & (ltv["location"] == "C") &
                          (ltv["k"] == k)])                       # (T, nL) coastal owner mass
        p_low = _safe_div(w[:, low[:w.shape[1]]].sum(axis=1), tot)
        p_high = _safe_div(w[:, ~low[:w.shape[1]]].sum(axis=1), tot)
        fig, ax = plt.subplots(figsize=(7.0, 4.6), constrained_layout=True)
        _draw_ltv_split(ax, m["t"].to_numpy(), p_low, p_high, title, ltv_max)
        _ltv_legend(ax)
        _save(fig, f"coastal_ltv{round(100 * ltv_max)}_{fname}")
        axes.append(ax)
    return axes


def plot_coastal_ltv_split_by_e(cond, renter, grids, ltv_max=0.8):
    """As plot_coastal_ltv_split, but conditional on current income level e:
    shares of all households of type k with income e (HE only).
    One tall figure per belief type, income levels 1 (top) to 5 (bottom),
    meant to sit side by side (realists | sceptics) in LaTeX."""
    low = grids["vL_sim"] <= ltv_max
    d = cond[(cond["scenario"] == "HE") & (cond["var"] == "l")]   # owner mass over LTV, by k, e
    r = renter[renter["scenario"] == "HE"]
    es = sorted(d["e"].unique())

    figs = []
    for k, name in [(0, "realist"), (1, "sceptic")]:
        fig, axes = plt.subplots(len(es), 1, figsize=(7.0, 13.0), sharex=True,
                                 constrained_layout=True)
        for ax, e in zip(axes, es):
            sel = (d["k"] == k) & (d["e"] == e)
            w = _pivot_TN(d[sel & (d["location"] == "C")])        # (T, nL) coastal owner mass
            tot = (w.sum(axis=1)                                   # all households of type k, income e
                   + _pivot_TN(d[sel & (d["location"] == "NC")]).sum(axis=1)
                   + _pivot_TN(r[(r["k"] == k) & (r["e"] == e)]).sum(axis=1))
            p_low = _safe_div(w[:, low[:w.shape[1]]].sum(axis=1), tot)
            p_high = _safe_div(w[:, ~low[:w.shape[1]]].sum(axis=1), tot)
            _draw_ltv_split(ax, np.arange(len(tot)), p_low, p_high,
                            f"Income level {e + 1}", ltv_max)
        for ax in axes[:-1]:
            ax.set_xlabel("")                                      # "Year" only on the bottom panel
        _ltv_legend(axes[-1], y=-0.38)
        _save(fig, f"coastal_ltv{round(100 * ltv_max)}_HE_{name}_by_income")
        figs.append(fig)
    return figs


def _ltv_legend(ax, y=-0.2):
    """Section 8 legend below the axes, ordered top -> bottom of the stack."""
    h, lab = ax.get_legend_handles_labels()
    ax.legend(h[::-1], lab[::-1], frameon=False,
              fontsize=14, title_fontsize=14, ncol=3,
              loc="upper center", bbox_to_anchor=(0.5, y))


def _draw_ltv_split(ax, tt, p_low, p_high, title, ltv_max):
    """Shared look for section 8 on one axis: stacked bands + top line,
    model period tt -> calendar year."""
    dark, light, edge = "#1c5cab", "#86b6ef", "#0d366b"   # one blue hue
    thr = f"{ltv_max:g}"
    ok = ~np.isnan(p_low)                                    # drop periods with no mass
    t = 1998 + 2 * tt[ok]

    ax.stackplot(t, p_low[ok], p_high[ok], colors=[dark, light],
                 labels=[f"LTV ≤ {thr}", f"LTV > {thr}"],
                 edgecolor="white", linewidth=1.0)
    ax.plot(t, p_low[ok] + p_high[ok], color=edge, lw=2.0, label="All")

    ax.set_title(title, loc="left", fontsize=18, fontweight="semibold", pad=10)
    ax.set_xlabel("Year", fontsize=18)
    ax.set_ylabel("Probability", fontsize=18)
    ax.set_ylim(0, 1)
    ax.set_xlim(1998, 1998 + 2 * tt.max())
    ax.set_xticks([2000, 2020, 2040, 2060, 2080, 2100])
    ax.grid(axis="y", color="#D9D9D9", lw=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#888888")
    ax.tick_params(axis="both", labelsize=16, colors="#444444")


# ---------- driver ----------
def plot_all_distributions(path="plotting_inputs.xlsx"):
    """Load reduced inputs from Excel and plot every diagnostic for HE and RE."""
    sheets, grids = load_inputs(path)
    masses, defaults = sheets["masses"], sheets["defaults"]
    cond, renter, ltv = sheets["cond"], sheets["renter"], sheets["ltv"]
    vTime = grids.get("vTime")

    for tag in ("HE", "RE"):
        # 1) tenure shares
        plot_tenure_shares(masses, tag, vTime)
        plot_coastal_share_lines(masses, tag, vTime)

        # 2) conditional dists + LTV heatmaps, per loc and k
        ks = sorted(masses[masses["scenario"] == tag]["k"].unique())
        for loc in ("C", "NC"):
            for k in ks:
                plot_conditional_dists(cond, grids, tag, loc, k)
                plot_ltv_heatmap(ltv, grids, tag, loc, k, vTime)

        # renters
        plot_renter_savings(renter, grids, tag)

        # 3) moment paths
        for loc in ("C", "NC"):
            for var in ("l", "m", "h"):
                plot_moment_paths(cond, grids, tag, loc, var, vTime)

        # 5) default rates
        plot_default_rates(masses, defaults, tag, vTime)

    plot_coastal_owner_share(masses, "HE")

    # 8) coastal ownership split by LTV (threshold 0.8), aggregate and by income level
    plot_coastal_ltv_split(masses, ltv, grids, 0.8)
    plot_coastal_ltv_split_by_e(cond, renter, grids, 0.8)

    plt.show()


if __name__ == "__main__":
    plot_all_distributions()