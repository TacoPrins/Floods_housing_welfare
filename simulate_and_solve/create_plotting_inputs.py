# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 14:24:48 2026

@author: tprins
"""

# -*- coding: utf-8 -*-
"""
create_plotting_inputs.py

Purpose:
    Take the full (large) distribution arrays produced by collect_results and
    reduce them to the small arrays that the plots actually consume, then write
    those to an Excel file. No plotting happens here.

    plot_dist.py reads the Excel file this writes and generates the figures, so
    the enormous 6D distributions never have to be reloaded to re-plot.

Reduced quantities (computed for both scenarios, HE and RE):
    - tenure masses mc / mnc / mr                       (T, K)
    - default masses def_c / def_nc                     (T, K)
    - conditional marginals over l / m / h, per (k, e)  (T, n_grid)  [C and NC]
    - renter marginal over x, per (k, e)                (T, nx)
    - LTV heatmap density per k (e aggregated)          (T, nL)      [C and NC]

Moment paths (mean, 10/90) are derived from the conditional marginals inside
plot_dist, so they need no separate storage.

Layout: one sheet per logical quantity, in long/tidy form with index columns
(scenario, loc, k, e, var, t) + value.  plot_dist filters each sheet with pandas.
"""

import numpy as np
import pandas as pd

SCENARIOS = ("HE", "RE")
VARS = {"l": 4, "m": 2, "h": 3}          # var name -> owner-array axis
GRID_NAMES = ("vTime", "vM_sim", "vH", "vL_sim", "vX_sim")


# ---------- reduction helpers ----------
def _marg(d, k, e, axis_keep):
    """Marginal of owner dist over one of m(2)/h(3)/l(4), given k and e -> (T, n)."""
    w = d[:, k, :, :, :, e]                      # (T, m, h, l)
    drop = tuple(a for a in (1, 2, 3) if a != axis_keep - 1)
    return w.sum(axis=drop)                       # (T, n_axis)


def _tenure_mass(dc, dnc, dr):
    mc = dc.sum(axis=(2, 3, 4, 5))                # (T, K)
    mnc = dnc.sum(axis=(2, 3, 4, 5))
    mr = dr.sum(axis=(2, 3))
    return mc, mnc, mr


def _long(rows, cols):
    """Build a tidy DataFrame from a list of row-dicts (empty-safe)."""
    return pd.DataFrame(rows, columns=cols)


# ---------- per-scenario reduction into row lists ----------
def _reduce_scenario(tag, dc, dnc, dr, defc, defnc, sink):
    """Append this scenario's reduced rows into the dict-of-lists `sink`."""
    T, K, E = dc.shape[0], dc.shape[1], dc.shape[5]

    # tenure + default masses (T, K)
    mc, mnc, mr = _tenure_mass(dc, dnc, dr)
    defc = np.asarray(defc); defnc = np.asarray(defnc)
    for k in range(K):
        for t in range(T):
            sink["masses"].append(
                {"scenario": tag, "k": k, "t": t,
                 "mc": mc[t, k], "mnc": mnc[t, k], "mr": mr[t, k]})
        for t in range(defc.shape[0]):
            sink["defaults"].append(
                {"scenario": tag, "k": k, "t": t,
                 "def_c": defc[t, k], "def_nc": defnc[t, k]})

    # owner conditional marginals over l/m/h, per (loc, k, e)  -> (T, n)
    for loc, d in (("C", dc), ("NC", dnc)):
        for var, axis_keep in VARS.items():
            for k in range(K):
                for e in range(E):
                    marg = _marg(d, k, e, axis_keep)          # (T, n)
                    for t in range(T):
                        for i, v in enumerate(marg[t]):
                            sink["cond"].append(
                                {"scenario": tag, "location": loc, "var": var,
                                 "k": k, "e": e, "t": t, "i": i, "value": v})

    # renter marginal over x, per (k, e)  -> (T, nx)
    for k in range(K):
        for e in range(E):
            for t in range(T):
                for i, v in enumerate(dr[t, k, :, e]):
                    sink["renter"].append(
                        {"scenario": tag, "k": k, "e": e, "t": t,
                         "i": i, "value": v})

    # LTV heatmap density per k (e aggregated) -> (T, nL), for C and NC
    for loc, d in (("C", dc), ("NC", dnc)):
        for k in range(K):
            w = d[:, k].sum(axis=(1, 2, 4))                   # (T, l)
            for t in range(T):
                for i, v in enumerate(w[t]):
                    sink["ltv"].append(
                        {"scenario": tag, "location": loc, "k": k, "t": t,
                         "i": i, "value": v})


# ---------- public entry point ----------
def create(dist_out, grids, path="plotting_inputs.xlsx"):
    """Reduce dist_out and write all plotting inputs to `path` (Excel)."""
    (vTime,
     dc_HE, dnc_HE, dr_HE, dc_RE, dnc_RE, dr_RE,
     defc_HE, defnc_HE, defc_RE, defnc_RE) = dist_out

    sink = {"masses": [], "defaults": [], "cond": [], "renter": [], "ltv": []}
    _reduce_scenario("HE", dc_HE, dnc_HE, dr_HE, defc_HE, defnc_HE, sink)
    _reduce_scenario("RE", dc_RE, dnc_RE, dr_RE, defc_RE, defnc_RE, sink)

    cols = {
        "masses":   ["scenario", "k", "t", "mc", "mnc", "mr"],
        "defaults": ["scenario", "k", "t", "def_c", "def_nc"],
        "cond":     ["scenario", "location", "var", "k", "e", "t", "i", "value"],
        "renter":   ["scenario", "k", "e", "t", "i", "value"],
        "ltv":      ["scenario", "location", "k", "t", "i", "value"],
    }

    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        for name, rows in sink.items():
            _long(rows, cols[name]).to_excel(xl, sheet_name=name, index=False)
        # grid vectors (x-axes), one column each, padded to equal length
        gvecs = {g: np.asarray(getattr(grids, g)).ravel() for g in GRID_NAMES}
        n = max(len(v) for v in gvecs.values())
        gdf = pd.DataFrame({g: np.append(v, [np.nan] * (n - len(v)))
                            for g, v in gvecs.items()})
        gdf.to_excel(xl, sheet_name="grids", index=False)

    print(f"saved plotting inputs to {path}")