# -*- coding: utf-8 -*-
"""
plot_welfare.py

(1) Newborns: plots the expenditure-equivalent welfare loss of sea level rise
    (sheet 'tax_equiv_newborns' in welfare_results.xlsx), one figure per belief type.
(2) Households alive in 1998: LaTeX table of the expenditure-equivalent tax
    (sheets 'tax_equiv_C', 'tax_equiv_NC', 'tax_equiv_renter' and their _RE versions) by income level,
    plus an average over income levels weighted by the t = 0 income distribution.

Output (in OUT_DIR): welfare_newborns_realists/sceptics (.png and .eps)
                     and welfare_SLR_table.tex (tabular only; \\input it in the paper)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import misc_functions as misc
import par_epsilons as parfile
import tauchen as tauch

FILE = "welfare_results.xlsx"
SHEET = "tax_equiv_newborns"
OUT_DIR = r"C:\Users\TPRINS\OneDrive - UvA\Documenten\Python files\New coding round July 2026\Plaatjes"
START_YEAR, STEP = 1998, 2
SHARE_Y = True   # same y-axis range in both panels, so they compare directly side by side

# one blue hue, light -> dark = income level 1 -> 5
COLORS = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]

plt.rcParams.update({
    "font.size": 18, "axes.labelsize": 20, "xtick.labelsize": 17,
    "ytick.labelsize": 17, "legend.fontsize": 15, "legend.title_fontsize": 15,
    "axes.spines.top": False, "axes.spines.right": False,
})
os.makedirs(OUT_DIR, exist_ok=True)

# ============ (1) newborns: figures ============
def load_newborns(sheet):
    """Newborn sheet -> DataFrame in percent, with a calendar-year column.
    t is only written on the first row of each (t, k) block, so fill it down."""
    d = pd.read_excel(FILE, sheet_name=sheet)
    d["t"] = d["t"].ffill().astype(int)
    d = d[d["t"] < d["t"].max()].copy()   # last period is never computed (all zeros)
    d["year"] = START_YEAR + STEP * d["t"]
    cols = [c for c in d.columns if c.startswith("e")]
    d[cols] = 100 * d[cols]               # to percent
    return d


def plot_newborns(df, re=None, suffix="", ylabel="Equivalent tax (%)", legend_loc="upper left"):
    """One figure per belief type: HE losses per income level as solid lines.
    If re is given, the RE losses (one type only) are added as dotted lines
    in the same colours, identical in both figures.
    Also used for the policy results (section 3), which can be negative."""
    ymax = df[e_cols].max().max()
    if re is not None:
        ymax = max(ymax, re[e_cols].max().max())
    ymin = min(0, df[e_cols].min().min())     # 0 unless some values are negative
    pad = 0.05 * (ymax - ymin)

    for k, name in [(0, "realists"), (1, "sceptics")]:
        d = df[df["k"] == k]
        fig, ax = plt.subplots(figsize=(7, 5))
        for i, col in enumerate(e_cols):
            ax.plot(d["year"], d[col], color=COLORS[i], lw=2.2, label=str(i + 1))
            if re is not None:            # stop RE where the HE line stops (no sceptics left)
                r = re[re["year"] <= d.loc[d[col].notna(), "year"].max()]
                ax.plot(r["year"], r[col], color=COLORS[i], lw=2.2, ls=":")
        ax.set_xlabel("Year")
        ax.set_ylabel(ylabel)
        ax.set_xlim(df["year"].min(), df["year"].max())
        if SHARE_Y:
            ax.set_ylim(ymin - (pad if ymin < 0 else 0), ymax + pad)
        if ymin < 0:
            ax.axhline(0, color="0.5", lw=0.8)
        ax.grid(axis="y", color="0.88", lw=0.8)
        loc = legend_loc[k] if isinstance(legend_loc, dict) else legend_loc   # per belief type if a dict
        leg = ax.legend(title="Income level", frameon=False, loc=loc)
        if re is not None:                # second legend: line style = equilibrium
            ax.add_artist(leg)
            style = [Line2D([], [], color="0.3", lw=2.2),
                     Line2D([], [], color="0.3", lw=2.2, ls=":")]
            ax.legend(style, ["HE", "RE"], frameon=False,
                      loc="upper left", bbox_to_anchor=(0.33, 1.0))   # right of the income legend
        fig.tight_layout()
        for ext in ("png", "eps"):
            fig.savefig(os.path.join(OUT_DIR, f"welfare_newborns_{name}{suffix}.{ext}"), dpi=300)
        plt.close(fig)


df = load_newborns(SHEET)
df_re = load_newborns(SHEET + "_RE")
e_cols = [c for c in df.columns if c.startswith("e")]

plot_newborns(df)                         # HE only (as before)
plot_newborns(df, df_re, suffix="_vs_RE") # HE (solid) and RE (dotted)

# ============ (2) households alive in 1998: table ============
TABLE_SHEETS = [("tax_equiv_C", "Coastal owners"),
                ("tax_equiv_NC", "Non-coastal owners"),
                ("tax_equiv_renter", "Renters")]

# income weights: population distribution over income levels at t = 0 (first row of mPi_E)
par = misc.construct_jitclass(parfile.par_dict)
mMarkov, vE = tauch.tauchen(par.dRho, par.dSigmaeps, par.iNumStates, par.iM, par.time_increment)
vPi_E = tauch.initial_dist(par, vE)
mPi_E = tauch.weight_matrix(par, vE, vPi_E, mMarkov)
w = np.asarray(mPi_E[0], dtype=float)
w = w / w.sum()

# rows: income levels 1..5; columns: (household group, Realists / Sceptics / RE); values in percent
cols = {}
for sheet, group in TABLE_SHEETS:
    d = pd.read_excel(FILE, sheet_name=sheet, index_col="k")
    for k, kname in [(0, "Realists"), (1, "Sceptics")]:
        cols[(group, kname)] = 100 * d.loc[k, e_cols].to_numpy(dtype=float)
    d_re = pd.read_excel(FILE, sheet_name=sheet + "_RE", index_col="k")   # RE: one type only
    cols[(group, "RE")] = 100 * d_re.loc[0, e_cols].to_numpy(dtype=float)
tab = pd.DataFrame(cols, index=[str(i + 1) for i in range(len(e_cols))])
tab.loc["Average"] = w @ tab.to_numpy()
print("Income weights:", np.round(w, 4))
print(tab.round(3))

# booktabs tabular; numbers in math mode so negatives get a proper minus sign
fmt = lambda v: f"${v:.2f}$"
n = tab.shape[1] // len(TABLE_SHEETS)     # columns per household group (Realists, Sceptics, RE)
lines = [r"\begin{tabular}{@{}l" + "r" * tab.shape[1] + "@{}}", r"\toprule",   # @{}: no padding at the outer edges
         " & " + " & ".join(rf"\multicolumn{{{n}}}{{c}}{{{g}}}" for _, g in TABLE_SHEETS) + r" \\",
         "".join(rf"\cmidrule(lr){{{2 + n * i}-{1 + n * (i + 1)}}}" for i in range(len(TABLE_SHEETS))),
         "Income level & " + " & ".join(kname for _, kname in tab.columns) + r" \\",
         r"\midrule"]
for label, row in tab.iterrows():
    if label == "Average":
        lines.append(r"\midrule")
    lines.append(f"{label} & " + " & ".join(fmt(v) for v in row) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]

with open(os.path.join(OUT_DIR, "welfare_SLR_table.tex"), "w") as fh:
    fh.write("\n".join(lines) + "\n")

# ============ (3) newborns: policies introduced in 2026 ============
POLICY_YEAR = 2026
T_POL = (POLICY_YEAR - START_YEAR) // STEP    # first period with the policy (t = 14)
slr = df.set_index(["t", "k"])[e_cols]        # SLR losses of newborns (%), by (t, k)


def policy_gains(tag):
    """Newborn welfare gain (%) from policy `tag` (BR or MP), and that gain as % of the
    SLR loss of the same newborns (same t, k, e). The sheet holds the tax on the no-policy
    (BAU) world that makes it as good as the policy world: negative = the policy is a gain,
    so gain = -tax."""
    gain = load_newborns(SHEET + "_" + tag)
    gain = gain[gain["t"] >= T_POL].copy()
    gain[e_cols] = -gain[e_cols].mask(gain[e_cols] <= -9.99)   # -10% = first grid point: no newborns of that type left
    share = gain.copy()
    share[e_cols] = 100 * gain[e_cols].to_numpy() / slr.loc[list(zip(gain["t"], gain["k"]))].to_numpy()
    return gain, share


# legend positions (gain figure, share figure): the empty corner in each figure, per belief type
POLICIES = {"BR": ("upper left", {0: "upper left", 1: "lower right"}),     # building standards
            "MP": ("lower left", {0: "lower right", 1: "lower left"})}     # mortgage premium

for tag, (loc_gain, loc_share) in POLICIES.items():
    gain, share = policy_gains(tag)
    plot_newborns(gain, suffix=f"_{tag}_gain", ylabel="Welfare gain (%)", legend_loc=loc_gain)
    plot_newborns(share, suffix=f"_{tag}_gain_share", ylabel="Gain as % of SLR loss", legend_loc=loc_share)
