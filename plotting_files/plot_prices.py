# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 14:04:17 2026

@author: tprins
"""
import matplotlib.pyplot as plt
import numpy as np
import LoM_epsilons as lom
import par_epsilons as parfile
import grid_creation as grid_creation
import misc_functions as misc
import coeff_io
"""
Inputs: coefficienten voor: (1) initial steady state, (2) baseline transition with sceptics, (3) baseline transition without sceptics (RE), and three sets of policy coefficients for (4) full information shock, (5) building restrictions, (6) mortgage premium

outputs: four distributions over individual state variables over time for the above 5 cases: owners C, owners NC, renters C, renters NC. Perhaps collapsed on the G, J dimensions (if we're not interested in this source of heterogeneity)
"""
def plot_pricepaths(
    par,
    grids,
    vCoeff_C_initial,
    vCoeff_NC_initial,
    vCoeff_C,
    vCoeff_NC,
    vCoeff_C_RE,
    vCoeff_NC_RE,
    vCoeff_C_terminal_RE,
    vCoeff_NC_terminal_RE,
    vCoeff_C_terminal_HE,
    vCoeff_NC_terminal_HE,
):
    normalisation = vCoeff_NC_initial[0]

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.figure(figsize=(7.0, 4.6))

    T = len(grids.vTime)
    t_indices = np.arange(T)
    years = par.starting_year + par.time_increment * t_indices

    yC_RE = np.array([
        lom.LoM(par, grids, t_index, vCoeff_C_RE)
        for t_index in t_indices
    ]) / normalisation

    yNC_RE = np.array([
        lom.LoM(par, grids, t_index, vCoeff_NC_RE)
        for t_index in t_indices
    ]) / normalisation

    yC_HE = np.array([
        lom.LoM(par, grids, t_index, vCoeff_C)
        for t_index in t_indices
    ]) / normalisation

    yNC_HE = np.array([
        lom.LoM(par, grids, t_index, vCoeff_NC)
        for t_index in t_indices
    ]) / normalisation

    lineC, = plt.plot(years, yC_RE, linestyle=":", linewidth=2)
    lineNC, = plt.plot(years, yNC_RE, linestyle=":", linewidth=2)

    plt.plot(
        years,
        yC_HE,
        linewidth=2,
        color=lineC.get_color(),
        label="Flood-exposed price trajectory",
    )

    plt.plot(
        years,
        yNC_HE,
        linewidth=2,
        color=lineNC.get_color(),
        label="Inland price trajectory",
    )

    # Initial prices
    x0_year = years[0]
    y_coastal = lom.LoM(par, grids, 0, vCoeff_C_initial) / normalisation
    y_inland = lom.LoM(par, grids, 0, vCoeff_NC_initial) / normalisation

    plt.scatter([x0_year], [y_coastal], zorder=5)
    plt.scatter([x0_year], [y_inland], zorder=5)

    #plt.annotate(
    #    "Initial flood-exposed price",
    #    (x0_year, y_coastal),
    #    xytext=(10, 10),
    #    textcoords="offset points",
    #    ha="center",
    #    fontsize=9,
    #)

    #plt.annotate(
    #    "Initial inland price",
    #    (x0_year, y_inland),
    #    xytext=(10, 10),
    #    textcoords="offset points",
    #    ha="center",
    #    fontsize=9,
    #)

    # Terminal prices
    xT_year = years[-1]

    yC_terminal_HE = (
        lom.LoM(par, grids, T - 1, vCoeff_C_terminal_HE)
        / normalisation
    )

    yNC_terminal_HE = (
        lom.LoM(par, grids, T - 1, vCoeff_NC_terminal_HE)
        / normalisation
    )

    plt.scatter(
        [xT_year],
        [yC_terminal_HE],
        color=lineC.get_color(),
        zorder=5,
    )

    plt.scatter(
        [xT_year],
        [yNC_terminal_HE],
        color=lineNC.get_color(),
        zorder=5,
    )

    #plt.annotate(
    #    "Terminal flood-exposed price",
    #    (xT_year, yC_terminal_HE),
    #    xytext=(-10, 10),
    #    textcoords="offset points",
    #    ha="right",
    #    fontsize=9,
    #)

    #plt.annotate(
    #    "Terminal inland price",
    #    (xT_year, yNC_terminal_HE),
    #    xytext=(-10, 10),
    #    textcoords="offset points",
    #    ha="right",
    #    fontsize=9,
    #)

    plt.vlines(x0_year, y_coastal, yC_HE[0],
               linestyles="dotted", linewidth=1)
    plt.vlines(x0_year, y_inland, yNC_HE[0],
               linestyles="dotted", linewidth=1)

    xticks = np.arange(int(years[0]), int(years[-1]) + 1, 20)
    plt.xticks(xticks)

    plt.xlabel("Year", fontsize=18)
    plt.ylabel("Price", fontsize=18)
    plt.title("House price trajectories", fontsize=18)
    plt.tick_params(labelsize=16)
    plt.legend(frameon=False, fontsize=16)
    plt.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.show()
    
def plot_price_transition_exp(
        par,
        grids,
        vCoeff_C_initial,
        vCoeff_NC_initial,
        vCoeff_C_baseline,
        vCoeff_NC_baseline,
        experiments,
        switch_index=14,
    ):
    """House prices with a policy introduced at switch_index: one separate figure per
    experiment, in the look of plot_pricepaths, all on the same y-range so they can be
    placed side by side. experiments: list of (title, vCoeff_C_experiment, vCoeff_NC_experiment).
    Prices are normalised by the initial inland price, as in plot_pricepaths.
    After the switch, the price without the policy is shown as a dotted line."""
    normalisation = vCoeff_NC_initial[0]      # initial inland price, as in plot_pricepaths

    T = len(grids.vTime)
    t_indices = np.arange(T)
    years = par.starting_year + par.time_increment * t_indices
    s = switch_index

    def path(vCoeff):
        return np.array([lom.LoM(par, grids, t_index, vCoeff) for t_index in t_indices]) / normalisation

    base = {"C": path(vCoeff_C_baseline), "NC": path(vCoeff_NC_baseline)}
    initial = {"C": lom.LoM(par, grids, 0, vCoeff_C_initial) / normalisation,
               "NC": lom.LoM(par, grids, 0, vCoeff_NC_initial) / normalisation}
    exps = [(title, {"C": path(cC), "NC": path(cNC)}) for title, cC, cNC in experiments]
    series = [("C", "tab:blue", "Flood-exposed"), ("NC", "tab:orange", "Inland")]

    # common y-range for all experiment figures
    values = np.concatenate([base["C"], base["NC"], [initial["C"], initial["NC"]]]
                            + [e[loc] for _, e in exps for loc in ("C", "NC")])
    pad = 0.05 * (values.max() - values.min())
    ylim = (values.min() - pad, values.max() + pad)

    plt.style.use("seaborn-v0_8-whitegrid")
    for title, exp in exps:
        plt.figure(figsize=(7.0, 4.6))
        for loc, color, name in series:
            b, e = base[loc], exp[loc]
            plt.plot(years[:s + 1], b[:s + 1], linewidth=2, color=color, label=name)   # before the policy
            plt.plot(years[s:], e[s:], linewidth=2, color=color)                       # with the policy
            plt.plot(years[s:], b[s:], linewidth=2, linestyle=":", color=color)        # without the policy
            plt.vlines(years[0], initial[loc], b[0], linestyles="dotted", linewidth=1, color=color)  # jump in 1998
            plt.vlines(years[s], b[s], e[s], linestyles="dotted", linewidth=1, color=color)          # jump at the switch
            plt.scatter([years[0], years[s], years[s], years[-1]],
                        [initial[loc], b[s], e[s], e[-1]], color=color, zorder=5)
        plt.axvline(years[s], linestyle=":", linewidth=1, color="0.5")
        plt.plot([], [], linewidth=2, linestyle=":", color="0.3", label="Without policy")   # legend entry only

        plt.xticks(np.arange(int(years[0]), int(years[-1]) + 1, 20))
        plt.ylim(*ylim)
        plt.xlabel("Year", fontsize=18)
        plt.ylabel("Price", fontsize=18)
        plt.title(title, fontsize=18)
        plt.tick_params(labelsize=16)
        plt.legend(frameon=False, fontsize=16)
        plt.grid(True, linestyle="--", alpha=0.4)

        plt.tight_layout()
        plt.show()


def rental_price(par, grids, t_index, P, P_prime, coastal):
    """User-cost rent given this period's price P and next period's price P_prime."""
    dmg = grids.vPi_S_median[t_index] * np.dot(grids.vPDF_z[1:], 1 - grids.vZ[1:]) if coastal else 0.0
    return par.dPsi + max(P - (1 - par.dDelta - dmg) / (1 + par.r) * P_prime, 0)


def rent_path(par, grids, vCoeff, coastal):
    """Rents along the transition; price is held constant in the last period (P' = P)."""
    T = len(grids.vTime)
    P = [lom.LoM(par, grids, t, vCoeff) for t in range(T)]
    P.append(P[-1])
    return np.array([rental_price(par, grids, t, P[t], P[t + 1], coastal) for t in range(T)])

def plot_rentpaths(par, grids, vCoeff_C_initial, vCoeff_NC_initial, vCoeff_C, vCoeff_NC,
                   vCoeff_C_RE, vCoeff_NC_RE, vCoeff_C_terminal_RE, vCoeff_NC_terminal_RE,
                   vCoeff_C_terminal_HE, vCoeff_NC_terminal_HE, end_year=2100):
    
    P0_NC = lom.LoM(par, grids, 0, vCoeff_NC_initial)
    normalisation = rental_price(par, grids, 0, P0_NC, P0_NC, False)   # initial inland rent                 # same as price graph
    T = len(grids.vTime)
    years = par.starting_year + par.time_increment * np.arange(T)
    n = int((end_year - par.starting_year) / par.time_increment) + 1   # periods up to end_year
    years = years[:n]
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.figure(figsize=(7.0, 4.6))

    specs = [  # (coastal, HE, HE terminal, RE, RE terminal, initial, name)
        (True,  vCoeff_C,  vCoeff_C_terminal_HE,  vCoeff_C_RE,  vCoeff_C_terminal_RE,  vCoeff_C_initial,  "flood-exposed"),
        (False, vCoeff_NC, vCoeff_NC_terminal_HE, vCoeff_NC_RE, vCoeff_NC_terminal_RE, vCoeff_NC_initial, "inland"),
    ]
    for coastal, cHE, cHE_T, cRE, cRE_T, c0, name in specs:
        rHE = rent_path(par, grids, cHE, coastal)[:n] / normalisation
        rRE = rent_path(par, grids, cRE, coastal)[:n] / normalisation
        P0 = lom.LoM(par, grids, 0, c0)                       # initial steady state: P' = P
        PT = lom.LoM(par, grids, T - 1, cHE_T)                # terminal steady state: P' = P
        r0 = rental_price(par, grids, 0, P0, P0, coastal) / normalisation
        rT = rental_price(par, grids, T - 1, PT, PT, coastal) / normalisation

        line, = plt.plot(years, rRE, linestyle=":", linewidth=2)
        col = line.get_color()
        plt.plot(years, rHE, linewidth=2, color=col, label=f"{name.capitalize()} rent trajectory")

        plt.scatter([years[0]], [r0], color=col, zorder=5)
        #plt.annotate(f"Initial {name} rent", (years[0], r0), xytext=(10, 10),
        #             textcoords="offset points", ha="center", fontsize=9)
        plt.vlines(years[0], r0, rHE[0], linestyles="dotted", linewidth=1)

        plt.scatter([years[-1]], [rT], color=col, zorder=5)
        #plt.annotate(f"Terminal {name} rent", (years[-1], rT), xytext=(-10, 10),
        #             textcoords="offset points", ha="right", fontsize=9)

    plt.xticks(np.arange(int(years[0]), int(years[-1]) + 1, 20))
    plt.xlabel("Year", fontsize=18)
    plt.ylabel("Rent", fontsize=18)
    plt.title("Rental price trajectories", fontsize=18)
    plt.tick_params(labelsize=16)
    plt.legend(frameon=False, fontsize=16)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    """############################################################################
    ### Plot price paths
    ############################################################################"""
    par = misc.construct_jitclass(parfile.par_dict)
    grids = grid_creation.create(par)

    # load solved coefficients by label (falls back to {} if file absent)
    C = coeff_io.load_coefficients()

    ### without experiments
    plot_pricepaths(
        par,
        grids,
        C["vCoeff_C_initial_HE"],
        C["vCoeff_NC_initial_HE"],
        C["vCoeff_C_HE"],
        C["vCoeff_NC_HE"],
        C["vCoeff_C_RE"],
        C["vCoeff_NC_RE"],
        C["vCoeff_C_terminal_RE"],
        C["vCoeff_NC_terminal_RE"],
        C["vCoeff_C_terminal_HE"],
        C["vCoeff_NC_terminal_HE"],
    )

    # Policy experiments: one figure each, same y-range (to place side by side in LaTeX)
    plot_price_transition_exp(
        par,
        grids,
        C["vCoeff_C_initial_HE"],
        C["vCoeff_NC_initial_HE"],
        C["vCoeff_C_HE"],
        C["vCoeff_NC_HE"],
        [("Building standards", C["vCoeff_C_BR"], C["vCoeff_NC_BR"]),
         ("Mortgage premium", C["vCoeff_C_MP"], C["vCoeff_NC_MP"])],
        switch_index=14,
    )
    
    plot_rentpaths(par, grids,
       C["vCoeff_C_initial_HE"], C["vCoeff_NC_initial_HE"],
       C["vCoeff_C_HE"], C["vCoeff_NC_HE"],
       C["vCoeff_C_RE"], C["vCoeff_NC_RE"],
       C["vCoeff_C_terminal_RE"], C["vCoeff_NC_terminal_RE"],
       C["vCoeff_C_terminal_HE"], C["vCoeff_NC_terminal_HE"])