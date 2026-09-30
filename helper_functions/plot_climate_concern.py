# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 12:05:48 2026

@author: tprins
"""

import pandas as pd
import matplotlib.pyplot as plt
 
# Path to the exported Excel (adjust as needed)
PATH = r"C:\Users\TPRINS\OneDrive - UvA\Documenten\Stata files\New STATA files\concern_forecast_new.xlsx"
 
# export excel writes no header row, so name the columns ourselves
df = pd.read_excel(PATH, header=None,
                   names=["cal_year", "p_concern", "p_concern_yearly"])
 
fig, ax = plt.subplots(figsize=(9, 5))
 
# Trend-model prediction (continuous line)
ax.plot(df["cal_year"], df["p_concern"],
        color="tab:blue", label="Baseline belief scenario")
 
# Per-year probit predictions (only sample years are non-missing)
yearly = df.dropna(subset=["p_concern_yearly"])
ax.scatter(yearly["cal_year"], yearly["p_concern_yearly"],
           facecolors="none", edgecolors="tab:red", zorder=5,
           label="CCAM yearly data")
 
# Sample-period boundaries
ax.axvline(2008, linestyle="--", color="gray")
ax.axvline(2024, linestyle="--", color="gray")
 
ax.set_xlabel("Year", fontsize=14)
ax.set_ylabel("Share of realists", fontsize=14)
ax.legend(loc="upper left", fontsize=12)
#ax.set_title("Extrapolated climate concern, age 40")
ax.set_ylim(0, 1)

plt.tight_layout()
plt.show()
 