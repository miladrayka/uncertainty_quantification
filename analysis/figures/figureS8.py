"""Draw UQ (calibrated) metrics subplots."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager

# Necessary for multibars plot.

BARWIDTH = 0.15
br1 = np.arange(5)
br2 = [x + BARWIDTH for x in br1]
br3 = [x + BARWIDTH for x in br2]
br4 = [x + BARWIDTH for x in br3]
br5 = [x + BARWIDTH for x in br4]

# 12, 18
FIG_HEIGHT = 6  # figure height
FIG_WIDTH = 18  # figure width

BARWIDTH = 0.1  # bar width
BAR_LW = 1  # bar linewidth
GRID_LW = 0.5  # grid linewidth

FACECOLOR1 = "#8ecae6"  # bar1 or Ensemble face color
FACECOLOR2 = "#219ebc"  # bar2 or MC-Dropout face color
FACECOLOR3 = "#023047"  # bar3 or Laplace Approximation face color
FACECOLOR4 = "#FFB703"  # bar4 or Bayes-by-Bayesian face color
FACECOLOR5 = "#FB8500"  # bar5 or Evidential Neural Network face color

EDGECOLOR1 = "black"  # bar1 or Ensemble edge color
EDGECOLOR2 = "black"  # bar2 or MC-Dropout edge color
EDGECOLOR3 = "black"  # bar3 or Laplace Approximation edge color
EDGECOLOR4 = "black"  # bar4 or Bayes-by-Bayesian edge color
EDGECOLOR5 = "black"  # bar5 or Evidential Neural Network edge color

font_manager.findfont("Helvetica Light")
plt.rc("font", family="Helvetica Light")
plt.rc("font", serif="Helvetica Light", size=25)
plt.rcParams["axes.linewidth"] = 1.25
plt.rcParams["xtick.major.size"] = 0
plt.rcParams["xtick.major.width"] = 1.25
plt.rcParams["ytick.major.size"] = 8
plt.rcParams["ytick.major.width"] = 1.25
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["legend.markerscale"] = 2

plt.rcParams["mathtext.it"] = "Helvetica Light:italic"
plt.rcParams["mathtext.rm"] = "Helvetica Light"
plt.rcParams["mathtext.default"] = "regular"

# Load UQ metrics results.

uq_test_df = pd.read_csv(r"../uncertainty_results/uq_metrics_lp_test.csv", index_col=0)
uq_bdb2020_df = pd.read_csv(
    r"../uncertainty_results/uq_metrics_bdb2020+.csv", index_col=0
)
uq_egfr_df = pd.read_csv(r"../uncertainty_results/uq_metrics_egfr.csv", index_col=0)
uq_mpro_df = pd.read_csv(r"../uncertainty_results/uq_metrics_mpro.csv", index_col=0)
uq_peptide_holdout_df = pd.read_csv(
    r"../uncertainty_results/uq_metrics_peptide_holdout.csv", index_col=0
)

# CMACE for each UQ methods on all test datasets.

cmace_ens = [
    df.loc["CMACE", "ensemble"]
    for df in [uq_test_df, uq_bdb2020_df, uq_egfr_df, uq_mpro_df, uq_peptide_holdout_df]
]
cmace_mc = [
    df.loc["CMACE", "mcdropout"]
    for df in [uq_test_df, uq_bdb2020_df, uq_egfr_df, uq_mpro_df, uq_peptide_holdout_df]
]
cmace_la = [
    df.loc["CMACE", "la"]
    for df in [uq_test_df, uq_bdb2020_df, uq_egfr_df, uq_mpro_df, uq_peptide_holdout_df]
]
cmace_bayes = [
    df.loc["CMACE", "bayes_by_bayesian"]
    for df in [uq_test_df, uq_bdb2020_df, uq_egfr_df, uq_mpro_df, uq_peptide_holdout_df]
]
cmace_enn = [
    df.loc["CMACE", "enn"]
    for df in [uq_test_df, uq_bdb2020_df, uq_egfr_df, uq_mpro_df, uq_peptide_holdout_df]
]

# CRMSCE for each UQ methods on all test datasets.

crmsce_ens = [
    df.loc["CRMSCE", "ensemble"]
    for df in [uq_test_df, uq_bdb2020_df, uq_egfr_df, uq_mpro_df, uq_peptide_holdout_df]
]
crmsce_mc = [
    df.loc["CRMSCE", "mcdropout"]
    for df in [uq_test_df, uq_bdb2020_df, uq_egfr_df, uq_mpro_df, uq_peptide_holdout_df]
]
crmsce_la = [
    df.loc["CRMSCE", "la"]
    for df in [uq_test_df, uq_bdb2020_df, uq_egfr_df, uq_mpro_df, uq_peptide_holdout_df]
]
crmsce_bayes = [
    df.loc["CRMSCE", "bayes_by_bayesian"]
    for df in [uq_test_df, uq_bdb2020_df, uq_egfr_df, uq_mpro_df, uq_peptide_holdout_df]
]
crmsce_enn = [
    df.loc["CRMSCE", "enn"]
    for df in [uq_test_df, uq_bdb2020_df, uq_egfr_df, uq_mpro_df, uq_peptide_holdout_df]
]

# Calibrated Miscalibration Area for each UQ methods on all test datasets.

cmca_ens = [
    df.loc["CMCA", "ensemble"]
    for df in [uq_test_df, uq_bdb2020_df, uq_egfr_df, uq_mpro_df, uq_peptide_holdout_df]
]
cmca_mc = [
    df.loc["CMCA", "mcdropout"]
    for df in [uq_test_df, uq_bdb2020_df, uq_egfr_df, uq_mpro_df, uq_peptide_holdout_df]
]
cmca_la = [
    df.loc["CMCA", "la"]
    for df in [uq_test_df, uq_bdb2020_df, uq_egfr_df, uq_mpro_df, uq_peptide_holdout_df]
]
cmca_bayes = [
    df.loc["CMCA", "bayes_by_bayesian"]
    for df in [uq_test_df, uq_bdb2020_df, uq_egfr_df, uq_mpro_df, uq_peptide_holdout_df]
]
cmca_enn = [
    df.loc["CMCA", "enn"]
    for df in [uq_test_df, uq_bdb2020_df, uq_egfr_df, uq_mpro_df, uq_peptide_holdout_df]
]

# Subplots.

fig, ax = plt.subplots(nrows=1, ncols=3)

fig.set_figheight(FIG_HEIGHT)
fig.set_figwidth(FIG_WIDTH)

# CMACE vs test datasets.

ax[0].bar(
    x=br1,
    height=cmace_ens,
    width=BARWIDTH,
    label="Deep Ensemble",
    edgecolor=EDGECOLOR1,
    facecolor=FACECOLOR1,
    linewidth=BAR_LW,
)
ax[0].bar(
    x=br2,
    height=cmace_mc,
    width=BARWIDTH,
    label="MC-Dropout",
    edgecolor=EDGECOLOR2,
    facecolor=FACECOLOR2,
    linewidth=BAR_LW,
)
ax[0].bar(
    x=br3,
    height=cmace_la,
    width=BARWIDTH,
    label="Laplace Approximation",
    edgecolor=EDGECOLOR3,
    facecolor=FACECOLOR3,
    linewidth=BAR_LW,
)
ax[0].bar(
    x=br4,
    height=cmace_bayes,
    width=BARWIDTH,
    label="Bayes by Backprop",
    edgecolor=EDGECOLOR4,
    facecolor=FACECOLOR4,
    linewidth=BAR_LW,
)
ax[0].bar(
    x=br5,
    height=cmace_enn,
    width=BARWIDTH,
    label="Evidential Neural Network",
    edgecolor=EDGECOLOR5,
    facecolor=FACECOLOR5,
    linewidth=BAR_LW,
)
ax[0].grid(visible=True, ls="--", lw=GRID_LW)
ax[0].set_ylabel("CMACE")
ax[0].set_xticks(
    np.arange(0.25, 5, 1),
    ["LP-Test", "BDB2020+", "EGFR", "$M^{Pro}$", "Peptide-Holdout"],
)
ax[0].set_xticklabels(
    ["LP-Test", "BDB2020+", "EGFR", "$M^{Pro}$", "Peptide-Holdout"], rotation=90
)

# CRMSCE vs test datasets.

ax[1].bar(
    x=br1,
    height=crmsce_ens,
    width=BARWIDTH,
    edgecolor=EDGECOLOR1,
    facecolor=FACECOLOR1,
    linewidth=BAR_LW,
)
ax[1].bar(
    x=br2,
    height=crmsce_mc,
    width=BARWIDTH,
    edgecolor=EDGECOLOR2,
    facecolor=FACECOLOR2,
    linewidth=BAR_LW,
)
ax[1].bar(
    x=br3,
    height=crmsce_la,
    width=BARWIDTH,
    edgecolor=EDGECOLOR3,
    facecolor=FACECOLOR3,
    linewidth=BAR_LW,
)
ax[1].bar(
    x=br4,
    height=crmsce_bayes,
    width=BARWIDTH,
    edgecolor=EDGECOLOR4,
    facecolor=FACECOLOR4,
    linewidth=BAR_LW,
)
ax[1].bar(
    x=br5,
    height=crmsce_enn,
    width=BARWIDTH,
    edgecolor=EDGECOLOR5,
    facecolor=FACECOLOR5,
    linewidth=BAR_LW,
)
ax[1].grid(visible=True, ls="--", lw=GRID_LW)
ax[1].set_ylabel("CRMSCE")
ax[1].set_xticks(
    np.arange(0.25, 5, 1),
    ["LP-Test", "BDB2020+", "EGFR", "$M^{Pro}$", "Peptide-Holdout"],
)
ax[1].set_xticklabels(
    ["LP-Test", "BDB2020+", "EGFR", "$M^{Pro}$", "Peptide-Holdout"], rotation=90
)

# CMCA vs test datasets.

ax[2].bar(
    x=br1,
    height=cmca_ens,
    width=BARWIDTH,
    edgecolor=EDGECOLOR1,
    facecolor=FACECOLOR1,
    linewidth=BAR_LW,
)
ax[2].bar(
    x=br2,
    height=cmca_mc,
    width=BARWIDTH,
    edgecolor=EDGECOLOR2,
    facecolor=FACECOLOR2,
    linewidth=BAR_LW,
)
ax[2].bar(
    x=br3,
    height=cmca_la,
    width=BARWIDTH,
    edgecolor=EDGECOLOR3,
    facecolor=FACECOLOR3,
    linewidth=BAR_LW,
)
ax[2].bar(
    x=br4,
    height=cmca_bayes,
    width=BARWIDTH,
    edgecolor=EDGECOLOR4,
    facecolor=FACECOLOR4,
    linewidth=BAR_LW,
)
ax[2].bar(
    x=br5,
    height=cmca_enn,
    width=BARWIDTH,
    edgecolor=EDGECOLOR5,
    facecolor=FACECOLOR5,
    linewidth=BAR_LW,
)
ax[2].grid(visible=True, ls="--", lw=GRID_LW)
ax[2].set_ylabel("CMCA")

ax[2].set_xticks(
    np.arange(0.25, 5, 1),
    ["LP-Test", "BDB2020+", "EGFR", "$M^{Pro}$", "Peptide-Holdout"],
)
ax[2].set_xticklabels(
    ["LP-Test", "BDB2020+", "EGFR", "$M^{Pro}$", "Peptide-Holdout"], rotation=90
)

fig.legend(loc=(0.06, 0.87), ncols=3)

plt.subplots_adjust(wspace=0.33)
plt.savefig(
    "uq_calibrated_metrics_plot.png", dpi=300, bbox_inches="tight", pad_inches=0.3
)
plt.show()
