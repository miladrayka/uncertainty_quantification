"""Draw UQ (uncalibrated) metrics subplots."""

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

FIG_HEIGHT = 15  # figure height
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

uq_test_df = pd.read_csv(
    "../uncertainty_results/uq_metrics_lp_test.csv", index_col=0)
uq_bdb2020_df = pd.read_csv(
    "../uncertainty_results/uq_metrics_bdb2020+.csv", index_col=0
)
uq_egfr_df = pd.read_csv(
    "../uncertainty_results/uq_metrics_egfr.csv", index_col=0)
uq_mpro_df = pd.read_csv(
    "../uncertainty_results/uq_metrics_mpro.csv", index_col=0)
uq_peptide_holdout_df = pd.read_csv(
    "../uncertainty_results/uq_metrics_peptide_holdout.csv", index_col=0
)

# Spearman's rank correlation coefficient for each UQ methods on all test datasets.

sp_ens = [
    df.loc["Spearman", "ensemble"]
    for df in [uq_test_df, uq_bdb2020_df, uq_egfr_df, uq_mpro_df, uq_peptide_holdout_df]
]
sp_mc = [
    df.loc["Spearman", "mcdropout"]
    for df in [uq_test_df, uq_bdb2020_df, uq_egfr_df, uq_mpro_df, uq_peptide_holdout_df]
]
sp_la = [
    df.loc["Spearman", "la"]
    for df in [uq_test_df, uq_bdb2020_df, uq_egfr_df, uq_mpro_df, uq_peptide_holdout_df]
]
sp_bayes = [
    df.loc["Spearman", "bayes_by_bayesian"]
    for df in [uq_test_df, uq_bdb2020_df, uq_egfr_df, uq_mpro_df, uq_peptide_holdout_df]
]
sp_enn = [
    df.loc["Spearman", "enn"]
    for df in [uq_test_df, uq_bdb2020_df, uq_egfr_df, uq_mpro_df, uq_peptide_holdout_df]
]

# Sharpness for each UQ methods on all test datasets.

sh_ens = [
    df.loc["Sharpness", "ensemble"]
    for df in [uq_test_df, uq_bdb2020_df, uq_egfr_df, uq_mpro_df, uq_peptide_holdout_df]
]
sh_mc = [
    df.loc["Sharpness", "mcdropout"]
    for df in [uq_test_df, uq_bdb2020_df, uq_egfr_df, uq_mpro_df, uq_peptide_holdout_df]
]
sh_la = [
    df.loc["Sharpness", "la"]
    for df in [uq_test_df, uq_bdb2020_df, uq_egfr_df, uq_mpro_df, uq_peptide_holdout_df]
]
sh_bayes = [
    df.loc["Sharpness", "bayes_by_bayesian"]
    for df in [uq_test_df, uq_bdb2020_df, uq_egfr_df, uq_mpro_df, uq_peptide_holdout_df]
]
sh_enn = [
    df.loc["Sharpness", "enn"]
    for df in [uq_test_df, uq_bdb2020_df, uq_egfr_df, uq_mpro_df, uq_peptide_holdout_df]
]

# MACE for each UQ methods on all test datasets.

mace_ens = [
    df.loc["MACE", "ensemble"]
    for df in [uq_test_df, uq_bdb2020_df, uq_egfr_df, uq_mpro_df, uq_peptide_holdout_df]
]
mace_mc = [
    df.loc["MACE", "mcdropout"]
    for df in [uq_test_df, uq_bdb2020_df, uq_egfr_df, uq_mpro_df, uq_peptide_holdout_df]
]
mace_la = [
    df.loc["MACE", "la"]
    for df in [uq_test_df, uq_bdb2020_df, uq_egfr_df, uq_mpro_df, uq_peptide_holdout_df]
]
mace_bayes = [
    df.loc["MACE", "bayes_by_bayesian"]
    for df in [uq_test_df, uq_bdb2020_df, uq_egfr_df, uq_mpro_df, uq_peptide_holdout_df]
]
mace_enn = [
    df.loc["MACE", "enn"]
    for df in [uq_test_df, uq_bdb2020_df, uq_egfr_df, uq_mpro_df, uq_peptide_holdout_df]
]

# RMSCE for each UQ methods on all test datasets.

rmsce_ens = [
    df.loc["RMSCE", "ensemble"]
    for df in [uq_test_df, uq_bdb2020_df, uq_egfr_df, uq_mpro_df, uq_peptide_holdout_df]
]
rmsce_mc = [
    df.loc["RMSCE", "mcdropout"]
    for df in [uq_test_df, uq_bdb2020_df, uq_egfr_df, uq_mpro_df, uq_peptide_holdout_df]
]
rmsce_la = [
    df.loc["RMSCE", "la"]
    for df in [uq_test_df, uq_bdb2020_df, uq_egfr_df, uq_mpro_df, uq_peptide_holdout_df]
]
rmsce_bayes = [
    df.loc["RMSCE", "bayes_by_bayesian"]
    for df in [uq_test_df, uq_bdb2020_df, uq_egfr_df, uq_mpro_df, uq_peptide_holdout_df]
]
rmsce_enn = [
    df.loc["RMSCE", "enn"]
    for df in [uq_test_df, uq_bdb2020_df, uq_egfr_df, uq_mpro_df, uq_peptide_holdout_df]
]

# Miscalibration Area for each UQ methods on all test datasets.

mca_ens = [
    df.loc["MCA", "ensemble"]
    for df in [uq_test_df, uq_bdb2020_df, uq_egfr_df, uq_mpro_df, uq_peptide_holdout_df]
]
mca_mc = [
    df.loc["MCA", "mcdropout"]
    for df in [uq_test_df, uq_bdb2020_df, uq_egfr_df, uq_mpro_df, uq_peptide_holdout_df]
]
mca_la = [
    df.loc["MCA", "la"]
    for df in [uq_test_df, uq_bdb2020_df, uq_egfr_df, uq_mpro_df, uq_peptide_holdout_df]
]
mca_bayes = [
    df.loc["MCA", "bayes_by_bayesian"]
    for df in [uq_test_df, uq_bdb2020_df, uq_egfr_df, uq_mpro_df, uq_peptide_holdout_df]
]
mca_enn = [
    df.loc["MCA", "enn"]
    for df in [uq_test_df, uq_bdb2020_df, uq_egfr_df, uq_mpro_df, uq_peptide_holdout_df]
]

# Negative Log-Likelihood for each UQ methods on all test datasets.

nll_ens = [
    df.loc["NLL", "ensemble"]
    for df in [uq_test_df, uq_bdb2020_df, uq_egfr_df, uq_mpro_df, uq_peptide_holdout_df]
]
nll_mc = [
    df.loc["NLL", "mcdropout"]
    for df in [uq_test_df, uq_bdb2020_df, uq_egfr_df, uq_mpro_df, uq_peptide_holdout_df]
]
nll_la = [
    df.loc["NLL", "la"]
    for df in [uq_test_df, uq_bdb2020_df, uq_egfr_df, uq_mpro_df, uq_peptide_holdout_df]
]
nll_bayes = [
    df.loc["NLL", "bayes_by_bayesian"]
    for df in [uq_test_df, uq_bdb2020_df, uq_egfr_df, uq_mpro_df, uq_peptide_holdout_df]
]
nll_enn = [
    df.loc["NLL", "enn"]
    for df in [uq_test_df, uq_bdb2020_df, uq_egfr_df, uq_mpro_df, uq_peptide_holdout_df]
]

# Subplots.

fig, ax = plt.subplots(nrows=3, ncols=2, sharex=True)

fig.set_figheight(FIG_HEIGHT)
fig.set_figwidth(FIG_WIDTH)

# Spearman's rank correlation coefficient vs test datasets.

ax[0, 0].bar(
    x=br1,
    height=sp_ens,
    width=BARWIDTH,
    label="Deep Ensemble",
    edgecolor=EDGECOLOR1,
    facecolor=FACECOLOR1,
    linewidth=BAR_LW,
)
ax[0, 0].bar(
    x=br2,
    height=sp_mc,
    width=BARWIDTH,
    label="MC-Dropout",
    edgecolor=EDGECOLOR2,
    facecolor=FACECOLOR2,
    linewidth=BAR_LW,
)
ax[0, 0].bar(
    x=br3,
    height=sp_la,
    width=BARWIDTH,
    label="Laplace Approximation",
    edgecolor=EDGECOLOR3,
    facecolor=FACECOLOR3,
    linewidth=BAR_LW,
)
ax[0, 0].bar(
    x=br4,
    height=sp_bayes,
    width=BARWIDTH,
    label="Bayes by Backprop",
    edgecolor=EDGECOLOR4,
    facecolor=FACECOLOR4,
    linewidth=BAR_LW,
)
ax[0, 0].bar(
    x=br5,
    height=sp_enn,
    width=BARWIDTH,
    label="Evidential Neural Network",
    edgecolor=EDGECOLOR5,
    facecolor=FACECOLOR5,
    linewidth=BAR_LW,
)
ax[0, 0].grid(visible=True, ls="--", lw=GRID_LW)
ax[0, 0].set_ylabel("$R_{S}$")

# Sharpness vs test datasets.

ax[0, 1].bar(
    x=br1,
    height=sh_ens,
    width=BARWIDTH,
    edgecolor=EDGECOLOR1,
    facecolor=FACECOLOR1,
    linewidth=BAR_LW,
)
ax[0, 1].bar(
    x=br2,
    height=sh_mc,
    width=BARWIDTH,
    edgecolor=EDGECOLOR2,
    facecolor=FACECOLOR2,
    linewidth=BAR_LW,
)
ax[0, 1].bar(
    x=br3,
    height=sh_la,
    width=BARWIDTH,
    edgecolor=EDGECOLOR3,
    facecolor=FACECOLOR3,
    linewidth=BAR_LW,
)
ax[0, 1].bar(
    x=br4,
    height=sh_bayes,
    width=BARWIDTH,
    edgecolor=EDGECOLOR4,
    facecolor=FACECOLOR4,
    linewidth=BAR_LW,
)
ax[0, 1].bar(
    x=br5,
    height=sh_enn,
    width=BARWIDTH,
    edgecolor=EDGECOLOR5,
    facecolor=FACECOLOR5,
    linewidth=BAR_LW,
)
ax[0, 1].grid(visible=True, ls="--", lw=GRID_LW)
ax[0, 1].set_ylabel("Sh")

# MACE vs test datasets.

ax[1, 0].bar(
    x=br1,
    height=mace_ens,
    width=BARWIDTH,
    edgecolor=EDGECOLOR1,
    facecolor=FACECOLOR1,
    linewidth=BAR_LW,
)
ax[1, 0].bar(
    x=br2,
    height=mace_mc,
    width=BARWIDTH,
    edgecolor=EDGECOLOR2,
    facecolor=FACECOLOR2,
    linewidth=BAR_LW,
)
ax[1, 0].bar(
    x=br3,
    height=mace_la,
    width=BARWIDTH,
    edgecolor=EDGECOLOR3,
    facecolor=FACECOLOR3,
    linewidth=BAR_LW,
)
ax[1, 0].bar(
    x=br4,
    height=mace_bayes,
    width=BARWIDTH,
    edgecolor=EDGECOLOR4,
    facecolor=FACECOLOR4,
    linewidth=BAR_LW,
)
ax[1, 0].bar(
    x=br5,
    height=mace_enn,
    width=BARWIDTH,
    edgecolor=EDGECOLOR5,
    facecolor=FACECOLOR5,
    linewidth=BAR_LW,
)
ax[1, 0].grid(visible=True, ls="--", lw=GRID_LW)
ax[1, 0].set_ylabel("MACE")


# RMSCE vs test datasets.

ax[1, 1].bar(
    x=br1,
    height=rmsce_ens,
    width=BARWIDTH,
    edgecolor=EDGECOLOR1,
    facecolor=FACECOLOR1,
    linewidth=BAR_LW,
)
ax[1, 1].bar(
    x=br2,
    height=rmsce_mc,
    width=BARWIDTH,
    edgecolor=EDGECOLOR2,
    facecolor=FACECOLOR2,
    linewidth=BAR_LW,
)
ax[1, 1].bar(
    x=br3,
    height=rmsce_la,
    width=BARWIDTH,
    edgecolor=EDGECOLOR3,
    facecolor=FACECOLOR3,
    linewidth=BAR_LW,
)
ax[1, 1].bar(
    x=br4,
    height=rmsce_bayes,
    width=BARWIDTH,
    edgecolor=EDGECOLOR4,
    facecolor=FACECOLOR4,
    linewidth=BAR_LW,
)
ax[1, 1].bar(
    x=br5,
    height=rmsce_enn,
    width=BARWIDTH,
    edgecolor=EDGECOLOR5,
    facecolor=FACECOLOR5,
    linewidth=BAR_LW,
)
ax[1, 1].grid(visible=True, ls="--", lw=GRID_LW)
ax[1, 1].set_ylabel("RMSCE")

# Miscalibration Area vs test datasets.

ax[2, 0].bar(
    x=br1,
    height=mca_ens,
    width=BARWIDTH,
    edgecolor=EDGECOLOR1,
    facecolor=FACECOLOR1,
    linewidth=BAR_LW,
)
ax[2, 0].bar(
    x=br2,
    height=mca_mc,
    width=BARWIDTH,
    edgecolor=EDGECOLOR2,
    facecolor=FACECOLOR2,
    linewidth=BAR_LW,
)
ax[2, 0].bar(
    x=br3,
    height=mca_la,
    width=BARWIDTH,
    edgecolor=EDGECOLOR3,
    facecolor=FACECOLOR3,
    linewidth=BAR_LW,
)
ax[2, 0].bar(
    x=br4,
    height=mca_bayes,
    width=BARWIDTH,
    edgecolor=EDGECOLOR4,
    facecolor=FACECOLOR4,
    linewidth=BAR_LW,
)
ax[2, 0].bar(
    x=br5,
    height=mca_enn,
    width=BARWIDTH,
    edgecolor=EDGECOLOR5,
    facecolor=FACECOLOR5,
    linewidth=BAR_LW,
)
ax[2, 0].grid(visible=True, ls="--", lw=GRID_LW)
ax[2, 0].set_ylabel("MCA")

# Negative Log-Likelihood vs test datasets.

ax[2, 1].bar(
    x=br1,
    height=nll_ens,
    width=BARWIDTH,
    edgecolor=EDGECOLOR1,
    facecolor=FACECOLOR1,
    linewidth=BAR_LW,
)
ax[2, 1].bar(
    x=br2,
    height=nll_mc,
    width=BARWIDTH,
    edgecolor=EDGECOLOR2,
    facecolor=FACECOLOR2,
    linewidth=BAR_LW,
)
ax[2, 1].bar(
    x=br3,
    height=nll_la,
    width=BARWIDTH,
    edgecolor=EDGECOLOR3,
    facecolor=FACECOLOR3,
    linewidth=BAR_LW,
)
ax[2, 1].bar(
    x=br4,
    height=nll_bayes,
    width=BARWIDTH,
    edgecolor=EDGECOLOR4,
    facecolor=FACECOLOR4,
    linewidth=BAR_LW,
)
ax[2, 1].bar(
    x=br5,
    height=nll_enn,
    width=BARWIDTH,
    edgecolor=EDGECOLOR5,
    facecolor=FACECOLOR5,
    linewidth=BAR_LW,
)
ax[2, 1].grid(visible=True, ls="--", lw=GRID_LW)
ax[2, 1].set_ylabel("NLL")

fig.legend(loc=(0.21, 0.89), ncols=2)

ax[2, 0].set_xticks(
    np.arange(0.25, 5, 1),
    ["LP-Test", "BDB2020+", "EGFR", "$M^{Pro}$", "Peptide-Holdout"],
)

ax[2, 0].set_xticklabels(
    ["LP-Test", "BDB2020+", "EGFR", "$M^{Pro}$", "Peptide-Holdout"], rotation=90)

ax[2, 1].set_xticks(
    np.arange(0.25, 5, 1),
    ["LP-Test", "BDB2020+", "EGFR", "$M^{Pro}$", "Peptide-Holdout"],
)

ax[2, 1].set_xticklabels(
    ["LP-Test", "BDB2020+", "EGFR", "$M^{Pro}$", "Peptide-Holdout"], rotation=90)

plt.subplots_adjust(hspace=0.03, wspace=0.15)
plt.savefig("uq_uncalibrated_metrics_plot.png", dpi=600, bbox_inches="tight")
plt.show()
