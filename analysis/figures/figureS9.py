"""Plot miscalibration diagram."""

from matplotlib import font_manager
import uncertainty_toolbox as uct
from uncertainty_toolbox.metrics_calibration import get_proportion_lists
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Tuple, Dict

font_manager.findfont("Helvetica Light")
plt.rc("font", family="Helvetica Light")
plt.rc("font", serif="Helvetica Light", size=21)
plt.rcParams["axes.linewidth"] = 1.5
plt.rcParams["xtick.major.size"] = 8
plt.rcParams["xtick.major.width"] = 1.5
plt.rcParams["ytick.major.size"] = 8
plt.rcParams["ytick.major.width"] = 1.5
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["legend.markerscale"] = 2

plt.rcParams["mathtext.it"] = "Helvetica Light:italic"
plt.rcParams["mathtext.rm"] = "Helvetica Light"
plt.rcParams["mathtext.default"] = "regular"


def pred_var_cal(path: str, var: bool = False) -> Tuple[np.array, np.array]:
    """Calculate residual error and variance.

    Parameters
    ----------
    path : str
        Path to a csv file.
    var : bool, optional
        If variance available sets it to True, by default False
    Returns
    -------
    Tuple[np.array, np.array]
        Residual error and variance of each data instance.
    """
    if var:
        df = pd.read_csv(path, index_col=0)
        y_true = df.loc[:, "Y"].to_numpy()
        y_pred = df.iloc[:, 1].to_numpy()
        y_var = df.iloc[:, 2].to_numpy()

    else:
        df = pd.read_csv(path, index_col=0)
        y_true = df.loc[:, "Y"].to_numpy()
        y_pred = df.iloc[:, 1:].mean(axis=1).to_numpy()
        y_var = df.iloc[:, 1:].var(axis=1).to_numpy()

    return y_pred, np.sqrt(y_var), y_true


def pred_std_cal_all(dataset_name: str) -> Dict[str, Tuple]:
    """Return y_pred, y_std, and y_true for all UQ methods for
    a given dataset.

    Parameters
    ----------
    dataset_name : str
        Dataset name.

    Returns
    -------
    Dict[str, Tuple]
        Return y_pred, y_std, and y_true for all UQ methods.
    """

    path_ensemble = rf"../uncertainty_results/ffnn_ecif_ensemble_{dataset_name}.csv"
    path_mcdropout = rf"../uncertainty_results/ffnn_ecif_mcdropout_{dataset_name}.csv"
    path_la = rf"../uncertainty_results/ffnn_ecif_la_{dataset_name}.csv"
    path_bayes_by_bayesian = (
        rf"../uncertainty_results/ffnn_ecif_bayes_by_bayesian_{dataset_name}.csv"
    )
    path_enn = rf"../uncertainty_results/ffnn_ecif_enn_{dataset_name}.csv"

    results = {}

    results["ens"] = pred_var_cal(path_ensemble)
    results["mcdropout"] = pred_var_cal(path_mcdropout)
    results["la"] = pred_var_cal(path_la, var=True)
    results["bayes"] = pred_var_cal(path_bayes_by_bayesian, var=True)
    results["enn"] = pred_var_cal(path_enn, var=True)

    return results


COLOR0 = "black"  # Ideal
COLOR1 = "#8ecae6"  # Ensemble face color
COLOR2 = "#219ebc"  # MC-Dropout face color
COLOR3 = "#023047"  # Laplace Approximation face color
COLOR4 = "#FFB703"  # Bayes-by-Bayesian face color
COLOR5 = "#FB8500"  # Evidential Neural Network face color

FIG_HEIGHT = 15  # figure height
FIG_WIDTH = 15  # figure width

LW = 3  # linewidth

fig, axs = plt.subplots(2, 3, figsize=(FIG_WIDTH, FIG_HEIGHT), sharex=True, sharey=True)

axs[1, 2].axis("off")  # Turn off the bottom-right subplot

xshift = 0.025

axs[0, 0].annotate("LP-Test", xy=(0.02, 0.89 + xshift), textcoords=("axes fraction"))
axs[0, 1].annotate("BDB2020+", xy=(0.02, 0.89 + xshift), textcoords=("axes fraction"))
axs[0, 2].annotate("M$^{Pro}$", xy=(0.02, 0.89 + xshift), textcoords=("axes fraction"))
axs[1, 0].annotate("EGFR", xy=(0.02, 0.89 + xshift), textcoords=("axes fraction"))
axs[1, 1].annotate(
    "Peptide-Holdout", xy=(0.02, 0.89 + xshift), textcoords=("axes fraction")
)

for name, ax in [
    ("LP-Test", axs[0, 0]),
    ("BDB2020+", axs[0, 1]),
    ("Mpro", axs[0, 2]),
    ("EGFR", axs[1, 0]),
    ("Peptide-Holdout", axs[1, 1]),
]:
    results = pred_std_cal_all(name)

    val_results = pred_std_cal_all("LP-Val")

    exp_props, obs_props = uct.metrics_calibration.get_proportion_lists_vectorized(*val_results["ens"])
    recal_model = uct.recalibration.iso_recal(exp_props, obs_props)

    exp_proportions, obs_proportions = get_proportion_lists(*results["ens"], recal_model=recal_model)
    ax.plot(exp_proportions, obs_proportions, label="Deep Ensemble", color=COLOR1, lw=LW)

    exp_props, obs_props = uct.metrics_calibration.get_proportion_lists_vectorized(*val_results["mcdropout"])
    recal_model = uct.recalibration.iso_recal(exp_props, obs_props)

    exp_proportions, obs_proportions = get_proportion_lists(*results["mcdropout"], recal_model=recal_model)
    ax.plot(exp_proportions, obs_proportions, label="MC-Dropout", color=COLOR2, lw=LW)

    exp_props, obs_props = uct.metrics_calibration.get_proportion_lists_vectorized(*val_results["la"])
    recal_model = uct.recalibration.iso_recal(exp_props, obs_props)

    exp_proportions, obs_proportions = get_proportion_lists(*results["la"], recal_model=recal_model)
    ax.plot(exp_proportions, obs_proportions, label="Laplace Approximation", color=COLOR3, lw=LW)

    exp_props, obs_props = uct.metrics_calibration.get_proportion_lists_vectorized(*val_results["bayes"])
    recal_model = uct.recalibration.iso_recal(exp_props, obs_props)

    exp_proportions, obs_proportions = get_proportion_lists(*results["bayes"], recal_model=recal_model)
    ax.plot(exp_proportions, obs_proportions, label="Bayes by Backprop", color=COLOR4, lw=LW)

    exp_props, obs_props = uct.metrics_calibration.get_proportion_lists_vectorized(*val_results["enn"])
    recal_model = uct.recalibration.iso_recal(exp_props, obs_props)

    exp_proportions, obs_proportions = get_proportion_lists(*results["enn"], recal_model=recal_model)
    ax.plot(exp_proportions, obs_proportions, label="Evidential Neural Network", color=COLOR5, lw=LW)

    ax.plot([0, 1], [0, 1], "--", label="Ideal", color=COLOR0)
    ax.axis("square")

    ax.set_xlim([0 - 0.015, 1 + 0.015])
    ax.set_ylim([0 - 0.015, 1 + 0.015])

    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(["0", "0.5", "1"])
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels(["0", "0.5", "1"])

handles, labels = axs[0, 0].get_legend_handles_labels()
leg = fig.legend(
    handles,
    labels,
    loc="upper right",
    bbox_to_anchor=(0.915, 0.45),
    handlelength=1.8,
    fontsize=19,
    frameon=False,
)

for line in leg.get_lines():
    line.set_linewidth(4.0)

fig.supylabel("Observed Proportion in Interval", x=0.066, fontsize=21)
fig.supxlabel("Predicted Proportion in Interval", x=0.51, y=0.189, fontsize=21)
fig.delaxes(axs[1, 2])

plt.subplots_adjust(wspace=0.039, hspace=-0.48)
plt.savefig("recalibrated_miscalibration_plot.png", dpi=300, bbox_inches="tight", pad_inches=0.03)

plt.show()
