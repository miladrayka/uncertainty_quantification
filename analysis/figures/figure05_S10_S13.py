"""Plot redicted Values and Intervals vs Index (Ordered by Observed Value) diagram."""

from typing import Tuple, Dict, List, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager


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

COLOR0 = "black"  # Ideal
COLOR1 = "#8ecae6"  # Ensemble face color
COLOR2 = "#219ebc"  # MC-Dropout face color
COLOR3 = "#023047"  # Laplace Approximation face color
COLOR4 = "#FFB703"  # Bayes-by-Bayesian face color
COLOR5 = "#FB8500"  # Evidential Neural Network face color

COLORS = [COLOR1, COLOR2, COLOR3, COLOR4, COLOR5]

FIG_HEIGHT = 15  # figure height
FIG_WIDTH = 15  # figure width

GRID_LW = 0.5  # grid linewidth
LW = 2  # linewidth

SUBSET = 100  # number of data to display


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


def filter_subset(input_list: List[List[Any]], n_subset: int) -> List[List[Any]]:
    """Keep only n_subset random indices from all lists given in input_list.

    Args:
        input_list: list of lists.
        n_subset: Number of points to plot after filtering.

    Returns:
        List of all input lists with sizes reduced to n_subset.
    """
    assert type(n_subset) is int
    n_total = len(input_list[0])
    idx = np.random.choice(range(n_total), n_subset, replace=False)
    idx = np.sort(idx)
    output_list = []
    for inp in input_list:
        outp = inp[idx]
        output_list.append(outp)
    return output_list


for num, uq in enumerate(["ens", "mcdropout", "la", "bayes", "enn"]):

    fig, axs = plt.subplots(3, 2, figsize=(FIG_WIDTH, FIG_HEIGHT))

    xshift = 0.0
    axs[0, 0].annotate(
        "LP-Test", xy=(0.02, 0.89 + xshift), textcoords=("axes fraction")
    )
    axs[0, 1].annotate(
        "BDB2020+", xy=(0.02, 0.89 + xshift), textcoords=("axes fraction")
    )
    axs[1, 0].annotate("M$^{Pro}$", xy=(0.02, 11.4))
    axs[1, 1].annotate("EGFR", xy=(0.02, 10.11))

    # Add the middle subplot in the last row
    ax_center = fig.add_axes([0.33, 0.097, 0.36, 0.24])
    ax_center.annotate(
        "Peptide-Holdout", xy=(0.02, 0.89 + xshift), textcoords=("axes fraction")
    )

    for name, ax in [
        ("LP-Test", axs[0, 0]),
        ("BDB2020+", axs[0, 1]),
        ("Mpro", axs[1, 0]),
        ("EGFR", axs[1, 1]),
        ("Peptide-Holdout", ax_center),
    ]:
        if name not in ["EGFR", "Mpro"]:
            results = pred_std_cal_all(name)
            y_pred, y_std, y_true = filter_subset([*results[uq]], SUBSET)
        else:
            results = pred_std_cal_all(name)
            y_pred, y_std, y_true = results[uq]

        order = np.argsort(y_true.flatten())
        y_pred, y_std, y_true = y_pred[order], y_std[order], y_true[order]
        xs = np.arange(len(order))
        intervals = y_std

        ax.errorbar(
            xs,
            y_pred,
            intervals,
            fmt="o",
            ls="none",
            linewidth=1.5,
            c=COLORS[num],
            alpha=0.5,
        )
        ax.plot(xs, y_pred, "o", color=COLORS[num], lw=LW, label="Predicted Values")
        ax.plot(xs, y_true, "--", color=COLOR0, lw=LW, label="Observed Values")
        ax.grid(visible=True, ls="--", lw=GRID_LW)

    num += 1

    fig.supylabel("Predicted Values and Intervals", x=0.07, fontsize=21)
    fig.supxlabel("Index (Ordered by Observed Value)", x=0.51, y=0.05, fontsize=21)
    fig.delaxes(axs[2, 0])
    fig.delaxes(axs[2, 1])
    plt.legend(loc=(1.05, 0.785), fontsize=18)
    plt.subplots_adjust(wspace=0.14, hspace=0.13)
    plt.savefig(f"{uq}_interval_plot.png", dpi=600, bbox_inches="tight")

    plt.show()
