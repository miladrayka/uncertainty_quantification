"""Produce binding affinity KDE plot and return the statistics."""

from collections import defaultdict

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

font_manager.findfont("Helvetica Light")
plt.rc("font", family="Helvetica Light")
plt.rc("font", serif="Helvetica Light", size=25)
plt.rcParams["axes.linewidth"] = 1.25
plt.rcParams["xtick.major.size"] = 8
plt.rcParams["xtick.minor.size"] = 2
plt.rcParams["xtick.major.width"] = 1.25
plt.rcParams["xtick.minor.width"] = 1.25
plt.rcParams["ytick.major.size"] = 8
plt.rcParams["ytick.minor.size"] = 2
plt.rcParams["ytick.major.width"] = 1.25
plt.rcParams["ytick.minor.width"] = 1.25
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.direction"] = "in"

plt.rcParams["mathtext.it"] = "Helvetica Light:italic"
plt.rcParams["mathtext.rm"] = "Helvetica Light"
plt.rcParams["mathtext.default"] = "regular"

COLOR0 = "#577590"  # LP-Train
COLOR1 = "#D4520C"  # LP-Val
COLOR2 = "#DA7B07"  # LP-Test
COLOR3 = "#D99A08"  # BDB2020+   F9C74F
COLOR4 = "#6B9B46"  # EGFR #90BE6D
COLOR5 = "#3A9278"  # MPro  43AA8B
COLOR6 = "#ED070B"  # Peptide-Holdout F94144

ALPHA = 0.36

COLORS = [COLOR0, COLOR1, COLOR2, COLOR3, COLOR4, COLOR5, COLOR6]

FIG_HEIGHT = 15  # figure height
FIG_WIDTH = 15  # figure

GRID_LW = 0.5  # grid linewidth


def statistics_cal(
    values_list: list,
    labels_list: list,
    path_to_save: str = None,
    len_smiles: bool = False,
    len_seq: bool = False,
) -> pd.DataFrame:
    """Calculate min, max, q25, q50, q75, mean, and std of provided values.

    Parameters
    ----------
    values_list : list
        List of the provided values.
    labels_list : list
        List of the prefered labels.
    caption: str
        Title of the dataframe.
    path_to_save : str, optional
        Filename path to save the plot, by default None
    len_smiles: bool, optional
        Add 90th percentile to final dataframe, by default False
    len_sq: bool, optional
        Add 80th percentile to final dataframe, by default False

    Returns
    -------
    pd.DataFrame
        Dataframe of calculated statistics.
    """
    stat_dict = defaultdict(list)
    for item in values_list:

        if len_smiles:
            min, q25, q50, q75, q90, max = np.percentile(
                item, [0, 25, 50, 75, 90, 100])
        elif len_seq:
            min, q25, q50, q75, q80, max = np.percentile(
                item, [0, 25, 50, 75, 90, 100])
        else:
            min, q25, q50, q75, max = np.percentile(item, [0, 25, 50, 75, 100])

        mean = np.mean(item)
        std = np.std(item)
        stat_dict["MIN"].append(min)
        stat_dict["Q1"].append(q25)
        stat_dict["Median"].append(q50)
        stat_dict["Q3"].append(q75)
        if len_smiles:
            stat_dict["P90"].append(q90)
        elif len_seq:
            stat_dict["P80"].append(q80)
        else:
            pass
        stat_dict["MAX"].append(max)
        stat_dict["Mean"].append(mean)
        stat_dict["STD"].append(std)

    df = pd.DataFrame.from_dict(stat_dict, orient="index").round(3)
    df.columns = labels_list

    if path_to_save:
        df.to_csv(path_to_save, index=True)

    return df


# Paths for different dataset .csv file.

lp_train_path = "../binding_data/LP_train.csv"
lp_val_path = "../binding_data/LP_val.csv"
lp_test_path = "../binding_data/LP_test.csv"
bdb2020_path = "../binding_data/BDB2020+_test.csv"
egfr_path = "../binding_data/EGFR.csv"
mpro_path = "../binding_data/Mpro.csv"
peptide_holdout_path = "../binding_data/Peptide_holdout.csv"

path_dict = {
    "LP-Train": lp_train_path,
    "LP-Val": lp_val_path,
    "LP-Test": lp_test_path,
    "BDB2020+": bdb2020_path,
    "EGFR": egfr_path,
    "M$^{pro}$": mpro_path,
    "Peptide-Holdout": peptide_holdout_path,
}

path_pdbid_dict = {
    "LP-Train": None,
    "LP-Val": "../binding_data/LP-Val_intersection_pdbids.txt",
    "LP-Test": "../binding_data/LP-Test_intersection_pdbids.txt",
    "BDB2020+": "../binding_data/BDB2020+_intersection_pdbids.txt",
    "EGFR": "../binding_data/EGFR_intersection_pdbids.txt",
    "M$^{pro}$": "../binding_data/M$^{pro}$_intersection_pdbids.txt",
    "Peptide-Holdout": "../binding_data/Peptide-Holdout_intersection_pdbids.txt",
}

ba_values_dict = {}
for name, path in path_dict.items():
    if name != "LP-Train":
        with open(path_pdbid_dict[name], "r") as file:
            pdbids = file.readlines()
        pdbids = [i.rstrip() for i in pdbids]
        ba_values_dict[name] = (
            pd.read_csv(path_dict[name], index_col="pdbid")
            .loc[pdbids, "value"]
            .tolist()
        )
    else:
        ba_values_dict[name] = (
            pd.read_csv(path_dict[name],
                        index_col="pdbid").loc[:, "value"].tolist()
        )

ba_values_list = list(ba_values_dict.values())
ba_labels = list(ba_values_dict.keys())

# Binding affinity distribution plot.

fig, axes = plt.subplots(
    3, 2, sharex=True, sharey=True, figsize=(FIG_WIDTH, FIG_HEIGHT)
)
i = 1
for ax1 in [0, 1, 2]:
    for ax2 in [0, 1]:
        sns.kdeplot(
            ba_values_list[0],
            ax=axes[ax1, ax2],
            fill=True,
            multiple="layer",
            color=COLORS[0],
            label="LP-Train"
        )
        sns.kdeplot(
            ba_values_list[i],
            ax=axes[ax1, ax2],
            fill=True,
            multiple="layer",
            color=COLORS[i], alpha=ALPHA,
            label=f"{ba_labels[i]}"
        )
        axes[ax1, ax2].set_ylabel(ylabel="")
        axes[ax1, ax2].legend(loc=1, handlelength=1)
        axes[ax1, ax2].grid(visible=True, ls="--", lw=GRID_LW)
        axes[ax1, ax2].set_yticks([0.0, 0.2, 0.4])
        axes[ax1, ax2].set_ylim(0, 0.45)
        i += 1

fig.supylabel("Density", x=0.021, y=0.54, fontsize=25)
fig.supxlabel("$pK = -log_{10}(K_{i}/K_{d}/IC50)$",
              x=0.55, y=0.033, fontsize=25)

plt.tight_layout()
plt.subplots_adjust(hspace=0.03, wspace=0.03)
plt.savefig("binding_affinity_dist.png", bbox_inches="tight", dpi=600)

df = statistics_cal(
    ba_values_list,
    ba_labels,
    path_to_save="binding_affinity_dist.csv",
)

df.to_latex(
    "binding_affinity_dist_latex.txt",
    index=True,
    float_format="{:.3f}".format,
)
