"""Plot all metrics."""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import font_manager

FIG_HEIGHT = 10  # figure height
FIG_WIDTH = 20  # figure width

SIZE = 15  # radius of the point.
LW = 0.99  # witdth of the point.

COLORS = ["#fd5901", "#f78104", "#faab36", "#249ea0", "#008083", "#005f60"]

font_manager.findfont("Helvetica Light")
plt.rc("font", family="Helvetica Light")
plt.rc("font", serif="Helvetica Light", size=24)
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

# Load performance metrics results.

ffnn_df = (
    pd.read_csv("../metrics_reuslts/ffnn_ecif_results.csv", index_col=0)
    .iloc[:, ::2]
    .iloc[:, [3, 2, 0, 1, 4, 5]]
)
cnn_df = (
    pd.read_csv("../metrics_reuslts/cnn_ms_oic_results.csv", index_col=0)
    .iloc[:, ::2]
    .iloc[:, [3, 2, 0, 1, 4, 5]]
)
deepdta_df = (
    pd.read_csv("../metrics_reuslts/deepdta_results.csv", index_col=0)
    .iloc[:, ::2]
    .iloc[:, [3, 2, 0, 1, 4, 5]]
)
ign_df = (
    pd.read_csv("../metrics_reuslts/ign_results.csv", index_col=0)
    .iloc[:, ::2]
    .iloc[:, [3, 2, 0, 1, 4, 5]]
)

df_list = {}
for i in ["RP", "SP", "MSE", "RMSE", "MAE", "Q95"]:
    df1 = pd.DataFrame({i: ffnn_df.loc[i, :], "Model": ["FFNN-ECIF"] * 6}).reset_index(
        names="Dataset"
    )
    df2 = pd.DataFrame({i: cnn_df.loc[i, :], "Model": ["CNN-MS-OIC"] * 6}).reset_index(
        names="Dataset"
    )
    df3 = pd.DataFrame({i: deepdta_df.loc[i, :], "Model": ["DeepDTA"] * 6}).reset_index(
        names="Dataset"
    )
    df4 = pd.DataFrame({i: ign_df.loc[i, :], "Model": ["IGN"] * 6}).reset_index(
        names="Dataset"
    )
    df5 = pd.concat([df1, df2, df3, df4])
    df5.loc[:, "Dataset"] = df5.loc[:, "Dataset"].str[:-5]
    df_list[i] = df5

fig, axes = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=False)

fig.set_figheight(FIG_HEIGHT)
fig.set_figwidth(FIG_WIDTH)

sns.stripplot(
    df_list["RP"],
    x="Model",
    y="RP",
    hue="Dataset",
    jitter=False,
    palette=COLORS,
    size=SIZE,
    edgecolor="black",
    linewidth=LW,
    ax=axes[0, 0],
    legend=False,
)

axes[0, 0].set_ylabel("R$_{P}$")

sns.stripplot(
    df_list["SP"],
    x="Model",
    y="SP",
    hue="Dataset",
    jitter=False,
    palette=COLORS,
    size=SIZE,
    edgecolor="black",
    linewidth=LW,
    ax=axes[0, 1],
    legend=False,
)

axes[0, 1].set_ylabel("R$_{S}$")
axes[0, 2].set_yticks([2.0, 4.0, 6.0])
axes[0, 2].set_yticklabels(["2.0", "4.0", "6.0"])


axes[1, 2].set_yticks([2.0, 3.0, 4.0, 5.0])
axes[1, 2].set_yticklabels(["2.0", "3.0", "4.0", "5.0"])


sns.stripplot(
    df_list["MSE"],
    x="Model",
    y="MSE",
    hue="Dataset",
    jitter=False,
    palette=COLORS,
    size=SIZE,
    edgecolor="black",
    linewidth=LW,
    ax=axes[0, 2],
    legend=False,
)

sns.stripplot(
    df_list["RMSE"],
    x="Model",
    y="RMSE",
    hue="Dataset",
    jitter=False,
    palette=COLORS,
    size=SIZE,
    edgecolor="black",
    linewidth=LW,
    ax=axes[1, 0],
    legend=False,
)

sns.stripplot(
    df_list["MAE"],
    x="Model",
    y="MAE",
    hue="Dataset",
    jitter=False,
    palette=COLORS,
    size=SIZE,
    edgecolor="black",
    linewidth=LW,
    ax=axes[1, 1],
    legend=False,
)
axes[1, 1].set_xlabel("")
axes[1, 1].set_xticklabels(["FFNN-ECIF", "CNN-MS-OIC", "DeepDTA", "IGN"], rotation=90)

sns.stripplot(
    df_list["Q95"],
    x="Model",
    y="Q95",
    hue="Dataset",
    jitter=False,
    palette=COLORS,
    size=SIZE,
    edgecolor="black",
    linewidth=LW,
    ax=axes[1, 2],
    legend=True,
)

axes[1, 2].set_ylabel("Q$_{95}$")
axes[1, 2].set_xlabel("")
axes[1, 2].set_xticklabels(["FFNN-ECIF", "CNN-MS-OIC", "DeepDTA", "IGN"], rotation=90)
axes[1, 0].set_xlabel("")

axes[1, 0].set_xticklabels(["FFNN-ECIF", "CNN-MS-OIC", "DeepDTA", "IGN"], rotation=90)

plt.legend(
    ncols=6,
    bbox_to_anchor=(1.035, 2.25, 0, 0),
    handletextpad=0.18,
    columnspacing=1.93,
    frameon=True,
)

plt.subplots_adjust(hspace=0.03, wspace=0.25)
plt.savefig("all_metric_plot.png", dpi=600, bbox_inches="tight")
plt.show()
