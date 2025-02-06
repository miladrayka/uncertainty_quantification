"""Return normality, metric, and p-value results."""

import pathlib
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, shapiro, skew, mannwhitneyu

from utils import cal_metrics


def metrics_and_normality(path1: str, path2: str, name: str) -> None:
    """Calculate metrics and normality test values for each model.

    Parameters
    ----------
    path1 : str
        Path to a folder that contains csv files with label and prediction values.
    path2 : str
        Path to a folder that contains txt files with pdbids.
    name : str
        Name to save results.
    """
    path_fv = pathlib.Path(path1)
    path_pdbid = pathlib.Path(path2)

    results_dict = defaultdict(list)
    normality_dict = defaultdict(list)
    dataset_names = []

    for path in path_fv.glob("*.csv"):
        dataset_name = str(path).split("_")[-1][:-4]
        dataset_names.append(dataset_name)
        df = pd.read_csv(path, index_col="PDBID")

        for item in path_pdbid.glob("*.txt"):
            if dataset_name in str(item):
                with open(str(item), "r") as file:
                    pdbids = file.readlines()
                pdbids = [i.rstrip() for i in pdbids]
                df = df.loc[pdbids, :]

                metrics_results = []

                shapiro_wilk_w, p_value = shapiro(df["Y"])
                skewness_value = skew(df["Y"])
                kurtosis_value = kurtosis(df["Y"])
                normality_dict[f"Label-ShapiroW"].append(shapiro_wilk_w)
                normality_dict[f"Label-ShapiroP"].append(p_value)
                normality_dict[f"Label-Skeweness"].append(skewness_value)
                normality_dict[f"Label-Kurtosis"].append(kurtosis_value)

                residual_error_shapiros = []
                residual_error_p = []
                residual_error_skewnesses = []
                residual_error_kurtosises = []

                pred_error_shapiros = []
                pred_error_p = []
                pred_error_skewnesses = []
                pred_error_kurtosises = []

                for i in range(1, 6):
                    result = cal_metrics(df["Y"], df[f"Y_{i}"])
                    metrics_results.append(result)

                    residual_error = df["Y"] - df[f"Y_{i}"]
                    shapiro_wilk_w, p_value = shapiro(residual_error)
                    skewness_value = skew(residual_error)
                    kurtosis_value = kurtosis(residual_error)

                    residual_error_shapiros.append(shapiro_wilk_w)
                    residual_error_p.append(p_value)
                    residual_error_skewnesses.append(skewness_value)
                    residual_error_kurtosises.append(kurtosis_value)

                    shapiro_wilk_w, p_value = shapiro(df[f"Y_{i}"])
                    skewness_value = skew(df[f"Y_{i}"])
                    kurtosis_value = kurtosis(df[f"Y_{i}"])

                    pred_error_shapiros.append(shapiro_wilk_w)
                    pred_error_p.append(p_value)
                    pred_error_skewnesses.append(skewness_value)
                    pred_error_kurtosises.append(kurtosis_value)

                normality_dict[f"Pred-ShapiroW"].append(np.mean(pred_error_shapiros))
                normality_dict[f"Pred-ShapiroP"].append(np.mean(pred_error_p))
                normality_dict[f"Pred-Skeweness"].append(np.mean(pred_error_skewnesses))
                normality_dict[f"Pred-Kurtosis"].append(np.mean(pred_error_kurtosises))

                normality_dict[f"Residual-ShapiroW"].append(
                    np.mean(residual_error_shapiros)
                )
                normality_dict[f"Residual-ShapiroP"].append(np.mean(residual_error_p))
                normality_dict[f"Residual-Skeweness"].append(
                    np.mean(residual_error_skewnesses)
                )
                normality_dict[f"Residual-Kurtosis"].append(
                    np.mean(residual_error_kurtosises)
                )

                results_dict[f"{dataset_name}-Mean"] = np.mean(metrics_results, axis=0)
                results_dict[f"{dataset_name}-STD"] = np.std(metrics_results, axis=0)

    pd.DataFrame(
        index=["RP", "SP", "MSE", "RMSE", "MAE", "Q95"], data=results_dict
    ).to_csv(f"{name}_results.csv", index=True)
    pd.DataFrame(index=dataset_names, data=normality_dict).to_csv(
        f"{name}_normality_test.csv", index=True
    )


def pred_mean(path1: str, path2: str) -> np.array:
    """_summary_

    Parameters
    ----------
    path1 : str
        Path to a csv file with label and prediction values.
    path2 : str
        Path to a txt file with pdbids.

    Returns
    -------
    np.array
        Average array of 5 predictions.
    """

    df_pred = pd.read_csv(path1, index_col="PDBID")

    with open(path2, "r") as file:
        pdbids = file.readlines()
    pdbids = [i.rstrip() for i in pdbids]

    mean_array = df_pred.loc[pdbids, :].iloc[:, 1:].mean(axis=1).to_numpy()

    return mean_array


metrics_and_normality(
    r"./ffnn",
    r"./pdbid",
    "ffnn_ecif",
)
metrics_and_normality(
    r"./cnn",
    r"./pdbid",
    "cnn_ms_oic",
)
metrics_and_normality(
    r"./deepdta",
    r"./pdbid",
    "deepdta",
)
metrics_and_normality(
    r"./ign",
    r"./pdbid",
    "ign",
)

path_pred_ffnn = r"./ffnn/ffnn_ecif_LP-Val.csv"
path_pred_cnn = r"./cnn/cnn_ms_oic_LP-Val.csv"
path_pred_deepdta = r"./deepdta/deepdta_LP-Val.csv"
path_pred_ign = r"./ign/ign_LP-Val.csv"
path_test_pdbid = r"./pdbid/LP-Val_intersection_pdbids.txt"

results = {}
for i in [path_pred_ffnn, path_pred_cnn, path_pred_deepdta, path_pred_ign]:
    mean_array = pred_mean(i, path_test_pdbid)
    name = i.split("/")[-1][:-11]
    results[f"{name}"] = mean_array

p_value_results = {}
for i in ["ffnn_ecif", "cnn_ms_oic", "deepdta", "ign"]:
    for j in ["ffnn_ecif", "cnn_ms_oic", "deepdta", "ign"]:
        _, p_value = mannwhitneyu(results[i], results[j])
        p_value_results[f"{i}_{j}"] = p_value

p_value_matrix = np.array(list(p_value_results.values())).reshape((4, 4))

df = pd.DataFrame(
    p_value_matrix,
    columns=["ffnn_ecif", "cnn_ms_oic", "deepdta", "ign"],
    index=["ffnn_ecif", "cnn_ms_oic", "deepdta", "ign"],
)
df.round(3).to_csv("p_value_metric.csv", index=True)
