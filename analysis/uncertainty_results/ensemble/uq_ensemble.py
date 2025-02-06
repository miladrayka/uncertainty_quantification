"""Uncertainty quantification metrics for Deep Ensemble method."""

from typing import Dict, Any

import numpy as np
import pandas as pd
from scipy import stats
import uncertainty_toolbox as uct


def uncertainty_quantification(
    csv_file: str, csv_file_calibration: str
) -> Dict[str, Any]:

    df = pd.read_csv(csv_file)

    ensemble_means = df.iloc[:, 2:].mean(axis=1).to_numpy()
    ensemble_stds = df.iloc[:, 2:].std(axis=1).to_numpy()
    ensemble_vars = df.iloc[:, 2:].var(axis=1).to_numpy()
    ground_truth = df.iloc[:, 1].to_numpy()
    residuals = ground_truth - ensemble_means
    abs_residuals = np.abs(residuals)

    df_calib = pd.read_csv(csv_file_calibration)

    calib_ensemble_means = df_calib.iloc[:, 2:].mean(axis=1).to_numpy()
    calib_ensemble_stds = df_calib.iloc[:, 2:].std(axis=1).to_numpy()
    calib_ground_truth = df_calib.iloc[:, 1].to_numpy()

    exp_props, obs_props = uct.metrics_calibration.get_proportion_lists_vectorized(
        calib_ensemble_means,
        calib_ensemble_stds,
        calib_ground_truth,
    )

    recal_model = uct.recalibration.iso_recal(exp_props, obs_props)

    sp = stats.spearmanr(abs_residuals, ensemble_vars)[0]
    sharpness = uct.metrics.sharpness(ensemble_stds)
    nll = uct.metrics.nll_gaussian(ensemble_means, ensemble_stds, ground_truth)

    mace = uct.metrics.mean_absolute_calibration_error(
        ensemble_means, ensemble_stds, ground_truth
    )
    rmsce = uct.metrics.root_mean_squared_calibration_error(
        ensemble_means, ensemble_stds, ground_truth
    )
    mis_area = uct.metrics.miscalibration_area(
        ensemble_means, ensemble_stds, ground_truth
    )

    calib_mace = uct.metrics.mean_absolute_calibration_error(
        ensemble_means, ensemble_stds, ground_truth, recal_model=recal_model
    )
    calib_rmsce = uct.metrics.root_mean_squared_calibration_error(
        ensemble_means, ensemble_stds, ground_truth, recal_model=recal_model
    )
    calib_mis_area = uct.metrics.miscalibration_area(
        ensemble_means, ensemble_stds, ground_truth, recal_model=recal_model
    )

    results = {
        "Spearman": sp,
        "Sharpness": sharpness,
        "MACE": mace,
        "CMACE": calib_mace,
        "RMSCE": rmsce,
        "CRMSCE": calib_rmsce,
        "MCA": mis_area,
        "CMCA": calib_mis_area,
        "NLL": nll,
    }

    return results


uq_dict = {}

for item in ["LP-Test", "BDB2020+", "EGFR", "M$^{pro}$", "Peptide-Holdout"]:
    uq = uncertainty_quantification(f"ffnn_ecif_ensemble_{item}.csv", f"ffnn_ecif_ensemble_LP-Val.csv")
    uq_dict[item] = uq

pd.DataFrame(uq_dict).round(3).to_csv("uq_metrics_ensemble.csv", index=True)
