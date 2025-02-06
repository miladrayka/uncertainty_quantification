"""This script provides some useful functions for weights initialization,
early stopping, and calculating metrics."""

from typing import List, Tuple

import torch
import torch.nn as nn
import numpy as np
from scipy import stats
from sklearn import metrics


def init_weights(m: nn.Linear, nonlinearity: str) -> None:
    """Initialize weights of a neural network.

    Parameters
    ----------
    m : nn.Linear
        A torch.nn.Linear layer.
    nonlinearity : str
        An activation function.
    """
    if isinstance(m, nn.Linear):
        if nonlinearity in ["relu", "elu"]:
            torch.nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            m.bias.data.fill_(0.01)
        if nonlinearity == "leakyrelu":
            torch.nn.init.kaiming_normal_(
                m.weight, mode="fan_in", nonlinearity="leaky_relu", a=0.01
            )
            m.bias.data.fill_(0.01)


class EarlyStopper:
    def __init__(self, patience: str = 15) -> None:
        """Initiate an early stopping object.

        Parameters
        ----------
        patience : str, optional
            Number of epochs that the early stopping waits, by default 15
        """

        self.patience = patience
        self.counter = 0
        self.min_validation_mse = float("inf")

    def early_stop(self, validation_mse: float) -> bool:
        if validation_mse < self.min_validation_mse:
            self.min_validation_mse = validation_mse
            self.counter = 0
        elif validation_mse > (self.min_validation_mse):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def cal_metrics(
    y_list: List[float], y_pred_list: List[float]
) -> Tuple[float, float, float, float, float]:
    """Calculate five metrics between lists of true labels and
    predicted labels. Metrics are: Pearson correlation coeficient,
    Spearman rank correlation coeficient, mean squared error, root
    mean squared error, and mean absolute error.

    Parameters
    ----------
    y_list : list[float]
        A list of true labels.
    y_pred_list : list[float]
        A list of predicted labels.

    Returns
    -------
    tuple[float, float, float, float, float]
        Calculated metrics in the following order:
        Pearson correlation coeficient, Spearman rank correlation coeficient,
        mean squared error, root mean squared error, and mean absolute error.
    """
    pr = stats.pearsonr(y_pred_list, y_list)[0]
    sp = stats.spearmanr(y_pred_list, y_list)[0]
    mse = metrics.mean_squared_error(y_pred_list, y_list, squared=True)
    rmse = metrics.mean_squared_error(y_pred_list, y_list, squared=False)
    mae = metrics.mean_absolute_error(y_pred_list, y_list)
    q95 = np.percentile(np.abs(y_pred_list - y_list), 95)

    return pr, sp, mse, rmse, mae, q95
