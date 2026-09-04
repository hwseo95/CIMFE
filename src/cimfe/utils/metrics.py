import numpy as np
from sklearn.metrics import r2_score

EPS = 1e-8


def MAE(pred, true):
    return np.mean(np.abs(pred - true))


def MSE(pred, true):
    return np.mean((pred - true) ** 2)


def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))


def MAPE(pred, true):
    return np.mean(np.abs((pred - true) / (true + EPS)))


def MSPE(pred, true):
    return np.mean(np.square((pred - true) / (true + EPS)))


def metric(pred, true):
    mse = MSE(pred, true)
    mae = MAE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    r2 = r2_score(true, pred)

    return mse, mae, rmse, mape, mspe, r2
