import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def tween(x1, x2, fn, n_points=50):
    if np.isscalar(x1) and np.isscalar(x2):
        xx = np.linspace(x1, x2, n_points).reshape(-1, 1)
        scaler = MinMaxScaler().fit(xx)
        linspace = (
            np.linspace(0, 1, n_points) if x1 < x2 else np.linspace(1, 0, n_points)
        )
        return scaler.inverse_transform(
            np.array([fn(x) for x in linspace]).reshape(-1, 1)
        ).ravel()
    xx_linear = np.linspace(x1, x2, n_points)
    scaler = MinMaxScaler().fit(xx_linear)
    xx_minmax = scaler.transform(xx_linear)
    # because rounding, sometimes we end up w/ numbers like 1.0000000002
    xx_minmax = np.where(xx_minmax > 1, 1, xx_minmax)
    xx_minmax_t = pd.DataFrame(xx_minmax).apply(lambda col: [fn(x) for x in col]).values
    return scaler.inverse_transform(xx_minmax_t)
