import numpy as np
import pandas as pd

def raar(expected_loss: float, loss_space: np.ndarray | pd.Series) -> float:
    mean = loss_space.mean()
    minimum = loss_space.min()
    return (mean - expected_loss) / (mean - minimum)

def tts_p(p_star, l_p: np.ndarray = None) -> np.ndarray:
    l_p = l_p or np.ones(p_star.shape[0])

    assert p_star.shape == l_p.shape

    with np.errstate(divide='ignore', invalid='ignore'): # suppress divide by zero error - np.where filters this
        return np.where(p_star > 0, l_p * np.ceil(np.log(0.01) / (np.log(1 - p_star))), np.inf)

def min_tts(p_star: np.ndarray, l_p: np.ndarray = None, return_tts_p = False):
    tts_p_ = tts_p(p_star, l_p)

    p_min_0_based = tts_p_.argmin()

    if return_tts_p:
        return p_min_0_based + 1, tts_p_[p_min_0_based], tts_p_

    return p_min_0_based + 1, tts_p_[p_min_0_based]