import numpy as np
import pandas as pd
import pickle
from qaoa_pipeline import QAOA_circuit
def raar(expected_loss: float, loss_space: np.ndarray | pd.Series) -> float:
    mean = loss_space.mean()
    minimum = loss_space.min()
    return (mean - expected_loss) / (mean - minimum)

def tts_p(p_star, l_p: np.ndarray = None) -> np.ndarray:
    l_p = l_p if l_p is not None else np.ones_like(p_star)

    assert p_star.shape == l_p.shape

    with np.errstate(divide='ignore', invalid='ignore'): # suppress divide by zero error - np.where filters this
        return np.where(p_star > 0, l_p * np.ceil(np.log(0.01) / (np.log(1 - p_star))), np.inf)

def min_tts(p_star: np.ndarray, l_p: np.ndarray = None, return_tts_p = False, axis=-1):
    tts_p_ = tts_p(p_star, l_p)

    p_min_0_based = tts_p_.argmin(axis=axis)
    indices_expanded = np.expand_dims(p_min_0_based, axis=-1)
    min_values = np.take_along_axis(tts_p_, indices_expanded, axis=-1).squeeze(axis=-1)

    if return_tts_p:
        return p_min_0_based + 1, min_values, tts_p_

    return min_values

def raar_qaoa(bf_file, params_file):
    df = pd.read_csv(bf_file)
    costs = df['loss'].astype(float).values
    # Min-max normalization
    costs = (costs - np.min(costs)) / (np.max(costs) - np.min(costs))
    mean = np.mean(costs)
    min = np.min(costs)
    # Take expectation value w.r.t. trial state psi(gammas, betas)
    with open(params_file, 'rb') as f:
        pkl = pickle.load(f)
    gammas = pkl['gammas']
    betas = pkl['betas']
    n = int(np.log2(pkl['num_generators']))
    p = len(gammas) # can generally be larger than p0 due to Iterative Interpolation
    qaoa_circuit = QAOA_circuit(n=n, p=p, costs=costs)
    trial_energy = qaoa_circuit.cost_function((gammas, betas))
    # return RAAR 
    return (mean - trial_energy)/(mean - min)

def ar_qaoa(bf_file, params_file):
    df = pd.read_csv(bf_file)
    costs = df['loss'].astype(float).values
    # Min-max normalization
    costs = (costs - np.min(costs)) / (np.max(costs) - np.min(costs))
    min = np.min(costs)
    max = np.max(costs)
    with open(params_file, 'rb') as f:
        pkl = pickle.load(f)
    gammas = pkl['gammas']
    betas = pkl['betas']
    n = int(np.log2(pkl['num_generators']))
    p = len(gammas) # can generally be larger than p0 due to Iterative Interpolation
    qaoa_circuit = QAOA_circuit(n=n, p=p, costs=costs)
    trial_energy = qaoa_circuit.cost_function((gammas, betas))
    
    return 1 - (trial_energy - min) / (max - min)

def tts_qaoa(bf_file, params_file):
    df = pd.read_csv(bf_file)
    costs = df['loss'].astype(float).values
    # Min-max normalization
    costs = (costs - np.min(costs)) / (np.max(costs) - np.min(costs))
    min = np.min(costs)
    optimal_indices = np.where(np.isclose(costs, min))[0]

    with open(params_file, 'rb') as f:
        pkl = pickle.load(f)
    gammas = pkl['gammas']
    betas = pkl['betas']
    n = int(np.log2(pkl['num_generators']))
    p = len(gammas) # can generally be larger than p0 due to Iterative Interpolation
    qaoa_circuit = QAOA_circuit(n=n, p=p, costs=costs)
    
    probs = qaoa_circuit.distribution((gammas, betas))
    p_star = np.sum(probs[optimal_indices])

    return np.where(
        p_star > 0,
        p * np.ceil(np.log(0.01) / np.log(1 - p_star)),
        np.inf
        )
