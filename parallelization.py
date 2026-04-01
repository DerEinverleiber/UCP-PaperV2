import itertools

import numpy as np

from cl_optimizer import SimulatedAnnealing


def construct_parallelization_arg_list(optimizers: list[SimulatedAnnealing], depth: int, step: int = 1, batch_size: int = 10) -> list[tuple[tuple[int, SimulatedAnnealing], np.ndarray]]:
    """
    :param optimizers: a set of optimizers (corresponding to different problems)
    :param depth: optimization depth (num. temperature iterations for Simulated Annealing)
    :param step: step size taken towards depth (default: 1)
    :param batch_size: size of batch that should be dispatched to a single thread (default: 10)
    :return: arg_list: list of arguments to be used for parallel execution of optimization tasks
    """
    assert step * batch_size <= depth, "step * batch_size must be smaller than depth (otherwise arg_list will be empty)"
    num_batches = depth / (batch_size * step)

    iterations = np.arange(1, depth + 1, step=step)
    iterations_reverse = iterations[::-1]

    stacked = np.array_split(np.vstack((iterations, iterations_reverse)),  2 * num_batches, axis=1)
    temp_iteration_batches = [a.flatten() for a in stacked][:len(stacked)//2] # first half contains all iterations

    return list(itertools.product(list(enumerate(optimizers)), temp_iteration_batches))


def extract_sorted_losses_and_temps(results: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    :param results: np.ndarray of shape (n + 1,) containing:
        column 0: id (corresponding to an optimizer process/dataset)
        columns [1, n/2]: temperature iterations
        columns [n/2 + 1, n]: losses corresponding to temperature iterations
    :return:
        sorted_temps: np.ndarray of shape (n/2,) containing: sorted temperature iterations (asc)
        sorted_losses: np.ndarray of shape (n/2,) containing: losses sorted by temperature iterations (asc)
    """
    group_values = results[:, 0]
    unique_values, inverse_indices = np.unique(group_values, return_inverse=True)
    grouped_arrays = [results[inverse_indices == i][:, 1:] for i in range(len(unique_values))]

    temperatures_and_losses = [tuple(b.flatten() for b in np.split(a, 2, axis=1)) for a in grouped_arrays]
    sort_indices = [np.array(np.argsort(temps), dtype=int) for temps, _ in temperatures_and_losses]
    sorted_losses = np.array([losses[sort_indices[i]] for i, (_, losses) in enumerate(temperatures_and_losses)])
    sorted_temps = np.array([temps[sort_indices[i]] for i, (temps, _) in enumerate(temperatures_and_losses)])

    return sorted_temps, sorted_losses