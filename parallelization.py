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
    num_batches = np.ceil(depth / (1.0 * batch_size * step))

    iterations = np.arange(1, depth + 1, step=step)
    iterations_reverse = iterations[::-1]

    stacked = np.array_split(np.vstack((iterations, iterations_reverse)),  2 * num_batches, axis=1)
    temp_iteration_batches = [np.unique(a.flatten()) for a in stacked][:len(stacked)//2] # first half contains all iterations

    return list(itertools.product(list(enumerate(optimizers)), temp_iteration_batches))


def get_execute_sim_ann(init_temp: int, end_temp: int):
    """
    Prepares a function to run simulated annealing optimization in parallel.
    :param init_temp: initial temperature for geometric temperature schedule (e.g. 1000)
    :param end_temp: final temperature for geometric temperature schedule (e.g. 1)
    :return:
    """

    def execute_sim_ann(df_id: int, sim_ann: SimulatedAnnealing, temp_iterations: list[int]) -> tuple[int, np.ndarray]:
        optima = []
        for iterations in temp_iterations:
            print(f"Temp. Iterations {iterations}")
            temp_schedule = SimulatedAnnealing.geometric_temp_schedule(init_temp, end_temp, iterations)
            _, loss = sim_ann.optimize(
                temp_schedule=temp_schedule,
                verbose=False
            )
            optima.append(loss)

        return df_id, np.dstack((temp_iterations, optima))

    return execute_sim_ann