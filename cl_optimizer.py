from typing import Callable

import numpy as np
import pandas as pd


def geometric_temp_schedule(init_temp: int, end_temp: int, num_iterations: int) -> np.ndarray:
    return np.geomspace(init_temp, end_temp, num=num_iterations)


def bounds_to_neighborhood_function(bounds: list[tuple[int, int]]) -> Callable[[list[int]], list[int]]:
    def neighborhood_function(x: list[int]) -> list[int]:
        for i in range(1000):
            index = np.random.choice(range(len(x)), size=1)[0]
            up_or_down = np.random.choice([-1, 1], size=1)[0]
            if bounds[index][0] <= x[index] + up_or_down < bounds[index][1]: # should lower bound be exclusive as well?
                x[index] += up_or_down
                return x

        raise Exception(f"No valid neighbor found for {x} with bounds {bounds} after {i} tries.")

    return neighborhood_function


def simulated_annealing(num_bits: int, neighborhood_fun: Callable[[list[int]], list[int]], lookup_table: pd.DataFrame, temp_schedule: np.ndarray) -> tuple[list[int], float]:
    x = np.random.choice([0, 1], size=num_bits)
    loss = lookup_table.loc[lookup_table['candidate'] == str(x), 'loss'].iloc[0]

    for temp in temp_schedule:
        x_new = neighborhood_fun(x)

        entry = lookup_table.loc[lookup_table['candidate'] == str(x_new)]
        loss_new = entry.loc[:, 'loss'].iloc[0]
        net_power_io_diff = entry.loc[:, 'abs. diff. net power IO'].iloc[0]

        print(f'Candidate: {x_new}; Loss: {round(loss_new, 5)}; abs. diff. net power IO: {round(net_power_io_diff, 5)} --- Best Candidate {x_new}; Best loss: {loss_new}')

        if loss_new < loss or np.random.rand() > np.exp(- (loss_new - loss)/temp):
            x = x_new
            loss = loss_new

    return x, loss


if __name__ == "__main__":
    lookup_table = pd.read_csv('data/brute_force/candidate_space_256_instances_2026-03-14_15-30-37.csv')
    temp_schedule = geometric_temp_schedule(100, 1, 80)
    neighborhood_function = bounds_to_neighborhood_function(bounds=[(0, 4)] * 4)

    best_candidate, loss = simulated_annealing(4, neighborhood_function, lookup_table, temp_schedule)
    print("Best candidate:", best_candidate)
    print("Loss:", loss)