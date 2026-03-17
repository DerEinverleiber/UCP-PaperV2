from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd


def geometric_temp_schedule(init_temp: int, end_temp: int, num_steps: int) -> np.ndarray:
    return np.geomspace(init_temp, end_temp, num=num_steps)


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


def simulated_annealing(
        num_bits: int,
        neighborhood_fun: Callable[[list[int]], list[int]],
        lookup_table: pd.DataFrame,
        temp_schedule: np.ndarray,
        temp_iterations: int,
        return_history: bool = False,
) -> tuple[list[int], float] | tuple[list[int], float, list[tuple[list[int], float]]]:
    x = np.random.choice([0, 1], size=num_bits)
    loss = lookup_table.loc[lookup_table['candidate'] == str(x), 'loss'].iloc[0]
    best_x = x
    best_loss = loss

    history = [(x, loss)]

    for temp in temp_schedule:
        for _ in range(temp_iterations):
            x_new = neighborhood_fun(x)

            entry = lookup_table.loc[lookup_table['candidate'] == str(x_new)]
            loss_new = entry.loc[:, 'loss'].iloc[0]
            net_power_io_diff = entry.loc[:, 'abs. diff. net power IO'].iloc[0]

            if loss_new < loss or np.random.rand() < np.exp(-(loss_new - loss) / temp):
                x, loss = x_new, loss_new
                if loss_new < best_loss:
                    best_x, best_loss = x_new, loss_new
            
            history.append((x_new.copy(), loss_new))
            print(f'Candidate: {x_new}; Loss: {round(loss_new, 5)}; abs. diff. net power IO: {round(net_power_io_diff, 5)} --- Best Candidate {best_x}; Best loss: {best_loss}')


    if return_history:
        return best_x, best_loss, history
    else:
        return best_x, best_loss


if __name__ == "__main__":
    init_temp = 100
    end_temp = 1
    num_steps = 100
    temp_iterations = 80

    lookup_table = pd.read_csv('data/brute_force/candidate_space_256_instances_2026-03-14_15-30-37.csv')
    temp_schedule = geometric_temp_schedule(init_temp, end_temp, num_steps)
    neighborhood_function = bounds_to_neighborhood_function(bounds=[(0, 4)] * 4)

    best_candidate, loss, history = simulated_annealing(
        4,
        neighborhood_function,
        lookup_table,
        temp_schedule,
        temp_iterations,
        return_history=True,
    )
    pd.DataFrame(history, columns=['candidate', 'loss']).to_csv(
        f'data/simulated_annealing/path_init_{init_temp}_end_{end_temp}_steps_{num_steps}_iter_{temp_iterations}'
        f'__256_instances_2026-03-14_15-30-37.csv',
    )
    print("Best candidate:", best_candidate)
    print("Loss:", loss)