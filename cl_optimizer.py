from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

class SimulatedAnnealing:
    
    def __init__(self, lookup_table: pd.DataFrame | str, bounds: list[tuple[int, int]] | int, temp_schedule: np.ndarray = None) -> None:
        if isinstance(lookup_table, str):
            try:
                self.lookup_table = pd.read_csv(lookup_table)
            except FileNotFoundError:
                print(f"Error: File not found at {lookup_table}")
                raise
            except Exception as e:
                print(f"Error reading CSV: {e}")
                raise
        else:
            self.lookup_table = lookup_table

        self.lookup_table = self.lookup_table.set_index('candidate') # set index for O(1) lookup

        if type(bounds) is int:
            self.bounds = [(0, 1)] * bounds
        else:
            self.bounds = bounds
        self.temp_schedule = temp_schedule
        self.history: list[tuple[list[int], float]] = []

    @classmethod
    def geometric_temp_schedule(cls, init_temp: int, end_temp: int, temp_iterations: int) -> np.ndarray:
        return np.geomspace(init_temp, end_temp, num=temp_iterations)

    def bounds_to_neighborhood_function(self) -> Callable[[list[int]], list[int]]:
        def neighborhood_function(x: list[int]) -> list[int]:
            for i in range(1000):
                index = np.random.randint(0, len(x))
                up_or_down = np.random.choice([-1, 1], size=1)[0]
                if self.bounds[index][0] <= x[index] + up_or_down <= self.bounds[index][1]: # should lower bound be exclusive as well?
                    x[index] += up_or_down
                    return x
    
            raise Exception(f"No valid neighbor found for {x} with self.bounds {self.bounds} after {i} tries.")
    
        return neighborhood_function


    def optimize(
            self,
            temp_schedule: np.ndarray = None,
            neighborhood_fun: Callable[[list[int]], list[int]] = None,
            verbose: bool = True,
    ) -> tuple[list[int], float]:
        if temp_schedule is None:
            if self.temp_schedule is None:
                raise Exception("Temp schedule must be provided either in constructor or when calling optimize.")
            else:
                temp_schedule = self.temp_schedule
        neighborhood_fun = neighborhood_fun or self.bounds_to_neighborhood_function()

        x = np.array([np.random.choice(np.arange(low, high + 1), size=1)[0] for low, high in self.bounds]) # check again
        loss = self.lookup_table.loc[str(x), 'loss']
        best_x = x
        best_loss = loss
    
        self.history = [(x, loss)]
    
        for temp in temp_schedule:
            x_new = neighborhood_fun(x)

            entry = self.lookup_table.loc[str(x_new)]
            loss_new = entry['loss']
            net_power_io_diff = entry['abs. diff. net power IO']

            if loss_new < loss or np.random.rand() < np.exp(-(loss_new - loss) / temp):
                x, loss = x_new, loss_new
                if loss_new < best_loss:
                    best_x, best_loss = x_new, loss_new

            self.history.append((x_new.copy(), loss_new))
            if verbose:
                print(f'Candidate: {x_new}; Loss: {round(loss_new, 5)}; abs. diff. net power IO: {round(net_power_io_diff, 5)} --- Best Candidate {best_x}; Best loss: {best_loss}')
    
        return best_x, best_loss