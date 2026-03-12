import itertools

import numpy as np

from powergrid import PowerGrid


def brute_force(grid: PowerGrid, bounds: list[tuple[int, int]], c: list[int]) -> tuple[list[int], float, float]:
    assert all([low < high for low, high in bounds]), \
        f"Are you sure bounds are correctly set? Some lower bounds seem to be greater/equal to the higher bounds."

    best_candidate = None
    best_loss = np.inf
    best_net_power_io_diff = None

    ranges = [range(low, high) for low, high in bounds]

    for x in itertools.product(*ranges):
        loss, net_power_io_diff = grid.loss_function(list(x), c, return_net_power_io_diff=True)
        if loss < best_loss:
            best_loss = loss
            best_candidate = x
            best_net_power_io_diff = net_power_io_diff
        print(f'Candidate: {x}; Loss: {round(loss, 5)}; abs. diff. net power IO: {round(net_power_io_diff, 5)} --- Best Candidate {best_candidate}; Best loss: {best_loss}')

    return best_candidate, best_loss, best_net_power_io_diff


if __name__ == '__main__':
    grid = PowerGrid.ieee57()
    generators = grid.get_generator_indices()
    print("Generators indices:", generators)
    best_candidate, best_loss, best_dis = brute_force(grid, bounds=[(0, 4)] * len(generators), c=np.random.uniform(0, 1, size=80))

    print("Best candidate:", best_candidate)
    print("Best loss:", best_loss)
    print("Best discrepancy", best_dis)

