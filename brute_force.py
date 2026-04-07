import csv
import itertools
from datetime import datetime

import numpy as np
from joblib import Parallel, delayed

from powergrid import PowerGrid


def brute_force(
        grid: PowerGrid, bounds: list[tuple[int, int]],
        c: list[int], write_to_file: bool | str = True,
        num_chunks: int = 20,
        start_from: tuple[int] = None,
        verbose: int = 2
) -> tuple[list[int], float, float]:
    assert all([low < high for low, high in bounds]), \
        f"Are you sure bounds are correctly set? Some lower bounds seem to be greater/equal to the higher bounds."

    best_candidate = None
    best_loss = np.inf
    best_net_power_io_diff = None

    ranges = [range(low, high + 1) for low, high in bounds]
    cartesian_product = np.array(list(itertools.product(*ranges)))
    if start_from:
        assert len(start_from) == cartesian_product.shape[1], "start_from candidate has invalid length"
        mask = np.all(cartesian_product == start_from, axis=1)
        first_match_idx = np.argmax(mask)
        cartesian_product = cartesian_product[first_match_idx:]
    chunks = np.array_split(cartesian_product, num_chunks)

    buffer = []
    file = None
    writer = None

    if write_to_file:
        file_name = write_to_file if isinstance(write_to_file, str) else \
            f"candidate_space_{len(cartesian_product)}_instances_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv"
        file = open(f"data/brute_force/{file_name}", 'w', newline='')
        writer = csv.writer(file)
        writer.writerow(['candidate', 'loss', 'abs. diff. net power IO'])

    for i, chunk in enumerate(chunks):
        for x in chunk:
            loss, net_power_io_diff = grid.loss_function(list(x), c, return_net_power_io_diff=True)
            if loss < best_loss:
                best_loss = loss
                best_candidate = x
                best_net_power_io_diff = net_power_io_diff
            buffer.append((x, loss, net_power_io_diff))
            if verbose > 1:
                print(f'Candidate: {x}; Loss: {round(loss, 5)}; abs. diff. net power IO: {round(net_power_io_diff, 5)} --- Best Candidate {best_candidate}; Best loss: {best_loss}')

        if file:
            for row in buffer:
                writer.writerow(row)
        buffer = []
        if verbose > 0:
            print(f"Chunk {i + 1}/{len(chunks)} completed.")


    return best_candidate, best_loss, best_net_power_io_diff


if __name__ == '__main__':
    seeds = list(range(1, 5))

    num_generatorss = list(range(1, 41))

    cartesian_product = list(itertools.product(num_generatorss, seeds))

    def execute_brute_force(n: int, num_generators: int, seed: int) -> None:
        np.random.seed(seed)
        grid = PowerGrid.random(n=n, num_generators=num_generators,  seed=seed)
        print("Generators indices:", num_generators)

        rng = np.random.default_rng(seed)

        file_name = f"test/candidate_space_{2**num_generators}_candidates_{n}_buses_{seed}_seed_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
        best_candidate, best_loss, best_dis = brute_force(
            grid,
            bounds=[(0, 1)] * num_generators,
            c=rng.uniform(0, 1, size=sum(grid.num_outgoing_branches())),
            write_to_file=file_name,
            verbose=1
        )

        print("Best candidate:", best_candidate)
        print("Best loss:", best_loss)
        print("Best discrepancy", best_dis)

    Parallel(n_jobs=-1, verbose=11)(delayed(execute_brute_force)(n=40, num_generators=num_generators, seed=seed) for num_generators, seed in cartesian_product)

