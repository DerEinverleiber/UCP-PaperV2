import csv
import itertools
from datetime import datetime

import numpy
import numpy as np

from powergrid import PowerGrid


def brute_force(grid: PowerGrid, bounds: list[tuple[int, int]], c: list[int], write_to_file: bool | str = True, num_chunks: int = 20, start_from: tuple[int] = None) -> tuple[list[int], float, float]:
    assert all([low < high for low, high in bounds]), \
        f"Are you sure bounds are correctly set? Some lower bounds seem to be greater/equal to the higher bounds."

    best_candidate = None
    best_loss = np.inf
    best_net_power_io_diff = None

    ranges = [range(low, high) for low, high in bounds]
    cartesian_product = numpy.array(list(itertools.product(*ranges)))
    if start_from:
        assert len(start_from) == cartesian_product.shape[1], "start_from candidate has invalid length"
        mask = np.all(cartesian_product == start_from, axis=1)
        first_match_idx = np.argmax(mask)
        cartesian_product = cartesian_product[first_match_idx:]
    chunks = numpy.array_split(cartesian_product, num_chunks)

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
            print(f'Candidate: {x}; Loss: {round(loss, 5)}; abs. diff. net power IO: {round(net_power_io_diff, 5)} --- Best Candidate {best_candidate}; Best loss: {best_loss}')

        if file:
            for row in buffer:
                writer.writerow(row)
        print(f"Chunk {i}/{len(chunks)} completed.")


    return best_candidate, best_loss, best_net_power_io_diff


if __name__ == '__main__':
    grid = PowerGrid.ieee57()
    generators = grid.get_generator_indices()
    print("Generators indices:", generators)
    best_candidate, best_loss, best_dis = brute_force(
        grid,
        bounds=[(0, 4)] * len(generators),
        c=np.random.uniform(0, 1, size=80),
        write_to_file=True,
        start_from=(1, 1, 1, 1, 1)
    )

    print("Best candidate:", best_candidate)
    print("Best loss:", best_loss)
    print("Best discrepancy", best_dis)

