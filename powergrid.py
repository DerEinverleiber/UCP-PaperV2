import numpy as np
from dataclasses import dataclass
from scipy.sparse import csr_matrix
from scipy.sparse import diags
from scipy.sparse.csgraph import laplacian
from scipy.sparse.linalg import spsolve
from scipy.linalg import inv

@dataclass 
class Bus:
    idx: int
    load: float
    generation: float
 
@dataclass 
class Branch:
    from_bus: int
    to_bus: int
    reactance: float

class PowerGrid:
    def __init__(self, busses: list[Bus], branches: list[Branch], reference_bus_id: int = 0): 
        self.busses = busses
        self.branches = branches
        self.reference_bus_id = reference_bus_id
        self.n = len(self.busses) # Number of Busses
        self.graph = self.susceptance_graph(self.branches) # change attribute name?
        diag = np.array(self.graph.sum(axis=1)).flatten()
        self.B = diags(diag) - self.graph # Susceptibility Matrix
        self.P = self.net_power(self.busses)
    
    @classmethod
    def ieee57(cls) -> "PowerGrid":
        """
        The power grid is modeled as a graph, where:
        - Vertices (nodes) represent buses, which may have loads, generators, or both.
        - Edges represent branches (transmission lines, transformers, or phase shifters) that connect buses. 
        Each branch is defined by its Tap Bus Number and Z Bus Number, which correspond to the two endpoints of the edge.
        - Edge weights are given by the branch reactances, representing the impedance of the connection. 
        These weights are used in calculations such as power flow, short-circuit analysis, and network optimization.

        Reactences X per unit (pu), dimensionless
        """
        branch_data = np.loadtxt('data/ieee57_branch.csv', delimiter=',', skiprows=1, dtype=str)
        tap_bus_number = np.array(branch_data[:, 0], dtype=int) # units?
        z_bus_number = np.array(branch_data[:, 1], dtype=int) #
        branch_reactance = np.array(branch_data[:, 7], dtype=float) 

        """
        Loads in MW
        Generation in MW
        Generation in MVAR (isn't this only relevant for AC systems?)

        A bus with zero generation corresponds to a pure electrical consumer
        """
        bus_data = np.loadtxt('data/bus_data_short.csv', delimiter=',', skiprows=1, dtype=float)
        bus_number, bus_load, bus_generation, bus_generation_MVAR = bus_data.T

        branches = [Branch(tap_bus_number[i], z_bus_number[i], branch_reactance[i]) for i in range(len(branch_data))] 
        busses = [Bus(bus_number[i], bus_load[i], bus_generation[i]) for i in range(len(bus_data))]

        return cls(busses, branches)

    def num_outgoing_branches(self) -> list[int]:
        num_outgoing_branches = [0] * self.n
        for branch in self.branches:
            num_outgoing_branches[branch.from_bus - 1] += 1 # switch to 0-indexing

        assert sum(num_outgoing_branches) == len(self.branches)
        return num_outgoing_branches

    def get_generator_indices(self) -> list[int]:
        return [bus.idx for bus in self.busses if bus.generation > 0]

    def get_neighbor_nodes(self, bus_idx: int) -> list[int]:
        return [branch.to_bus for branch in self.branches if branch.from_bus == bus_idx]

    def get_num_generators(self) -> int:
        return len(self.get_generator_indices())

    @classmethod
    def random(cls, n: int) -> "PowerGrid":
        pass

    def susceptance_graph(self, branches: list[Branch]):
        edges = [(branch.from_bus-1, branch.to_bus-1) for branch in branches] # switch to 0-indexing
        rows, cols = zip(*edges)
        susceptances = [1/(branch.reactance) for branch in branches] # Assuming small resistances
        graph = csr_matrix((susceptances, (rows, cols)), shape=(self.n, self.n))  # Undirected Graph with weights corresponding to susceptances 
        graph += graph.T
        return graph
    
    def net_power(self, busses: list[Bus]) -> csr_matrix:
        return csr_matrix([bus.generation - bus.load for bus in busses]).T # shape=(n, 1)

    def solve_lse(self):
        non_slack = [i for i in range(self.n) if i != self.reference_bus_id]

        B_red = self.B[non_slack, :][:, non_slack]
        P_red = self.P[non_slack]
        theta_red = spsolve(B_red, P_red.toarray().flatten()) # need to convert Sprase P to dense P to solve

        theta = np.zeros(self.n)
        theta[non_slack] = theta_red

        return theta
    
    """
    c: List of cost parameters 
    x: List of decision variables (MINLP)
    """
    def loss_function(self, x: list[int], c: list[float] = None, return_net_power_io_diff: bool = False) -> float | tuple[float, float]:
        num_edges = np.array(self.num_outgoing_branches(), dtype=int) # d1, ..., dN

        inv_reduced_graph_laplacian = self.prepare_inverse_reduced_graph_laplacian()
        power_vector = self.apply_decision_variables(x)

        reduced_power_vector = np.delete(power_vector, self.reference_bus_id)
        theta_reduced = np.dot(inv_reduced_graph_laplacian, reduced_power_vector)

        b_prime = self.prepare_b_prime(num_edges)
        theta_prime = np.repeat(theta_reduced, np.delete(num_edges, self.reference_bus_id), axis=0)
        rho = np.dot(b_prime, theta_prime)

        if c is None:
            c = np.zeros(theta_prime.shape[0])
        else:
            c = np.asarray(c)

        total_generation_consumption_discrepancy = np.abs(np.sum(power_vector))
        penalty_term = 10000 * total_generation_consumption_discrepancy # total power IO diff. should approx. be 0
        loss =  np.dot(c, np.abs(rho)) + penalty_term

        if return_net_power_io_diff:
            return loss, total_generation_consumption_discrepancy
        else:
            return loss

    def prepare_b_prime(self, num_edges: np.ndarray[int]) -> csr_matrix:
        e = lambda v, j: self.get_neighbor_nodes(v + 1)[j]  # return j-th entry of neighbor nodes

        # diagonal entries
        b_prime_diags_row_indices = np.concatenate(
            [[i] * d for i, d in enumerate(num_edges) if i != self.reference_bus_id]) # 0-based
        b_prime_diags_col_indices = np.concatenate(
            [np.arange(0, d) for i, d in enumerate(num_edges) if i != self.reference_bus_id]) # 0-based
        b_prime_diags_col_indices = np.array(
            [e(row, col) for row, col in zip(b_prime_diags_row_indices, b_prime_diags_col_indices, strict=True)] # 1-based
        )

        b_prime_diags = self.B[b_prime_diags_row_indices, b_prime_diags_col_indices - 1].A1 # e uses 1-based indexing
        b_prime = diags(b_prime_diags).toarray()

        # off-diagonal entries
        for i in range(self.n - 1): # reduced graph has only n - 1 nodes/busses
            for j in range(num_edges[i]):
                for k in range(b_prime.shape[1]):
                    if i * j != k:
                        b_prime[i, k] = self.B[i, e(i, j) - 1]

        return b_prime

    def prepare_inverse_reduced_graph_laplacian(self) -> csr_matrix:
        graph = self.graph.toarray()

        graph_laplacian = laplacian(graph)
        reduced_graph_laplacian = np.delete(graph_laplacian, self.reference_bus_id, axis=0)
        reduced_graph_laplacian = np.delete(reduced_graph_laplacian, self.reference_bus_id, axis=1)
        return inv(reduced_graph_laplacian)

    def apply_decision_variables(self, x: list[int]) -> csr_matrix:
        generator_indices = np.array(self.get_generator_indices(), dtype=int) - 1 # 0-based indexing
        mask = np.zeros(len(self.busses), dtype=bool)
        mask[generator_indices] = True

        power_vector = self.P.toarray().flatten()
        power_vector[mask] *= x
        return power_vector

        