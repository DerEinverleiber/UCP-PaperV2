import numpy as np
from dataclasses import dataclass
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import spsolve

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
    resistance: float # New: added attribute

class PowerGrid():
    """
    Representation of a power grid network using the DC power flow approximation.

    Attributes
    ----------
    busses : list[Bus]
        List of buses in the network (loads, generators).
    branches : list[Branch]
        List of branches (transmission lines) connecting buses.
    reference_bus_id : int
        Index of the slack/reference bus (default 0).
    n : int
        Number of buses in the network.
    graph : csr_matrix
        Weighted adjacency matrix of branch susceptances.
    B : csr_matrix
        Susceptance (Laplacian-like) matrix for DC power flow.
    P : csr_matrix
        Net power injections vector (P_gen - P_load) assuming all generators on.

    Methods
    -------
    ieee57()
        Load the IEEE 57-bus test system.
    random(n)
        Generate a random network of n buses (placeholder).
    susceptance_graph(branches)
        Build adjacency matrix of branch susceptances from branch data.
    net_power(busses, x=None)
        Compute net injections for a given on/off vector x.
    solve_lse(P_update=None)
        Solve B θ = P for bus voltage angles θ.
    loss_function(x=None, c=None)
        Compute total generation cost for a given on/off vector x.
    """
    def __init__(self, busses: list[Bus], branches: list[Branch], reference_bus_id: int = 0): 
        self.busses = busses
        self.branches = branches
        self.reference_bus_id = reference_bus_id
        self.n = len(self.busses) # Number of Busses
        # Zero-based bus positions that actually contain generation.
        self.generator_positions = [i for i, bus in enumerate(self.busses) if bus.generation > 0]
        self.num_generators = len(self.generator_positions)
        self.graph = self.susceptance_graph(self.branches) # change attribute name?
        diag = np.array(self.graph.sum(axis=1)).flatten()
        self.B = diags(diag) - self.graph # Susceptibility Matrix
        self.P = self.net_power(self.busses) # initial power vector (all generators are turned on)

    @classmethod
    def ieee57(cls) -> "PowerGrid":
        """
        Load the IEEE 57-bus test system.

        The network is modeled as a graph:
        - Nodes correspond to buses with loads and/or generators.
        - Edges represent transmission lines connecting buses.
        - Edge weights are branch reactances used in DC power flow calculations.

        Data sources:
        - Branch data: 'ieee57_branch.csv' (includes resistance and reactance)
        - Bus data: 'bus_data_short.csv' (loads and generation)

        Notes
        -----
        - Generation in MW, loads in MW.
        - Reactive power and AC effects are ignored (DC approximation).
        - A bus with zero generation is treated as a pure load.
        """
        branch_data = np.loadtxt('data/ieee57_branch.csv', delimiter=',', skiprows=1, dtype=str)
        tap_bus_number = np.array(branch_data[:, 0], dtype=int) # units?
        z_bus_number = np.array(branch_data[:, 1], dtype=int) #
        branch_reactance = np.array(branch_data[:, 7], dtype=float) 
        branch_resistance = np.array(branch_data[:, 6], dtype=float)

        bus_data = np.loadtxt('data/bus_data_short.csv', delimiter=',', skiprows=1, dtype=float)
        bus_number, bus_load, bus_generation, bus_generation_MVAR = bus_data.T

        branches = [Branch(tap_bus_number[i], z_bus_number[i], branch_reactance[i], branch_resistance[i]) for i in range(len(branch_data))]
        busses = [Bus(bus_number[i], bus_load[i], bus_generation[i]) for i in range(len(bus_data))]

        return cls(busses, branches)
    
    @classmethod
    def random(cls, n: int, num_generators: int, max_ratio: float = 10e-1, min_reactance :float =10e-2) -> "PowerGrid":
        """
        Generate a random, connected power grid network for DC power flow experiments.

        The network is created by first building a spanning tree to ensure connectivity,
        then adding extra random edges. Branch reactances and resistances, as well as
        bus loads and generator capacities, are sampled from uniform distributions.

        Important notes
        ---------------
        - Reactances X are strictly positive (X > 0) and sampled uniformly in [min_reactance, 1].
        - Resistances R are small relative to X, sampled uniformly in [0, max_ratio].
        - Generator outputs and bus loads are sampled uniformly in [0, 1].
        - No negative reactances are allowed.
        - The graph is connected but not fully connected; sparsity is controlled by
        the number of extra edges added.

        Parameters
        ----------
        n : int
            Number of buses in the network.
        max_ratio : float, optional
            Maximum resistance relative to reactance (default 0.1).
        min_reactance : float, optional
            Minimum reactance value (default 0.01).
        seed : int, optional
            Random seed for reproducibility (default 1234).

        Returns
        -------
        PowerGrid
            Instance of PowerGrid with randomly generated buses and branches.
        """

        generator_indices = np.random.choice(list(range(n)), size=num_generators, replace=False)
        generations = np.zeros(n, dtype=float)
        generations[generator_indices] += np.random.uniform(1e-2, 1, size=num_generators)

        loads = np.random.uniform(0, 1, n)

        edges = []
        nodes = list(range(n))
        np.random.shuffle(nodes)
        for i in range(1, n):
            edges.append((nodes[i-1], nodes[i]))

        num_extra = n // 2
        for _ in range(num_extra):
            a, b = np.random.choice(n, size=2, replace=False)
            if (a, b) not in edges and (b, a) not in edges:
                edges.append((a,b))

        num_edges = len(edges)
        reactances = np.random.uniform(min_reactance, 1, num_edges)
        resistances = np.random.uniform(0, max_ratio, num_edges)

        branches = [Branch(from_bus=i+1, to_bus=j+1, reactance=reactances[k],
                           resistance=resistances[k]) for k, (i, j) in enumerate(edges)]
        busses = [Bus(idx=i+1, load=loads[i], generation=generations[i]) for i in range(n)]

        return cls(busses, branches)



    def susceptance_graph(self, branches: list[Branch]):
        """
        Build the weighted adjacency matrix of branch susceptances.

        Parameters
        ----------
        branches : list[Branch]
            List of transmission lines with reactance and resistance.

        Returns
        -------
        csr_matrix
            Sparse symmetric matrix of susceptances (shape = n x n).
        """
        edges = [(branch.from_bus-1, branch.to_bus-1) for branch in branches] # switch to 0-indexing
        rows, cols = zip(*edges)
        susceptances = [branch.reactance/(branch.resistance**2 + branch.reactance**2) for branch in branches] # New: we are no longer using taylor expansion
        graph = csr_matrix((susceptances, (rows, cols)), shape=(self.n, self.n))  # Undirected Graph with weights corresponding to susceptances 
        graph += graph.T
        return graph
    

    def expand_generator_bits(self, x: list[int] | np.ndarray | None = None) -> np.ndarray:
        """
        Expand a compressed generator-only bit string to a full bus-length vector.

        Parameters
        ----------
        x : array_like, optional
            Generator on/off vector of length self.num_generators. If None, all generators are on.

        Returns
        -------
        np.ndarray
            Full bus-length vector with zeros on pure load buses.
        """
        if x is None:
            x = np.ones(self.num_generators, dtype=float)
        x = np.asarray(x, dtype=float)

        if x.ndim != 1:
            raise ValueError(f'x must be a 1D array, got shape {x.shape}.')
        if len(x) != self.num_generators:
            raise ValueError(
                f'x must have length equal to the number of generators ({self.num_generators}), '
                f'got {len(x)}.'
            )

        x_full = np.zeros(self.n, dtype=float)
        x_full[self.generator_positions] = x
        return x_full

    '''
    def net_power(self, busses: list[Bus], x=None, load_factor: float = 1.0) -> csr_matrix:
        """
        Compute net power injections: P_i = x_i * P_gen,i - P_load,i.

        Parameters
        ----------
        busses : list[Bus] — buses with load and generation.
        x : array_like, optional — generator on/off vector (1 = on, 0 = off). Defaults to all on.

        Returns
        -------
        csr_matrix
            Sparse column vector of net injections (n x 1).
        """
        gen = np.array([bus.generation for bus in busses], dtype=float)
        load = np.array([bus.load for bus in busses], dtype=float)
        if x is None:
            x = np.ones(self.n, dtype=float)
        x = np.asarray(x, dtype=float)
        x_full = self.expand_generator_bits(x)
        return x_full * gen - load_factor * load
    '''

    def net_power(self, busses: list[Bus], x=None, load_factor: float = 1.0) -> np.ndarray:
        gen = np.array([bus.generation for bus in busses], dtype=float)
        load = np.array([bus.load for bus in busses], dtype=float)
        x_full = self.expand_generator_bits(x)
        return x_full * gen - load_factor * load


    def get_generator_indices(self) -> list[int]:
        return [bus.idx for bus in self.busses if bus.generation > 0]
    
    def get_generator_positions(self) -> list[int]:
        """Return 0-based array positions of generator buses."""
        return self.generator_positions.copy()


   
    def solve_lse(self, P_update: np.ndarray = None):
        """
        Solve the linear system B θ = P for bus voltage angles (DC approximation).

        Parameters
        ----------
        P_update : csr_matrix, optional — net injections vector. Defaults to self.P.

        Returns
        -------
        np.ndarray
            Voltage angles θ (radians) at all buses (length n), with reference bus at 0.
        """
        if P_update is None:
            P_update = self.P.copy()
        else:
            P_update = np.asarray(P_update, dtype=float)
        
        non_slack = [i for i in range(self.n) if i != self.reference_bus_id]

        B_red = self.B[non_slack, :][:, non_slack]
        P_red = P_update[non_slack]
        theta_red = spsolve(B_red, P_red) # need to convert Sprase P to dense P to solve

        theta = np.zeros(self.n)
        theta[non_slack] = theta_red

        return theta
    
    def compute_line_flows(self, theta: np.ndarray) -> np.ndarray:
        rho = np.zeros(len(self.branches), dtype=float)

        for m, br in enumerate(self.branches):
            i = br.from_bus - 1
            j = br.to_bus - 1
            b_ij = br.reactance / (br.resistance**2 + br.reactance**2)
            rho[m] = b_ij * (theta[i] - theta[j])

        return rho

    def loss_function(self, x: list[int], c: list[float] = None, load_factor: float = 1.0, mismatch_penalty = 1e4, return_net_power_io_diff: bool = True) -> float | tuple[float, float]:
        """
        Compute total generation cost for a given generator on/off vector.

        Parameters
        ----------
        x : list[int], optional — generator on/off vector (1 = on, 0 = off). Defaults to all on.
        c : list[float], optional — cost per generator. Defaults to zeros.

        Returns
        -------
        float
            Total generation cost.
            :param return_net_power_io_diff:
        """
        if x is None:
            x = np.ones(self.num_generators) # all generators are on
        x = np.asarray(x)

        if c is None:
            c = np.ones(len(self.branches), dtype=float)
        c = np.asarray(c, dtype=float)

        # Determine Power Vector 
        power_vector = self.net_power(x=x, busses=self.busses, load_factor=load_factor)
        # Determine theta
        theta = self.solve_lse(power_vector)
        # Determine line flows
        rho = self.compute_line_flows(theta)
        
        # Determine penalty
        mismatch = np.sum(power_vector)
        penalty_term = np.abs(mismatch*mismatch_penalty)

        loss = np.dot(c, np.abs(rho)) + penalty_term

        if return_net_power_io_diff:
            return loss, mismatch
        else:
            return loss
        
    def num_outgoing_branches(self) -> list[int]:
        num_outgoing_branches = [0] * self.n
        for branch in self.branches:
            num_outgoing_branches[branch.from_bus - 1] += 1 # switch to 0-indexing

        assert sum(num_outgoing_branches) == len(self.branches)
        return num_outgoing_branches



if __name__ == '__main__':
    import networkx as nx
    from matplotlib import pyplot as plt


    def plot_grid(pg: PowerGrid):
        G = nx.from_scipy_sparse_array(pg.graph)
        pos = nx.spring_layout(G)  # layout of the nodes
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color='skyblue', node_size=500)
        # Draw edges
        nx.draw_networkx_edges(G, pos, width=2)
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=12)
        plt.axis('off')
        plt.show()


    # plt.show()
    pg = PowerGrid.random(n=10, num_generators=2)

    print(pg.get_generator_indices())

    plot_grid(pg)
