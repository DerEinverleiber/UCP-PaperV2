import numpy as np
from dataclasses import dataclass
from scipy.sparse import csr_matrix
from scipy.sparse import diags
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

class PowerGrid(): 
    def __init__(self, busses: list[Bus], branches: list[Branch], reference_bus_id: int = 0): 
        self.busses = busses
        self.branches = branches
        self.reference_bus_id = reference_bus_id
        self.n = len(self.busses) # Number of Busses
        self.graph = self.susceptance_graph(self.branches) # change attribute name?
        diag = np.array(self.graph.sum(axis=1)).flatten()
        self.B = diags(diag) - self.graph # Suscepibility Matrix
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
        tap_bus_number = np.array(branch_data[:, 0], dtype=float) # units?
        z_bus_number = np.array(branch_data[:, 1], dtype=float) # 
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
    """
    def loss_function(self, c: list[float] = None) -> float:
        if c == None:
            c = np.zeros(self.n)
        else:
            c = np.asarray(c)

        
        # compute laplacian 

        

        