import numpy as np
from dataclasses import dataclass
from scipy.sparse import csr_matrix
from scipy.sparse import csgraph

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

class PowerGrid(): # rough outline 
    def __init__(self, busses: list[Bus], branches: list[Branch]): 
        self.busses = busses
        self.branches = branches
        self.n = len(busses) # Number of Busses
        edges = [(branch.from_bus-1, branch.to_bus-1) for branch in branches] # switch to 0-indexing
        rows, cols = zip(*edges)
        susceptances = [1/(branch.reactance) for branch in branches] # Assuming small resistances
        graph = csr_matrix((susceptances, (rows, cols)), shape=(self.n, self.n))  # Undirected Graph with weights corresponding to susceptances 
        self.graph = graph + graph.T
        diag = np.array(self.graph.sum(axis=1)).flatten()
        self.B = csr_matrix(np.diag(diag)- self.graph) # Suscepibility Matrix
    
   
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


    def loss_function(costs: list[float]) -> float:
        assert len(costs) == len()
        

        