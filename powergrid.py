import numpy as np
from dataclasses import dataclass

@dataclass 
class Bus:
    idx: int
    # more

@dataclass
class Generator:
    idx: int
    #more

@dataclass 
class Branch:
    from_bus: int
    to_bus: int
    reactance: float
    #capacity/ cost?

class PowerGrid(): # rough outline 
    def __init__(self, buses: list[Bus], branches: list[Branch], generators: list[Generator]): # do i want these arguments?
        pass

    @classmethod
    def random() -> "PowerGrid":
        pass

    @classmethod
    def ieee57() -> "PowerGrid":
        pass

    def loss_function() -> float:
        pass

        