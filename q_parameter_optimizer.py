from abc import ABC, abstractmethod

class Parameterization(ABC):
    """
    Abstract Class that serves as a structure outline for individual ways of optimizing QAOA parameters
    For more details, see A. Apte, et al. (arXiv:2504.01694v1)
    """
    @abstractmethod
    def init():
        pass
    @abstractmethod
    def optimize():
        pass

class LinearParameterization(Parameterization):
    pass

class ChebyshevParameterization(Parameterization):
    pass

