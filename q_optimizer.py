class QOptimizer():
    """
    Class Implementing the quantun optimization architecture presented in J. Stein, et al. (https://arxiv.org/abs/2305.08482) Fig. 6
    """
    def __init__(self, n: int, p: int, parameterization: Parameterization):
        """
        n system size
        p circuit depth
        """
        self.n = n
        self.p = p
        self.psi = self.init_superposition(n) # variational state psi is modified in place
        # parameters from PowerGrid necessary for update
        #...

    def init_superposition():
        # |+> ⊗ |+> ⊗ ... ⊗ |+>
        # use Pennylane
        pass 
    
    def U_mix():
        # use Pennylane
        pass

    def U_cost(): 
        # Use Pennylane
        # use brute force solution for diagonal entries
        pass

    def QAOA_update():
        #apply U_M
        #apply U_C
        pass

    def init_parameters(): 
        # Chebechev stuff (paper, thesis)
        pass 

    
    def parameter_update(): # or simply init_parameters gives list of parameters
        # gamma_i -> gamma_i+1
        # beta_i -> beta_i+1
        pass


    def run(): # executes optimization procedure
        pass
