import pennylane as qml
import numpy as np
# We will be using the pennylane qml.qaoa class https://docs.pennylane.ai/en/stable/code/qml_qaoa.html

class QOptimizer():
    """
    Class Implementing the quantun optimization architecture presented in J. Stein, et al. (https://arxiv.org/abs/2305.08482) Fig. 6
    """
    def __init__(self, n: int, p: int, costs, parameters = None, use_gd_optimizer: bool = False, max_iter=100):
        """
        n: system size
        p: circuit depth
        costs: array of cost function values for all bit strings x shape (2**n,)
        init_parameters (default None): [gammas, betas]
        use_gd_optimizer (detault False): if set to false, Chebyshev Optimizer is Used
        """
        self.n = n
        self.p = p
        self.costs = np.array(costs)
        self.wires = range(self.n)
        self.max_iter = max_iter
        assert costs.shape == (2**n,)
        self.dev = qml.device("default.qubit", wires=len(self.wires)) # or lightning?
        self.cost_h = qml.Hermitian(np.diag(self.costs), wires=self.wires)
        self.mix_h = qml.qaoa.mixers.x_mixer(wires=range(self.n))
        if parameters is None:
            self.gammas, self.betas = self.init_parameters()
        else:
            parameters = np.array(parameters)
            assert parameters.shape == (2, self.p)
            self.gammas, self.betas = parameters 
        self.params = np.array([self.gammas, self.betas], requires_grad=True)
        self.use_gd_optimizer = use_gd_optimizer
        self.optimizer = qml.GradientDescentOptimizer() if self.use_gd_optimizer else None # INSERT CHEBYSEV
        self.qnode = qml.QNode(self._qnode, self.dev)
    
    def _qnode(self, gammas, betas):
        self.circuit(gammas, betas)
        return qml.expval(self.cost_h)

    def init_parameters(self):
        gammas = np.linspace(0, 1, self.p)
        betas = np.linspace(1, 0, self.p)
        return gammas, betas
    
    # We follow the tutorial https://pennylane.ai/qml/demos/tutorial_qaoa_intro for the QAOA circuit implementation

    def qaoa_layer(self, gamma, beta):
        qml.qaoa.layers.cost_layer(gamma, self.cost_h)
        qml.qaoa.layers.mixer_layer(beta, self.mix_h)

    def circuit(self, gammas, betas):
        # initialize superposition
        for w in self.wires:
                qml.Hadamard(wires=w)
        # apply for p layers
        qml.layer(self.qaoa_layer, self.p, gammas, betas)
 

    def cost_function(self, params):
        gammas, betas = params
        return self.qnode(gammas, betas)

    # parameter optimization
    def optimize(self, return_history=False):
        params_history = []
        if self.use_gd_optimizer:
            # Gradient Descent Optimization
            for i in range(self.max_iter):
                self.params = self.optimizer.step(self.cost_function, self.params) # parameters are modified in place
        else: 
            #Chebyshev optimization
            pass

    # Return history?