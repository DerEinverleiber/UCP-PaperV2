import pennylane as qml
import numpy as np
# We will be using the pennylane qml.qaoa class https://docs.pennylane.ai/en/stable/code/qml_qaoa.html

class QOptimizer():
    """
    Class Implementing the quantun optimization architecture presented in J. Stein, et al. (https://arxiv.org/abs/2305.08482) Fig. 6
    """
    def __init__(self, n: int, p: int, costs):
        """
        n: system size
        p: circuit depth
        costs: array of cost function values for all bit strings x shape (2**n,)
        """
        self.n = n
        self.p = p
        self.costs = np.array(costs)
        self.wires = range(self.n)
        assert costs.shape == (2**n,)
        self.dev = qml.device("default.qubit", wires=len(self.wires)) # or lightning?
        self.cost_h = qml.Hermitian(np.diag(self.costs), wires=self.wires)
        self.mix_h = qml.qaoa.mixers.x_mixer(wires=range(self.n))

    def qaoa_layer(self, gamma, beta):
        qml.qaoa.layers.cost_layer(gamma, self.cost_h)
        qml.qaoa.layers.mixer_layer(beta, self.mix_h)

    def circuit(self, gammas, betas): 
        # initialize supoerposition
        for w in self.wires:
            qml.Hadamard(wires=w)
        # apply p layers
        for i in range(self.p):
            self.qaoa_layer(gammas[i], betas[i])
    

    def cost_function(self, params):
        @qml.qnode(self.dev)
        def qnode(gammas, betas):
            self.circuit(gammas, betas)
            return qml.expval(self.cost_h)
        gammas, betas = params
        return qnode(gammas, betas)
