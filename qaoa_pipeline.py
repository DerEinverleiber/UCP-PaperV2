import pennylane as qml
import numpy as np

class QAOA_circuit:
    """
    Class Implementing the quantun circuit presented in J. Stein, et al. (https://arxiv.org/abs/2305.08482) Fig. 6
    We will work along the tutorial https://pennylane.ai/qml/demos/tutorial_qaoa_intro for the QAOA circuit implementation
    """
    def __init__(self, n: int, p: int, costs):
        """
        n: system size
        p: circuit depth
        costs: array of cost function values for all bit strings x shape (2**n,)
        parameters: [gammas, betas] shape (2, n)
        """
        self.n = n
        self.p = p
        self.costs = np.array(costs)
        self.max_cost = np.max(self.costs) # needed for AR later
        self.min_cost = np.min(self.costs)
        self.wires = range(self.n)
        assert costs.shape == (2**n,)
        
        self.cost_h = qml.Hermitian(np.diag(self.costs), wires=self.wires)
        self.mix_h = qml.qaoa.mixers.x_mixer(wires=range(self.n))
        self.dev = qml.device("default.qubit", wires=len(self.wires)) # or lightning?
        self.qnode = qml.QNode(self._qnode, self.dev)
    
    def _qnode(self, gammas, betas):
        self.circuit(gammas, betas)
        return qml.expval(self.cost_h)

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

'''
Used like
qaoa = QAOA_circuit(n=4, p=3, costs=cost_array)
params = (gammas, betas)
energy = qaoa.cost_function(params)
'''

class ChebyshevOptimizer:
    def __init__(self, qaoa_circuit: QAOA_circuit, num_coeffs: int, stepsize: float =0.1):
        self.qaoa_circuit = qaoa_circuit
        #self.p = self.qaoa_circuit.p
        self.num_coeffs = num_coeffs
        self.stepsize = stepsize
        self.optimizer = qml.GradientDescentOptimizer(stepsize)

    # adapted from: https://github.com/jpmorganchase/QOKit/blob/main/qokit/parameter_utils.py
    def to_chebyshev(self, gammas: np.ndarray, betas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Convert gamma, beta angles in standard parameterizing QAOA to the Chebyshev basis
        
        Parameters
        ----------
        gammas : list-like
        betas : list-like
        num_coeffs : int
            p // 2 is recommended

        Returns
        -------
        u, v : np.array
            QAOA parameters in Chebyshev basis
        """
        assert len(gammas) == len(betas)
        fit_interval = np.linspace(-1, 1, len(gammas))
        u = np.polynomial.chebyshev.chebfit(fit_interval, gammas, deg=self.num_coeffs - 1)  # offset of 1 due to fitting convention
        v = np.polynomial.chebyshev.chebfit(fit_interval, betas, deg=self.num_coeffs - 1)
        return u, v

    # adapted from: https://github.com/jpmorganchase/QOKit/blob/main/qokit/parameter_utils.py
    def to_angles(self, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Convert u,v in Chebyshev basis of functions
        to gamma, beta angles of QAOA schedule

        Parameters
        ----------
        u : list-like
        v : list-like

        Returns
        -------
        gamma, beta : np.array
            QAOA angles parameters in standard parameterization
        """
        
        assert len(u) == len(v)
        fit_interval = np.linspace(-1, 1, self.qaoa_circuit.p)
        gamma = np.polynomial.chebyshev.chebval(fit_interval, u)
        beta = np.polynomial.chebyshev.chebval(fit_interval, v)
        return gamma, beta
    
    def wrapped_cost_function(self, coeffs):
        u, v = coeffs 
        gammas, betas = self.to_angles(u, v)
        return self.qaoa_circuit.cost_function((gammas, betas))

    def step(self, coeffs):
        return self.optimizer.step(
            lambda p: self.wrapped_cost_function(p),
            coeffs
        )
    
class IterativeInterpolation:
    """Following the Iterative Interpolation (II) algorithm, as proposed by A. Apte, et al. (arXiv:2504.01694v1)
    """
    def __init__(self, init_params, qaoa_circuit: QAOA_circuit, cheby_optimizer: ChebyshevOptimizer,
                p0: int, C: int, 
                epsilon: float, tau: int, delta_p=5, target_AR=0.95, opt_steps :int = 100
                ):
        self.init_params = init_params
        self.qaoa_circuit = qaoa_circuit
        self.min_cost = qaoa_circuit.min_cost
        self.max_cost = qaoa_circuit.max_cost
        self.cheby_optimizer = cheby_optimizer
        self.p0 = p0
        self.p_max = qaoa_circuit.p # or simply QAOA depth
        self.delta_p = delta_p # choose for convenience
        self.C = C # choose p // 2 ?
        self.epsilon = epsilon
        self.tau = tau
        self.target_AR = target_AR
        self.opt_steps = opt_steps # number of steps for Chebyshev optimization

    def approximation_ratio(self, energy):
        return 1 - (energy - self.min_cost)/(self.max_cost - self.min_cost)
        
    def _pad(self, coeffs):
        u, v = coeffs
        u = np.append(u, 0.0)
        v = np.append(v, 0.0)
        return np.array([u, v], requires_grad=True)

    
    def _interpolate(self, coeffs, p):
        self.qaoa_circuit = QAOA_circuit(
            n=self.qaoa_circuit.n,
            p=p,
            costs=self.qaoa_circuit.costs
        )
        self.cheby_optimizer = ChebyshevOptimizer(
            qaoa_circuit=self.qaoa_circuit,
            num_coeffs=len(coeffs[0]),
            stepsize=self.cheby_optimizer.stepsize
        )
        return coeffs
    
    def run(self):
        c_pat = 0 # initialize patience counter
        p = self.p0
        AR_old, AR_current = 0.0, 0.0 #
        gammas, betas = self.init_params
        u, v = self.cheby_optimizer.to_chebyshev(gammas, betas) # transform (gamma^p, beta^p) to functional basis
        coeffs = np.array([u, v], requires_grad=True)
        while p <= self.p_max and AR_current < self.target_AR: #and i <= self.max_iter:
           # initialize parameters?    
            for _ in range(self.opt_steps): # optimize the first C coefficients
                coeffs = self.cheby_optimizer.step(coeffs)
            u, v = coeffs
            # current energy
            gammas, betas = self.cheby_optimizer.to_angles(u, v)
            energy = self.qaoa_circuit.cost_function((gammas, betas))
            AR_current = self.approximation_ratio(energy)
            delta = np.abs(AR_current-AR_old) # compute relative performance increase
            if delta <= self.epsilon:
                # increase number of coefficients C to be tuned
                c_pat += 1
            else :
                c_pat = 0
            if c_pat >= self.tau:
                coeffs = self._pad(coeffs)
                self.C += 1
                self.cheby_optimizer.num_coeffs = len(coeffs[0])
                c_pat = 0
            # perform interpolation
            AR_old = AR_current 
            p+=self.delta_p
            if p <= self.p_max:
                coeffs = self._interpolate(coeffs, p)
                u, v = coeffs
                gammas, betas = self.cheby_optimizer.to_angles(u, v)
        return gammas, betas, energy, AR_current



