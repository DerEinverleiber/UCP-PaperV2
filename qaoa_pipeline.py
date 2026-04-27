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
        
        self.mix_h = qml.qaoa.mixers.x_mixer(wires=range(self.n))
        self.dev = qml.device("default.qubit", wires=len(self.wires)) 
        self.prob_qnode = qml.QNode(self._prob_qnode, self.dev)  

    def _prob_qnode(self, gammas, betas):  
        self.circuit(gammas, betas)
        return qml.probs(wires=self.wires)
    
    def qaoa_layer(self, gamma, beta):   
        qml.DiagonalQubitUnitary(qml.math.exp(-1j * gamma * self.costs), wires=self.wires)
        qml.qaoa.layers.mixer_layer(beta, self.mix_h)

    def circuit(self, gammas, betas):
        # initialize superposition
        for w in self.wires:
                qml.Hadamard(wires=w)
        # apply for p layers
        qml.layer(self.qaoa_layer, self.p, gammas, betas)
 
    def cost_function(self, params):
        gammas, betas = params
        probs = self.prob_qnode(gammas, betas)
        return np.dot(probs, self.costs)
     
    def distribution(self, params, as_dict=False):   # NEW
        """
        Return computational basis probabilities:
        [p(000...0), p(000...1), ..., p(111...1)]

        If as_dict=True, returns:
        {'000...0': p0, '000...1': p1, ..., '111...1': p_last}
        """
        gammas, betas = params
        probs = self.prob_qnode(gammas, betas)

        if as_dict:
            bitstrings = [format(i, f"0{self.n}b") for i in range(2**self.n)]
            return dict(zip(bitstrings, probs))
        return probs

'''
Used like
qaoa = QAOA_circuit(n=4, p=3, costs=cost_array)
params = (gammas, betas)
energy = qaoa.cost_function(params)
'''

class ChebyshevOptimizer:
    def __init__(self, qaoa_circuit: QAOA_circuit, num_coeffs: int, stepsize: float =0.005):
        self.qaoa_circuit = qaoa_circuit
        #self.p = self.qaoa_circuit.p
        self.num_coeffs = num_coeffs
        self.stepsize = stepsize
        """
        AdamOptimizer turned out to work out a lot better than plane gradient GradientDescent!
        The JPMorgan paper relied on gradient free BOBYQA optimization (they also used the Approxmation Ratio
        directly as their objective function). 
        """
        self.optimizer = qml.AdamOptimizer(stepsize)

    # adapted from: https://github.com/jpmorganchase/QOKit/blob/main/qokit/parameter_utils.py
    def to_chebyshev(self, gammas: np.ndarray, betas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Convert gamma, beta angles in standard parameterizing QAOA to the Chebyshev basis
        
        Parameters
        ----------
        gammas : list-like
        betas : list-like

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
        fit_interval = np.linspace(-1, 1, self.qaoa_circuit.p) # -1 to 1 because Cbebyshev polynomials are stable on that domain
        # we can generally obtain negative parameters that way, but that's okay.
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
                p0: int, p_max: int, C: int, 
                epsilon: float, tau: int, delta_p=5, target_AR=0.95, opt_steps :int = 100
                ):
        self.init_params = init_params
        self.qaoa_circuit = qaoa_circuit
        self.min_cost = qaoa_circuit.min_cost
        self.max_cost = qaoa_circuit.max_cost
        self.cheby_optimizer = cheby_optimizer
        self.p0 = p0
        self.p_max = p_max # or simply QAOA depth
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
        self.C += 1
        self.cheby_optimizer.num_coeffs = len(u) # update optimizer
        return qml.numpy.array([u, v], requires_grad=True)

    
    def _interpolate(self, coeffs, p):
        self.qaoa_circuit = QAOA_circuit( # update circuit
            n=self.qaoa_circuit.n,
            p=p,
            costs=self.qaoa_circuit.costs
        )
        self.cheby_optimizer = ChebyshevOptimizer( # update optimizer
            qaoa_circuit=self.qaoa_circuit,
            num_coeffs=len(coeffs[0]),
            stepsize=self.cheby_optimizer.stepsize
        )
        # return updated gammas, betas
        u, v = coeffs
        gammas, betas = self.cheby_optimizer.to_angles(u, v)
        return (gammas, betas)
    
    def run(self):
        c_pat = 0 # initialize patience counter
        p = self.p0
        AR_old, AR_current = 0.0, 0.0 #
        ARs = {}
        energies = {}
        gammas, betas = self.init_params # initial parameters of circuit with depth p0
        
        while p <= self.p_max and AR_current < self.target_AR: #and i <= self.max_iter:
            print(f'>> II: p={p} of {self.p_max}, AR_curent={AR_current}', flush=True)
            u, v = self.cheby_optimizer.to_chebyshev(gammas, betas) # transform (gamma^p, beta^p) to functional basis
            coeffs = qml.numpy.array([u, v], requires_grad=True)
           # initialize parameters?    
            print('    >> Start Chebyshev Opt.', flush=True)
            for _ in range(self.opt_steps): # optimize the first C coefficients
                coeffs = self.cheby_optimizer.step(coeffs)
            u, v = coeffs
            print('    >> End Chebyshev Opt.', flush=True)
            energy = self.cheby_optimizer.wrapped_cost_function((u, v))
            AR_current = self.approximation_ratio(energy)
            delta = 0 if AR_old == 0 else np.abs((AR_current-AR_old)/AR_old) # compute relative performance increase
            if delta <= self.epsilon:
                # increase number of coefficients C to be tuned
                c_pat += 1
            else :
                c_pat = 0
            if c_pat >= self.tau:
                coeffs = self._pad(coeffs)    
                c_pat = 0
            # perform interpolation
            ARs[p] = AR_current
            energies[p] = energy
            AR_old = AR_current 
            p+=self.delta_p
            if p <= self.p_max:
                gammas, betas = self._interpolate(coeffs, p) # updated p and optimized coeffs
        return gammas, betas, energies, ARs



