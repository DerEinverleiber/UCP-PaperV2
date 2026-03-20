import numpy as np
import scipy

from q_optimizer import QOptimizer


class IterativeInterpolation:

    def __init__(self, qaoa: QOptimizer, init_gamma: np.ndarray, init_beta: np.ndarray, max_energy: float, min_energy: float):
        self.qaoa = qaoa
        self.init_gamma = init_gamma
        self.init_beta = init_beta
        self.max_energy = max_energy
        self.min_energy = min_energy

    # adapted from: https://github.com/jpmorganchase/QOKit/blob/main/qokit/parameter_utils.py
    def to_chebyshev_basis_coefficients(self, gamma: np.ndarray, beta: np.ndarray, depth: int) -> tuple[np.ndarray, np.ndarray]:
        fit_interval = np.linspace(-1, 1, len(gamma))
        u = np.polynomial.chebyshev.chebfit(fit_interval, gamma, deg=depth - 1)  # offset of 1 due to fitting convention
        v = np.polynomial.chebyshev.chebfit(fit_interval, beta, deg=depth - 1)
        return u, v
#
    # adapted from: https://github.com/jpmorganchase/QOKit/blob/main/qokit/parameter_utils.py
    def to_qaoa_angles(self, u: np.ndarray, v: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
        fit_interval = np.linspace(-1, 1, p)
        gamma = np.polynomial.chebyshev.chebval(fit_interval, u)
        beta = np.polynomial.chebyshev.chebval(fit_interval, v)
        return gamma, beta

    def optimize_chebyshev_basis_coefficients(self, u: np.ndarray, v: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray, float]:
        # use some optimization algorithm (I guess) to find better gamma_cheby, beta_cheby (probably need loss here??)
        num_coeff = len(u)
        def to_optimize(candidate):
            u, v = candidate[:num_coeff], candidate[num_coeff:]
            gamma, beta = self.to_qaoa_angles(u, v, p)
            _, best_loss = self.qaoa.run(gamma, beta)
            return best_loss

        res = scipy.optimize.minimize(to_optimize, x0=np.concatenate((u, v)), method="COBYLA", options={"maxiter": 100})
        approximation_ratio = (res.fun - self.min_energy) / (self.max_energy - self.min_energy)
        u, v = res.x[:num_coeff], res.x[num_coeff:]
        return u, v, approximation_ratio

    def interpolate_schedule(self, gamma: np.ndarray, beta: np.ndarray, delta_p: int):
        # transform gamma and beta to have len(gamma) + delta_p entries (extrapolate schedule)
        return np.concatenate((gamma, [1] * delta_p)), np.concatenate((beta, [1] * delta_p))


    def run(
            self,
            p_0: int,
            delta_p: int,
            p_max: int,
            improvement_threshold: float,
            num_coefficients: int,
            patience: int = 10,
            desired_approx_ratio: float = 0.95
    ) -> tuple[np.ndarray, np.ndarray]:
        patience_counter = 0
        p = p_0
        current_approximation_ratio = 0

        gamma, beta = self.init_gamma, self.init_beta

        while p <= p_max:
            u, v = self.to_chebyshev_basis_coefficients(gamma, beta, depth=num_coefficients)
            u, v, approximation_ratio = self.optimize_chebyshev_basis_coefficients(u, v, p)
            gamma, beta = self.to_qaoa_angles(u, v, p)


            if approximation_ratio >= desired_approx_ratio:
                return gamma, beta

            if approximation_ratio - current_approximation_ratio < improvement_threshold:
                patience_counter += 1
                if patience_counter == patience:
                    num_coefficients += 1
                    patience_counter = 0

            gamma, beta = self.interpolate_schedule(gamma, beta, delta_p=delta_p)
            p += delta_p
            current_approximation_ratio = approximation_ratio

        return gamma, beta


class QAOAMock:
    def __init__(self, gamma, beta):
        self.gamma = np.array(gamma)
        self.beta = np.array(beta)

    def run(self, gamma, beta):
        return 1, 1

    def get_angles(self):
        return self.gamma, self.beta


if __name__ == '__main__':
    qaoa = QAOAMock([0.1, 0.001, 0.5], [0.2, 0.3, 0.012])
    iterative_interpolation = IterativeInterpolation(
        qaoa=qaoa,
        init_gamma=np.array([0.1, 0.001, 0.5]),
        init_beta=np.array([0.2, 0.3, 0.012]),
        max_energy = 1,
        min_energy = 1
    )

    gamma, beta = iterative_interpolation.run(
        p_0=1,
        delta_p=1,
        p_max=10,
        improvement_threshold=0.000001,
        num_coefficients=10,
        patience=10,
    )

    print("Gamma", gamma)
    print("Beta", beta)
