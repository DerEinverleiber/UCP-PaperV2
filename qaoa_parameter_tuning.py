import numpy as np

import q_optimizer


def to_chebyshev_basis_coefficients(gamma: np.ndarray, beta: np.ndarray, depth: int) -> tuple[np.ndarray, np.ndarray]:
    # adapted from: https://github.com/jpmorganchase/QOKit/blob/main/qokit/parameter_utils.py
    fit_interval = np.linspace(-1, 1, len(gamma))
    u = np.polynomial.chebyshev.chebfit(fit_interval, gamma,
                                        deg=depth - 1)  # offset of 1 due to fitting convention
    v = np.polynomial.chebyshev.chebfit(fit_interval, beta, deg=depth - 1)
    return u, v


def optimize_chebyshev_basis_coefficients(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # use some optimization algorithm (I guess) to find better gamma_cheby, beta_cheby (probably need loss here??)
    return u, v


def to_qaoa_angles(u: np.ndarray, v: np.ndarray, p: int) -> tuple[np.ndarray, np.ndarray]:
    # adapted from: https://github.com/jpmorganchase/QOKit/blob/main/qokit/parameter_utils.py
    fit_interval = np.linspace(-1, 1, p)
    gamma = np.polynomial.chebyshev.chebval(fit_interval, u)
    beta = np.polynomial.chebyshev.chebval(fit_interval, v)
    return gamma, beta


def interpolate_schedule(gamma: np.ndarray, beta: np.ndarray, delta_p: int):
    # transform gamma and beta to have len(gamma) + delta_p entries (extrapolate schedule)
    return np.concatenate((gamma, [1] * delta_p)), np.concatenate((beta, [1] * delta_p))


def iterative_interpolation(
        qaoa: q_optimizer,
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

    gamma, beta = qaoa.get_angles()

    while p <= p_max: # implement something like this
        u, v = to_chebyshev_basis_coefficients(gamma, beta, depth=num_coefficients)
        u, v = optimize_chebyshev_basis_coefficients(u, v)
        gamma, beta = to_qaoa_angles(u, v, p)
        _, best_loss = qaoa.run(gamma, beta)
        approximation_ratio = -1 # not yet implemented - do we use the brute force data to calculate this?

        if approximation_ratio >= desired_approx_ratio:
            return gamma, beta

        if approximation_ratio - current_approximation_ratio < improvement_threshold:
            patience_counter += 1
            if patience_counter == patience:
                num_coefficients += 1

        gamma, beta = interpolate_schedule(gamma, beta, delta_p=delta_p)
        p += delta_p

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

    gamma, beta = iterative_interpolation(
        qaoa=qaoa,
        p_0=1,
        delta_p=1,
        p_max=10,
        improvement_threshold=0.000001,
        num_coefficients=10,
        patience=10,
    )

    print("Gamma", gamma)
    print("Beta", beta)
