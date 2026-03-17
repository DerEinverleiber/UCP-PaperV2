import q_optimizer


def to_chebychev_basis_coefficients(gamma: float, beta: float, depth: int) -> tuple[list[float], list[float]]:
    # return first {depth} coefficients of gamma and beta in the chebychev basis
    # use system of linear equations Ax = b with x being cheby coefficients, b qaoa angles and A cheby basis function evaluations
    # A_i_j = f_j(i/p)
    # probably solving two (one for beta, one for gamma) or a large one
    pass


def optimize_chebychev_basis_coefficients(gamma_cheby, beta_cheby):
    # use some optimization algorithm (I guess) to find better gamma_cheby, beta_cheby (probably need loss here??)
    pass


def to_qaoa_angles(gamma_cheby, beta_cheby):
    # return sum{i}(coefficient_i * cheby_chev_basis_function_i)
    pass


def interpolate_schedule(gamma, beta, delta_p):
    # transform gamma and beta to have len(gamma) + delta_p entries (extrapolate schedule)
    pass


def iterative_interpolation(
        qaoa: q_optimizer,
        p_0: int,
        delta_p: int,
        p_max: int,
        improvement_threshold: float,
        num_coefficients: int,
        patience: int,
        desired_approx_ratio: float
) -> tuple[list[float], list[float]]:
    patience_counter = 0
    p = p_0
    current_approximation_ratio = 0

    gamma, beta = qaoa.get_angles()

    while p <= p_max: # implement something like this
        gamma_cheby, beta_cheby = to_chebychev_basis_coefficients(gamma, beta, depth=num_coefficients)
        gamma_cheby, beta_cheby = optimize_chebychev_basis_coefficients(gamma_cheby, beta_cheby)
        gamma, beta = to_qaoa_angles(gamma_cheby, beta_cheby)
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