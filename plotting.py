import numpy as np
import seaborn as sns
from scipy.optimize import curve_fit


def plot_with_errorbar(ax, xs: np.ndarray, means: np.ndarray, stds: np.ndarray, labels: tuple[str, str],
                       color: tuple[float, float, float], regplot: bool | str = False, set_xticks: bool = False, logy: bool = False):
    if logy:
        ax.set_yscale('log')

    if regplot:
        mask = np.isfinite(means)
        means = means[mask]
        stds = stds[mask]

    ax.plot(
        xs,
        means,
        alpha=0.7,
        linewidth=1.6,
        markersize=3.5,
        color=color
    )

    if set_xticks:
        ax.set_xticks(xs)

    ax.errorbar(
        xs,
        y=means,
        yerr=stds,
        fmt="none",
        capsize=3,
        color=color,
        alpha=0.7,
        linewidth=1.6
    )
    ax.set_ylabel(labels[0])
    ax.set_xlabel(labels[1])
    ax.grid(visible=True)

    if regplot:
        reg_logx = True if regplot == 'log' else False
        sns.regplot(x=xs, y=means, color=color, ax=ax, line_kws={'linestyle': "--", 'linewidth': 1.0}, ci=None,
                    logx=reg_logx, dropna=True)

def plot_per_load_factor(ax, xs: np.ndarray, means: np.ndarray, stds: np.ndarray,
                         load_factors, labels: tuple[str, str], palette: list[tuple[float, float, float]],
                         regplot: bool | str = False, set_xticks: bool = False, logy: bool = False) -> None:
    if logy:
        ax.set_yscale('log')

    if regplot:
        mask = np.isfinite(means).all(axis=0)
        means = means[:, mask]
        stds = stds[:, mask]
        xs = xs[mask]

    for i, (load_factor, mean, std) in enumerate(zip(load_factors, means, stds)):

        ax.plot(
            xs,
            mean,
            alpha=0.7,
            linewidth=1.6,
            markersize=3.5,
            color=palette[i],
            label=f'Load: {load_factor}'
        )

        if set_xticks:
            ax.set_xticks(xs)

        ax.errorbar(
            xs,
            y=mean,
            yerr=std,
            fmt="none",
            capsize=3,
            color=palette[i],
            alpha=0.7,
            linewidth=1.6
        )

        if regplot:
            reg_logx = True if regplot == 'log' else False
            sns.regplot(x=xs, y=mean, color=palette[i], ax=ax,
                        line_kws={'linestyle': "--", 'linewidth': 1.0}, ci=None, logx=reg_logx, dropna=True)

    ax.set_ylabel(labels[0])
    ax.set_xlabel(labels[1])
    ax.grid(visible=True)
    ax.legend()


def log_plot_with_fits(ax, xs: np.ndarray, means: np.ndarray, stds: np.ndarray,
                    labels: tuple[str, str], palette: list[tuple[float, float, float]], set_xticks: bool = False) -> None:
    ax.set_yscale('log')

    mask = np.isfinite(means)
    means = means.clip(min=1)[mask] # clip for log scale
    log_mean_clipped = np.log10(means)
    xs = xs[mask]
    stds = stds[mask]

    def linear_model(x, a, b):
        return a * x + b

    def log_model(x, a, b, c, d, n):
        arg = b * x + c
        arg = np.clip(arg, 1e-10, None)  # prevent log of zero or negative
        return a * (np.log(arg) / np.log(n)) + d

    ax.plot(
        xs,
        means,
        alpha=0.7,
        linewidth=1.6,
        markersize=3.5,
        color=palette[0],
    )

    if set_xticks:
        ax.set_xticks(xs)

    lower = np.clip(means - stds, 1, None)
    ax.errorbar(
        xs,
        y=means,
        yerr=[means - lower, stds], # prevent -inf values do to log scale
        fmt="none",
        capsize=3,
        color=palette[0],
        alpha=0.7,
        linewidth=1.6
    )

    popt_linear, pcov_linear = curve_fit(f=linear_model, xdata=xs, ydata=log_mean_clipped, nan_policy='omit')
    popt_log, pcov_log = curve_fit(
        f=log_model, xdata=xs, ydata=log_mean_clipped,
        bounds=(
            [-np.inf, 1e-6, -np.inf, -np.inf, 1.01],
            [np.inf, np.inf, np.inf, np.inf, np.inf]
        ),
        nan_policy='omit',
        maxfev=10000, p0=[1, 1, 1, 1, 10]
    )

    perr_linear = np.mean(np.sqrt(np.diag(pcov_linear)))
    perr_log = np.mean(np.sqrt(np.diag(pcov_log)))

    if perr_linear < perr_log:
        ax.plot(xs, 10**linear_model(xs, *popt_linear),
                      label='Linear fit: a=%5.3f, b=%5.3f' % tuple(popt_linear), linestyle='dashed')
    else:
        ax.plot(xs, 10**log_model(xs, *popt_log),
                      label='Log fit: a=%5.3f, b=%5.3f, c=%5.3f, d=%5.3f, n=%5.3f' % tuple(popt_log),
                      linestyle='dashed')

    ax.set_ylabel(labels[0])
    ax.set_xlabel(labels[1])
    ax.grid(visible=True)
    ax.legend()


def log_plot_per_load_factor_with_fits(ax, xs: np.ndarray, means: np.ndarray, stds: np.ndarray,
                         load_factors, labels: tuple[str, str], palette: list[tuple[float, float, float]], set_xticks: bool = False) -> None:
    ax.set_yscale('log')

    mask = np.isfinite(means).all(axis=0)
    means = means.clip(min=1)[:, mask] # clip for log scale
    log_means_clipped = np.log10(means)
    xs = xs[mask]
    stds = stds[:, mask]

    def linear_model(x, a, b):
        return a * x + b

    def log_model(x, a, b, c, d, n):
        arg = b * x + c
        arg = np.clip(arg, 1e-10, None)  # prevent log of zero or negative
        return a * (np.log(arg) / np.log(n)) + d

    for i, (load_factor, mean, std, log_mean_clipped) in enumerate(zip(load_factors, means, stds, log_means_clipped)):
        ax.plot(
            xs,
            mean,
            alpha=0.7,
            linewidth=1.6,
            markersize=3.5,
            color=palette[i],
            label=f'Load: {load_factor}'
        )

        if set_xticks:
            ax.set_xticks(xs)

        lower = np.clip(mean - std, 1, None)
        ax.errorbar(
            xs,
            y=mean,
            yerr=[mean - lower, std], # prevent -inf values do to log scale
            fmt="none",
            capsize=3,
            color=palette[i],
            alpha=0.7,
            linewidth=1.6
        )

        popt_linear, pcov_linear = curve_fit(f=linear_model, xdata=xs, ydata=log_mean_clipped, nan_policy='omit')
        popt_log, pcov_log = curve_fit(
            f=log_model, xdata=xs, ydata=log_mean_clipped,
            bounds=(
                [-np.inf, 1e-6, -np.inf, -np.inf, 1.01],
                [np.inf, np.inf, np.inf, np.inf, np.inf]
            ),
            nan_policy='omit',
            maxfev=10000, p0=[1, 1, 1, 1, 10]
        )

        perr_linear = np.mean(np.sqrt(np.diag(pcov_linear)))
        perr_log = np.mean(np.sqrt(np.diag(pcov_log)))

        if perr_linear < perr_log:
            ax.plot(xs, 10**linear_model(xs, *popt_linear),
                          label='Linear fit: a=%5.3f, b=%5.3f' % tuple(popt_linear), linestyle='dashed')
        else:
            ax.plot(xs, 10**log_model(xs, *popt_log),
                          label='Log fit: a=%5.3f, b=%5.3f, c=%5.3f, d=%5.3f, n=%5.3f' % tuple(popt_log),
                          linestyle='dashed')

    ax.set_ylabel(labels[0])
    ax.set_xlabel(labels[1])
    ax.grid(visible=True)
    ax.legend()