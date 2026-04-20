import numpy as np
import seaborn as sns


def plot_with_errorbar(ax, xs: np.ndarray, means: np.ndarray, stds: np.ndarray, labels: tuple[str, str],
                       color: tuple[int, int, int], regplot: bool = False):
    ax.plot(
        xs,
        means,
        alpha=0.7,
        linewidth=1.6,
        markersize=3.5,
        color=color
    )

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
        sns.regplot(x=xs, y=means, color=color, ax=ax, line_kws={'linestyle': "--", 'linewidth': 1.0}, ci=None,
                    logx=True)

def plot_per_load_factor(ax, xs: np.ndarray, means: np.ndarray, stds: np.ndarray, load_factors, labels: tuple[str, str], palette: list[tuple[int, int, int]], regplot: bool = False) -> None:
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
            sns.regplot(x=xs, y=mean, color=palette[i], ax=ax,
                        line_kws={'linestyle': "--", 'linewidth': 1.0}, ci=None, logx=True)

    ax.set_ylabel(labels[0])
    ax.set_xlabel(labels[1])
    ax.grid(visible=True)
    ax.legend()