import ipywidgets
import matplotlib as mpl
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
import os

def plot_results(ps, zs, cov, actual=None, std_scale=1,
                 plot_P=True, y_lim=None, 
                 xlabel='time', ylabel='position',
                 title='Kalman Filter'):
    """
    Combines measurements and filter plots into a single figure 
    and saves covariance plots as a separate figure.
    """
    # Ensure the output folder exists
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_folder = os.path.join(script_dir, "kalmanplots")
    os.makedirs(output_folder, exist_ok=True)
    print(f"Saving plots to: {os.path.abspath(output_folder)}")
    
    count = len(zs)
    zs = np.asarray(zs)
    cov = np.asarray(cov)

    # Combine measurements and filter results in a single plot
    fig = plot_combined_measurements_and_filter(
        xs=range(1, count + 1),
        zs=zs,
        ps=ps,
        cov=cov,
        std_scale=std_scale,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        y_lim=y_lim
    )

    # Save combined plot
    output_file = os.path.join(output_folder, "combined_plot.png")
    fig.savefig(output_file)
    print(f"Saved combined plot to: {output_file}")
    plt.close(fig)

    # Create and save covariance plots
    if plot_P:
        fig = plot_covariance_fig(cov)
        output_file = os.path.join(output_folder, "covariance_plots.png")
        fig.savefig(output_file)
        print(f"Saved covariance plots to: {output_file}")
        plt.close(fig)


def plot_combined_measurements_and_filter(xs, zs, ps, cov, std_scale=1, 
                                          xlabel='time', ylabel='position', 
                                          title='Kalman Filter', y_lim=None,  figsize=(12, 8)):
    """
    Creates a combined plot with measurements, filter results, and optional variance shading.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot measurements
    plot_measurements(xs, zs, ax=ax, label='Measurements', color='k', lines=True)

    # Plot filter
    plot_filter(xs, ps, ax=ax, label='Filter', color='C0', std_scale=std_scale) # might do something like var=cov[:, 0, 0] but need to figure out which var etc

    # Customize axes and legend
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    if y_lim is not None:
        ax.set_ylim(y_lim)

    return fig


def plot_covariance_fig(cov):
    """
    Creates a figure with subplots for position and velocity variance.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Position variance
    axes[0].set_title(r"$\sigma^2_x$ (pos variance)")
    plot_covariance(cov, index=(0, 0), ax=axes[0])

    # Velocity variance
    axes[1].set_title(r"$\sigma^2_\dot{x}$ (vel variance)")
    plot_covariance(cov, index=(1, 1), ax=axes[1])

    return fig


def plot_measurements(xs, ys=None, ax=None, color='k', lw=1, label='Measurements', 
                      lines=False, **kwargs):
    """
    Helper to plot measurements on a given axis or a new figure.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    if ys is None:
        ys = xs
        xs = range(len(ys))

    if lines:
        ax.plot(xs, ys, color=color, lw=lw, ls='--', label=label, **kwargs)
    else:
        ax.scatter(xs, ys, edgecolor=color, facecolor='none', lw=2, label=label, **kwargs)

    return fig


def plot_filter(xs, ys=None, ax=None, color='C0', label='Filter', var=None, std_scale=1, **kwargs):
    """
    Helper to plot filter results with optional variance shading on a given axis or new figure.
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.get_figure()

    if ys is None:
        ys = xs
        xs = range(len(ys))

    # Plot the main filter line
    ax.plot(xs, ys, color=color, label=label, **kwargs)

    # Optionally add variance shading
    if var is not None:
        std = std_scale * np.sqrt(var)
        std_top = ys + std
        std_btm = ys - std

        ax.fill_between(xs, std_btm, std_top, facecolor='green', alpha=0.2, label="Variance")

    return fig


def plot_covariance(P, index=(0, 0), ax=None):
    """
    Plot covariance values at the specified index on the given axis.
    """
    ps = [p[index[0], index[1]] for p in P]
    if ax is None:
        ax = plt.gca()
    ax.plot(ps)
    ax.set_xlabel("Time")
    ax.set_ylabel("Covariance")