import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_context('notebook')


# Generic scatter plot. X is a two column matrix, each row corresponding
# to a single data point.
def scatter_2d(X, labels=None,
               show=True, title=None, tight=True):
    fig, ax = plt.subplots(figsize=(16, 16))
    fig = sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=labels, ax=ax)

    if title is not None:
        fig.set_title(title)
    if tight:
        plt.tight_layout()
    if show:
        plt.show()


# `data` can be any T-by-D matrix representing a D-channel time-series
# running for T steps. All other arguments control some display feature.
def multi_channel_plot(data,
                       show=True,
                       title=None,
                       tight=True,
                       context=None,
                       overlay=False,
                       save_to=None,
                       figsize=(16, 4),
                       standardize=False):
    sns.set_context(context)

    if len(data.shape) == 1:
        data = data.reshape([-1, 1])

    n, d = data.shape

    if d == 1:
        overlay = True

    dim = 1 if overlay else d
    ts = range(n)
    fig_x, fig_y = figsize
    fig, axes = plt.subplots(dim, 1,
                             figsize=(fig_x, fig_y*dim),
                             squeeze=False,
                             sharex=True,
                             sharey=True)
    if title is not None:
        fig.suptitle(title)

    if standardize:
        data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

    channel_palette = sns.hls_palette(n_colors=d)
    for j in range(d):
        channel = data[:, j]
        color = channel_palette[j]
        if not overlay:
            ax = axes[j, 0]
            ax.set_title('Channel {}'.format(j+1))
        else:
            ax = axes[0, 0]
        sns.lineplot(x=ts, y=channel, ax=ax, color=color)

    sns.set_style('ticks')

    if tight:
        plt.tight_layout()

    if save_to is not None:
        plt.savefig(save_to)

    if show:
        plt.show()


# Display the RCs `R`, which is the output of `CMSSA.transform(X, collapse=False)`.
def plot_rcs(R, show=True, title=None, tight=False,
             show_sum=True,
             hide_rcs=False,
             up_to=float('inf'),
             save_to=None):
    if hide_rcs and not show_sum:
        raise Exception('i need to show *something*, right?')

    num_comp, n, d = R.shape
    d = min([d, up_to])
    ts = range(n)

    bit = 1 if show_sum else 0
    nc = 0 if hide_rcs else num_comp

    fig, axes = plt.subplots(nc+bit, d,
                             figsize=(16, 3*(nc+bit)),
                             squeeze=False,
                             sharex=True,
                             sharey=True)
    if title is not None:
        fig.suptitle(title)
    R_sum = np.sum(R, axis=0)

    channel_palette = sns.hls_palette(n_colors=d)
    for j in range(d):
        color = channel_palette[j]
        palette = sns.dark_palette(color, n_colors=nc+bit, reverse=True)

        if show_sum:
            ax = axes[0, j]
            ax.set_title('Summed RCs 1-{}, Channel {}'.format(num_comp, j+1))
            sns.lineplot(x=ts, y=R_sum[:, j],
                         ax=ax,
                         color=palette[0])

        if hide_rcs:
            continue

        for k in range(num_comp):
            ax = axes[k+bit, j]
            ax.set_title('RC {}, Channel {}'.format(k+1, j+1))
            sns.lineplot(x=ts, y=R[k, :, j],
                         ax=ax,
                         color=palette[k+bit])

    sns.set_style('ticks')

    if tight:
        plt.tight_layout()

    if save_to is not None:
        plt.savefig(save_to)

    if show:
        plt.show()
