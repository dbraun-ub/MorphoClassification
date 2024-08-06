import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Polygon


def scatterplot_matrix(data, names, figsize=(15, 15), fontsize=14, **kwargs):
    """
    Adapted from https://stackoverflow.com/questions/7941207/\
    is-there-a-function-to-make-scatterplot-matrices-in-matplotlib

    Plots a scatterplot matrix of subplots. Each row of "data" is plotted
    against other rows, resulting in a nrows by nrows grid of subplots with the
    diagonal subplots labeled with "names". Additional keyword arguments are
    passed on to matplotlib's "plot" command. Returns the matplotlib figure
    object containing the subplot grid.
    """
    numvars = data.shape[1]
    fig, axes = plt.subplots(nrows=numvars, ncols=numvars, figsize=figsize)
    fig.subplots_adjust(hspace=0.05, wspace=0.05)

    # Simplify tick settings and visibility
    for ax in axes.flat:
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

    # Plot the data and annotate diagonal
    for i in range(numvars):
        for j in range(numvars):
            ax = axes[i, j]
            if i == j:
                ax.annotate(
                    names[i],
                    (0.5, 0.5),
                    xycoords="axes fraction",
                    ha="center",
                    va="center",
                    fontsize=fontsize,
                )
            else:
                ax.plot(data[:, j], data[:, i], **kwargs)
                ax.xaxis.set_visible(j == numvars - 1)
                ax.yaxis.set_visible(i == 0)

    # Adjust tick positions for the edge subplots only
    for ax in axes[:, 0]:  # First column
        ax.yaxis.set_ticks_position("left")
    for ax in axes[:, -1]:  # Last column
        ax.yaxis.set_ticks_position("right")
    for ax in axes[0, :]:  # First row
        ax.xaxis.set_ticks_position("top")
    for ax in axes[-1, :]:  # Last row
        ax.xaxis.set_ticks_position("bottom")

    return fig


def my_plotshapes(
    data,
    joins,
    ax,
    landmark=True,
    landmark_label=False,
    joinline=False,
    closed=True,
    marker="o",
    markersize=5,
    color="black",
    alpha=1,
    invert_y=False,
    x_label=None,
    y_label=None,
    title=None,
):

    joins_idx = joins - 1
    data = data[joins_idx]
    labels = joins

    n_shapes = data.shape[2]

    for i in range(n_shapes):
        for j, array in enumerate(data):
            if landmark:
                ax.plot(
                    array[0][i],
                    array[1][i],
                    marker=marker,
                    mfc="none",
                    color=color,
                    markersize=markersize,
                    alpha=alpha,
                )

            else:
                ax.scatter(array[0][i], array[1][i], facecolors="none")

            if landmark_label:
                ax.annotate(
                    str(labels[j]), (array[0][i], array[1][i]), ha="right", va="bottom",
                )

        if joinline:
            ax.add_patch(Polygon(data[:, :, i], closed=closed, fill=False))

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)
    if invert_y:
        plt.gca().invert_yaxis()


def plot_rotated_conf_mat(conf_mat, proc):
    plt.figure(figsize=[10, 10])

    ax = plt.subplot(121)
    for i in range(conf_mat.shape[0]):
        ax.scatter(conf_mat[i, 0, :], conf_mat[i, 1, :], alpha=0.1)
    plt.gca().invert_yaxis()

    ax = plt.subplot(122)
    for i in range(conf_mat.shape[0]):
        ax.scatter(proc.rotated[i, 0, :], proc.rotated[i, 1, :], alpha=0.1)

    ax.scatter(proc.mshape[:, 0], proc.mshape[:, 1], marker="*", c="black", s=80)

    plt.gca().invert_yaxis()
    plt.show()


def plot_raw_conf_mat(conf_mat, alpha=0.1):
    plt.figure(figsize=[10, 10])
    for i in range(conf_mat.shape[0]):
        plt.scatter(conf_mat[i, 0, :], conf_mat[i, 1, :], alpha=alpha)
    # plt.ylim(400,800)
    plt.gca().invert_yaxis()
    plt.show()


def plot_decision_regions(X, y, classifier, resolution=0.02):
    """
    The function is defined in Raschka & Mirjalili (2019) 3rd Ed., Ch. 2, 3 and 5.
    """

    # setup marker generator and color map
    markers = ("s", "x", "o", "^", "v")
    colors = ("red", "blue", "lightgreen", "gray", "cyan")
    cmap = ListedColormap(colors[: len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution)
    )
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot examples by class
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(
            x=X[y == cl, 0],
            y=X[y == cl, 1],
            alpha=0.6,
            color=cmap(idx),
            edgecolor="black",
            marker=markers[idx],
            label=cl,
        )
