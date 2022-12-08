"""Visualization functions."""

import numpy as np
from matplotlib import pyplot as plt
from src.decision_tree import DecisionTreeNode


def _tree_depth(tree_root):
    if isinstance(tree_root, DecisionTreeNode):
        return max(_tree_depth(tree_root.left), _tree_depth(tree_root.right)) + 1
    return 1


def _draw_tree_rec(tree_root, x_left, x_right, y_all):
    x_center = (x_right - x_left) / 2 + x_left
    if isinstance(tree_root, DecisionTreeNode):
        x_center = (x_right - x_left) / 2 + x_left
        x_left = _draw_tree_rec(tree_root.left, x_left, x_center, y_all - 1)
        plt.plot((x_center, x_left), (y_all - 0.1, y_all - 0.9), c=(0, 0, 0))
        x_right = _draw_tree_rec(tree_root.right, x_center, x_right, y_all - 1)
        plt.plot((x_center, x_right), (y_all - 0.1, y_all - 0.9), c=(0, 0, 0))
        plt.text(
            x_center,
            y_all,
            "x[%i] < %f" % (tree_root.split_dim, tree_root.split_value),
            horizontalalignment="center",
        )
    else:
        plt.text(x_center, y_all, str(tree_root.y), horizontalalignment="center")
    return x_center


def draw_tree(tree, save_path=None):
    """Tree visualization."""
    td = _tree_depth(tree.root)
    plt.figure(figsize=(0.33 * 2**td, 2 * td))
    plt.xlim(-1, 1)
    plt.ylim(0.95, td + 0.05)
    plt.axis("off")
    _draw_tree_rec(tree.root, -1, 1, td)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.show()


def plot_roc_curve(y_test, p_pred):
    """Roc curve visualization."""
    positive_samples = sum(1 for y in y_test if y == 0)
    tpr = []
    fpr = []
    for segment in np.arange(-0.01, 1.02, 0.01):
        y_pred = [(0 if p.get(0, 0) > segment else 1) for p in p_pred]
        tpr.append(
            sum(1 for yp, yt in zip(y_pred, y_test) if yp == 0 and yt == 0)
            / positive_samples
        )
        fpr.append(
            sum(1 for yp, yt in zip(y_pred, y_test) if yp == 0 and yt != 0)
            / (len(y_test) - positive_samples)
        )
    plt.figure(figsize=(7, 7))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.xlim(-0.01, 1.01)
    plt.ylim(-0.01, 1.01)
    plt.tight_layout()
    plt.show()


def _rectangle_bounds(bounds):
    return (
        (bounds[0][0], bounds[0][0], bounds[0][1], bounds[0][1]),
        (bounds[1][0], bounds[1][1], bounds[1][1], bounds[1][0]),
    )


def _plot_2d_tree(tree_root, bounds, colors):
    if isinstance(tree_root, DecisionTreeNode):
        if tree_root.split_dim:
            _plot_2d_tree(
                tree_root.left,
                [bounds[0], [bounds[1][0], tree_root.split_value]],
                colors,
            )
            _plot_2d_tree(
                tree_root.right,
                [bounds[0], [tree_root.split_value, bounds[1][1]]],
                colors,
            )
            plt.plot(
                bounds[0], (tree_root.split_value, tree_root.split_value), c=(0, 0, 0)
            )
        else:
            _plot_2d_tree(
                tree_root.left,
                [[bounds[0][0], tree_root.split_value], bounds[1]],
                colors,
            )
            _plot_2d_tree(
                tree_root.right,
                [[tree_root.split_value, bounds[0][1]], bounds[1]],
                colors,
            )
            plt.plot(
                (tree_root.split_value, tree_root.split_value), bounds[1], c=(0, 0, 0)
            )
    else:
        x, y = _rectangle_bounds(bounds)
        plt.fill(x, y, c=colors[tree_root.y] + [0.2])


def plot_2d(tree, x_data, y_data):
    """2d-Tree visualization."""
    plt.figure(figsize=(9, 9))
    colors = dict((c, list(np.random.random(3))) for c in np.unique(y_data))
    bounds = list(zip(np.min(x_data, axis=0), np.max(x_data, axis=0)))
    plt.xlim(*bounds[0])
    plt.ylim(*bounds[1])
    _plot_2d_tree(
        tree.root, list(zip(np.min(x_data, axis=0), np.max(x_data, axis=0))), colors
    )
    for value in np.unique(y_data):
        plt.scatter(
            x_data[y_data == value, 0],
            x_data[y_data == value, 1],
            c=[colors[value]],
            label=value,
        )
    plt.legend()
    plt.tight_layout()
    plt.show()
