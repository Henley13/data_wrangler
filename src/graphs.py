# -*- coding: utf-8 -*-

""" Different functions to plot """

# libraries
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import manifold
from sklearn.decomposition import TruncatedSVD
from toolbox.utils import (get_config_tag, load_sparse_csr, load_numpy_matrix,
                           _check_graph_folders)
print("\n")


def plot_extension_cleaned(result_directory, df_log):
    """
    Function to plot extension distribution among cleaned files
    :param result_directory: string
    :param df_log: pandas
    :param df_log: pandas Dataframe
    :return:
    """
    print("plot extension distribution", "\n")

    # paths
    path_png = os.path.join(result_directory, "graphs", "png")
    path_pdf = os.path.join(result_directory, "graphs", "pdf")
    path_jpeg = os.path.join(result_directory, "graphs", "jpeg")
    path_svg = os.path.join(result_directory, "graphs", "svg")

    # settings
    sns.set(style="white",
            color_codes=True,
            palette="muted",
            font='sans-serif',
            font_scale=1,
            rc=None)

    # plot
    data_counts = df_log['extension'].value_counts()
    extension_names = data_counts.index
    extension_counts = data_counts.get_values()
    y_pos = np.arange(len(extension_names))
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.set_xlabel("Number of files (log scale)", fontsize=7)
    ax.set_position([0.35, 0.15, 0.6, 0.8])  # left, bottom, witdh, height
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(extension_names)
    ax.invert_yaxis()
    ax.barh(bottom=y_pos,
            width=extension_counts,
            align='center',
            color='darkcyan',
            edgecolor="black",
            alpha=0.8,
            fill=True,
            log=True)
    for i, v in enumerate(y_pos):
        ax.text(3, i + 0.1, str(extension_counts[i]), color='black', fontsize=7)

    # save figures
    path = os.path.join(path_jpeg, "extension.jpeg")
    plt.savefig(path)
    path = os.path.join(path_pdf, "extension.pdf")
    plt.savefig(path)
    path = os.path.join(path_png, "extension.png")
    plt.savefig(path)
    path = os.path.join(path_svg, "extension.svg")
    plt.savefig(path)

    plt.close("all")

    return


def plot_size_cleaned(result_directory, df_log):
    """
    Function to plot size distribution of cleaned files
    :param result_directory: string
    :param df_log: pandas Dataframe
    :return:
    """
    print("plot size distribution", "\n")

    # paths
    path_png = os.path.join(result_directory, "graphs", "png")
    path_pdf = os.path.join(result_directory, "graphs", "pdf")
    path_jpeg = os.path.join(result_directory, "graphs", "jpeg")
    path_svg = os.path.join(result_directory, "graphs", "svg")

    # settings
    sns.set(style="white",
            color_codes=True,
            palette="muted",
            font='sans-serif',
            font_scale=1,
            rc=None)

    # plots with histograms
    if True:
        sns.set_context("paper",
                        rc={"font.size": 7,
                            "axes.titlesize": 7,
                            "axes.labelsize": 7})
        g = sns.JointGrid(x="n_col",
                          y="n_row",
                          data=df_log,
                          size=4,
                          ratio=5,
                          space=0.2,
                          dropna=True,
                          xlim=[0.9, max(df_log["n_col"]) * 2],
                          ylim=[0.1, max(df_log["n_row"]) * 2])
        g.plot_joint(plt.scatter,
                     color='darkcyan',
                     edgecolor="black",
                     s=15,
                     alpha=0.8)
        mybins_log = np.logspace(0, np.log(100), 60)
        g.plot_marginals(sns.distplot,
                         hist=True,
                         kde=False,
                         rug=True,
                         color='darkcyan',
                         bins=mybins_log)
        ax = g.ax_joint
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel("Number of columns (log scale)", fontsize=7)
        ax.set_ylabel("Number of rows (log scale)", fontsize=7)
        ax_margx = g.ax_marg_x
        ax_margy = g.ax_marg_y
        ax_margx.set_xscale('log')
        ax_margy.set_yscale('log')

        # save figures
        path = os.path.join(path_jpeg, "size_histo.jpeg")
        g.savefig(path)
        path = os.path.join(path_pdf, "size_histo.pdf")
        g.savefig(path)
        path = os.path.join(path_png, "size_histo.png")
        g.savefig(path)
        path = os.path.join(path_svg, "size_histo.svg")
        g.savefig(path)

    # plots with color
    if True:
        groups = df_log.groupby("extension")
        # plot
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Number of columns (log scale)", fontsize=7)
        ax.set_ylabel("Number of rows (log scale)", fontsize=7)
        ax.set_position([0.12, 0.14, 0.8, 0.8]) # left, bottom, witdh, hight
        # TODO get the legend back
        plt.xticks(fontsize=7)
        plt.yticks(fontsize=7)
        for name, group in groups:
            ax.scatter(x=group.n_col,
                       y=group.n_row,
                       s=17,
                       c=None,
                       marker="o",
                       cmap=None,
                       norm=None,
                       vmin=None,
                       vmax=None,
                       alpha=0.8,
                       linewidths=None,
                       verts=None,
                       edgecolors=None,
                       label=name)
        plt.legend()
        ax.legend(prop={'size': 7})
        # save figures
        path = os.path.join(path_jpeg, "size.jpeg")
        plt.savefig(path)
        path = os.path.join(path_pdf, "size.pdf")
        plt.savefig(path)
        path = os.path.join(path_png, "size.png")
        plt.savefig(path)
        path = os.path.join(path_svg, "size.svg")
        plt.savefig(path)

    plt.close("all")

    return


def prepocessed_tsne(result_directory, df_log):
    """
    Function to preprocess data in order to plot a t-SNE embedding of the count
    matrix
    :param result_directory: string
    :param df_log: pandas Dataframe
    :return: numpy array [n_samples, n_components], pandas Dataframe
    """
    print("preprocessed t-SNE embedding", "\n")

    # path
    path_count = os.path.join(result_directory, "count_normalized_stem.npz")
    path_graph = os.path.join(result_directory, "graphs")

    # get data
    X = load_sparse_csr(path_count)
    print("X shape :", X.shape)

    # dimensionality reduction
    tsvd = TruncatedSVD(n_components=30,
                        algorithm='randomized',
                        n_iter=5,
                        random_state=0,
                        tol=0.0)
    X_reduced = tsvd.fit_transform(X)
    path = os.path.join(result_directory, "count_tsvd")
    np.savetxt(path, X_reduced, delimiter="\t")
    path = os.path.join(path_graph, "count_tsvd")
    np.savetxt(path, X_reduced, delimiter="\t")
    print("X reduced shape :", X_reduced.shape)

    # preprocessed metadata
    df = df_log[["extension", "zipfile", "title_page", "title_producer"]]
    path = os.path.join(result_directory, "df_metadata")
    df.to_csv(path, sep="\t", encoding="utf-8", index=False, header=True)
    path = os.path.join(path_graph, "df_metadata")
    df.to_csv(path, sep="\t", encoding="utf-8", index=False, header=True)
    print("metadata shape :", df.shape, "\n")

    return X_reduced, df


def plot_tsne(result_directory, df_log):
    return


def plot_mds(result_directory, df_log):
    """
        Function to compute and plot a MDS embedding of the count matrix
        :param result_directory:
        :param df_log: pandas Dataframe
        :return:
        """
    print("computing MDS embedding", "\n")

    # paths
    path_png = os.path.join(result_directory, "graphs", "png")
    path_pdf = os.path.join(result_directory, "graphs", "pdf")
    path_jpeg = os.path.join(result_directory, "graphs", "jpeg")
    path_svg = os.path.join(result_directory, "graphs", "svg")
    path_count = os.path.join(result_directory, "count.npz")

    # get data
    X = load_sparse_csr(path_count)
    print("X shape :", X.shape)

    # MDS embedding of the count matrix
    mds = manifold.MDS(n_components=2,
                       metric=True,
                       n_init=4,
                       max_iter=300,
                       verbose=10,
                       eps=0.001,
                       n_jobs=10,
                       random_state=0,
                       dissimilarity='euclidean')
    X_mds = mds.fit_transform(X.todense())
    path = os.path.join(result_directory, "x_mds.npy")
    np.save(path, X_mds)
    print("X MDS shape :", X_mds.shape)

    # plot X_mds
    df_mds = pd.DataFrame({"component_1": X_mds[:, 0],
                            "component_2": X_mds[:, 1],
                            "extension": df_log["extension"]})
    groups = df_mds.groupby("extension")
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.set_xlabel("")
    ax.set_ylabel("")
    # ax.set_position([0.12, 0.14, 0.8, 0.8])  # left, bottom, witdh, hight
    plt.xticks([], [])
    plt.yticks([], [])
    for name, group in groups:
        ax.scatter(x=group.component_1,
                   y=group.component_2,
                   s=10,
                   c=None,
                   marker="o",
                   cmap=None,
                   norm=None,
                   vmin=None,
                   vmax=None,
                   alpha=0.8,
                   linewidths=None,
                   verts=None,
                   edgecolors=None,
                   label=name)
    plt.legend()
    ax.legend(prop={'size': 7})

    # save figures
    path = os.path.join(path_jpeg, "X_MDS.jpeg")
    plt.savefig(path)
    path = os.path.join(path_pdf, "X_MDS.pdf")
    plt.savefig(path)
    path = os.path.join(path_png, "X_MDS.png")
    plt.savefig(path)
    path = os.path.join(path_svg, "X_MDS.svg")
    plt.savefig(path)

    plt.close("all")

    return


def plot_score_nmf(result_directory):
    return


def make_table_source_topic(result_directory):
    return


def main(result_directory):
    # paths
    path_log = os.path.join(result_directory, "log_final")

    # get data
    df_log = pd.read_csv(path_log, header=0, encoding="utf-8", sep=";",
                         index_col=False)
    print("log length :", df_log.shape[0], "\n")

    # check folders
    _check_graph_folders(result_directory)

    # preprocessed data
    prepocessed_tsne(result_directory, df_log)

    # table
    make_table_source_topic(result_directory)

    # plot
    plot_extension_cleaned(result_directory, df_log)
    plot_size_cleaned(result_directory, df_log)
    # plot_mds(result_directory, df_log)
    # plot_score_nmf(result_directory)

    return

if __name__ == "__main__":

    # paths
    result_directory = get_config_tag("result", "cleaning")

    # run
    main(result_directory)
