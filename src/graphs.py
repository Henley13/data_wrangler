# -*- coding: utf-8 -*-

""" Different functions to plot """

# libraries
import os
import pandas as pd
import numpy as np
import seaborn as sns
import random
import joblib
from tqdm import tqdm
from sklearn import manifold
from sklearn.decomposition import TruncatedSVD
from toolbox.utils import (get_config_tag, load_sparse_csr, check_graph_folders)
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
print("\n")


def _get_path_cachedir(result_directory):
    """
    Function to get the path of the cache directory
    :param result_directory: string
    :return: string
    """
    path_cache = os.path.join(result_directory, "cache")
    if os.path.isdir(path_cache):
        pass
    else:
        os.mkdir(path_cache)
    return path_cache


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
    extension_names = []
    extension_counts = data_counts.get_values()
    for i in range(len(data_counts.index)):
        name = data_counts.index[i]
        count = extension_counts[i]
        extension_names.append(name + " (%i)" % count)
    y_pos = np.arange(len(extension_names))
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlabel("Number of files (log scale)", fontsize=15)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=10)
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
    # for i, v in enumerate(y_pos):
    #     ax.text(800, i + 0.1, str(extension_counts[i]), color='black',
    #             fontsize=10)
    plt.tight_layout()

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

    # plot with histograms
    if True:
        sns.set_context("paper",
                        rc={"font.size": 7,
                            "axes.titlesize": 7,
                            "axes.labelsize": 7})
        g = sns.JointGrid(x="n_col",
                          y="n_row",
                          data=df_log,
                          size=6,
                          ratio=5,
                          space=0.,
                          dropna=True,
                          xlim=[0.5, max(df_log["n_col"]) * 2],
                          ylim=[0.5, max(df_log["n_row"]) * 2])
        g.plot_joint(plt.scatter,
                     color='#4CB391',
                     edgecolor="black",
                     s=15,
                     alpha=0.8,
                     marker='.')
        mybins_log = np.logspace(0, np.log(100), 60)
        g.plot_marginals(sns.distplot,
                         hist=True,
                         kde=False,
                         rug=True,
                         color='#4CB391',
                         bins=mybins_log)
        ax = g.ax_joint
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel("Number of columns (log scale)", fontsize=15)
        ax.set_ylabel("Number of rows (log scale)", fontsize=15)
        ax_margx = g.ax_marg_x
        ax_margy = g.ax_marg_y
        ax_margx.set_xscale('log')
        ax_margy.set_yscale('log')
        plt.tight_layout()

        # save figures
        path = os.path.join(path_jpeg, "size_histo.jpeg")
        g.savefig(path)
        path = os.path.join(path_pdf, "size_histo.pdf")
        g.savefig(path)
        path = os.path.join(path_png, "size_histo.png")
        g.savefig(path)
        path = os.path.join(path_svg, "size_histo.svg")
        g.savefig(path)

    # plot with hexbin
    if True:
        sns.set_context("paper",
                        rc={"font.size": 7,
                            "axes.titlesize": 7,
                            "axes.labelsize": 7})
        mybins_log = np.logspace(0, np.log(100), 60)
        g = sns.jointplot(x=df_log["n_col"].apply(np.log),
                          y=df_log["n_row"].apply(np.log),
                          size=6,
                          ratio=5,
                          space=0.,
                          kind="hex",
                          stat_func=spearmanr,
                          color="#4CB391",
                          marginal_kws={"hist": True,
                                        "kde": False,
                                        "rug": True,
                                        "color": '#4CB391',
                                        "bins": mybins_log},
                          xlim=[np.log(0.5), np.log(max(df_log["n_col"])) * 2],
                          ylim=[np.log(0.5), np.log(max(df_log["n_row"])) * 2])
        ax = g.ax_joint
        ax.set_xlabel("Number of columns (log scale)", fontsize=15)
        ax.set_ylabel("Number of rows (log scale)", fontsize=15)
        plt.tight_layout()

        # save figures
        path = os.path.join(path_jpeg, "size_hexbin.jpeg")
        g.savefig(path)
        path = os.path.join(path_pdf, "size_hexbin.pdf")
        g.savefig(path)
        path = os.path.join(path_png, "size_hexbin.png")
        g.savefig(path)
        path = os.path.join(path_svg, "size_hexbin.svg")
        g.savefig(path)

    # plot with color
    if True:
        groups = df_log.groupby("extension")
        # plot
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Number of columns (log scale)", fontsize=15)
        ax.set_ylabel("Number of rows (log scale)", fontsize=15)
        plt.xticks(fontsize=7)
        plt.yticks(fontsize=7)
        for name, group in groups:
            ax.scatter(x=group.n_col,
                       y=group.n_row,
                       s=17,
                       c=None,
                       marker=".",
                       cmap=None,
                       norm=None,
                       vmin=None,
                       vmax=None,
                       alpha=1,
                       linewidths=None,
                       verts=None,
                       edgecolors=None,
                       label=name)
        plt.legend()
        ax.legend(prop={'size': 13})
        plt.tight_layout()

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
    df = df_log[["extension", "zipfile", "title_page", "title_producer",
                 "tags_page"]]
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
        Function to compute and plot a MDS embedding of the topic matrix
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
    path_w = os.path.join(result_directory, "w.npy")

    # get data
    W = np.load(path_w)
    print("W shape :", W.shape)

    # MDS embedding of the topic matrix
    mds = manifold.MDS(n_components=2,
                       metric=True,
                       n_init=4,
                       max_iter=300,
                       verbose=10,
                       eps=0.001,
                       n_jobs=10,
                       random_state=0,
                       dissimilarity='euclidean')
    W_mds = mds.fit_transform(W)
    path = os.path.join(result_directory, "W_mds.npy")
    np.save(path, W_mds)
    print("W MDS shape :", W_mds.shape)

    # plot W_mds
    df_mds = pd.DataFrame({"component_1": W_mds[:, 0],
                            "component_2": W_mds[:, 1],
                            "extension": df_log["extension"]})
    groups = df_mds.groupby("extension")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.xticks([], [])
    plt.yticks([], [])
    for name, group in groups:
        ax.scatter(x=group.component_1,
                   y=group.component_2,
                   s=10,
                   c=None,
                   marker=".",
                   cmap=None,
                   norm=None,
                   vmin=None,
                   vmax=None,
                   alpha=1,
                   linewidths=None,
                   verts=None,
                   edgecolors=None,
                   label=name)
    plt.legend()
    ax.legend(prop={'size': 13})
    plt.tight_layout()

    # save figures
    path = os.path.join(path_jpeg, "W_mds.jpeg")
    plt.savefig(path)
    path = os.path.join(path_pdf, "W_mds.pdf")
    plt.savefig(path)
    path = os.path.join(path_png, "W_mds.png")
    plt.savefig(path)
    path = os.path.join(path_svg, "W_mds.svg")
    plt.savefig(path)

    plt.close("all")

    return


def plot_score_nmf(result_directory):
    """
    Function to plot the scores from the NMF algorithm
    :param result_directory: string
    :return: string
    """
    print("plot NMF scores", "\n")

    # paths
    path_w = os.path.join(result_directory, "w.npy")
    path_png = os.path.join(result_directory, "graphs", "png")
    path_pdf = os.path.join(result_directory, "graphs", "pdf")
    path_jpeg = os.path.join(result_directory, "graphs", "jpeg")
    path_svg = os.path.join(result_directory, "graphs", "svg")

    # get data
    w = np.load(path_w)
    print("shape W:", w.shape)

    # compute sparsity of w
    n = np.count_nonzero(w)
    x, y = w.shape
    sparsity = 1 - (n / (x * y))
    print("sparsity rate of W:", sparsity, "\n")

    # get values and labels
    values = []
    names = []
    for i in range(y):
        values.append(w[:, i])
        names.append("topic %i" % i)

    # plot with boxplot
    fig, ax = plt.subplots(figsize=(5, 5))
    bp = ax.boxplot(values,
                    notch=False,
                    vert=False,
                    manage_xticks=False,
                    patch_artist=True)
    for box in bp["boxes"]:
        box.set(facecolor="darkcyan", linewidth=1, alpha=0.8)
    for median in bp["medians"]:
        median.set(color="darkred", alpha=0.8)
    ax.set_xlim(left=-0.1, right=w.max() + 0.1)
    xticks = ax.xaxis.get_major_ticks()
    xticks[0].label1.set_visible(False)
    xticks[-1].label1.set_visible(False)
    ax.set_yticklabels([""] + names, fontsize=10)
    ax.set_xlabel("NMF weights (sparsity rate : %s)"
                  % str(round(sparsity, 2)), fontsize=15)
    plt.tight_layout()

    # save figures
    path = os.path.join(path_jpeg, "NMF scores boxplot.jpeg")
    plt.savefig(path)
    path = os.path.join(path_pdf, "NMF scores boxplot.pdf")
    plt.savefig(path)
    path = os.path.join(path_png, "NMF scores boxplot.png")
    plt.savefig(path)
    path = os.path.join(path_svg, "NMF scores boxplot.svg")
    plt.savefig(path)

    plt.close("all")

    return


def make_table_source_topic(result_directory):
    return


def _split_str(s):
    stopwords = ["passerelle_inspire", "donnees_ouvertes",
                 "geoscientific_information", "grand_public"]
    if isinstance(s, str):
        s = s.split(" ")
        for word in stopwords:
            if word in s:
                s.remove(word)
        return s
    elif isinstance(s, float) and np.isnan(s):
        return []
    else:
        print(type(s), s)
        raise ValueError("wrong type")


def _compute_distances(df_log, w):
    """
    Function to randomly compute distances between files
    :param df_log: pandas Dataframe
    :param w: matrix [n_samples, n_topics]
    :return: list of floats, list of floats, list of floats, list of floats,
             list of floats, list of floats, list of floats
    """
    # collect several random couples of files
    same_tag = []
    same_producer = []
    same_page = []
    same_reuse = []
    same_extension = []
    other = []
    all_ = []
    for i in tqdm(range(df_log.shape[0])):
        i_tag = _split_str(df_log.at[i, "tags_page"])
        i_producer = df_log.at[i, "title_producer"]
        i_page = df_log.at[i, "title_page"]
        i_reuse = _split_str(df_log.at[i, "reuse"])
        i_extension = df_log.at[i, "extension"]
        i_topic = w[i, :]
        partners = random.sample(
            [j for j in range(df_log.shape[0]) if j != i],
            k=100)
        for j in partners:
            j_tag = _split_str(df_log.at[j, "tags_page"])
            j_producer = df_log.at[j, "title_producer"]
            j_page = df_log.at[j, "title_page"]
            j_reuse = _split_str(df_log.at[j, "reuse"])
            j_extension = df_log.at[j, "extension"]
            j_topic = w[j, :]
            distance_ij = cosine(i_topic, j_topic)

            if not np.isnan(distance_ij) and np.isfinite(distance_ij):
                c = True
                all_.append(distance_ij)
                if i_page == j_page:
                    same_page.append(distance_ij)
                    continue
                if len(set(i_tag).intersection(j_tag)) > 0:
                    same_tag.append(distance_ij)
                    c = False
                if i_producer == j_producer:
                    same_producer.append(distance_ij)
                    c = False
                if len(set(i_reuse).intersection(j_reuse)) > 0:
                    same_reuse.append(distance_ij)
                    c = False
                if i_extension == j_extension:
                    same_extension.append(distance_ij)
                    c = False
                if c:
                    other.append(distance_ij)

    return (same_tag, same_producer, same_page, same_reuse, same_extension,
            other, all_)


def plot_distribution_distance(result_directory, df_log):
    """
    Function to plot the distribution of distance
    :param df_log: pandas Dataframe
    :param result_directory: string
    :return:
    """
    print("plot distribution distances", "\n")

    # memory cache
    memory = joblib.Memory(cachedir=_get_path_cachedir(result_directory),
                           verbose=0)
    _compute_distances_cached = memory.cache(_compute_distances)

    # paths
    path_w = os.path.join(result_directory, "w.npy")
    path_png = os.path.join(result_directory, "graphs", "png")
    path_pdf = os.path.join(result_directory, "graphs", "pdf")
    path_jpeg = os.path.join(result_directory, "graphs", "jpeg")
    path_svg = os.path.join(result_directory, "graphs", "svg")

    # load data
    w = np.load(path_w)

    # collect several random couples of files
    (same_tag, same_producer, same_page, same_reuse, same_extension, other,
     all_) = _compute_distances_cached(df_log, w)
    values = [same_tag, same_producer, same_page, same_reuse, same_extension]
    name_other = ["other \n(%s%%)" % str(round(len(other) * 100 / len(all_),
                                                2))]
    name_all = ["all \n(%i pairs)" % len(all_)]
    names = ["common tags \n(%s%%)" % str(
                 round(len(same_tag) * 100 / len(all_), 2)),
             "same producer \n(%s%%)" % str(
                 round(len(same_producer) * 100 / len(all_), 2)),
             "same page \n(%s%%)" % str(
                 round(len(same_page) * 100 / len(all_), 2)),
             "common reuses \n(%s%%)" % str(
                 round(len(same_reuse) * 100 / len(all_), 2)),
             "same extension \n(%s%%)" % str(
                 round(len(same_extension) * 100 / len(all_), 2))]
    print("same page :", len(same_page))
    print("same tag :", len(same_tag))
    print("same producer :", len(same_producer))
    print("same reuse :", len(same_reuse))
    print("same extension :", len(same_extension))
    print("other :", len(other))
    print("all :", len(all_), "\n")

    # plot with boxplot...
    fig = plt.figure(figsize=(5, 5))
    ax1 = plt.subplot2grid((7, 1), (0, 0), rowspan=1)
    ax2 = plt.subplot2grid((7, 1), (1, 0), rowspan=5)
    ax3 = plt.subplot2grid((7, 1), (6, 0), rowspan=1)
    for ax in fig.axes:
        ax.set_facecolor('white')
        ax.axvline(x=0, color='grey', linestyle='--', linewidth=0.8)
        ax.axvline(x=0.2, color='grey', linestyle='--', linewidth=0.8)
        ax.axvline(x=0.4, color='grey', linestyle='--', linewidth=0.8)
        ax.axvline(x=0.6, color='grey', linestyle='--', linewidth=0.8)
        ax.axvline(x=0.8, color='grey', linestyle='--', linewidth=0.8)
        ax.axvline(x=1, color='grey', linestyle='--', linewidth=0.8)
        ax.set_xlim(left=-0.1, right=1.1)

    # ...for all the files...
    bp = ax1.boxplot(all_,
                     notch=False,
                     vert=False,
                     manage_xticks=False,
                     patch_artist=True)
    for box in bp["boxes"]:
        box.set(facecolor="darkcyan", linewidth=1, alpha=0.8)
    for median in bp["medians"]:
        median.set(color="darkred", alpha=0.8)
    ax1.set_yticklabels([""] + name_all, fontsize=10,
                        multialignment='center')
    ax1.axes.get_xaxis().set_visible(False)

    # ...for shared characteristics files...
    bp = ax2.boxplot(values,
                     notch=False,
                     vert=False,
                     manage_xticks=False,
                     patch_artist=True)
    for box in bp["boxes"]:
        box.set(facecolor="darkcyan", linewidth=1, alpha=0.8)
    for median in bp["medians"]:
        median.set(color="darkred", alpha=0.8)
    ax2.axes.get_xaxis().set_visible(False)
    ax2.set_yticklabels([""] + names, fontsize=10, multialignment='center')

    # ... and for files with nothing in common
    bp = ax3.boxplot(other,
                     notch=False,
                     vert=False,
                     manage_xticks=False,
                     patch_artist=True)
    for box in bp["boxes"]:
        box.set(facecolor="darkcyan", linewidth=1, alpha=0.8)
    for median in bp["medians"]:
        median.set(color="darkred", alpha=0.8)
    xticks = ax3.xaxis.get_major_ticks()
    xticks[0].label1.set_visible(False)
    xticks[-1].label1.set_visible(False)
    ax3.set_xlabel("cosine distance", fontsize=15)
    ax3.set_yticklabels([""] + name_other, fontsize=10,
                        multialignment='center')

    plt.text(-0.1, 0.7, "close in the \ntopic space", fontsize=9,
             multialignment='center')
    plt.text(0.85, 0.7, "distant in the \ntopic space", fontsize=9,
             multialignment='center')
    plt.tight_layout()

    # save figures
    path = os.path.join(path_jpeg, "distance distribution boxplot.jpeg")
    plt.savefig(path)
    path = os.path.join(path_pdf, "distance distribution boxplot.pdf")
    plt.savefig(path)
    path = os.path.join(path_png, "distance distribution boxplot.png")
    plt.savefig(path)
    path = os.path.join(path_svg, "distance distribution boxplot.svg")
    plt.savefig(path)

    # plot with violinplot
    data = pd.DataFrame()
    distance = []
    category = []
    values = [other] + values + [all_]
    names = name_other + names + name_all
    for i in range(len(values)):
        distance += values[i]
        l = [names[i]] * len(values[i])
        category += l
    data["distance"] = distance
    data["category"] = category
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_facecolor('white')
    ax.axvline(x=0, color='grey', linestyle='--', linewidth=0.8)
    ax.axvline(x=0.2, color='grey', linestyle='--', linewidth=0.8)
    ax.axvline(x=0.4, color='grey', linestyle='--', linewidth=0.8)
    ax.axvline(x=0.6, color='grey', linestyle='--', linewidth=0.8)
    ax.axvline(x=0.8, color='grey', linestyle='--', linewidth=0.8)
    ax.axvline(x=1, color='grey', linestyle='--', linewidth=0.8)
    ax.set_xlim(left=-0.1, right=1.1)
    sns.set_context("paper")
    sns.violinplot(x="distance",
                   y="category",
                   hue=None,
                   data=data,
                   order=reversed(names),
                   hue_order=None,
                   bw='scott',
                   cut=0,
                   scale='width',
                   scale_hue=True,
                   gridsize=100,
                   width=0.8,
                   inner=None,
                   split=False,
                   orient="h",
                   linewidth=None,
                   color=None,
                   palette=None,
                   saturation=0.8,
                   ax=ax)
    ax.set_xlabel("cosine distance", fontsize=15)
    ax.yaxis.label.set_visible(False)
    plt.text(-0.1, 7.3, "close in the \ntopic space", fontsize=9,
             multialignment='center')
    plt.text(0.85, 7.3, "distant in the \ntopic space", fontsize=9,
             multialignment='center')
    plt.tight_layout()

    # save figures
    path = os.path.join(path_jpeg, "distance distribution violin plot.jpeg")
    plt.savefig(path)
    path = os.path.join(path_pdf, "distance distribution violin plot.pdf")
    plt.savefig(path)
    path = os.path.join(path_png, "distance distribution violin plot.png")
    plt.savefig(path)
    path = os.path.join(path_svg, "distance distribution violin plot.svg")
    plt.savefig(path)

    plt.close("all")

    return


def main(result_directory):
    """
    Function to run all the script
    :param result_directory: string
    :return:
    """

    # paths
    path_log = os.path.join(result_directory, "log_final_reduced_with_reuse")

    # get data
    df_log = pd.read_csv(path_log, header=0, encoding="utf-8", sep=";",
                         index_col=False)
    print("log length :", df_log.shape[0], "\n")

    # check folders
    check_graph_folders(result_directory)

    # preprocessed data
    #prepocessed_tsne(result_directory, df_log)

    # table
    #make_table_source_topic(result_directory)

    # plot
    #plot_extension_cleaned(result_directory, df_log)
    #plot_size_cleaned(result_directory, df_log)
    #plot_mds(result_directory, df_log)
    #plot_score_nmf(result_directory)
    plot_distribution_distance(result_directory, df_log)

    return

if __name__ == "__main__":

    # paths
    result_directory = get_config_tag("result", "cleaning")

    # run
    main(result_directory)
