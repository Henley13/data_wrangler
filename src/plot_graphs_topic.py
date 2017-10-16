# -*- coding: utf-8 -*-

""" Different functions to plot """

# libraries
import os
import pandas as pd
import numpy as np
import random
import joblib
import scipy.sparse as sp
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import precision_recall_curve
from toolbox.utils import (get_config_tag, load_sparse_csr, save_sparse_csr,
                           split_str, get_path_cachedir, save_dictionary,
                           print_top_words)
from scipy.spatial.distance import cosine
from wordcloud.wordcloud import WordCloud
from metric_transformation import (compute_topic_space_nmf,
                                   compute_topic_space_svd,
                                   get_all_reused_pairs, get_x_y_balanced,
                                   get_auc, get_x_score, learn_metric,
                                   transform_space)
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
print("\n")

# memory cache
general_directory = get_config_tag("data", "general")
memory = joblib.Memory(cachedir=get_path_cachedir(general_directory), verbose=0)


@memory.cache()
def compute_best_topic_space(result_directory, tfidf, df_log, d_best):
    # parameters
    model = d_best["model"]
    n_topics = d_best["n_topics"]
    balance_reuse = d_best["balance_reuse"]
    same_page = d_best["same_page"]
    max_reuse = d_best["max_reuse"]
    learning = d_best["learning"]
    norm = d_best["norm"]

    # reshape data
    df_log_reset = df_log.reset_index(drop=False, inplace=False)

    # get topic space
    if model == "nmf":
        w, model_fitted = compute_topic_space_nmf(tfidf, n_topics)
    else:
        w, model_fitted = compute_topic_space_svd(tfidf, n_topics)

    # get all reused pairs
    df_pairs_reused = get_all_reused_pairs(df_log_reset)

    #  build a dataset Xy
    x, y = get_x_y_balanced(df_pairs_reused, df_log_reset, w, same_page,
                            balance_reuse, max_reuse)

    # compute auc, precision, recall and threshold
    auc = get_auc(x, y, norm)
    x_score = get_x_score(x, norm)
    precision, recall, threshold = precision_recall_curve(y, x_score)

    # transform topic space
    if learning:
        l = learn_metric(x, y)
        w = transform_space(w, l)

    # save topic space
    path_w = os.path.join(result_directory, "best_w.npy")
    np.save(path_w, w)

    return w, auc, precision, recall, threshold, model_fitted


###############################################################################


@memory.cache()
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
        i_tag = split_str(df_log.at[i, "tags_page"])
        i_producer = df_log.at[i, "title_producer"]
        i_page = df_log.at[i, "title_page"]
        i_reuse = split_str(df_log.at[i, "reuses"])
        i_extension = df_log.at[i, "extension"]
        i_topic = w[i, :]
        partners = random.sample(
            [j for j in range(df_log.shape[0]) if j != i],
            k=100)
        for j in partners:
            j_tag = split_str(df_log.at[j, "tags_page"])
            j_producer = df_log.at[j, "title_producer"]
            j_page = df_log.at[j, "title_page"]
            j_reuse = split_str(df_log.at[j, "reuses"])
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


def plot_distribution_distance(result_directory, df_log, w):
    """
    Function to plot the distribution of distance
    :param df_log: pandas Dataframe
    :param result_directory: string
    :param w: numpy matrix
    :return:
    """
    print("plot distribution distances", "\n")

    # paths
    path_png = os.path.join(result_directory, "graphs", "png")
    path_pdf = os.path.join(result_directory, "graphs", "pdf")
    path_jpeg = os.path.join(result_directory, "graphs", "jpeg")
    path_svg = os.path.join(result_directory, "graphs", "svg")

    # collect several random couples of files
    (same_tag, same_producer, same_page, same_reuse, same_extension, other,
     all_) = _compute_distances(df_log, w)
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
    print()
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


###############################################################################


def count_tag(result_directory, df_log):
    """
    Function to count the tag per file
    :param result_directory: string
    :param df_log: pandas Dataframe
    :return: sparse csr matrix [n_samples, n_tags], dict{word:index}
    """
    print("--- count tags ---", "\n")

    # count tags
    d = {}
    tag_name = []
    row_indices = []
    col_indices = []
    total_tag_counted = 0
    for row in tqdm(range(df_log.shape[0])):
        tags = df_log.at[row, "tags_page"]
        if isinstance(tags, str):
            tags = tags.split(" ")
        elif isinstance(tags, float):
            if np.isnan(tags):
                continue
            else:
                raise ValueError("wrong type for the tag")
        else:
            raise ValueError("wrong type for the tag")
        for tag in tags:
            if tag in tag_name:
                col = tag_name.index(tag)
            else:
                tag_name.append(tag)
                col = len(tag_name) - 1
            total_tag_counted += 1
            d[tag] = col
            row_indices.append(row)
            col_indices.append(col)

    print("col_indice :", len(col_indices))
    print("number of unique tags :", len(d))
    print("total number of tags counted :", total_tag_counted)

    data = np.ones((total_tag_counted,))
    row_indices = np.array(row_indices)
    col_indices = np.array(col_indices)

    # compute a sparse csr matrix
    matrix = sp.coo_matrix((data, (row_indices, col_indices)),
                           shape=(df_log.shape[0], len(d)))
    matrix = matrix.tocsr()
    print("count shape :", matrix.shape, "\n")

    # save matrix and vocabulary
    path = os.path.join(result_directory, "tag_count.npz")
    save_sparse_csr(path, matrix)
    path = os.path.join(result_directory, "tag_vocabulary")
    save_dictionary(d, path, ["word", "index"])

    return matrix, d


def uniqueness_reuse(df_reuse):
    # check uniqueness
    d = defaultdict(lambda: [])
    for i in range(df_reuse.shape[0]):
        id_reuse = df_reuse.at[i, "id_reuse"]
        title_reuse = df_reuse.at[i, "title_reuse"]
        d[title_reuse].append(id_reuse)
    l_title = []

    # correct duplicates
    for i in range(df_reuse.shape[0]):
        id_reuse = df_reuse.at[i, "id_reuse"]
        title_reuse = df_reuse.at[i, "title_reuse"]
        l_id = list(set(d[title_reuse]))
        if len(l_id) == 1:
            l_title.append(title_reuse)
        else:
            n = l_id.index(id_reuse)
            title_reuse = title_reuse + "_" + str(n)
            l_title.append(title_reuse)

    # change df
    df_reuse["title_reuse"] = l_title
    return df_reuse


def count_reuse(general_directory, result_directory, df_log):
    """
    Function to count the reuses per file.

    Parameters
    ----------
    general_directory : str
    result_directory : str
    df_log : pandas Dataframe

    Returns
    -------
    matrix : sparse csr matrix [n_samples, n_reuse]

    d : dict{word:index}
    """
    print("--- count reuses ---", "\n")

    # load log reuse
    path_reuse = os.path.join(general_directory, "metadata_page_reuse.csv")
    df_reuse = pd.read_csv(path_reuse, header=0, encoding="utf-8", sep=";",
                           index_col=False)
    df_reuse = uniqueness_reuse(df_reuse)
    df_reuse.set_index("id_reuse", drop=True, inplace=True)

    # count reuses
    d = {}
    reuses_id = []
    row_indices = []
    col_indices = []
    total_reuse_counted = 0
    for row in tqdm(range(df_log.shape[0])):
        reuses = df_log.at[row, "reuses"]
        if isinstance(reuses, str):
            reuses = reuses.split(" ")
        elif isinstance(reuses, float):
            if np.isnan(reuses):
                continue
            else:
                raise ValueError("wrong type for the reuse")
        else:
            raise ValueError("wrong type for the reuse")
        for reuse in reuses:
            if reuse in reuses_id:
                col = reuses_id.index(reuse)
            else:
                reuses_id.append(reuse)
                col = len(reuses_id) - 1

            title_reuse = df_reuse.loc[reuse, "title_reuse"]
            if isinstance(title_reuse, str):
                title_reuse = title_reuse.replace(";", "")
            else:
                title_reuse = title_reuse.values[0].replace(";", "")
            d[title_reuse] = col
            row_indices.append(row)
            col_indices.append(col)
            total_reuse_counted += 1

    # TODO new way to get names
    print("col_indice :", len(col_indices))
    print("number of unique reuses :", len(d))
    print("total number of reuses counted :", total_reuse_counted)

    data = np.ones((total_reuse_counted,))
    row_indices = np.array(row_indices)
    col_indices = np.array(col_indices)
    print(max(row_indices))
    print(max(col_indices))

    # compute a sparse csr matrix
    matrix = sp.coo_matrix((data, (row_indices, col_indices)),
                           shape=(df_log.shape[0], len(d)))
    matrix = matrix.tocsr()
    print("count shape :", matrix.shape, "\n")

    # save matrix and vocabulary
    path = os.path.join(result_directory, "reuse_count.npz")
    save_sparse_csr(path, matrix)
    path = os.path.join(result_directory, "reuse_vocabulary")
    save_dictionary(d, path, ["word", "index"])

    return matrix, d


def count_producer(result_directory, df_log):
    """
    Function to count the producer per file
    :param result_directory: string
    :param df_log: pandas Dataframe
    :return: sparse csr matrix [n_samples, n_producers], dict{word:index}
    """
    print("--- count producers ---", "\n")

    # count producers
    d = {}
    producer_name = []
    row_indices = []
    col_indices = []
    total_producer_counted = 0
    for row in tqdm(range(df_log.shape[0])):
        producer = df_log.at[row, "title_producer"]
        if isinstance(producer, str):
            pass
        elif isinstance(producer, float):
            if np.isnan(producer):
                continue
            else:
                raise ValueError("wrong type for the producer")
        else:
            raise ValueError("wrong type for the producer")
        if producer in producer_name:
            col = producer_name.index(producer)
        else:
            producer_name.append(producer)
            col = len(producer_name) - 1
        total_producer_counted += 1
        d[producer] = col
        row_indices.append(row)
        col_indices.append(col)

    print("number of unique producers :", len(d))
    print("total number of producers counted :", total_producer_counted)

    data = np.ones((total_producer_counted,))
    row_indices = np.array(row_indices)
    col_indices = np.array(col_indices)

    # compute a sparse csr matrix
    matrix = sp.coo_matrix((data, (row_indices, col_indices)),
                           shape=(df_log.shape[0], len(d)))
    matrix = matrix.tocsr()
    print("count shape :", matrix.shape, "\n")

    # save matrix and vocabulary
    path = os.path.join(result_directory, "producer_count.npz")
    save_sparse_csr(path, matrix)
    path = os.path.join(result_directory, "producer_vocabulary")
    save_dictionary(d, path, ["word", "index"])

    return matrix, d


def count_extension(result_directory, df_log):
    """
    Function to count the extension per file
    :param result_directory: string
    :param df_log: pandas Dataframe
    :return: sparse csr matrix [n_samples, n_extensions], dict{word:index}
    """
    print("--- count extensions ---", "\n")

    # count extensions
    d = {}
    extension_name = []
    row_indices = []
    col_indices = []
    total_extension_counted = 0
    for row in tqdm(range(df_log.shape[0])):
        extension = df_log.at[row, "extension"]
        if isinstance(extension, str):
            pass
        elif isinstance(extension, float):
            if np.isnan(extension):
                continue
            else:
                raise ValueError("wrong type for the extension")
        else:
            raise ValueError("wrong type for the extension")
        if extension in extension_name:
            col = extension_name.index(extension)
        else:
            extension_name.append(extension)
            col = len(extension_name) - 1
        total_extension_counted += 1
        d[extension] = col
        row_indices.append(row)
        col_indices.append(col)

    print("number of unique extensions :", len(d))
    print("total number of extensions counted :", total_extension_counted)

    data = np.ones((total_extension_counted,))
    row_indices = np.array(row_indices)
    col_indices = np.array(col_indices)

    # compute a sparse csr matrix
    matrix = sp.coo_matrix((data, (row_indices, col_indices)),
                           shape=(df_log.shape[0], len(d)))
    matrix = matrix.tocsr()
    print("count shape :", matrix.shape, "\n")

    # save matrix and vocabulary
    path = os.path.join(result_directory, "extension_count.npz")
    save_sparse_csr(path, matrix)
    path = os.path.join(result_directory, "extension_vocabulary")
    save_dictionary(d, path, ["word", "index"])

    return matrix, d


def topic_tag(result_directory, w):
    """
    Function to embed the topic vectors within the tag space
    :param result_directory: string
    :param w: matrix [n_samples, n_topics]
    :return: sparse csr matrix [n_topics, n_tags]
    """

    # paths
    path_tag = os.path.join(result_directory, "tag_count.npz")

    # get data
    tag = load_sparse_csr(path_tag)
    print("w shape :", w.shape)
    print("tag shape :", tag.shape)

    # multiply matrices
    w = sp.csr_matrix(w)
    w = w.transpose().tocsr()
    res = w.dot(tag).tocsr()
    print("topic x tag shape :", res.shape, "\n")

    # save matrix
    path = os.path.join(result_directory, "topic_tag.npz")
    save_sparse_csr(path, res)

    return res


def topic_reuse(result_directory, w):
    # paths
    path_reuse = os.path.join(result_directory, "reuse_count.npz")

    # get data
    reuse = load_sparse_csr(path_reuse)
    print("w shape :", w.shape)
    print("reuse shape :", reuse.shape)

    # multiply matrices
    w = sp.csr_matrix(w)
    w = w.transpose().tocsr()
    res = w.dot(reuse).tocsr()
    print("topic x reuse shape :", res.shape, "\n")

    # save matrix
    path = os.path.join(result_directory, "topic_reuse.npz")
    save_sparse_csr(path, res)

    return res


def topic_producer(result_directory, w):
    """
    Function to embed the topic vectors within the extension space
    :param result_directory: string
    :return: sparse csr matrix [n_topics, n_extensions]
    """

    # paths
    path_producer = os.path.join(result_directory, "producer_count.npz")

    # get data
    producer = load_sparse_csr(path_producer)
    print("w shape :", w.shape)
    print("producer shape :", producer.shape)

    # multiply matrices
    w = sp.csr_matrix(w)
    w = w.transpose().tocsr()
    res = w.dot(producer).tocsr()
    print("topic x producer shape :", res.shape, "\n")

    # save matrix
    path = os.path.join(result_directory, "topic_producer.npz")
    save_sparse_csr(path, res)

    return res


def topic_extension(result_directory, w):
    """
    Function to embed the topic vectors within the extension space
    :param result_directory: string
    :return: sparse csr matrix [n_topics, n_extensions]
    """

    # paths
    path_extension = os.path.join(result_directory, "extension_count.npz")

    # get data
    extension = load_sparse_csr(path_extension)
    print("w shape :", w.shape)
    print("extension shape :", extension.shape)

    # multiply matrices
    w = sp.csr_matrix(w)
    w = w.transpose().tocsr()
    res = w.dot(extension).tocsr()
    print("topic x extension shape :", res.shape, "\n")

    # save matrix
    path = os.path.join(result_directory, "topic_extension.npz")
    save_sparse_csr(path, res)

    return res


def print_top_object(topic, feature_names, n_top_words):
    """
    Function to print the most important object for one topic
    :param topic: numpy array [1, n_objects]
    :param feature_names: list of string (with the right indexation)
    :param n_top_words: integer
    :return:
    """
    print("\n".join([feature_names[i] + " " + str(topic[i])
                     for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()
    return


def print_source_topic(embedding_tag, embedding_reuse, embedding_producer,
                       embedding_extension, tag_names, reuse_names,
                       producer_names, extension_names, n_top_words):
    """
    Function to print the most important objects for on topic
    :param embedding_tag: csr sparse matrix [n_topics, n_tags]
    :param embedding_reuse: csr sparse matrix [n_topics, n_reuses]
    :param embedding_producer: csr sparse matrix [n_topics, n_producers]
    :param embedding_extension: csr sparse matrix [n_topics, n_extensions]
    :param tag_names: list of tags name with the right index
    :param reuse_names: list of reuses name with the right index
    :param producer_names: list of producers name with the right index
    :param extension_names: list of extensions name with the right index
    :param n_top_words: integer
    :return:
    """
    tag = np.asarray(embedding_tag.todense())
    reuse = np.asarray(embedding_reuse.todense())
    producer = np.asarray(embedding_producer.todense())
    extension = np.asarray(embedding_extension.todense())
    z = zip(tag, reuse, producer, extension)
    for topic_idx, (topic_tag, topic_reuse, topic_producer,
                    topic_extension) in enumerate(z):
        print("Topic #%d:" % topic_idx, "\n")
        print("--- tag ---")
        print_top_object(topic_tag, tag_names, n_top_words)
        print("--- reuse ---")
        print_top_object(topic_reuse, reuse_names, n_top_words)
        print("--- producer ---")
        print_top_object(topic_producer, producer_names, n_top_words)
        print("--- extension ---")
        print_top_object(topic_extension, extension_names, n_top_words)
        print("---------------------------------------------")
    print("\n")
    return


def make_wordcloud_object(result_directory, n_top_words):
    """
    Function to build wordclouds
    :param result_directory: string
    :param n_top_words: integer
    :return:
    """

    print("wordcloud...", "\n")

    # paths
    path_tag = os.path.join(result_directory, "topic_tag.npz")
    path_reuse = os.path.join(result_directory, "topic_reuse.npz")
    path_producer = os.path.join(result_directory, "topic_producer.npz")
    path_extension = os.path.join(result_directory, "topic_extension.npz")
    path_vocabulary_tag = os.path.join(result_directory, "tag_vocabulary")
    path_vocabulary_reuse = os.path.join(result_directory, "reuse_vocabulary")
    path_vocabulary_producer = os.path.join(result_directory,
                                            "producer_vocabulary")
    path_vocabulary_extension = os.path.join(result_directory,
                                             "extension_vocabulary")
    path_png = os.path.join(result_directory, "graphs", "png")
    path_pdf = os.path.join(result_directory, "graphs", "pdf")
    path_jpeg = os.path.join(result_directory, "graphs", "jpeg")

    # load tag data
    df_tag = pd.read_csv(path_vocabulary_tag, header=0, encoding="utf-8",
                         sep=";", index_col=False)
    df_tag.sort_values(by="index", axis=0, ascending=True, inplace=True)
    tag_names = list(df_tag["word"])
    topic_tag = load_sparse_csr(path_tag)

    # load reuse data
    df_reuse = pd.read_csv(path_vocabulary_reuse, header=0, encoding="utf-8",
                           sep=";", index_col=False)
    df_reuse.sort_values(by="index", axis=0, ascending=True, inplace=True)
    reuse_names = list(df_reuse["word"])
    topic_reuse = load_sparse_csr(path_reuse)

    # load producer data
    df_producer = pd.read_csv(path_vocabulary_producer, header=0,
                              encoding="utf-8", sep=";", index_col=False)
    df_producer.sort_values(by="index", axis=0, ascending=True, inplace=True)
    producer_names = list(df_producer["word"])
    topic_producer = load_sparse_csr(path_producer)

    # load extension data
    df_extension = pd.read_csv(path_vocabulary_extension, header=0,
                               encoding="utf-8", sep=";", index_col=False)
    df_extension.sort_values(by="index", axis=0, ascending=True, inplace=True)
    extension_names = list(df_extension["word"])
    topic_extension = load_sparse_csr(path_extension)

    # print
    print_source_topic(topic_tag, topic_reuse, topic_producer, topic_extension,
                       tag_names, reuse_names, producer_names, extension_names,
                       n_top_words)

    # build wordclouds
    stopwords = {'passerelle_inspire', 'donnees_ouvertes',
                 'geoscientific_information', 'grand_public'}
    l_object = [(topic_tag, tag_names, "tag", "Oranges"),
                (topic_reuse, reuse_names, "reuse", "Reds"),
                (topic_producer, producer_names, "producer", "Blues"),
                (topic_extension, extension_names, "extension", "Greens")]
    for topic_ind in range(topic_tag.shape[0]):
        print("Topic #%i" % topic_ind)
        for object, object_names, object_string, color in l_object:
            print("---", object_string)
            topic_object = np.asarray(object.todense())
            topic = topic_object[topic_ind, ]
            d = {}
            for i in range(len(object_names)):
                word = object_names[i]
                weight = topic[i]
                if weight > 0:
                    d[word] = weight
            wc = WordCloud(width=1000, height=500, margin=2,
                           prefer_horizontal=1,
                           background_color="white", colormap=color,
                           stopwords=stopwords)
            wc = wc.fit_words(d)
            plt.figure()
            plt.imshow(wc, interpolation="bilinear")
            plt.axis("off")
            plt.title("Wordcloud topic %i (%s)" % (topic_ind, object_string),
                      fontweight="bold")
            ax = plt.gca()
            ttl = ax.title
            ttl.set_position([.5, 1.06])
            plt.tight_layout()

            path = os.path.join(path_jpeg, "topic %i (%s).jpeg" %
                                (topic_ind, object_string))
            if os.path.isfile(path):
                os.remove(path)
            wc.to_file(path)
            path = os.path.join(path_pdf, "topic %i (%s).pdf" %
                                (topic_ind, object_string))
            if os.path.isfile(path):
                os.remove(path)
            wc.to_file(path)
            path = os.path.join(path_png, "topic %i (%s).png" %
                                (topic_ind, object_string))
            if os.path.isfile(path):
                os.remove(path)
            wc.to_file(path)

            plt.close()

    return


###############################################################################


def make_wordcloud_vocabulary(result_directory, model_fitted, n_top_words):
    """
    Function to build wordclouds from vocabulary
    :param result_directory: string
    :param model_fitted: sklearn fitted model
    :param n_top_words: integer
    :return:
    """

    print("wordcloud...", "\n")

    # paths
    path_vocabulary = os.path.join(result_directory, "token_vocabulary_unstem")
    path_png = os.path.join(result_directory, "graphs", "png")
    path_pdf = os.path.join(result_directory, "graphs", "pdf")
    path_jpeg = os.path.join(result_directory, "graphs", "jpeg")

    # load data
    df = pd.read_csv(path_vocabulary, header=0, encoding="utf-8", sep=";",
                     index_col=False)
    df.sort_values(by="index", axis=0, ascending=True, inplace=True)
    feature_names = list(df["word"])
    print_top_words(model_fitted, feature_names, n_top_words)

    # build wordclouds
    for topic_ind, topic in enumerate(model_fitted.components_):
        print("topic #%i" % topic_ind)
        d = {}
        for i in range(len(feature_names)):
            word = feature_names[i]
            weight = topic[i]
            if weight > 0:
                d[word] = weight
        wc = WordCloud(width=1000, height=500, margin=2, prefer_horizontal=1,
                       background_color='white', colormap="viridis")
        wc = wc.fit_words(d)
        plt.figure()
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title("Wordcloud topic %i" % topic_ind, fontweight="bold")
        ax = plt.gca()
        ttl = ax.title
        ttl.set_position([.5, 1.06])
        plt.tight_layout()

        # save figures
        path = os.path.join(path_jpeg, "topic %i.jpeg" % topic_ind)
        if os.path.isfile(path):
            os.remove(path)
        wc.to_file(path)
        path = os.path.join(path_pdf, "topic %i.pdf" % topic_ind)
        if os.path.isfile(path):
            os.remove(path)
        wc.to_file(path)
        path = os.path.join(path_png, "topic %i.png" % topic_ind)
        if os.path.isfile(path):
            os.remove(path)
        wc.to_file(path)

        plt.close()

    return


###############################################################################


def main(general_directory, result_directory, d_best, n_top_words):
    """
    Function to run all the script
    :param general_directory: string
    :param result_directory: string
    :param d_best: dict
    :param n_top_words: int
    :return:
    """

    # paths
    path_log = os.path.join(result_directory, "log_final_reduced")

    # get data
    df_log = pd.read_csv(path_log, header=0, encoding="utf-8", sep=";",
                         index_col=False)
    path_tfidf = os.path.join(result_directory, "tfidf.npz")
    tfidf = load_sparse_csr(path_tfidf)
    print("df_log shape :", df_log.shape)
    print("tfidf shape :", tfidf.shape, "\n")

    # compute topic space
    (w, auc, precision, recall, threshold,
     model_fitted) = compute_best_topic_space(result_directory, tfidf, df_log,
                                              d_best)
    print("w shape :", w.shape)
    print("best auc :", auc, "(%i topics)" % d_best["n_topics"], "\n")

    # plot distribution distance within the topic space
    plot_distribution_distance(result_directory, df_log, w)

    # plot topic wordclouds
    count_tag(result_directory, df_log)
    topic_tag(result_directory, w)
    count_reuse(general_directory, result_directory, df_log)
    topic_reuse(result_directory, w)
    count_producer(result_directory, df_log)
    topic_producer(result_directory, w)
    count_extension(result_directory, df_log)
    topic_extension(result_directory, w)
    make_wordcloud_object(result_directory, n_top_words)
    make_wordcloud_vocabulary(result_directory, model_fitted, n_top_words)

    return


if __name__ == "__main__":

    # paths
    general_directory = get_config_tag("data", "general")
    result_directory = get_config_tag("result", "cleaning")

    # parameters
    d_best = dict()
    d_best["model"] = "nmf"
    d_best["balance_reuse"] = 0.3
    d_best["same_page"] = False
    d_best["max_reuse"] = 30000
    d_best["learning"] = True
    d_best["norm"] = "l2"
    d_best["n_topics"] = 25
    n_top_words = 5

    # run
    main(general_directory=general_directory,
         result_directory=result_directory,
         d_best=d_best,
         n_top_words=n_top_words)
