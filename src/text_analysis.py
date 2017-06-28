# -*- coding: utf-8 -*-

""" Analyze text elements, extract topics and make a wordcloud."""

# libraries
import os
import pandas as pd
import numpy as np
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from wordcloud.wordcloud import WordCloud
from sklearn.externals import joblib
from text_extraction import get_ordered_features
from toolbox.utils import (print_top_words, load_sparse_csr, get_config_tag,
                           _check_graph_folders)
from scipy.spatial.distance import cosine
print("\n")


def origin_word(words, n_top, result_directory):
    """
    Function to find the origin of a specific word.
    :param words: string or list of string
    :param n_top: integer
    :param result_directory: string
    :return:
    """
    # TODO make it consistent with the reduced version of log
    # paths
    path_vocabulary = os.path.join(result_directory,
                                   "token_vocabulary_unstem_bis")
    path_count = os.path.join(result_directory, "count_normalized.npz")
    path_log = os.path.join(result_directory, "log_final")

    # get data
    df = pd.read_csv(path_vocabulary, header=0, encoding="utf-8", sep=";",
                     index_col=False)
    df.sort_values(by="index", axis=0, ascending=True, inplace=True)
    feature_names = list(df["word"])
    df_metadata = pd.read_csv(path_log, header=0, encoding="utf-8",
                              sep=";", index_col=False)
    matrix = load_sparse_csr(path_count)
    if isinstance(words, str):
        words = [words]
    if not isinstance(words, list):
        raise ValueError("The query isn't a string or a list of string")
    # get word counts
    for word in words:
        print("query : %s" % word)
        ind = feature_names.index(word)
        col = matrix.getcol(ind).tocsc()
        print("appears %i times in %i different files" % (col.sum(),
                                                          len(col.data)), "\n")
        rows = [col.indices[i] for i in col.data.argsort()[:-n_top - 1:-1]]
        values = [col.data[i] for i in col.data.argsort()[:-n_top - 1:-1]]
        df0 = df_metadata.ix[rows]
        for j in range(len(rows)):
            row = rows[j]
            value = values[j]
            print("*** top source %i :" % j, df0.at[row, "matrix_name"])
            print("    extension :", df0.at[row, "extension"])
            print("    source file :", df0.at[row, "source_file"])
            print("    producer :", df0.at[row, "title_producer"])
            print("    file :", df0.at[row, "title_file"])
            print("    page :", df0.at[row, "title_page"])
            print("    count :", value)
            print("    size :", df0.at[row, "size"])
            print("    row :", df0.at[row, "n_row"])
            print("    col :", df0.at[row, "n_col"])
            print("    header :", df0.at[row, "header"])
            print("    features :", df0.at[row, "header_name"], "\n")
        print('\n', "------------", "\n")
    return


def make_wordcloud(result_directory, n_top_words):
    """
    Function to build wordclouds
    :param result_directory: string
    :param n_top_words: integer
    :return:
    """

    print("wordcloud...", "\n")

    # paths
    path_vocabulary = os.path.join(result_directory,
                                   "token_vocabulary_unstem_bis")
    path_nmf = os.path.join(result_directory, "nmf.pkl")
    path_png = os.path.join(result_directory, "graphs", "png")
    path_pdf = os.path.join(result_directory, "graphs", "pdf")
    path_jpeg = os.path.join(result_directory, "graphs", "jpeg")

    # load data
    nmf = joblib.load(path_nmf)
    df = pd.read_csv(path_vocabulary, header=0, encoding="utf-8", sep=";",
                     index_col=False)
    df.sort_values(by="index", axis=0, ascending=True, inplace=True)
    feature_names = list(df["word"])
    print_top_words(nmf, feature_names, n_top_words)

    # build wordclouds
    for topic_ind, topic in enumerate(nmf.components_):
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


def find_kneighbors_random(result_directory, n_queries, n_neighbors):
    """
    Function to find closest kneighbors
    :param result_directory: string
    :param n_queries: integer
    :param n_neighbors: integer
    :return:
    """
    print("kneighbors", "\n")

    # paths
    path_log = os.path.join(result_directory, "log_final_reduced")
    path_vocabulary = os.path.join(result_directory,
                                   "token_vocabulary_unstem_bis")
    path_tfidf = os.path.join(result_directory, "tfidf.npz")
    path_knn = os.path.join(result_directory, "knn.pkl")
    path_w = os.path.join(result_directory, "w.npy")

    # load data
    tfidf = load_sparse_csr(path_tfidf)
    w = np.load(path_w)
    print("tfidf shape", tfidf.shape)
    print("W shape :", w.shape, "\n")
    knn = joblib.load(path_knn)
    df_log = pd.read_csv(path_log, sep=";", encoding="utf-8", index_col=False)
    feature_names = get_ordered_features(path_vocabulary)

    # search for kneighbors
    queries = random.choices([i for i in range(df_log.shape[0])],
                             k=n_queries)
    for n in queries:
        print("query :", df_log.at[n, "matrix_name"])
        print("source file :", df_log.at[n, "source_file"])
        print("topic scores :", w[n].reshape(1, -1)[0])
        print(" ".join([feature_names[tfidf[n].indices[j]] for j in
                        tfidf[n].data.argsort()[:-20 - 1:-1]]))
        print("header :", df_log.at[n, "header_name"], "\n")
        kneighbors_test = knn.kneighbors(w[n].reshape(1, -1),
                                         n_neighbors=n_neighbors)
        id_kneighbors = kneighbors_test[1][0]
        for i in range(len(id_kneighbors)):
            id = id_kneighbors[i]
            if df_log.at[id, "matrix_name"] == df_log.at[n, "matrix_name"]:
                continue
            else:
                print("--- kneighbors %i :" % i, df_log.at[id, "matrix_name"])
                print("----- source file :", df_log.at[id, "source_file"])
                print("----- distance :", kneighbors_test[0][0][i])
                print("----- header :", df_log.at[id, "header_name"])
                print("-----", " ".join([feature_names[tfidf[id].indices[j]]
                                         for j in tfidf[id].data.argsort()
                                         [:-20 - 1:-1]]), "\n")
        print("\n")
    return


def plot_kneighbors(result_directory):
    """
    Function to plot the topic distance vs the tag distance between two files
    :param result_directory: string
    :return:
    """
    print("distance plots", "\n")

    # paths
    path_log = os.path.join(result_directory, "log_final_reduced")
    path_w = os.path.join(result_directory, "w.npy")
    path_png = os.path.join(result_directory, "graphs", "png")
    path_pdf = os.path.join(result_directory, "graphs", "pdf")
    path_jpeg = os.path.join(result_directory, "graphs", "jpeg")
    path_svg = os.path.join(result_directory, "graphs", "svg")

    # load data and model
    df_log = pd.read_csv(path_log, sep=";", encoding="utf-8", index_col=False)
    w = np.load(path_w)

    # collect several random couple of files
    df = df_log.query("extension != 'geojson'")
    df.reset_index(drop=False, inplace=True)
    indexes = [i for i in range(df.shape[0])]

    # compute distances between couples
    l_topic = []
    l_tag = []
    l_extension = []
    for i in tqdm(range(100000)):
        couple = random.sample(indexes, k=2)
        i_a = couple[0]
        i_b = couple[1]
        a_topic = w[i_a, :]
        b_topic = w[i_b, :]
        a_tag = str(df.at[i_a, "tags_page"]).split(" ")
        b_tag = str(df.at[i_b, "tags_page"]).split(" ")
        distance_topic = cosine(a_topic, b_topic)
        n_tag_union = len(set(a_tag + b_tag))
        n_tag_inter = len(set(a_tag).intersection(b_tag))
        distance_tag = (1 + n_tag_union - n_tag_inter) / (n_tag_union + 1)
        l_topic.append(distance_topic)
        l_tag.append(distance_tag)
        if df.at[i_a, "extension"] == df.at[i_b, "extension"]:
            l_extension.append(df.at[i_a, "extension"])
        else:
            l_extension.append("different")

    # reshape data
    df = pd.DataFrame({"topic_distance": l_topic,
                       "tag_distance": l_tag,
                       "extension": l_extension})
    groups = df.groupby("extension")

    # plot a scatter plot
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_xlabel("Topic distance", fontsize=15)
    ax.set_ylabel("Tags distance", fontsize=15)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    for name, group in groups:
        ax.scatter(x=group.topic_distance,
                   y=group.tag_distance,
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
    path = os.path.join(path_jpeg, "topic tag distances.jpeg")
    if os.path.isfile(path):
        os.remove(path)
    plt.savefig(path)
    path = os.path.join(path_pdf, "topic tag distances.pdf")
    if os.path.isfile(path):
        os.remove(path)
    plt.savefig(path)
    path = os.path.join(path_png, "topic tag distances.png")
    if os.path.isfile(path):
        os.remove(path)
    plt.savefig(path)
    path = os.path.join(path_svg, "topic tag distances.svg")
    if os.path.isfile(path):
        os.remove(path)
    plt.savefig(path)

    plt.close()

    return


def main(result_directory, n_top_words, wordcloud_bool, kneighbors_bool,
         distance_plot_bool, n_queries=5, n_neighbors=5):
    """
    Function to run all the script
    :param result_directory: string
    :param n_top_words: integer
    :param wordcloud_bool: boolean
    :param kneighbors_bool: boolean
    :param distance_plot_bool: boolean
    :param n_queries: integer
    :param n_neighbors: integer
    :return:
    """
    _check_graph_folders(result_directory)
    if wordcloud_bool:
        make_wordcloud(result_directory, n_top_words)
    if kneighbors_bool:
        find_kneighbors_random(result_directory, n_queries, n_neighbors)
    if distance_plot_bool:
        plot_kneighbors(result_directory)
    return

if __name__ == "__main__":

    # parameters
    n_top_words = get_config_tag("n_top_words", "text_extraction")
    wordcloud_bool = get_config_tag("wordcloud", "text_analysis")
    kneighbors_bool = get_config_tag("kneighbors", "text_analysis")
    distance_plot_bool = get_config_tag("distance_plot", "text_analysis")

    # path
    result_directory = get_config_tag("result", "cleaning")

    # run code
    main(result_directory=result_directory,
         n_top_words=n_top_words,
         wordcloud_bool=wordcloud_bool,
         kneighbors_bool=kneighbors_bool,
         distance_plot_bool=distance_plot_bool,
         n_queries=5,
         n_neighbors=5)

    # specific queries
    # words = ["sefip", "epm", "pse", "dap", "faa", "qcne", "loos"]
    # origin_word(words, 5, result_directory)
