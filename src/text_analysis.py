# -*- coding: utf-8 -*-

""" Analyze text elements, extract topics and make a wordcloud."""

# libraries
import time
import os
import pandas as pd
import numpy as np
import random
import pickle
from wordcloud.wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from toolbox.utils import print_top_words, load_sparse_csr, get_config_tag
print("\n")


def origin_word(words, n_top, result_directory):
    """
    Function to find the origin of a specific word.
    :param words: string or list of string
    :param n_top: integer
    :param result_directory: string
    :return:
    """

    # paths
    path_vocabulary = os.path.join(result_directory, "token_vocabulary")
    path_count = os.path.join(result_directory, "count.npz")
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
    path_vocabulary = os.path.join(result_directory, "token_vocabulary")
    path_nmf = os.path.join(result_directory, "nmf.pkl")
    path_graph = os.path.join(result_directory, "graphs")

    # check graph directory exists
    if os.path.isdir(path_graph):
        pass
    else:
        os.mkdir(path_graph)

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
        wc = WordCloud(width=1000, height=500, margin=2, prefer_horizontal=0.9,
                       background_color='white', colormap="viridis")
        wc = wc.fit_words(d)
        plt.figure()
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title("Wordcloud topic %i" % topic_ind, fontweight="bold")
        ax = plt.gca()
        ttl = ax.title
        ttl.set_position([.5, 1.06])
        # plt.show()
        path = os.path.join(path_graph, "topic %i.png" % topic_ind)
        os.remove(path)
        wc.to_file(path)

    return


def find_kneighbors(result_directory):

    print("kneighbors", "\n")

    # paths
    path_log = os.path.join(result_directory, "log_final")
    path_vocabulary = os.path.join(result_directory, "token_vocabulary")
    path_tfidf = os.path.join(result_directory, "tfidf.npz")
    path_nmf = os.path.join(result_directory, "nmf.pkl")
    path_knn = os.path.join(result_directory, "knn.pkl")
    path_distance = os.path.join(result_directory, "distances.pkl")
    path_w = os.path.join(result_directory, "w.npy")

    # load data
    nmf = joblib.load(path_nmf)
    tfidf = load_sparse_csr(path_tfidf)
    w = nmf.transform(tfidf)
    print("W shape :", W.shape, "\n")
    knn = joblib.load(path_knn_w)
    time.sleep(5)
    df_log = pd.read_csv(path_log, sep=";", encoding="utf-8", index_col=False)
    df = pd.read_csv(path_vocabulary, header=0, encoding="utf-8", sep=";",
                     index_col=False)
    df.sort_values(by="index", axis=0, ascending=True, inplace=True)
    feature_names = list(df["word"])

    # search for kneighbors
    queries = random.choices([i for i in range(df_log.shape[0])], k=5)
    for n in queries:
        print("topic scores :", W[n].reshape(1, -1)[0])
        kneighbors_test = knn.kneighbors(W[n].reshape(1, -1))
        print("query :", df_log.at[n, "filename"])
        print("source file :", df_log.at[n, "source_file"])
        print(" ".join([feature_names[tfidf[n].indices[j]] for j in
                        tfidf[n].data.argsort()[:-20 - 1:-1]]), "\n")
        print("header :", df_log.at[n, "header_name"])

        id_kneighbors = kneighbors_test[1][0]
        for i in range(len(id_kneighbors)):
            id = id_kneighbors[i]
            print("--- kneighbors %i :" % i, df_log.at[id, "filename"])
            print("----- source file :", df_log.at[id, "source_file"])
            print("----- distance :", kneighbors_test[0][0][i])
            print("----- header :", df_log.at[id, "header_name"])
            print("-----", " ".join([feature_names[tfidf[id].indices[j]]
                                     for j in tfidf[id].data.argsort()
                                     [:-20 - 1:-1]]), "\n")
        print("\n")

    d = {}
    # for k in [3, 5, 10, 15, 20, 30]:
    for k in [i for i in range(5, 300, 5)]:
        mean_distance = []
        for n in range(df_log.shape[0]):
            kneighbors_test = knn.kneighbors(W[n].reshape(1, -1), n_neighbors=k)
            id_kneighbors = kneighbors_test[1][0]
            distance = []
            for i in range(len(id_kneighbors)):
                distance.append(kneighbors_test[0][0][i])
            mean_distance.append(np.mean(distance))
        d[k] = mean_distance

    joblib.dump(d, path_distance, compress=0, protocol=pickle.HIGHEST_PROTOCOL)

# parameters
n_top_words = get_config_tag("n_top_words", "text_extraction")
wordcloud_bool = get_config_tag("wordcloud", "text_extraction")
kneighbors_bool = get_config_tag("kneighbors", "text_extraction")
distance_plot_bool = get_config_tag("distance_plot", "text_extraction")

# path
result_directory = get_config_tag("result", "cleaning")
path_log = os.path.join(result_directory, "log_final")

# others
path_vocabulary = os.path.join(result_directory, "token_vocabulary")
path_tfidf = os.path.join(result_directory, "tfidf.npz")
path_count = os.path.join(result_directory, "count.npz")
path_nmf = os.path.join(result_directory, "nmf.pkl")
path_knn_tfidf = os.path.join(result_directory, "knn_tfidf.pkl")
path_knn_w = os.path.join(result_directory, "knn_w.pkl")
path_graph = os.path.join(result_directory, "graphs")
path_distance = os.path.join(result_directory, "distances.pkl")

start = time.clock()

# check graph directory exists
if os.path.isdir(path_graph):
    pass
else:
    os.mkdir(path_graph)

if wordcloud_bool:

    print("wordcloud", "\n")

    # load data
    nmf = joblib.load(path_nmf)
    df = pd.read_csv(path_vocabulary, header=0, encoding="utf-8", sep=";",
                     index_col=False)
    df.sort_values(by="index", axis=0, ascending=True, inplace=True)
    feature_names = list(df["word"])
    print_top_words(nmf, feature_names, n_top_words)

    for topic_ind, topic in enumerate(nmf.components_):
        print("topic #%i" % topic_ind)
        d = {}
        for i in range(len(feature_names)):
            word = feature_names[i]
            weight = topic[i]
            if weight > 0:
                d[word] = weight
        wc = WordCloud(width=1000, height=500, margin=2, prefer_horizontal=0.9,
                       background_color='white', colormap="viridis")
        wc = wc.fit_words(d)
        plt.figure()
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title("Wordcloud topic %i" % topic_ind, fontweight="bold")
        ax = plt.gca()
        ttl = ax.title
        ttl.set_position([.5, 1.06])
        # plt.show()
        path = os.path.join(path_graph, "topic %i.png" % topic_ind)
        wc.to_file(path)

    print("\n", "#######################", "\n")

if kneighbors_bool:

    print("kneighbors", "\n")

    # load data
    nmf = joblib.load(path_nmf)
    tfidf = load_sparse_csr(path_tfidf)
    W = nmf.transform(tfidf)
    print("W shape :", W.shape, "\n")
    knn = joblib.load(path_knn_w)
    time.sleep(5)
    df_log = pd.read_csv(path_log, sep=";", encoding="utf-8", index_col=False)
    df = pd.read_csv(path_vocabulary, header=0, encoding="utf-8", sep=";",
                     index_col=False)
    df.sort_values(by="index", axis=0, ascending=True, inplace=True)
    feature_names = list(df["word"])

    # search for kneighbors
    queries = random.choices([i for i in range(df_log.shape[0])], k=5)
    for n in queries:
        print("topic scores :", W[n].reshape(1, -1)[0])
        kneighbors_test = knn.kneighbors(W[n].reshape(1, -1))
        print("query :", df_log.at[n, "filename"])
        print("source file :", df_log.at[n, "source_file"])
        print(" ".join([feature_names[tfidf[n].indices[j]] for j in
                        tfidf[n].data.argsort()[:-20 - 1:-1]]), "\n")
        print("header :", df_log.at[n, "header_name"])

        id_kneighbors = kneighbors_test[1][0]
        for i in range(len(id_kneighbors)):
            id = id_kneighbors[i]
            print("--- kneighbors %i :" % i, df_log.at[id, "filename"])
            print("----- source file :", df_log.at[id, "source_file"])
            print("----- distance :", kneighbors_test[0][0][i])
            print("----- header :", df_log.at[id, "header_name"])
            print("-----", " ".join([feature_names[tfidf[id].indices[j]]
                                     for j in tfidf[id].data.argsort()
                                     [:-20 - 1:-1]]), "\n")
        print("\n")

    d = {}
    # for k in [3, 5, 10, 15, 20, 30]:
    for k in [i for i in range(5, 300, 5)]:
        mean_distance = []
        for n in range(df_log.shape[0]):
            kneighbors_test = knn.kneighbors(W[n].reshape(1, -1), n_neighbors=k)
            id_kneighbors = kneighbors_test[1][0]
            distance = []
            for i in range(len(id_kneighbors)):
                distance.append(kneighbors_test[0][0][i])
            mean_distance.append(np.mean(distance))
        d[k] = mean_distance

    joblib.dump(d, path_distance, compress=0, protocol=pickle.HIGHEST_PROTOCOL)

    print("\n", "#######################", "\n")

if distance_plot_bool:

    print("distance plots", "\n")

    df_log = pd.read_csv(path_log, sep=";", encoding="utf-8", index_col=False)
    d = joblib.load(path_distance)
    data = []
    labels = []
    for key in d:
        print("number of kneighbors :", key)
        data.append(d[key])
        if key % 10 == 0:
            labels.append(str(key))
        else:
            labels.append("")

    plt.figure()
    plt.boxplot(data, labels=labels)
    plt.title("KNN distances (%i matrices)" % df_log.shape[0],
              fontweight="bold")
    path = os.path.join(path_graph, "boxplot knn distances.png")
    plt.xlabel("Number of neighbors")
    plt.ylabel("Mean distance")
    plt.savefig(path)
    # plt.show()

    print("\n", "#######################", "\n")

words = ["sefip", "epm", "pse", "dap", "faa", "qcne", "loos"]
origin_word(words, 5, path_vocabulary, path_count, path_log)



if __name__ == "__main__":



    # parameters
    n_top_words = get_config_tag("n_top_words", "text_extraction")
    wordcloud_bool = get_config_tag("wordcloud", "text_extraction")
    kneighbors_bool = get_config_tag("kneighbors", "text_extraction")
    distance_plot_bool = get_config_tag("distance_plot", "text_extraction")

    # path
    result_directory = get_config_tag("result", "cleaning")
    path_log = os.path.join(result_directory, "log_final")

    # others
    path_vocabulary = os.path.join(result_directory, "token_vocabulary")
    path_tfidf = os.path.join(result_directory, "tfidf.npz")
    path_count = os.path.join(result_directory, "count.npz")
    path_nmf = os.path.join(result_directory, "nmf.pkl")
    path_knn_tfidf = os.path.join(result_directory, "knn_tfidf.pkl")
    path_knn_w = os.path.join(result_directory, "knn_w.pkl")
    path_graph = os.path.join(result_directory, "graphs")
    path_distance = os.path.join(result_directory, "distances.pkl")

    path_vocabulary = os.path.join(result_directory, "token_vocabulary")
    path_count = os.path.join(result_directory, "count.npz")
    path_log = os.path.join(result_directory, "log_final")


