# -*- coding: utf-8 -*-

""" Source the different topics extracted through NMF using mutual
    information between topics and tags/producers """

# libraries
import os
import pandas as pd
import numpy as np
import scipy.sparse as sp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from wordcloud.wordcloud import WordCloud
from toolbox.utils import get_config_tag, save_sparse_csr, load_sparse_csr
from toolbox.utils import save_dictionary
from tqdm import tqdm
print("\n")


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


def topic_tag(result_directory):
    """
    Function to embed the topic vectors within the tag space
    :param result_directory: string
    :return: sparse csr matrix [n_topics, n_tags]
    """

    # paths
    path_w = os.path.join(result_directory, "w.npy")
    path_tag = os.path.join(result_directory, "tag_count.npz")

    # get data
    w = sp.csr_matrix(np.load(path_w))
    tag = load_sparse_csr(path_tag)
    print("w shape :", w.shape)
    print("tag shape :", tag.shape)

    # multiply matrices
    w = w.transpose().tocsr()
    res = w.dot(tag).tocsr()
    print("topic x tag shape :", res.shape, "\n")

    # save matrix
    path = os.path.join(result_directory, "topic_tag.npz")
    save_sparse_csr(path, res)

    return res


def topic_producer(result_directory):
    """
    Function to embed the topic vectors within the extension space
    :param result_directory: string
    :return: sparse csr matrix [n_topics, n_extensions]
    """

    # paths
    path_w = os.path.join(result_directory, "w.npy")
    path_producer = os.path.join(result_directory, "producer_count.npz")

    # get data
    w = sp.csr_matrix(np.load(path_w))
    producer = load_sparse_csr(path_producer)

    print("w shape :", w.shape)
    print("producer shape :", producer.shape)

    # multiply matrices
    w = w.transpose().tocsr()
    res = w.dot(producer).tocsr()
    print("topic x producer shape :", res.shape, "\n")

    # save matrix
    path = os.path.join(result_directory, "topic_producer.npz")
    save_sparse_csr(path, res)

    return res


def topic_extension(result_directory):
    """
    Function to embed the topic vectors within the extension space
    :param result_directory: string
    :return: sparse csr matrix [n_topics, n_extensions]
    """

    # paths
    path_w = os.path.join(result_directory, "w.npy")
    path_extension = os.path.join(result_directory, "extension_count.npz")

    # get data
    w = sp.csr_matrix(np.load(path_w))
    extension = load_sparse_csr(path_extension)

    print("w shape :", w.shape)
    print("extension shape :", extension.shape)

    # multiply matrices
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


def print_source_topic(embedding_tag, embedding_producer, embedding_extension,
                       tag_names, producer_names, extension_names, n_top_words):
    """
    Function to print the most important objects for on topic
    :param embedding_tag: csr sparse matrix [n_topics, n_tags]
    :param embedding_producer: csr sparse matrix [n_topics, n_producers]
    :param embedding_extension: csr sparse matrix [n_topics, n_extensions]
    :param tag_names: list of tags name with the right index
    :param producer_names: list of producers name with the right index
    :param extension_names: list of extensions name with the right index
    :param n_top_words: integer
    :return:
    """
    tag = np.asarray(embedding_tag.todense())
    producer = np.asarray(embedding_producer.todense())
    extension = np.asarray(embedding_extension.todense())
    z = zip(tag, producer, extension)
    for topic_idx, (topic_tag, topic_producer, topic_extension) in enumerate(z):
        print("Topic #%d:" % topic_idx, "\n")
        print("--- tag ---")
        print_top_object(topic_tag, tag_names, n_top_words)
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
    path_producer = os.path.join(result_directory, "topic_producer.npz")
    path_extension = os.path.join(result_directory, "topic_extension.npz")
    path_vocabulary_tag = os.path.join(result_directory, "tag_vocabulary")
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
    print_source_topic(topic_tag, topic_producer, topic_extension, tag_names,
                       producer_names, extension_names, n_top_words)

    # build wordclouds
    stopwords = {'passerelle_inspire', 'donnees_ouvertes',
                 'geoscientific_information', 'grand_public'}
    l_object = [(topic_tag, tag_names, "tag", "Oranges"),
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


def main(result_directory, n_top_words):
    """
    Function to run all the script
    :param result_directory: string
    :param n_top_words: integer
    :return:
    """
    # paths
    path_log = os.path.join(result_directory, "log_final_reduced")

    # get data
    df_log = pd.read_csv(path_log, header=0, encoding="utf-8", sep=";",
                         index_col=False)
    print("log length :", df_log.shape[0], "\n")

    # functions
    count_tag(result_directory, df_log)
    topic_tag(result_directory)
    count_producer(result_directory, df_log)
    topic_producer(result_directory)
    count_extension(result_directory, df_log)
    topic_extension(result_directory)
    make_wordcloud_object(result_directory, n_top_words)
    return

if __name__ == "__main__":

    # paths
    result_directory = get_config_tag("result", "cleaning")

    # run
    main(result_directory, 5)
