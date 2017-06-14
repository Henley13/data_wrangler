# -*- coding: utf-8 -*-

""" Source the different topics extracted through NMF using mutual
    information between topics and tags/producers """

# libraries
import os
import pandas as pd
import numpy as np
import scipy.sparse as sp
from collections import defaultdict
from toolbox.utils import get_config_tag, save_sparse_csr, load_sparse_csr
print("\n")


def count_tag(result_directory):
    """
    Function to count the tag per file
    :param result_directory: string
    :return: sparse csr matrix [n_samples, n_tags]
    """
    print("--- count tags ---", "\n")

    # paths
    path_log = os.path.join(result_directory, "log_final")

    # get data
    df_log = pd.read_csv(path_log, header=0, encoding="utf-8", sep=";",
                         index_col=False)
    print("log length :", df_log.shape[0])

    # collect tags
    d = defaultdict(lambda: 0)
    total_tag = 0
    for i in range(df_log.shape[0]):
        tags = df_log.at[i, "tags_page"]
        if isinstance(tags, str):
            tags = tags.split(" ")
        elif isinstance(tags, float):
            if np.isnan(tags):
                tags = []
            else:
                raise ValueError("wrong type for the tag")
        else:
            raise ValueError("wrong type for the tag")
        for tag in tags:
            d[tag] += 1
            total_tag += 1
    # l = dict_to_list(d, reversed=True)
    print("number of unique tags :", len(d))
    print("total number of tags :", total_tag)

    # count tags
    l = []
    data = np.ones((total_tag,))
    row_indices = []
    col_indices = []
    for row in range(df_log.shape[0]):
        tags = df_log.at[row, "tags_page"]
        if isinstance(tags, str):
            tags = tags.split(" ")
        else:
            continue
        for tag in tags:
            if tag in l:
                col = l.index(tag)
            else:
                l.append(tag)
                col = len(l) - 1
            row_indices.append(row)
            col_indices.append(col)
    row_indices = np.array(row_indices)
    col_indices = np.array(col_indices)

    # compute a sparse csr matrix
    matrix = sp.coo_matrix((data, (row_indices, col_indices)),
                           shape=(df_log.shape[0], len(d)))
    matrix = matrix.tocsr()
    print("count shape :", matrix.shape, "\n")

    # save matrix
    path = os.path.join(result_directory, "tag_count.npz")
    save_sparse_csr(path, matrix)

    return matrix


def count_producer(result_directory):
    """
        Function to count the producer per file
        :param result_directory: string
        :return: sparse csr matrix [n_samples, n_producer]
        """
    print("--- count producers ---", "\n")

    # paths
    path_log = os.path.join(result_directory, "log_final")

    # get data
    df_log = pd.read_csv(path_log, header=0, encoding="utf-8", sep=";",
                         index_col=False)
    print("log length :", df_log.shape[0])

    # collect tags
    d = defaultdict(lambda: 0)
    total_producer = 0
    for i in range(df_log.shape[0]):
        producer = df_log.at[i, "title_producer"]
        if isinstance(producer, str):
            total_producer += 1
            d[producer] += 1
        elif isinstance(producer, float):
            if np.isnan(producer):
                print("nan")
            else:
                print(producer)
                raise ValueError("wrong type for the producer")
        else:
            print(producer)
            raise ValueError("wrong type for the producer")
    print("number of unique producers :", len(d))
    print("total number of producers :", total_producer)

    # count producers
    l = []
    data = np.ones((total_producer,))
    row_indices = []
    col_indices = []
    for row in range(df_log.shape[0]):
        producer = df_log.at[row, "title_producer"]
        if isinstance(producer, str):
            if producer in l:
                col = l.index(producer)
            else:
                l.append(producer)
                col = len(l) - 1
            row_indices.append(row)
            col_indices.append(col)
        else:
            continue
    row_indices = np.array(row_indices)
    col_indices = np.array(col_indices)

    # compute a sparse csr matrix
    matrix = sp.coo_matrix((data, (row_indices, col_indices)),
                           shape=(df_log.shape[0], len(d)))
    matrix = matrix.tocsr()
    print("count shape :", matrix.shape, "\n")

    # save matrix
    path = os.path.join(result_directory, "producer_count.npz")
    save_sparse_csr(path, matrix)

    return matrix


def topic_tag(result_directory):
    """
    Function to concatenate topic and tag matrices
    :param result_directory: string
    :return: sparse csr matrix [n_samples, n_topics + n_tags]
    """

    # paths
    path_w = os.path.join(result_directory, "w.npy")
    path_tag = os.path.join(result_directory, "tag_count.npz")

    # get data
    w = sp.csr_matrix(np.load(path_w))
    tag = load_sparse_csr(path_tag)

    print("w shape :", w.shape)
    print("tag shape :", tag.shape)

    # merge matrices
    matrix = sp.hstack([w, tag], format="csr")
    print(matrix.shape, "\n")

    # save matrix
    path = os.path.join(result_directory, "topic_tag.npz")
    save_sparse_csr(path, matrix)

    return matrix


def topic_producer(result_directory):
    """
        Function to concatenate topic and producer matrices
        :param result_directory: string
        :return: sparse csr matrix [n_samples, n_topics + n_producers]
        """

    # paths
    path_w = os.path.join(result_directory, "w.npy")
    path_producer = os.path.join(result_directory, "producer_count.npz")

    # get data
    w = sp.csr_matrix(np.load(path_w))
    producer = load_sparse_csr(path_producer)

    print("w shape :", w.shape)
    print("producer shape :", producer.shape)

    # merge matrices
    matrix = sp.hstack([w, producer], format="csr")
    print(matrix.shape, "\n")

    # save matrix
    path = os.path.join(result_directory, "topic_producer.npz")
    save_sparse_csr(path, matrix)

    return matrix


def mutual_information():
    return


def main(result_directory):
    """
    Function to run all the script
    :param result_directory: string
    :return:
    """
    count_tag(result_directory)
    topic_tag(result_directory)
    count_producer(result_directory)
    topic_producer(result_directory)
    return

if __name__ == "__main__":

    # paths
    result_directory = get_config_tag("result", "cleaning")

    count_tag(result_directory)
    topic_tag(result_directory)
    count_producer(result_directory)
    topic_producer(result_directory)
