#!/bin/python3
# coding: utf-8

""" Extract text elements from a file and weight it with tf-idf. """

# libraries
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import scipy.sparse as sp
import pandas as pd
import numpy as np
import time
import os
from sklearn.decomposition import NMF
from sklearn.neighbors import NearestNeighbors
from sklearn.externals import joblib
from toolbox.utils import get_config_tag
print("\n")

# parameters
content_bool = get_config_tag("content", "text_extraction")
header_bool = get_config_tag("header", "text_extraction")
metadata_bool = get_config_tag("metadata", "text_extraction")
n_topics = get_config_tag("n_topics", "text_extraction")
n_top_words = get_config_tag("n_top_words", "text_extraction")

# paths
nltk_path = get_config_tag("nltk", "text_extraction")
result_directory = get_config_tag("result", "cleaning")
data_directory = os.path.join(result_directory, "data_fitted")
path_log = os.path.join(result_directory, "log_final")

# stopwords
nltk.data.path.append(nltk_path)
french_stopwords = list(set(stopwords.words('french')))
french_stopwords += ["unnamed", "http", "les", "des"]

# other
path_vocabulary = os.path.join(result_directory, "token_vocabulary")
path_tfidf = os.path.join(result_directory, "tfidf.npz")
path_count = os.path.join(result_directory, "count.npz")
path_nmf = os.path.join(result_directory, "nmf.pkl")
path_knn_tfidf = os.path.join(result_directory, "knn_tfidf.pkl")
path_knn_w = os.path.join(result_directory, "knn_w.pkl")


def text_content_extraction(path_log=path_log,
                            result_directory=result_directory):
    """
    Function to extract the textual content from the files
    :param path_log: string
    :param result_directory: string
    :return: list of strings
    """
    df_log = pd.read_csv(path_log, sep=";", encoding="utf-8", index_col=False)
    print("number of files: ", df_log.shape[0], "\n")
    full_text = []
    for i in range(df_log.shape[0]):
        if i % 100 == 0:
            print(i)
        filename = df_log.at[i, "matrix_name"]
        multi_header = df_log.at[i, "multiheader"]
        header = df_log.at[i, "header_name"]
        path = os.path.join(result_directory, "data_fitted", filename)
        index = [0]
        if multi_header:
            index = pd.MultiIndex.from_tuples(header)
        with open(path, mode="rt", encoding="utf-8") as f:
            for j in range(len(index)):
                f.readline()
            full_text.append(f.read())
    return full_text


def text_metadata_extraction(path_log=path_log,
                             result_directory=result_directory):
    """
    Function to extract the textual content from the metadata files
    :param path_log: string
    :param result_directory: string
    :return: list of strings
    """
    df_log = pd.read_csv(path_log, sep=";", encoding="utf-8", index_col=False)
    print("number of files: ", df_log.shape[0], "\n")
    full_text = []
    for i in range(df_log.shape[0]):
        if i % 100 == 0:
            print(i)
        metadata = df_log.at[i, "metadata"]
        if metadata:
            filename = df_log.at[i, "matrix_name"]
            path = os.path.join(result_directory, "metadata_cleaning", filename)
            with open(path, mode='rt', encoding='utf-8') as f:
                full_text.append(f.read())
        else:
            full_text.append("")
    return full_text


def text_header_extraction(path_log=path_log):
    """
    Function to extract the header from the files
    :param path_log: string
    :return: list of strings
    """
    df_log = pd.read_csv(path_log, sep=";", encoding="utf-8", index_col=False)
    print("number of files: ", df_log.shape[0], "\n")
    return list(df_log["header_name"])


def text_content_header_extraction(path_log=path_log,
                                   result_directory=result_directory):
    """
    Function to extract the textual content from the metadata files
    :param path_log: string
    :param result_directory: string
    :return: list of strings
    """
    df_log = pd.read_csv(path_log, sep=";", encoding="utf-8", index_col=False)
    print("number of files: ", df_log.shape[0], "\n")
    full_text = []
    for i in range(df_log.shape[0]):
        if i % 100 == 0:
            print(i)
        filename = df_log.at[i, "matrix_name"]
        path = os.path.join(result_directory, "data_fitted", filename)
        with open(path, mode="rt", encoding="utf-8") as f:
            full_text.append(f.read())
    return full_text


def text_full_extraction(path_log=path_log, result_directory=result_directory):
    """
    Function to extract the full textual content from the files
    :param path_log: string
    :param result_directory: string
    :return: list of strings
    """
    df_log = pd.read_csv(path_log, sep=";", encoding="utf-8", index_col=False)
    print("number of files: ", df_log.shape[0], "\n")
    full_text = []
    for i in range(df_log.shape[0]):
        if i % 100 == 0:
            print(i)
        text = []
        filename = df_log.at[i, "matrix_name"]
        path = os.path.join(result_directory, "data_fitted", filename)
        with open(path, mode="rt", encoding="utf-8") as f:
            text.append(f.read())
        metadata = df_log.at[i, "metadata"]
        if metadata:
            path_metadata = os.path.join(result_directory, "metadata_cleaning",
                                         filename)
            with open(path_metadata, mode='rt', encoding='utf-8') as f:
                text.append(f.read())
        else:
            text.append("")
        full_text.append(" ".join(text))
    return full_text


def text_extraction(path_log=path_log, result_directory=result_directory,
                    content_bool=content_bool, header_bool=header_bool,
                    metadata_bool=metadata_bool):
    """
    Function to select which textual part to extract from the files
    :param path_log: string
    :param result_directory: string
    :param content_bool: boolean
    :param header_bool: boolean
    :param metadata_bool: boolean
    :return: list of strings
    """
    if content_bool and header_bool and metadata_bool:
        return text_full_extraction(path_log, result_directory)
    elif content_bool and not header_bool and not metadata_bool:
        return text_content_extraction(path_log, result_directory)
    elif not content_bool and header_bool and not metadata_bool:
        return text_header_extraction(path_log)
    elif not content_bool and not header_bool and metadata_bool:
        return text_metadata_extraction(path_log, result_directory)
    elif content_bool and header_bool and not metadata_bool:
        return text_content_header_extraction(path_log, result_directory)
    elif not content_bool and header_bool and metadata_bool:
        raise ValueError("Extraction of header and metadata only has to be "
                         "implemented")
    elif content_bool and not header_bool and metadata_bool:
        raise ValueError(" Extraction of content and metadata only has to be "
                         "implemented")
    else:
        raise ValueError("No text extracted")


def tfidf_computation(path_log=path_log, result_directory=result_directory,
                      content_bool=content_bool, header_bool=header_bool,
                      stop_words=french_stopwords):
    """
    Function to compute tfidf matrix
    :param path_log: string
    :param result_directory: string
    :param content_bool: boolean
    :param header_bool: boolean
    :param stop_words: list of strings
    :return: sparse matrix [n_samples, n_features], sparse matrix
             [n_samples, n_features], dictionary
    """
    count_vect = CountVectorizer(input=u'content',
                                 encoding=u'utf-8',
                                 decode_error=u'strict',
                                 strip_accents='unicode',
                                 lowercase=True,
                                 preprocessor=None,
                                 tokenizer=None,
                                 stop_words=stop_words,
                                 token_pattern=r'\b[a-zA-Z][a-zA-Z][a-zA-Z]+\b',
                                 ngram_range=(1, 1),
                                 analyzer=u'word',
                                 max_df=1.0,
                                 min_df=5,
                                 max_features=None,
                                 vocabulary=None,
                                 binary=False)
    tfidf_transf = TfidfTransformer(norm=u'l2',
                                    use_idf=True,
                                    smooth_idf=True,
                                    sublinear_tf=False)
    print("text extraction...", "\n")
    full_text = text_extraction(path_log,
                                result_directory,
                                content_bool,
                                header_bool)
    print()
    print("counting process...", "\n")
    count_matrix = count_vect.fit_transform(full_text)
    print()
    print("tfidf computation...", "\n")
    tfidf = tfidf_transf.fit_transform(count_matrix)
    count_matrix = sp.csr_matrix(count_matrix)
    tfidf = sp.csr_matrix(tfidf)
    return count_matrix, tfidf, count_vect.vocabulary_


def save_sparse_csr(path, array):
    np.savez(path, data=array.data, indices=array.indices, indptr=array.indptr,
             shape=array.shape)
    return


def save_dictionary(dictionary, path, header):
    """
    Function to save a dictionary in a csv format
    :param dictionary: dictionary
    :param path: string
    :param header: list of strings
    :return:
    """
    with open(path, mode="wt", encoding="utf-8") as f:
        f.write(";".join(header))
        f.write("\n")
        for key in dictionary:
            s = ";".join([key, str(dictionary[key])])
            f.write(s)
            f.write("\n")
    return


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()
    return


###############################################################################
###############################################################################

start = time.clock()

# compute tfidf matrix
count_matrix, tfidf, vocabulary = tfidf_computation()
print("count shape :", count_matrix.shape)
print("tfidf shape :", tfidf.shape, "\n")

end = time.clock()
print("tf-idf :", round(end - start, 2), "seconds")

print("\n", "#######################", "\n")

start = time.clock()

# save results
# feature_names = sorted(vocabulary, key=vocabulary.get)
save_sparse_csr(path_count, count_matrix)
save_sparse_csr(path_tfidf, tfidf)
save_dictionary(vocabulary, path_vocabulary, ["word", "index"])

end = time.clock()
print("saving :", round(end - start, 2), "seconds")

print("\n", "#######################", "\n")

df_log = pd.read_csv(path_log, sep=";", encoding="utf-8", index_col=False)
df = pd.read_csv(path_vocabulary, header=0, encoding="utf-8", sep=";",
                 index_col=False)
print("number of unique word :", df.shape[0])
df.sort_values(by="index", axis=0, ascending=True, inplace=True)
feature_names = list(df["word"])

print("\n", "#######################", "\n")

start = time.clock()

# NMF
print("NMF", "\n")
nmf = NMF(n_components=n_topics, random_state=1, alpha=.1, l1_ratio=.5)
nmf = nmf.fit(tfidf)
joblib.dump(nmf, path_nmf)
print_top_words(nmf, feature_names, n_top_words)
W = nmf.transform(tfidf)

end = time.clock()
print("NMF :", round(end - start, 2), "seconds")

print("\n", "#######################", "\n")

start = time.clock()

# KNN
print("NearestNeighbors", "\n")
knn_tfidf = NearestNeighbors(n_neighbors=5,
                             radius=1.0,
                             algorithm='auto',
                             leaf_size=30,
                             metric='minkowski',
                             p=2,
                             metric_params=None,
                             n_jobs=1)
knn_tfidf = knn_tfidf.fit(tfidf)
joblib.dump(knn_tfidf, path_knn_tfidf)
knn_w = NearestNeighbors(n_neighbors=5,
                         radius=1.0,
                         algorithm='auto',
                         leaf_size=30,
                         metric='minkowski',
                         p=2,
                         metric_params=None,
                         n_jobs=1)
knn_w = knn_w.fit(W)
joblib.dump(knn_w, path_knn_w)

end = time.clock()
print("KNN :", round(end - start, 2), "seconds")

print("\n", "#######################", "\n")
