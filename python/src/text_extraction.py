#!/bin/python3
# coding: utf-8

""" Extract text elements from a file and weight it with tf-idf. """

# libraries
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
import time
import os
nltk.data.path.append("../data/nltk_data")
print("\n")

# parameters
content_bool = True
header_bool = True
french_stopwords = list(set(stopwords.words('french')))
french_stopwords += ["unnamed", "http", "les", "des"]

# path
data_directory = "../data/test_fitted"
path_log = "../data/log_cleaning"
path_vocabulary = "../data/token_vocabulary"
path_tfidf = "../data/tfidf.npz"

# other
# tokenizer = nltk.data.load('tokenizers/punkt/PY3/french.pickle')


def text_content_extraction(path_log=path_log, data_directory=data_directory):
    """
    Function to extract the textual content from the files
    :param path_log: string
    :param data_directory: string
    :return: list of strings
    """
    df_log = pd.read_csv(path_log, sep=";", encoding="utf-8", index_col=False)
    print("number of files: ", df_log.shape[0], "\n")
    full_text = []
    for i in range(df_log.shape[0]):
        if i % 100 == 0:
            print(i)
        filename = df_log.at[i, "filename"]
        multi_header = df_log.at[i, "multiheader"]
        header = df_log.at[i, "header_name"]
        path = os.path.join(data_directory, filename)
        index = [0]
        if multi_header:
            index = pd.MultiIndex.from_tuples(header)
        with open(path, mode="rt", encoding="utf-8") as f:
            for j in range(len(index)):
                f.readline()
            text = f.read()
        full_text.append(text)
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


def text_full_extraction(path_log=path_log, data_directory=data_directory):
    """
    Function to extract the full textual content from the files
    :param path_log: string
    :param data_directory: string
    :return: list of strings
    """
    df_log = pd.read_csv(path_log, sep=";", encoding="utf-8", index_col=False)
    print("number of files: ", df_log.shape[0], "\n")
    full_text = []
    for i in range(df_log.shape[0]):
        if i % 100 == 0:
            print(i)
        filename = df_log.at[i, "filename"]
        path = os.path.join(data_directory, filename)
        with open(path, mode="rt", encoding="utf-8") as f:
            text = f.read()
        full_text.append(text)
    return full_text


def text_extraction(path_log=path_log, data_directory=data_directory,
                    content_bool=content_bool, header_bool=header_bool):
    """
    Function to select which textual part to extract from the files
    :param path_log: string
    :param data_directory: string
    :param content_bool: boolean
    :param header_bool: boolean
    :return: list of strings
    """
    if content_bool and header_bool:
        return text_full_extraction(path_log, data_directory)
    elif content_bool and not header_bool:
        return text_content_extraction(path_log, data_directory)
    elif not content_bool and header_bool:
        return text_header_extraction(path_log)
    else:
        raise ValueError("No text extracted")


def tfidf_computation(path_log=path_log, data_directory=data_directory,
                      content_bool=content_bool, header_bool=header_bool,
                      stop_words=french_stopwords):
    """
    Function to compute tfidf matrix
    :param path_log: string
    :param data_directory: string
    :param content_bool: boolean
    :param header_bool: boolean
    :param stop_words: list of strings
    :return: sparse matrix [n_samples, n_features], dictionary
    """
    tfidf_vectorizer = TfidfVectorizer(input=u'content',
                                       encoding=u'utf-8',
                                       decode_error=u'strict',
                                       strip_accents='unicode',
                                       lowercase=True,
                                       preprocessor=None,
                                       tokenizer=None,
                                       analyzer=u'word',
                                       stop_words=stop_words,
                                       token_pattern=r'\b[a-zA-Z][a-zA-Z][a-zA-Z]+\b',
                                       ngram_range=(1, 1),
                                       max_df=1.0,
                                       min_df=1,
                                       max_features=None,
                                       vocabulary=None,
                                       binary=False,
                                       norm=u'l2',
                                       use_idf=True,
                                       smooth_idf=True,
                                       sublinear_tf=False)
    full_text = text_extraction(path_log,
                                data_directory,
                                content_bool,
                                header_bool)
    print()
    print("tfidf computation...", "\n")
    tfidf = tfidf_vectorizer.fit_transform(full_text)
    return tfidf, tfidf_vectorizer.vocabulary_


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


###############################################################################
###############################################################################

start = time.clock()

# compute tfidf matrix
tfidf, vocabulary = tfidf_computation()
print("tfidf shape :", tfidf.shape, "\n")

end = time.clock()
print("tf-idf :", round(end - start, 2), "seconds")

print("\n", "#######################", "\n")

start = time.clock()

# save results
# feature_names = sorted(vocabulary, key=vocabulary.get)
save_sparse_csr(path_tfidf, tfidf)
save_dictionary(vocabulary, path_vocabulary, ["word", "index"])

end = time.clock()
print("saving :", round(end - start, 2), "seconds")

print("\n", "#######################", "\n")

