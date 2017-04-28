#!/bin/python3
# coding: utf-8

""" Extract text elements from a file and weight it with tf-idf. """

# libraries
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
import pandas as pd
import numpy as np
import time
import os
import re
import unidecode
import scipy.sparse as sp
nltk.data.path.append("../data/nltk_data")
print("\n")

# parameters
content_bool = True
header_bool = True
french_stopwords = list(set(stopwords.words('french')))
french_stopwords += ["unnamed", "http", "nc", "le", "la", "les", "un", "des"]

# path
data_directory = "../data/test_fitted"
path_log = "../data/log_cleaning"
path_token = "../data/token_count.npz"
path_vocabulary = "../data/token_vocabulary"
path_tfidf = "../data/tfidf.npz"

# other
tokenizer = nltk.data.load('tokenizers/punkt/PY3/french.pickle')
for letter in "abcdefghijklmnopqrstuvwxyz":
    french_stopwords.append(letter)
delay = 0


def remove_accent(s):
    """
    Function to remove accent from a string
    :param s: string
    :return: string (ASCII)
    """
    # s = s.replace("é", "e").replace("è", "e").replace("à", "a").replace("ù", "u").replace("ê", "e").replace("â", "a")
    s = unidecode.unidecode(s)
    return s


def extraction(path_log, data_directory, content_bool, header_bool, french_stopwords):
    df_log = pd.read_csv(path_log, sep=";", encoding="utf-8", index_col=False)
    print("number of files: ", df_log.shape[0], "\n")
    # create an empty temporary text file
    path_temp = os.path.join(data_directory, "temp_tokens")
    if os.path.isfile(path_temp):
        os.remove(path_temp)
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
        l = []
        l_header = []
        with open(path, mode="rt", encoding="utf-8") as f:
            j = 0
            for row in f:
                row = remove_accent(row).lower().strip()
                if j not in index:
                    l.append(row)
                else:
                    l_header.append(row)
                j += 1
        tokens_file = []
        if content_bool:
            text_content = " ".join(l)
            tokens_file += nltk.word_tokenize(text_content)
        if header_bool:
            text_header = " ".join(l_header)
            tokens_file += nltk.word_tokenize(text_header)
        tokens_file = [token for token in tokens_file if
                       token not in french_stopwords and not re.search('[0-9;,:/\\().+-><%_\'?!§*$&^"=`#~@]', token)]
        # print(filename, ":", len(tokens_file), "mot(s)")
        with open(path_temp, mode="at", encoding="utf-8") as f:
            f.write(";".join(tokens_file))
            f.write("\n")
    size = os.path.getsize(path_temp)
    print("size tokens :", size)
    with open(path_temp, mode="rt", encoding="utf-8") as f:
        tokens = [row.strip().split(";") for row in f]
    os.remove(path_temp)
    return tokens


def tfidf_extraction(path_log, data_directory, content_bool, header_bool, french_stopwords):
    df_log = pd.read_csv(path_log, sep=";", encoding="utf-8", index_col=False)
    print("number of files: ", df_log.shape[0], "\n")
    for i in range(df_log.shape[0]):
        filename = df_log.at[i, "filename"]
        multi_header = df_log.at[i, "multiheader"]
        header = df_log.at[i, "header_name"]
        path = os.path.join(data_directory, filename)
        index = [0]
        if multi_header:
            index = pd.MultiIndex.from_tuples(header)
        l = []
        l_header = []
        with open(path, mode="rt", encoding="utf-8") as f:
            j = 0
            for row in f:
                row = remove_accent(row).lower().strip()
                if j not in index:
                    l.append(row)
                else:
                    l_header.append(row)
                j += 1
        tokens_file = []
        if content_bool:
            text_content = " ".join(l)
            tokens_file += nltk.word_tokenize(text_content)
        if header_bool:
            text_header = " ".join(l_header)
            tokens_file += nltk.word_tokenize(text_header)
        tokens_file = [token for token in tokens_file if
                       token not in french_stopwords and not re.search('[0-9;,:/\\().+-><%_\'?!§*$&^"=`#~@]', token)]
        # print(filename, ":", len(tokens_file), "mot(s)")
        content_file = " ".join(tokens_file)
    tfidf_vectorizer = TfidfVectorizer(input=u'content', encoding=u'utf-8', decode_error=u'strict', strip_accents=None, lowercase=True, preprocessor=None, tokenizer=None, analyzer=u'word', stop_words=None, token_pattern=u'(?u)\b\w\w+\b', ngram_range=(1, 1), max_df=1.0, min_df=1, max_features=None, vocabulary=None, binary=False, dtype= < type 'numpy.int64' >, norm=u'l2', use_idf=True, smooth_idf=True, sublinear_tf=False)
    tfidf = tfidf_vectorizer.fit_transform(os.)

def fit_sparse_matrix(token_document, indptr, indices, data, vocabulary):
    """
    Function to prepar a sparse matrix with word count
    :param token_document: list of lists of tokens (one per document)
    :param
    :return: dictionary, list of integers, list of integers, list of integers
    """

    indptr = [0]
    indices = []
    data = []
    vocabulary = {}
    for d in token_document:
        for token in d:
            index = vocabulary.setdefault(token, len(vocabulary))
            indices.append(index)
            data.append(1)
        indptr.append(len(indices))
    return vocabulary, indptr, indices, data


def save_sparse_csr(path, array):
    np.savez(path, data=array.data, indices=array.indices, indptr=array.indptr, shape=array.shape)
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

# extract text
tokens = extraction(path_log, data_directory, content_bool, header_bool, french_stopwords)

end = time.clock()
delay += end - start
print("text extraction :", round(end - start, 2), "seconds")

print("\n", "#######################", "\n")

start = time.clock()

# statistics
print("number of documents :", len(tokens), "\n")
vocabulary, indptr, indices, data = fit_sparse_matrix(tokens)
feature_names = sorted(vocabulary, key=vocabulary.get)

# count matrix
X_sparse = sp.csr_matrix((data, indices, indptr), dtype=int)
X = X_sparse.toarray()
print("count matrix shape :", X.shape, "\n")

end = time.clock()
delay += end - start
print("count matrix :", round(end - start, 2), "seconds")

print("\n", "#######################", "\n")

start = time.clock()

# tfidf
tfidf_vectorizer = TfidfTransformer()
tfidf = tfidf_vectorizer.fit_transform(X)
print("tfidf shape :", tfidf.shape, "\n")

end = time.clock()
delay += end - start
print("tf-idf :", round(end - start, 2), "seconds")

print("\n", "#######################", "\n")

start = time.clock()

# save results
save_sparse_csr(path_token, X_sparse)
save_sparse_csr(path_tfidf, tfidf)
save_dictionary(vocabulary, path_vocabulary, ["word", "index"])

end = time.clock()
delay += end - start
print("saving :", round(end - start, 2), "seconds")

print("\n", "#######################", "\n")

print("total time :", round(delay, 2), "seconds")

