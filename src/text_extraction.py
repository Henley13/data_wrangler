# -*- coding: utf-8 -*-

""" Count text elements from files, extract features and fit models. """

# libraries
import nltk
import scipy.sparse as sp
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict, Counter
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import NMF
from sklearn.neighbors import NearestNeighbors
from sklearn.externals import joblib
from toolbox.utils import get_config_tag, save_sparse_csr, save_dictionary
from toolbox.utils import print_top_words, dict_to_list
print("\n")


def _most_common(l):
    """
    Function to return the most common item from a list
    :param l: list
    :return: string (most common item)
    """
    data = Counter(l)
    return data.most_common(1)[0][0]


def _text_content_extraction(log, result_directory):
    """
    Function to extract the textual content from the files
    :param log: dataframe pandas
    :param result_directory: string
    :return: list of strings
    """
    # collect text
    full_text = []
    for i in tqdm(range(log.shape[0])):
        filename = log.at[i, "matrix_name"]
        multi_header = log.at[i, "multiheader"]
        header = log.at[i, "header_name"]
        path = os.path.join(result_directory, "data_fitted", filename)
        index = [0]
        if multi_header:
            index = pd.MultiIndex.from_tuples(header)
        with open(path, mode="rt", encoding="utf-8") as f:
            for j in range(len(index)):
                f.readline()
            full_text.append(f.read())

    return full_text


def _text_extradata_extraction(log, result_directory):
    """
    Function to extract the textual content from the extradata files
    :param log: dataframe pandas
    :param result_directory: string
    :return: list of strings
    """
    # collect text
    full_text = []
    for i in tqdm(range(log.shape[0])):
        if log.at[i, "extradata"]:
            filename = log.at[i, "matrix_name"]
            path = os.path.join(result_directory, "extradata_cleaning",
                                filename)
            with open(path, mode='rt', encoding='utf-8') as f:
                full_text.append(f.read())
        else:
            full_text.append("")

    return full_text


def _text_header_extraction(log):
    """
    Function to extract the header from the files
    :param log: dataframe pandas
    :return: list of strings
    """
    return list(log["header_name"])


def _text_content_header_extraction(log, result_directory):
    """
    Function to extract the textual content from the files
    :param log: dataframe pandas
    :param result_directory: string
    :return: list of strings
    """
    # collect text
    full_text = []
    for i in tqdm(range(log.shape[0])):
        filename = log.at[i, "matrix_name"]
        path = os.path.join(result_directory, "data_fitted", filename)
        with open(path, mode="rt", encoding="utf-8") as f:
            full_text.append(f.read())

    return full_text


def _text_full_extraction(log, result_directory):
    """
    Function to extract the full textual content from the files
    :param log: dataframe pandas
    :param result_directory: string
    :return: list of strings
    """
    # collect text
    full_text = []
    for i in tqdm(range(log.shape[0])):
        text = []
        filename = log.at[i, "matrix_name"]
        path = os.path.join(result_directory, "data_fitted", filename)
        with open(path, mode="rt", encoding="utf-8") as f:
            text.append(f.read())
        if log.at[i, "extradata"]:
            path_extradata = os.path.join(result_directory,
                                          "extradata_cleaning", filename)
            with open(path_extradata, mode='rt', encoding='utf-8') as f:
                text.append(f.read())
        else:
            text.append("")
        full_text.append(" ".join(text))

    return full_text


def _text_extraction(path_log, result_directory, content_bool, header_bool,
                     extradata_bool):
    """
    Function to select which textual part we need to extract from the files
    :param path_log: string
    :param result_directory: string
    :param content_bool: boolean
    :param header_bool: boolean
    :param extradata_bool: boolean
    :return: list of strings
    """
    df_log = pd.read_csv(path_log, sep=";", encoding="utf-8", index_col=False)
    print("number of files: ", df_log.shape[0])
    if content_bool and header_bool and extradata_bool:
        return _text_full_extraction(df_log, result_directory)
    elif content_bool and not header_bool and not extradata_bool:
        return _text_content_extraction(df_log, result_directory)
    elif not content_bool and header_bool and not extradata_bool:
        return _text_header_extraction(df_log)
    elif not content_bool and not header_bool and extradata_bool:
        return _text_extradata_extraction(df_log, result_directory)
    elif content_bool and header_bool and not extradata_bool:
        return _text_content_header_extraction(df_log, result_directory)
    elif not content_bool and header_bool and extradata_bool:
        raise ValueError("Extraction of header and extradata only has to be "
                         "implemented")
    elif content_bool and not header_bool and extradata_bool:
        raise ValueError("Extraction of content and extradata only has to be "
                         "implemented")
    else:
        raise ValueError("No text extracted")


def count_computation(path_log, result_directory, content_bool, header_bool,
                      extradata_bool, stop_words, path_count, path_vocabulary,
                      max_df, min_df):
    """
    Function to count words
    :param path_log: string
    :param result_directory: string
    :param content_bool: boolean
    :param header_bool: boolean
    :param extradata_bool: boolean
    :param stop_words: list of strings
    :param path_count: string
    :param path_vocabulary: string
    :param max_df: integer/float
    :param min_df: integer/float
    :return: sparse matrix [n_samples, n_features], dictionary
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
                                 max_df=max_df,
                                 min_df=min_df,
                                 max_features=None,
                                 vocabulary=None,
                                 binary=False)
    print("text extraction...")
    full_text = _text_extraction(path_log,
                                 result_directory,
                                 content_bool,
                                 header_bool,
                                 extradata_bool)
    print("counting process...")
    count_matrix = count_vect.fit_transform(full_text)
    count_matrix = sp.csr_matrix(count_matrix)
    vocabulary = count_vect.vocabulary_
    save_sparse_csr(path_count, count_matrix)
    save_dictionary(vocabulary, path_vocabulary, ["word", "index"])
    return count_matrix, vocabulary


def tfidf_computation(count_matrix):
    """
    Function to compute tfidf matrix
    :param count_matrix: sparse row matrix [n_samples, n_features]
    :return: sparse row matrix [n_samples, n_features]
    """
    tfidf_transf = TfidfTransformer(norm=u'l2',
                                    use_idf=True,
                                    smooth_idf=True,
                                    sublinear_tf=False)
    tfidf = tfidf_transf.fit_transform(count_matrix)
    tfidf = sp.csr_matrix(tfidf)
    return tfidf


def normalize_matrix(matrix, result_directory, title):
    """
    Function to normalize a matrix (the sum of each row gives the same value)
    :param matrix: sparse csr matrix [n_samples, n_features]
    :param result_directory: string
    :param title: string
    :return: sparse csr matrix [n_samples, n_features],
             numpy array [n_samples, 1]
    """
    # TODO improve this function
    sum = np.zeros((matrix.shape[0], 1))
    weight = np.zeros((matrix.shape[0], 1))
    for i in range(matrix.shape[0]):
        row = matrix.getrow(i)
        sum[i, 0] = row.sum()
    m = np.mean(sum)
    for i in range(matrix.shape[0]):
        if sum[i, 0] == 0:
            weight[i, 0] = 1
        else:
            weight[i, 0] = m / sum[i, 0]
    weight = sp.csr_matrix(weight)
    new_matrix = matrix.multiply(weight)
    path_count_normalized = os.path.join(result_directory, title)
    save_sparse_csr(path_count_normalized, new_matrix)
    return new_matrix, weight.todense()


def mix_matrices(matrices, weights, vocabularies):
    """
    Function to mix different count matrices
    :param matrices: list of sparse csr matrices [n_samples, n_features]
    :param weights: list of floats
    :param vocabularies: list of lists of strings
    :return: sparse csr matrix [n_samples, n_features], list of strings
    """
    # TODO improve this function
    l = []
    n_row = None
    n_matrices = len(matrices)
    d = defaultdict(lambda: [-1 for k in range(n_matrices)])
    for i in range(n_matrices):
        matrix = matrices[i]
        n_row = matrix.shape[0]
        weight = weights[i]
        l.append(matrix.power(weight))
        vocabulary = vocabularies[i]
        for ind, word in enumerate(vocabulary):
            d[word][i] = ind
    res = sp.lil_matrix((n_row, len(d)))
    col_index = 0
    features = []
    for word in d:
        features.append(word)
        col = sp.csr_matrix((n_row, 1))
        for k in range(n_matrices):
            col += l[k].getcol(d[word][k])
        res[:, col_index] = col
        col_index += 1
    return res.tocsr(), features


def stemming_matrix(matrix, features):
    """
    Function to concatenate the columns of a matrix by stemming and merging
    its features
    :param matrix: sparse csr matrix [n_samples, n_words]
    :param features: list of string (ordered with the right index)
    :return: sparse csr matrix [n_samples, n_stem_words], list of string
    """
    # TODO fix unstemming
    # TODO improve this function
    stemmer = SnowballStemmer("french")
    d_stem = defaultdict(lambda: [])
    d_index = defaultdict(lambda: [])
    features_stem = []
    for i, feature in tqdm(enumerate(features), desc="stemming"):
        if str(feature) == "nan":
            feature = "nan"
        stem = stemmer.stem(feature)
        d_stem[stem].append(feature)
        d_index[stem].append(i)
        if stem in features_stem:
            continue
        else:
            features_stem.append(stem)
    features_unstem = []
    for stem in features_stem:
        unstem = _most_common(d_stem[stem])
        features_unstem.append(unstem)

    print("length features :", len(features))
    print("length stem features :", len(features_stem))
    print("length unstem features :", len(features_unstem), "\n")

    new_matrix = []
    n_row = matrix.shape[0]
    for stem in tqdm(features_stem, desc="stacking stemmed columns"):
        index_to_stack = d_index[stem]
        columns_to_stack = []
        for i in index_to_stack:
            columns_to_stack.append(matrix.getcol(i))
        new_columns = sum(columns_to_stack)
        new_matrix.append(new_columns)
    matrix_stem = sp.lil_matrix((n_row, len(features_stem)))
    for col_index in tqdm(range(len(features_stem)),
                          desc="building stem matrix"):
        matrix_stem[:, col_index] = new_matrix[col_index]
    matrix_stem = matrix_stem.tocsr()

    print("shape matrix :", matrix.shape)
    print("type matrix :", type(matrix))
    print("shape stem matrix :", matrix_stem.shape)
    print("type stem matrix :", type(matrix_stem), "\n")

    return matrix_stem, features_unstem


def extract_features(path_log, result_directory, content_bool, header_bool,
                     extradata_bool, french_stopwords):
    """
    Function to extract, combine and save textual features
    :param path_log: string
    :param result_directory: string
    :param content_bool: boolean
    :param header_bool: boolean
    :param extradata_bool: boolean
    :param french_stopwords: list of strings
    :return: sparse row matrix [n_samples, n_words], dictionary,
             sparse row matrix [n_samples, n_words],
             list of sparse row matrices [n_samples, 1]
    """
    matrices = []
    weights = []
    vocabularies = []
    weighting_list = []
    # count
    if content_bool:
        print("--- extracting content ---")
        path_count = os.path.join(result_directory, "count_content.npz")
        path_vocabulary = os.path.join(result_directory,
                                       "token_vocabulary_content")
        count_matrix, vocab = \
            count_computation(path_log=path_log,
                              result_directory=result_directory,
                              content_bool=True,
                              header_bool=False,
                              extradata_bool=False,
                              stop_words=french_stopwords,
                              path_count=path_count,
                              path_vocabulary=path_vocabulary,
                              max_df=0.95,
                              min_df=2)
        print("normalization...")
        matrix, weighting = normalize_matrix(count_matrix, result_directory,
                                             "count_content_normalized")
        print("count shape :", matrix.shape)
        features = get_ordered_features(path_vocabulary)
        print("number of unique words :", len(features), "\n")
        matrices.append(matrix)
        weights.append(0.5)
        vocabularies.append(features)
        weighting_list.append(weighting)
    if header_bool:
        print("--- extracting header ---")
        path_count = os.path.join(result_directory, "count_header.npz")
        path_vocabulary = os.path.join(result_directory,
                                       "token_vocabulary_header")
        count_matrix, vocab = \
            count_computation(path_log=path_log,
                              result_directory=result_directory,
                              content_bool=False,
                              header_bool=True,
                              extradata_bool=False,
                              stop_words=french_stopwords,
                              path_count=path_count,
                              path_vocabulary=path_vocabulary,
                              max_df=0.95,
                              min_df=2)
        print("normalization...")
        matrix, weighting = normalize_matrix(count_matrix, result_directory,
                                             "count_header_normalized")
        print("count shape :", matrix.shape)
        features = get_ordered_features(path_vocabulary)
        print("number of unique word :", len(features), "\n")
        matrices.append(matrix)
        weights.append(0.25)
        vocabularies.append(features)
        weighting_list.append(weighting)
    if extradata_bool:
        print("--- extracting extradata ---")
        path_count = os.path.join(result_directory, "count_extradata.npz")
        path_vocabulary = os.path.join(result_directory,
                                       "token_vocabulary_extradata")
        count_matrix, vocab = \
            count_computation(path_log=path_log,
                              result_directory=result_directory,
                              content_bool=False,
                              header_bool=False,
                              extradata_bool=True,
                              stop_words=french_stopwords,
                              path_count=path_count,
                              path_vocabulary=path_vocabulary,
                              max_df=0.95,
                              min_df=2)
        print("normalization...")
        matrix, weighting = normalize_matrix(count_matrix, result_directory,
                                             "count_extradata_normalized")
        print("count shape :", matrix.shape)
        features = get_ordered_features(path_vocabulary)
        print("number of unique word :", len(features), "\n")
        matrices.append(matrix)
        weights.append(0.25)
        vocabularies.append(features)
        weighting_list.append(weighting)
    print("--- mixing matrices ---")
    count_matrix, features = mix_matrices(matrices, weights, vocabularies)
    path_count = os.path.join(result_directory, "count_normalized.npz")
    save_sparse_csr(path_count, count_matrix)
    d_features = {}
    for i, word in enumerate(features):
        d_features[word] = i
    path_vocabulary = os.path.join(result_directory, "token_vocabulary")
    save_dictionary(d_features, path_vocabulary, ["word", "index"])
    print("count shape :", count_matrix.shape, "\n")
    print("--- stemming matrix ---")
    count_stem_matrix, unstem_features = stemming_matrix(count_matrix,
                                                         features)
    path_count_stem = os.path.join(result_directory,
                                   "count_normalized_stem.npz")
    save_sparse_csr(path_count_stem, count_stem_matrix)
    d_features = {}
    for i, word in enumerate(unstem_features):
        d_features[word] = i
    path_vocabulary = os.path.join(result_directory, "token_vocabulary_unstem")
    save_dictionary(d_features, path_vocabulary, ["word", "index"])
    print("count shape :", count_stem_matrix.shape, "\n")
    # tfidf
    print("--- tfidf ---")
    tfidf = tfidf_computation(count_stem_matrix)
    path_tfidf = os.path.join(result_directory, "tfidf.npz")
    save_sparse_csr(path_tfidf, tfidf)
    print("tfidf shape :", tfidf.shape, "\n")

    return count_matrix, d_features, tfidf, weighting_list


def get_ordered_features(path_vocabulary):
    """
    Function to get a list a features with the right indexation
    :param path_vocabulary: string
    :return: list of strings
    """
    df = pd.read_csv(path_vocabulary, header=0, encoding="utf-8", sep=";",
                     index_col=False)
    df.sort_values(by="index", axis=0, ascending=True, inplace=True)
    return list(df["word"])


def compute_nmf(matrix, n_topics, feature_names, n_top_words, result_directory):
    """
    Function to compute nmf and save it
    :param matrix: sparse row matrix [n_samples, n_features]
    :param n_topics: integer
    :param feature_names: list of strings
    :param n_top_words: integer
    :param result_directory: string
    :return: sklearn fitted model, sparse row matrix [n_samples, n_topics]
    """
    print("--- NMF ---", "\n")
    nmf = NMF(n_components=n_topics, random_state=1, alpha=.1, l1_ratio=.5)
    nmf = nmf.fit(matrix)
    print_top_words(nmf, feature_names, n_top_words)
    w = nmf.transform(matrix)
    path_nmf = os.path.join(result_directory, "nmf.pkl")
    path_w = os.path.join(result_directory, "w.npy")
    joblib.dump(nmf, path_nmf)
    np.save(path_w, w)
    return nmf, w


def compute_knn(matrix, result_directory):
    """
    Function to compute knn and save it
    :param matrix: sparse row matrix [n_samples, n_features]
    :param result_directory: string
    :return: sklearn fitted model
    """
    print("--- KNN ---", "\n")
    knn = NearestNeighbors(n_neighbors=5,
                           radius=1.0,
                           algorithm='auto',
                           leaf_size=30,
                           metric='minkowski',
                           p=2,
                           metric_params=None,
                           n_jobs=1)
    knn = knn.fit(matrix)
    path_knn = os.path.join(result_directory, "knn.pkl")
    joblib.dump(knn, path_knn)
    return knn


def main(content_bool, header_bool, extradata_bool, n_topics, n_top_words,
         nltk_path, result_directory, path_log):
    """
    Function to run all the script
    :param content_bool: boolean
    :param header_bool: boolean
    :param extradata_bool: boolean
    :param n_topics: integer
    :param n_top_words: integer
    :param nltk_path: string
    :param result_directory: string
    :param path_log: string
    :return: sparse csr matrix [n_samples, n_features], dictionary,
             sparse csr matrix [n_samples, n_features],
             sparse csr matrix [n_samples, 1],
             sklearn fitted model,
             numpy matrix [n_samples, n_topics],
             sklearn fitted model
    """

    # stopwords
    nltk.data.path.append(nltk_path)
    french_stopwords = list(set(stopwords.words('french')))
    french_stopwords += ["unnamed", "http", "les", "des"]
    geojson = ['featurecollection', 'geometry', 'type', 'polygon',
               'coordinates', 'properties', 'feature', 'crs', 'urn', 'ogc',
               'def', 'libelle', 'multipolygon', 'name', 'point', 'linestring',
               'multilinestring', 'pprt', 'ppri', 'mvt', 'nda', 'bdx', 'gid',
               'pci', 'pprn']
    french_stopwords += geojson

    # extract textual features
    count_matrix, d_features, tfidf, weighting_list = \
        extract_features(path_log,
                         result_directory,
                         content_bool,
                         header_bool,
                         extradata_bool,
                         french_stopwords)
    features = dict_to_list(d_features)

    # fit models
    # nmf, w = compute_nmf(tfidf,
    #                      n_topics,
    #                      features,
    #                      n_top_words,
    #                      result_directory)
    # knn = compute_knn(w, result_directory)

    return count_matrix, d_features, tfidf, weighting_list

if __name__ == "__main__":

    # parameters
    content_bool = get_config_tag("content", "text_extraction")
    header_bool = get_config_tag("header", "text_extraction")
    extradata_bool = get_config_tag("extradata", "text_extraction")
    n_topics = get_config_tag("n_topics", "text_extraction")
    n_top_words = get_config_tag("n_top_words", "text_extraction")

    # paths
    nltk_path = get_config_tag("nltk", "text_extraction")
    result_directory = get_config_tag("result", "cleaning")
    path_log = os.path.join(result_directory, "log_final_reduced")

    # run code
    count_matrix, d_features, tfidf, weighting_list = \
        main(content_bool=content_bool,
             header_bool=header_bool,
             extradata_bool=extradata_bool,
             n_topics=n_topics,
             n_top_words=n_top_words,
             nltk_path=nltk_path,
             result_directory=result_directory,
             path_log=path_log)

    print("count_matrix shape", count_matrix.shape)
    print("tfidf shape", tfidf.shape)
    print("d_features length", len(d_features))
    # print("w shape", w.shape)
