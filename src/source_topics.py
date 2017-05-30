# -*- coding: utf-8 -*-

""" Source the different topics extracted through NMF. """

# libraries
import os
import pandas as pd
import numpy as np
import scipy.sparse as sp
from toolbox.utils import get_config_tag
from sklearn.externals import joblib
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