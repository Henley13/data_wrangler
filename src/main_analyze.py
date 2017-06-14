# -*- coding: utf-8 -*-

""" Main script to analyze data """

# libraries
import os
import text_extraction
import text_analysis
import source_topics
from toolbox.utils import get_config_tag
print("\n")

# parameters
content_bool = get_config_tag("content", "text_extraction")
header_bool = get_config_tag("header", "text_extraction")
metadata_bool = get_config_tag("metadata", "text_extraction")
n_topics = get_config_tag("n_topics", "text_extraction")
n_top_words = get_config_tag("n_top_words", "text_extraction")
wordcloud_bool = get_config_tag("wordcloud", "text_analysis")
kneighbors_bool = get_config_tag("kneighbors", "text_analysis")
distance_plot_bool = get_config_tag("distance_plot", "text_analysis")

# paths
nltk_path = get_config_tag("nltk", "text_extraction")
result_directory = get_config_tag("result", "cleaning")
path_log = os.path.join(result_directory, "log_final")

# extract text
count_matrix, d_features, tfidf, weighting_list, nmf, w, knn = \
    text_extraction.main(content_bool=content_bool,
                         header_bool=header_bool,
                         metadata_bool=metadata_bool,
                         n_topics=n_topics,
                         n_top_words=n_top_words,
                         nltk_path=nltk_path,
                         result_directory=result_directory,
                         path_log=path_log)

# analyze text extracted
text_analysis.main(result_directory=result_directory,
                   n_top_words=n_top_words,
                   wordcloud_bool=wordcloud_bool,
                   kneighbors_bool=kneighbors_bool,
                   distance_plot_bool=distance_plot_bool,
                   n_queries=5,
                   n_neighbors=5)

# source topics
source_topics.main(result_directory=result_directory)

