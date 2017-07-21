# -*- coding: utf-8 -*-

""" Main script to analyze data """

# libraries
import text_analysis
import source_topics
import graphs
from toolbox.utils import get_config_tag
print("\n")

# parameters
n_top_words = get_config_tag("n_top_words", "text_extraction")
wordcloud_bool = get_config_tag("wordcloud", "text_analysis")
kneighbors_bool = get_config_tag("kneighbors", "text_analysis")
distance_plot_bool = get_config_tag("distance_plot", "text_analysis")

# paths
result_directory = get_config_tag("result", "cleaning")

# analyze extracted text
text_analysis.main(result_directory=result_directory,
                   n_top_words=n_top_words,
                   wordcloud_bool=wordcloud_bool,
                   kneighbors_bool=kneighbors_bool,
                   distance_plot_bool=distance_plot_bool,
                   n_queries=5,
                   n_neighbors=5)

# source topics
source_topics.main(result_directory=result_directory,
                   n_top_words=5)

# plot graphs and build tables
graphs.main(result_directory=result_directory)
