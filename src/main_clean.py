# -*- coding: utf-8 -*-

""" Main script to clean data and edit metadata """

# libraries
import os

import clean_files
import text_extraction
from old import edit_metadata, distribution_files
from toolbox.utils import get_config_tag, get_config_trace

print("\n")

get_config_trace()

# path
input_directory = get_config_tag("input", "cleaning")
result_directory = get_config_tag("result", "cleaning")
metadata_directory = get_config_tag("output", "metadata")
path_log = os.path.join(result_directory, "log_final")
nltk_path = get_config_tag("nltk", "text_extraction")

# get parameters
workers = get_config_tag("n_jobs", "cleaning")
reset = get_config_tag("reset", "cleaning")
multi_bool = get_config_tag("multi", "cleaning")
dict_param = dict()
dict_param["threshold_n_row"] = get_config_tag("threshold_n_row", "cleaning")
dict_param["ratio_sample"] = get_config_tag("ratio_sample", "cleaning")
dict_param["max_sample"] = get_config_tag("max_sample", "cleaning")
dict_param["threshold_n_col"] = get_config_tag("threshold_n_col", "cleaning")
dict_param["check_header"] = get_config_tag("check_header", "cleaning")
dict_param["threshold_json"] = get_config_tag("threshold_json", "cleaning")
count_bool = get_config_tag("count", "distribution")
log_bool = get_config_tag("log", "distribution")
plot_bool = get_config_tag("plot", "distribution")
error_bool = get_config_tag("error", "distribution")
efficiency_bool = get_config_tag("efficiency", "distribution")
content_bool = get_config_tag("content", "text_extraction")
header_bool = get_config_tag("header", "text_extraction")
metadata_bool = get_config_tag("metadata", "text_extraction")
n_topics = get_config_tag("n_topics", "text_extraction")
n_top_words = get_config_tag("n_top_words", "text_extraction")

# clean files
clean_files.main(input_directory=input_directory,
                 result_directory=result_directory,
                 workers=workers,
                 reset=reset,
                 dict_param=dict_param,
                 multi=multi_bool)

# compute and show distribution
distribution_files.main(count_bool=count_bool,
                        log_bool=log_bool,
                        error_bool=error_bool,
                        efficiency_bool=efficiency_bool,
                        result_directory=result_directory,
                        files_directory=input_directory)

# edit metadata
edit_metadata.main(result_directory=result_directory,
                   metadata_directory=metadata_directory)

# extract text
text_extraction.main(content_bool=content_bool,
                     header_bool=header_bool,
                     metadata_bool=metadata_bool,
                     n_topics=n_topics,
                     n_top_words=n_top_words,
                     nltk_path=nltk_path,
                     result_directory=result_directory,
                     path_log=path_log)
