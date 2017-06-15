# -*- coding: utf-8 -*-

""" Main script to clean data and edit metadata """

# libraries
import clean_files
import distribution_files
import edit_metadata
from toolbox.utils import get_config_tag, get_config_trace
print("\n")

get_config_trace()

# path
input_directory = get_config_tag("input", "cleaning")
result_directory = get_config_tag("result", "cleaning")
metadata_directory = get_config_tag("output", "metadata")

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
extension_bool = get_config_tag("extension", "distribution")
count_bool = get_config_tag("count", "distribution")
log_bool = get_config_tag("log", "distribution")
plot_bool = get_config_tag("plot", "distribution")
error_bool = get_config_tag("error", "distribution")
efficiency_bool = get_config_tag("efficiency", "distribution")

# clean files
clean_files.main(input_directory=input_directory,
                 result_directory=result_directory,
                 workers=workers,
                 reset=reset,
                 dict_param=dict_param,
                 multi=multi_bool)

# compute and show distribution
distribution_files.main(extension_bool=extension_bool,
                        count_bool=count_bool,
                        log_bool=log_bool,
                        plot_bool=plot_bool,
                        error_bool=error_bool,
                        efficiency_bool=efficiency_bool,
                        result_directory=result_directory,
                        files_directory=input_directory)

# edit metadata
edit_metadata.main(result_directory=result_directory,
                   metadata_directory=metadata_directory)
