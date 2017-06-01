# -*- coding: utf-8 -*-

""" Main script to analyze data """

# libraries
import os
import sys
from joblib.format_stack import format_exc
from toolbox.utils import get_config_tag
from toolbox.clean import get_ready, cleaner
print("\n")

# parameters
input_directory = get_config_tag("input", "cleaning")
dict_param = dict()
dict_param["threshold_n_row"] = get_config_tag("threshold_n_row", "cleaning")
dict_param["ratio_sample"] = get_config_tag("ratio_sample", "cleaning")
dict_param["max_sample"] = get_config_tag("max_sample", "cleaning")
dict_param["threshold_n_col"] = get_config_tag("threshold_n_col", "cleaning")
dict_param["check_header"] = get_config_tag("check_header", "cleaning")
dict_param["threshold_json"] = get_config_tag("threshold_json", "cleaning")

result_directory = "../data/problem"

# problematic files
problems = []
for i, file in enumerate(os.listdir(input_directory)):
    if i >= 39400:
        problems.append(file)

# run code
output_directory, path_log, path_error, path_metadata = \
    get_ready(result_directory, reset=False)
n = 39400
for file in problems:
    print(n, file)
    try:
        cleaner(file, input_directory, output_directory, path_log, path_metadata,
                dict_param)
        print()
    except:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        text = format_exc(exc_type, exc_value, exc_traceback, context=5,
                          tb_offset=0)
        print(text, "\n")
    n += 1
