# -*- coding: utf-8 -*-

""" Main script to analyze data """

# libraries
import os
import sys
from joblib.format_stack import format_exc
from toolbox.utils import get_config_tag
from toolbox.clean import get_ready, cleaner
from clean_files import main
from shutil import copyfile, rmtree
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
new_input_directory = "../data/input_problem"

os.mkdir(new_input_directory)

# problematic files
problems = []
for i, file in enumerate(os.listdir(input_directory)):
    if i >= 39400:
        problems.append(file)
        path_src = os.path.join(input_directory, file)
        path_dst = os.path.join(new_input_directory, file)
        copyfile(path_src, path_dst)

# run code
output_directory, path_log, path_error, path_metadata = \
    get_ready(result_directory, reset=False)
main(new_input_directory, result_directory, 10, False, dict_param)

# remove files
rmtree(result_directory)
rmtree(new_input_directory)
