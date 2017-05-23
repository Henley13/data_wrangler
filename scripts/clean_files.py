# -*- coding: utf-8 -*-

""" Detect the file extension and reshape it. """

# libraries
import magic
import os
from joblib import Parallel, delayed
from src.toolbox.utils import log_error, get_config_tag
from src.toolbox.clean import cleaner, get_ready
print("\n")

# parameters
input_directory = get_config_tag("input", "cleaning")
result_directory = get_config_tag("result", "cleaning")
workers = get_config_tag("n_jobs", "cleaning")
reset = get_config_tag("reset", "cleaning")
threshold_n_row = get_config_tag("threshold_n_row", "cleaning")
ratio_sample = get_config_tag("ratio_sample", "cleaning")
max_sample = get_config_tag("max_sample", "cleaning")
threshold_n_col = get_config_tag("threshold_n_col", "cleaning")
check_header = get_config_tag("check_header", "cleaning")
threshold_json = get_config_tag("threshold_json", "cleaning")

n_files = len(os.listdir(input_directory))
print("number of files :", n_files, "\n")

# make the result directory ready
output_directory, path_log, path_error, path_metadata = \
    get_ready(result_directory, reset)


def worker_cleaning_activity(filename,
                             input_directory=input_directory,
                             output_directory=output_directory,
                             path_log=path_log,
                             threshold_n_row=threshold_n_row,
                             ratio_sample=ratio_sample,
                             max_sample=max_sample,
                             threshold_n_col=threshold_n_col,
                             check_header=check_header,
                             threshold_json=threshold_json,
                             path_error=path_error):
    """
    Function to encapsulate the process and use multiprocessing.
    :return:
    """
    try:
        cleaner(filename, input_directory, output_directory, path_log,
                threshold_n_row, ratio_sample, max_sample,
                threshold_n_col, check_header, threshold_json)
    except:
        path = os.path.join(input_directory, filename)
        size_file = os.path.getsize(path)
        extension = ""
        print(size_file)
        print(os.path.isfile(os.path.join(output_directory, filename)))
        if size_file > 0 and not os.path.isfile(os.path.join(output_directory,
                                                             filename)):
            extension = magic.Magic(mime=True).from_file(path)
        log_error(os.path.join(path_error, filename), [filename, extension])

# multiprocessing
Parallel(n_jobs=workers, verbose=20)(delayed(worker_cleaning_activity)
                                     (filename=file)
                                     for file in os.listdir(input_directory))

print("\n")
print("total number of files :", len(os.listdir(output_directory)))
