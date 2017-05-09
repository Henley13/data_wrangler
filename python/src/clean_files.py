#!/bin/python3
# coding: utf-8

""" Detect the file extension and reshape it. """

# libraries
import os
import magic
from .clean_files_functions import cleaner, get_ready
from .functions import log_error
from joblib import Parallel, delayed
print("\n")

##############
# parameters #
##############

##############################################################################
##############################################################################

# number of workers to use
workers = 2

# boolean to reset the output directory or not
reset = True

# paths
input_directory = "../data/data_collected_xml"
output_directory = "../data/test_fitted2"
path_log = "../data/log_cleaning2"
n_files = len(os.listdir(input_directory))
print("number of files :", n_files, "\n")

# minimum number of rows needed to analyze a sample of the file
# (otherwise, we use the entire file)
threshold_n_row = 100

# percentage of rows to extract from the file to build a sample
ratio_sample = 20

# maximum size of a sample
max_sample = 1000

# minimum frequency to reach in order to accept a number of columns
# (n_col = N if at least threshold_n_col * 100 % rows have N columns)
threshold_n_col = 0.8

# number of rows to analyze when we are searching for a consistent header
check_header = 10

# minimum frequency to reach for specific characters in order to classify
# a file as a json
threshold_json = 0.004

##############################################################################
##############################################################################

# make the output directory ready
path_error = get_ready(output_directory, path_log, reset)


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
