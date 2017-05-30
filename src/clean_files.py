# -*- coding: utf-8 -*-

""" Detect the file extension and clean it. """

import magic
import os
from joblib import Parallel, delayed
from toolbox.utils import log_error, get_config_tag, get_config_trace
from toolbox.clean import cleaner, get_ready
print("\n")


def worker_cleaning_activity(filename,
                             input_directory,
                             output_directory,
                             path_log,
                             path_metadata,
                             dict_param,
                             path_error):
    """
    Function to encapsulate the process and use multiprocessing.
    :param filename: string
    :param input_directory: string
    :param output_directory: string
    :param path_log: string
    :param path_metadata: string
    :param dict_param: dictionary
    :param path_error: string
    :return:
    """
    try:
        cleaner(filename, input_directory, output_directory, path_log,
                path_metadata, dict_param)
    except:
        path = os.path.join(input_directory, filename)
        size_file = os.path.getsize(path)
        extension = "empty file"
        if size_file > 0:
            extension = magic.Magic(mime=True).from_file(path)
        log_error(os.path.join(path_error, filename), [filename, extension])
    return


def main(input_directory, result_directory, workers, reset, dict_param):
    """
    Function to run all the script
    :param input_directory: string
    :param result_directory: string
    :param workers: integer
    :param reset: boolean
    :param dict_param: dictionary
    :return:
    """

    print("clean files...")
    n_files = len(os.listdir(input_directory))
    print("number of files :", n_files, "\n")

    # make the result directory ready
    output_directory, path_log, path_error, path_metadata = \
        get_ready(result_directory, reset)

    # multiprocessing
    Parallel(n_jobs=workers, verbose=20)(delayed(worker_cleaning_activity)
                                         (filename=file,
                                          input_directory=input_directory,
                                          output_directory=output_directory,
                                          path_log=path_log,
                                          path_metadata=path_metadata,
                                          dict_param=dict_param,
                                          path_error=path_error)
                                         for file in
                                         os.listdir(input_directory))

    print("\n")
    print("total number of files :", len(os.listdir(output_directory)), "\n")

if __name__ == "__main__":

    get_config_trace()

    # parameters
    input_directory = get_config_tag("input", "cleaning")
    result_directory = get_config_tag("result", "cleaning")
    workers = get_config_tag("n_jobs", "cleaning")
    reset = get_config_tag("reset", "cleaning")

    dict_param = dict()
    dict_param["threshold_n_row"] = get_config_tag("threshold_n_row",
                                                   "cleaning")
    dict_param["ratio_sample"] = get_config_tag("ratio_sample", "cleaning")
    dict_param["max_sample"] = get_config_tag("max_sample", "cleaning")
    dict_param["threshold_n_col"] = get_config_tag("threshold_n_col",
                                                   "cleaning")
    dict_param["check_header"] = get_config_tag("check_header", "cleaning")
    dict_param["threshold_json"] = get_config_tag("threshold_json", "cleaning")

    # run code
    main(input_directory=input_directory,
         result_directory=result_directory,
         workers=workers,
         reset=reset,
         dict_param=dict_param)
