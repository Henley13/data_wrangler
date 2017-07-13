# -*- coding: utf-8 -*-

""" Detect the file extension and clean it. """

import magic
import os
import pandas as pd
import shutil
from shutil import copyfile
from tqdm import tqdm
from joblib import Parallel, delayed
from toolbox.utils import log_error, get_config_tag, get_config_trace
from toolbox.clean import cleaner, get_ready, file_is_json
print("\n")


def worker_cleaning_activity(filename, input_directory, output_directory,
                             path_log, path_metadata, dict_param, path_error,
                             error_directory, temp_dir):
    """
    Function to encapsulate the process and use multiprocessing.
    :param filename: string
    :param input_directory: string
    :param output_directory: string
    :param path_log: string
    :param path_metadata: string
    :param dict_param: dictionary
    :param path_error: string (to store error log)
    :param error_directory: string (to store files)
    :param temp_dir: string
    :return:
    """
    try:
        cleaner(filename, input_directory, output_directory, path_log,
                path_metadata, dict_param, temp_dir)
    except Exception:
        path = os.path.join(input_directory, filename)
        if os.path.getsize(path) == 0:
            return
        extension = magic.Magic(mime=True).from_file(path)
        try:
            is_json = file_is_json(filename, input_directory, dict_param)
            if (extension in ["text/plain", "application/octet-stream"] and
                    is_json):
                extension = "json"
        except Exception:
            pass
        log_error(os.path.join(path_error, filename), [filename, extension])
        path_out = os.path.join(error_directory, filename)
        if not os.path.isfile(path_out):
            copyfile(path, path_out)
    return


def main(input_directory, result_directory, workers, reset, dict_param, multi):
    """
    Function to run all the script
    :param input_directory: string
    :param result_directory: string
    :param workers: integer
    :param reset: boolean
    :param dict_param: dictionary
    :param multi: boolean
    :return:
    """

    print("clean files...", "\n")

    # make the result directory ready
    (output_directory, path_log, path_error, path_metadata, error_directory,
     temporary_directory) = get_ready(result_directory, reset)

    if multi:
        # multiprocessing
        Parallel(n_jobs=workers, verbose=20)(delayed(worker_cleaning_activity)
                                             (filename=file,
                                              input_directory=input_directory,
                                              output_directory=output_directory,
                                              path_log=path_log,
                                              path_metadata=path_metadata,
                                              dict_param=dict_param,
                                              path_error=path_error,
                                              error_directory=error_directory,
                                              temp_dir=temporary_directory)
                                             for file in
                                             os.listdir(input_directory))
    else:
        for file in tqdm(os.listdir(input_directory)):
            worker_cleaning_activity(filename=file,
                                     input_directory=input_directory,
                                     output_directory=output_directory,
                                     path_log=path_log,
                                     path_metadata=path_metadata,
                                     dict_param=dict_param,
                                     path_error=path_error,
                                     error_directory=error_directory,
                                     temp_dir=temporary_directory)

    print()
    print("close temporary directory", "\n")
    shutil.rmtree(temporary_directory)

    print("total number of files cleaned (excel sheets included) :",
          len(os.listdir(output_directory)))
    df = pd.read_csv(path_log, sep=";", encoding="utf-8", index_col=False)
    print("total number of files saved in the log :", df.shape[0])
    print("total number of extra data :", len(os.listdir(path_metadata)))
    print("total number of failures :", len(os.listdir(path_error)), "\n")

    return

if __name__ == "__main__":

    get_config_trace()

    # parameters
    input_directory = get_config_tag("input", "cleaning")
    result_directory = get_config_tag("result", "cleaning")
    workers = get_config_tag("n_jobs", "cleaning")
    reset = get_config_tag("reset", "cleaning")
    multi = get_config_tag("multi", "cleaning")

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
         dict_param=dict_param,
         multi=multi)
