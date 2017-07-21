# -*- coding: utf-8 -*-

""" Detect the file extension and clean it. """

import magic
import os
import pandas as pd
import shutil
from shutil import copyfile
from tqdm import tqdm
from joblib import Parallel, delayed
from toolbox.utils import (log_error, get_config_tag, get_config_trace,
                           reset_log_error)
from toolbox.clean import cleaner, file_is_json,
print("\n")


def _check_directory(path_directory, reset):
    """
    Function to check and initialize a directory.

    Parameters
    ----------
    path_directory : str
        Path of the directory

    reset : bool
        Boolean to decide if the directory has to be reinitialized

    Returns
    -------
    """
    if not os.path.isdir(path_directory):
        os.mkdir(path_directory)
    else:
        if reset:
            shutil.rmtree(path_directory)
            os.mkdir(path_directory)
        else:
            pass
    return


def _get_ready(result_directory, reset):
    """
    Function to make the output directory ready to host cleaned data.

    Parameters
    ----------
    result_directory : str
        Path of the output directory

    reset : bool
        Boolean to decide if the former directory is preserved or not

    Returns
    -------
    """
    print("make directories ready", "\n")

    # paths
    output_directory = os.path.join(result_directory, "data_fitted")
    path_log = os.path.join(result_directory, "log_cleaning")
    error_log_directory = os.path.join(result_directory, "fit_errors")
    metadata_directory = os.path.join(result_directory, "metadata_cleaning")
    error_file_directory = os.path.join(result_directory, "files_to_fix")

    # check result directory
    if not os.path.isdir(result_directory):
        os.mkdir(result_directory)

    # check output directory
    _check_directory(output_directory, reset)

    # check the error log
    reset_log_error(error_log_directory, reset)

    # check the error directory
    _check_directory(error_file_directory, reset)

    # check the log
    if not reset and os.path.isfile(path_log):
        pass
    else:
        with open(path_log, mode="wt", encoding="utf-8") as f:
            f.write("matrix_name;file_name;source_file;n_row;n_col;integer;"
                    "float;object;metadata;time;header;multiheader;header_name;"
                    "extension;zipfile")
            f.write("\n")

    # check output directory for metadata
    _check_directory(metadata_directory, reset)

    return (output_directory, path_log, error_log_directory, metadata_directory,
            error_file_directory)


def worker_cleaning_activity(filename, input_directory, output_directory,
                             path_log, path_metadata, dict_param, path_error,
                             error_directory):
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
    :return:
    """
    # TODO add geojson here and in clean.py (using geopandas)
    # TODO link these errors with the metadata previously collected (producer, page, etc.)
    try:
        cleaner(filename, input_directory, output_directory, path_log,
                path_metadata, dict_param)
    except Exception:
        path = os.path.join(input_directory, filename)
        if os.path.getsize(path) == 0:
            extension = "zerobyte"
        else:
            extension = magic.Magic(mime=True).from_file(path)
            try:
                if extension in ["text/plain", "application/octet-stream"]:
                    is_json = file_is_json(filename, input_directory,
                                               dict_param)
                    if is_json:
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
    (output_directory, path_log, error_log_directory, metadata_directory,
     error_directory) = _get_ready(result_directory, reset)

    if multi:
        # multiprocessing
        Parallel(n_jobs=workers, verbose=20)(delayed(worker_cleaning_activity)
                                             (filename=file,
                                              input_directory=input_directory,
                                              output_directory=output_directory,
                                              path_log=path_log,
                                              path_metadata=metadata_directory,
                                              dict_param=dict_param,
                                              path_error=error_log_directory,
                                              error_directory=error_directory)
                                             for file in
                                             os.listdir(input_directory))
    else:
        for file in tqdm(os.listdir(input_directory)):
            worker_cleaning_activity(filename=file,
                                     input_directory=input_directory,
                                     output_directory=output_directory,
                                     path_log=path_log,
                                     path_metadata=metadata_directory,
                                     dict_param=dict_param,
                                     path_error=error_log_directory,
                                     error_directory=error_directory)

    print()

    print("total number of files cleaned (excel sheets included) :",
          len(os.listdir(output_directory)))
    df = pd.read_csv(path_log, sep=";", encoding="utf-8", index_col=False)
    print("total number of files saved in the log :", df.shape[0])
    print("total number of extra data :", len(os.listdir(metadata_directory)))
    print("total number of failures :", len(os.listdir(error_log_directory)),
          "\n")

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
