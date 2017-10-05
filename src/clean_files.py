# -*- coding: utf-8 -*-

""" Detect the file extension and clean it. """

import os
import shutil
from joblib import Parallel, delayed
from toolbox.utils import (get_config_tag, get_config_trace, check_directory)
from toolbox.clean import cleaner
print("\n")


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
    error_log_directory = os.path.join(result_directory, "error_cleaning")
    extradata_directory = os.path.join(result_directory, "extradata_cleaning")
    log_directory = os.path.join(result_directory, "log_cleaning")

    # check result directory
    check_directory(result_directory, reset)

    # check output directory
    check_directory(output_directory, reset)

    # check the error log directory
    check_directory(error_log_directory, reset)

    # check output directory for metadata
    check_directory(extradata_directory, reset)

    # check the log directory
    check_directory(log_directory, reset)

    return (output_directory, log_directory, error_log_directory,
            extradata_directory)


def worker_cleaning_activity(filename, input_directory, output_directory,
                             log_directory, extradata_directory, dict_param,
                             error_log_directory):
    """
    Function to encapsulate the cleaning process and use multiprocessing.

    Parameters
    ----------
    filename : str
        Name of the file to clean (most of the time its Id)

    input_directory : str
        Path of the collected data directory

    output_directory : str
        Path of the output directory

    log_directory : str
        Path of the log directory

    extradata_directory : str
        Path of the extradata directory

    dict_param : dict
        Dictionary with several parameters stored

    error_log_directory : str
        Path of the error log directory

    Returns
    -------
    """
    # TODO add geojson here and in clean.py (using geopandas)
    cleaner(filename, input_directory, output_directory, log_directory,
            extradata_directory, dict_param, error_log_directory)
    return


def remove_temporary(output_directory):
    """
    Function to remove the temporary directories from the output directory.

    Parameters
    ----------
    output_directory: str
        Path of the output directory

    Returns
    -------
    """
    print("closed temporary directories", "\n")

    for file in os.listdir(output_directory):
        path = os.path.join(output_directory, file)
        if os.path.isfile(path):
            pass
        else:
            if file[0:3] == "tmp":
                shutil.rmtree(path)
    return


def main(input_directory, result_directory, workers, reset, dict_param, multi):
    """
    Function to run all the script.

    Parameters
    ----------
    input_directory : str
        Path of the collected data directory

    result_directory : str
        Path of the result directory

    workers : int
        Number of workers to use

    reset : bool
        Boolean to decide if the former result directory is preserved or not

    dict_param : dict
        Dictionary with several parameters stored

    multi : bool
        Boolean to decide if multiprocessing is used

    Returns
    -------
    """
    print("clean files...", "\n")

    # make the result directory ready
    (output_directory, log_directory, error_log_directory,
     extradata_directory) = _get_ready(result_directory, reset)

    if multi:
        # multiprocessing
        Parallel(n_jobs=workers,
                 verbose=20)(delayed(worker_cleaning_activity)
                                    (filename=file,
                                     input_directory=input_directory,
                                     output_directory=output_directory,
                                     log_directory=log_directory,
                                     extradata_directory=extradata_directory,
                                     dict_param=dict_param,
                                     error_log_directory=error_log_directory)
                             for file in os.listdir(input_directory))
    else:
        for i, file in enumerate(sorted(os.listdir(input_directory))):
            print(i, "=>", file, "...")
            worker_cleaning_activity(filename=file,
                                     input_directory=input_directory,
                                     output_directory=output_directory,
                                     log_directory=log_directory,
                                     extradata_directory=extradata_directory,
                                     dict_param=dict_param,
                                     error_log_directory=error_log_directory)
            print("...", file, "done!")

    print()

    # remove the temporary directories from the output directory
    remove_temporary(output_directory)

    print("total number of files cleaned (excel sheets included) :",
          len(os.listdir(output_directory)))
    print("total number of files saved in the log :",
          len(os.listdir(log_directory)))
    print("total number of extra data :", len(os.listdir(extradata_directory)))
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
