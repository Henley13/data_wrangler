# -*- coding: utf-8 -*-

""" Functions to compute some basic statistics on the fitted files. """

# libraries
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from toolbox.utils import get_config_tag
print("\n")


def count(files_directory):
    """
    Function to show the distribution of our extensions
    :param files_directory: string
    :return:
    """
    print("### count ###", "\n")

    # test if the result exists
    main_directory = os.path.dirname(files_directory)
    sum_extension_path = os.path.join(main_directory, "sum_extension")
    if not os.path.isfile(sum_extension_path):
        raise FileNotFoundError("sum_extension doesn't exist.")

    # analyze data
    df_extension = pd.read_csv(sum_extension_path, sep=";", encoding="utf-8",
                               index_col=False)
    print(list(df_extension.columns))
    print(df_extension.shape, "\n")
    print("total files :")
    print(df_extension["extension"].value_counts(), "\n")
    print("unzipped files :")
    print(df_extension.query("zipfile == True")["extension"].value_counts(),
          "\n")
    print("direct files :")
    print(df_extension.query("zipfile == False")["extension"].value_counts(),
          "\n")
    return


def log(result_directory):
    """
    Function to show the results of our cleaning process
    :param result_directory: string
    :return:
    """
    print("### log ###", "\n")

    # get data
    path_log = os.path.join(result_directory, "log_cleaning")
    df_log = pd.read_csv(path_log, sep=";", encoding="utf-8", index_col=False)
    print(list(df_log.columns))
    print(df_log.shape, "\n")

    # large matrices
    n = 0
    for i in range(df_log.shape[0]):
        if df_log.at[i, "n_row"] >= 1000:
            n += 1
    print("number of matrices with more than 1000 rows :", n, "\n")

    # distribution
    print("row distribution")
    for i in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
        print("%ième percentile :" % i, np.percentile(df_log["n_row"], i))
    print("\n")

    print("col distribution")
    for i in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
        print("%ième percentile :" % i, np.percentile(df_log["n_col"], i))
    print("\n")

    print("integer :", sum(df_log["integer"]))
    print("float :", sum(df_log["float"]))
    print("object :", sum(df_log["object"]), "\n")

    return


def error(result_directory):
    """
    Function to count the different errors that occurred during the cleaning
    process
    :param result_directory: string
    :return:
    """

    print("### error ###", "\n")

    # reset file
    sum_error_path = os.path.join(result_directory, "sum_error")
    if os.path.isfile(sum_error_path):
        os.remove(sum_error_path)
    with open(sum_error_path, mode="wt", encoding="utf-8") as f:
        f.write("filename;extension;error;content")
        f.write("\n")

    # get data
    errors_directory = os.path.join(result_directory, "fit_errors")
    for filename in tqdm(os.listdir(errors_directory)):
        path = os.path.join(errors_directory, filename)
        with open(path, mode="rt", encoding="utf-8") as f:
            c = f.readlines()
            extension = c[2].strip()
            error = c[3].split(" ")[0]
            content = c[-2]
        with open(sum_error_path, mode="at", encoding="utf-8") as f:
            f.write(";".join([str(filename), str(extension), str(error),
                              str(content)]))

    # analyze data
    df_error = pd.read_csv(sum_error_path, sep=";", encoding="utf-8",
                           index_col=False)
    print(list(df_error.columns))
    print(df_error.shape, "\n")
    print(df_error['extension'].value_counts(), "\n")
    print(df_error['error'].value_counts(), "\n")

    print("--------------------", "\n")

    extensions = list(set(list(df_error["extension"])))
    for ext in extensions:
        print("extension :", ext, "\n")
        query = "extension == '%s'" % ext
        df_error_ext = df_error.query(query)
        print(df_error_ext["error"].value_counts(), "\n")
        max_e = df_error_ext["error"].value_counts().index.tolist()[0]
        print(df_error_ext.query("error == '%s'" % max_e)["content"].
              value_counts(), "\n")
        print("---", "\n")

    return


def efficiency(result_directory, files_directory):
    """
    Function to compute the efficiency of our cleaning process
    :param result_directory: string
    :param files_directory: string
    :return:
    """
    print("### efficiency ###", "\n")

    # paths
    main_directory = os.path.dirname(files_directory)
    path_log = os.path.join(result_directory, "log_cleaning")
    sum_extension_path = os.path.join(main_directory, "sum_extension")
    sum_error_path = os.path.join(result_directory, "sum_error")

    # get data
    df_log = pd.read_csv(path_log, sep=";", encoding="utf-8", index_col=False)
    df_extension = pd.read_csv(sum_extension_path, sep=";", encoding="utf-8",
                               index_col=False)
    df_error = pd.read_csv(sum_error_path, sep=";", encoding="utf-8",
                           index_col=False)

    # compute efficiency
    efficiency = {}
    for ext in list(set(list(df_extension["extension"]))):
        query = "extension == '%s'" % ext
        n_success = df_log.query(query).shape[0]
        n_fail = df_error.query(query).shape[0]
        if n_success + n_fail == 0:
            efficiency[ext] = 0.0
        else:
            efficiency[ext] = round(n_success / (n_success + n_fail) * 100, 2)
        print(efficiency[ext], "% ==>", ext)
    print("\n")

    return


def main(count_bool, log_bool, error_bool, efficiency_bool, result_directory,
         files_directory):
    """
    Function to run all the script
    :param count_bool: boolean
    :param log_bool: boolean
    :param error_bool: boolean
    :param efficiency_bool: boolean
    :param result_directory: string
    :param files_directory: string
    :return:
    """
    if count_bool:
        count(files_directory)
    if log_bool:
        log(result_directory)
    if error_bool:
        error(result_directory)
    if efficiency_bool:
        efficiency(result_directory, files_directory)
    return

if __name__ == "__main__":

    # paths
    files_directory = get_config_tag("input", "cleaning")
    result_directory = get_config_tag("result", "cleaning")

    # parameters
    extension_bool = get_config_tag("extension", "distribution")
    count_bool = get_config_tag("count", "distribution")
    log_bool = get_config_tag("log", "distribution")
    error_bool = get_config_tag("error", "distribution")
    efficiency_bool = get_config_tag("efficiency", "distribution")

    # run code
    main(count_bool=count_bool,
         log_bool=log_bool,
         error_bool=error_bool,
         efficiency_bool=efficiency_bool,
         result_directory=result_directory,
         files_directory=files_directory)
