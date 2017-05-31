# -*- coding: utf-8 -*-

""" Function to compute some basic statistics on the fitted files. """

# libraries
import pandas as pd
import numpy as np
import os
import magic
import zipfile
import matplotlib.pyplot as plt
from tempfile import TemporaryDirectory
from toolbox.utils import get_config_tag
print("\n")


def extension(result_directory, files_directory):
    """
    Function to compute the distribution of our extensions
    :param result_directory: string
    :param files_directory: string
    :return:
    """
    print("### extension ###", "\n")

    # initialize sum_extension file
    sum_extension_path = os.path.join(result_directory, "sum_extension")
    if os.path.isfile(sum_extension_path):
        os.remove(sum_extension_path)
    with open(sum_extension_path, mode="wt", encoding="utf-8") as f:
        f.write("extension;zipfile")
        f.write("\n")
    # collect data
    errors = 0
    for filename in os.listdir(files_directory):
        path_file = os.path.join(files_directory, filename)
        size_file = os.path.getsize(path_file)
        if size_file <= 0:
            return
        ext = magic.Magic(mime=True).from_file(path_file)
        if ext == "application/zip":
            try:
                z = zipfile.ZipFile(path_file)
                with TemporaryDirectory() as temp_directory:
                    z.extractall(temp_directory)
                    for file in z.namelist():
                        temp_path = os.path.join(temp_directory, file)
                        if os.path.isfile(temp_path):
                            ext = magic.Magic(mime=True) \
                                .from_file(temp_path)
                            with open(sum_extension_path, mode="at",
                                      encoding="utf-8") as f:
                                f.write(";".join([ext, "True"]))
                                f.write("\n")
            except:
                errors += 1
                continue
        else:
            with open(sum_extension_path, mode="at", encoding="utf-8") as f:
                f.write(";".join([ext, "False"]))
                f.write("\n")
    print("number of extensions not find :", errors, "\n")
    return


def count(result_directory):
    """
    Function to show the distribution of our extensions
    :param result_directory: string
    :return:
    """
    print("### count ###", "\n")

    # test if the result exists
    sum_extension_path = os.path.join(result_directory, "sum_extension")
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


def plot(result_directory):
    """
    Function to plot results
    :param result_directory: string
    :return:
    """
    print("### plot ###", "\n")

    # check if the graph directory exists
    graph_directory = os.path.join(result_directory, "graphs")
    if not os.path.isdir(graph_directory):
        os.mkdir(graph_directory)

    # get data
    path_log = os.path.join(result_directory, "log_cleaning")
    df_log = pd.read_csv(path_log, sep=";", encoding="utf-8", index_col=False)

    # draw scatter plots
    plt.scatter(df_log["n_col"], df_log["n_row"], c="green", alpha=0.8)
    plt.xlabel("Number of columns (log scale)")
    plt.ylabel("Number of rows (log scale)")
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Files size (%i files)" % df_log.shape[0])
    path = os.path.join(graph_directory, "size distribution (log log).png")
    if os.path.isfile(path):
        os.remove(path)
    plt.savefig(path)

    plt.scatter(df_log["n_col"], df_log["n_row"], c="green", alpha=0.8)
    plt.xlabel("Number of columns")
    plt.ylabel("Number of rows")
    plt.xscale("linear")
    plt.yscale("linear")
    plt.title("Files size (%i files)" % df_log.shape[0])
    path = os.path.join(graph_directory, "size distribution.png")
    if os.path.isfile(path):
        os.remove(path)
    plt.savefig(path)

    plt.scatter(df_log["n_col"], df_log["n_row"], c="green", alpha=0.8)
    plt.xlabel("Number of columns")
    plt.ylabel("Number of rows (log scale)")
    plt.xscale("linear")
    plt.yscale("log")
    plt.title("Files size (%i files)" % df_log.shape[0])
    path = os.path.join(graph_directory, "size distribution (log linear).png")
    if os.path.isfile(path):
        os.remove(path)
    plt.savefig(path)

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
    for filename in os.listdir(errors_directory):
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


def efficiency(result_directory):
    """
    Function to compute the efficiency of our cleaning process
    :param result_directory: string
    :return:
    """
    print("### efficiency ###", "\n")

    efficiency = {}
    path_log = os.path.join(result_directory, "log_cleaning")
    df_log = pd.read_csv(path_log, sep=";", encoding="utf-8", index_col=False)
    sum_extension_path = os.path.join(result_directory, "sum_extension")
    df_extension = pd.read_csv(sum_extension_path, sep=";", encoding="utf-8",
                               index_col=False)
    sum_error_path = os.path.join(result_directory, "sum_error")
    df_error = pd.read_csv(sum_error_path, sep=";", encoding="utf-8",
                           index_col=False)
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


def main(extension_bool, count_bool, log_bool, plot_bool, error_bool,
         efficiency_bool, result_directory, files_directory):
    """
    Function to run all the script
    :param extension_bool: boolean
    :param count_bool: boolean
    :param log_bool: boolean
    :param plot_bool: boolean
    :param error_bool: boolean
    :param efficiency_bool: boolean
    :param result_directory: string
    :param files_directory:string
    :return:
    """
    if extension_bool:
        extension(result_directory, files_directory)
    if count_bool:
        count(result_directory)
    if log_bool:
        log(result_directory)
    if plot_bool:
        plot(result_directory)
    if error_bool:
        error(result_directory)
    if efficiency_bool:
        efficiency(result_directory)
    return

if __name__ == "__main__":

    # paths
    files_directory = get_config_tag("input", "cleaning")
    result_directory = get_config_tag("result", "cleaning")

    # parameters
    extension_bool = get_config_tag("extension", "distribution")
    count_bool = get_config_tag("count", "distribution")
    log_bool = get_config_tag("log", "distribution")
    plot_bool = get_config_tag("plot", "distribution")
    error_bool = get_config_tag("error", "distribution")
    efficiency_bool = get_config_tag("efficiency", "distribution")

    # run code
    main(extension_bool=extension_bool,
         count_bool=count_bool,
         log_bool=log_bool,
         plot_bool=plot_bool,
         error_bool=error_bool,
         efficiency_bool=efficiency_bool,
         result_directory=result_directory,
         files_directory=files_directory)
