#!/bin/python3
# coding: utf-8

""" Function to compute some basic statistics on the fitted files. """

# libraries
import pandas as pd
import numpy as np
import os
import magic
import zipfile
import matplotlib.pyplot as plt
from collections import defaultdict
from tempfile import TemporaryDirectory
print("\n")

# path
path_files = "../data/data_collected"
sum_extension_path = "../data/sum_extension"
path_log = "../data/log_cleaning"
error_path = "../data/test_fitted/fit_errors"
sum_error_path = "../data/sum_error"

# parameters
extension_bool = True
count_bool = True
log_bool = True
plot_bool = True
error_bool = True
efficiency_bool = True

#############
# extension #
#############

if extension_bool:
    print("### extension ###", "\n")
    # collect data
    if os.path.isfile(sum_extension_path):
        os.remove(sum_extension_path)
    with open(sum_extension_path, mode="wt", encoding="utf-8") as f:
        f.write("extension;zipfile")
        f.write("\n")
    d_extensions = defaultdict(lambda x: 0)
    for filename in os.listdir(path_files):
        path_file = os.path.join(path_files, filename)
        size_file = os.path.getsize(path_file)
        if size_file > 0:
            ext = magic.Magic(mime=True).from_file(path_file)
            if ext == "application/zip":
                try:
                    z = zipfile.ZipFile(path_file)
                    with TemporaryDirectory() as temp_directory:
                        z.extractall(temp_directory)
                        for file in z.namelist():
                            temp_path = os.path.join(temp_directory, file)
                            if os.path.isfile(temp_path):
                                ext = magic.Magic(mime=True)\
                                    .from_file(temp_path)
                                with open(sum_extension_path, mode="at",
                                          encoding="utf-8") as f:
                                    f.write(";".join([ext, "True"]))
                                    f.write("\n")
                except:
                    continue
            else:
                with open(sum_extension_path, mode="at", encoding="utf-8") as f:
                    f.write(";".join([ext, "False"]))
                    f.write("\n")

    print("##########################################################", "\n")

if count_bool:
    print("### count ###", "\n")
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

    print("##########################################################", "\n")

#######
# log #
#######

if log_bool:
    print("### log ###", "\n")
    # get data
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

    # plot
    if plot_bool:
        plt.scatter(df_log["n_col"], df_log["n_row"], c="green", alpha=0.8)
        plt.xlabel("Number of columns")
        plt.ylabel("Number of rows")
        plt.title("Files size (%i files)" % df_log.shape[0])
        plt.savefig("fig_1.png")

    print("##########################################################", "\n")

#########
# error #
#########

if error_bool:
    print("### error ###", "\n")
    # reset file
    if os.path.isfile(sum_error_path):
        os.remove(sum_error_path)
    with open(sum_error_path, mode="wt", encoding="utf-8") as f:
        f.write("filename;extension;error;content")
        f.write("\n")

    # get data
    for filename in os.listdir(error_path):
        path = os.path.join(error_path, filename)
        with open(path, mode="rt", encoding="utf-8") as f:
            c = f.readlines()
            extension = c[2].strip()
            error = c[3].split(" ")[0]
            content = c[-2]
        with open(sum_error_path, mode="at", encoding="utf-8") as f:
            f.write(";".join([str(filename), str(extension), str(error),
                              str(content)]))
            f.write("\n")

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
        print(df_error.query(query)["error"].value_counts(), "\n")
        max_e = df_error.query(query)["error"].value_counts().index.tolist()[0]
        print(df_error.query("error == '%s'" % max_e)["content"].
              value_counts(), "\n")
        print("---", "\n")

    print("##########################################################", "\n")

##############
# efficiency #
##############

if efficiency_bool:
    print("### efficiency ###", "\n")
    efficiency = {}
    df_log = pd.read_csv(path_log, sep=";", encoding="utf-8", index_col=False)
    df_extension = pd.read_csv(sum_extension_path, sep=";", encoding="utf-8",
                               index_col=False)
    extensions = list(set(list(df_log["extension"])))
    for ext in extensions:
        query = "extension == '%s'" % ext
        # todo correct excel sheet
        efficiency[ext] = round(df_log.query(query).shape[0] / df_extension.
                                query(query).shape[0] * 100, 2)
        print(efficiency[ext], "% ==>", ext)
    print("\n")

    print("##########################################################", "\n")
