# -*- coding: utf-8 -*-

""" Different functions to plot """

# libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from toolbox.utils import get_config_tag, save_sparse_csr, load_sparse_csr
print("\n")


def plot_extension_cleaned(result_directory, df_log):
    print(df_log.columns)
    return


def plot_size_cleaned(result_directory):
    return


def plot_tsne(result_directory):
    return


def plot_mds(result_directory):
    return


def plot_score_nmf(result_directory):
    return


def make_table_source_topic(result_directory):
    return


def main(result_directory):
    # paths
    path_log = os.path.join(result_directory, "log_final")

    # get data
    df_log = pd.read_csv(path_log, header=0, encoding="utf-8", sep=";",
                         index_col=False)
    print("log length :", df_log.shape[0], "\n")

    # table
    make_table_source_topic(result_directory)

    # plot
    plot_extension_cleaned(result_directory, df_log)
    plot_size_cleaned(result_directory)
    plot_tsne(result_directory)
    plot_mds(result_directory)
    plot_score_nmf(result_directory)

    return

if __name__ == "__main__":

    # paths
    result_directory = get_config_tag("result", "cleaning")

    # run
    main(result_directory)