# -*- coding: utf-8 -*-

""" Different functions to plot graphs about data collected and cleaned. """

# libraries
import os
import pandas as pd
import numpy as np
from toolbox.utils import (get_config_tag, check_graph_folders)
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
print("\n")


def plot_extension_cleaned(result_directory, df_log):
    """
    Function to plot extension distribution among cleaned files
    :param result_directory: string
    :param df_log: pandas
    :param df_log: pandas Dataframe
    :return:
    """
    print("plot extension distribution", "\n")

    # paths
    path_png = os.path.join(result_directory, "graphs", "png")
    path_pdf = os.path.join(result_directory, "graphs", "pdf")
    path_jpeg = os.path.join(result_directory, "graphs", "jpeg")
    path_svg = os.path.join(result_directory, "graphs", "svg")

    # plot
    data_counts = df_log['extension'].value_counts()
    extension_names = []
    extension_counts = data_counts.get_values()
    for i in range(len(data_counts.index)):
        name = data_counts.index[i]
        count = extension_counts[i]
        extension_names.append(name + " (%i)" % count)
    y_pos = np.arange(len(extension_names))
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_facecolor('white')
    ax.set_xlabel("Number of files (log scale)", fontsize=15)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=10)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(extension_names)
    ax.invert_yaxis()
    ax.barh(bottom=y_pos,
            width=extension_counts,
            align='center',
            color='darkcyan',
            edgecolor="black",
            alpha=0.8,
            fill=True,
            log=True)
    plt.tight_layout()

    # save figures
    path = os.path.join(path_jpeg, "extension.jpeg")
    plt.savefig(path)
    path = os.path.join(path_pdf, "extension.pdf")
    plt.savefig(path)
    path = os.path.join(path_png, "extension.png")
    plt.savefig(path)
    path = os.path.join(path_svg, "extension.svg")
    plt.savefig(path)

    plt.close("all")

    return


def plot_size_cleaned(result_directory, df_log):
    """
    Function to plot size distribution of cleaned files
    :param result_directory: string
    :param df_log: pandas Dataframe
    :return:
    """
    print("plot size distribution", "\n")

    # paths
    path_png = os.path.join(result_directory, "graphs", "png")
    path_pdf = os.path.join(result_directory, "graphs", "pdf")
    path_jpeg = os.path.join(result_directory, "graphs", "jpeg")
    path_svg = os.path.join(result_directory, "graphs", "svg")

    # scatter plot
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_facecolor('white')
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of columns (log scale)", fontsize=15)
    ax.set_ylabel("Number of rows (log scale)", fontsize=15)
    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    color = ["steelblue", "forestgreen", "firebrick", "darkorchid", "black",
             "darkorange", "grey"]
    marker = ["^", "*", "x", "D", ".", "o", "s"]
    groups = df_log.groupby("extension")
    for i, (name, group) in enumerate(groups):
        ax.scatter(x=group.y,
                   y=group.x,
                   s=10,
                   c=color[i],
                   marker=marker[i],
                   label=name)
    plt.legend(loc="upper right",
               fontsize=10)
    plt.tight_layout()

    # save figures
    path = os.path.join(path_jpeg, "size.jpeg")
    plt.savefig(path)
    path = os.path.join(path_pdf, "size.pdf")
    plt.savefig(path)
    path = os.path.join(path_png, "size.png")
    plt.savefig(path)
    path = os.path.join(path_svg, "size.svg")
    plt.savefig(path)

    plt.close("all")

    # # violin plot row
    # fig, ax = plt.subplots(figsize=(5, 5))
    # ax.set_facecolor('white')
    # ax.axvline(x=0, color='grey', linestyle='--', linewidth=0.8)
    # ax.axvline(x=0.2, color='grey', linestyle='--', linewidth=0.8)
    # ax.axvline(x=0.4, color='grey', linestyle='--', linewidth=0.8)
    # ax.axvline(x=0.6, color='grey', linestyle='--', linewidth=0.8)
    # ax.axvline(x=0.8, color='grey', linestyle='--', linewidth=0.8)
    # ax.axvline(x=1, color='grey', linestyle='--', linewidth=0.8)
    # # ax.set_xlim(left=-0.1, right=1.1)
    # sns.set_context("paper")
    # sns.violinplot(x="x",
    #                y="extension",
    #                hue=None,
    #                data=df_log,
    #                order=None,
    #                hue_order=None,
    #                bw='scott',
    #                cut=0,
    #                scale='width',
    #                scale_hue=True,
    #                gridsize=100,
    #                width=0.8,
    #                inner="quartile",
    #                split=False,
    #                orient="h",
    #                linewidth=None,
    #                color=None,
    #                palette=None,
    #                saturation=0.8,
    #                ax=ax)
    # ax.set_xlabel("Number of rows", fontsize=15)
    # ax.yaxis.label.set_visible(False)
    # plt.tight_layout()
    #
    # # save figures
    # path = os.path.join(path_jpeg, "size row.jpeg")
    # plt.savefig(path)
    # path = os.path.join(path_pdf, "size row.pdf")
    # plt.savefig(path)
    # path = os.path.join(path_png, "size row.png")
    # plt.savefig(path)
    # path = os.path.join(path_svg, "size row.svg")
    # plt.savefig(path)
    #
    # plt.close("all")
    #
    # # violin plot columns
    # fig, ax = plt.subplots(figsize=(5, 5))
    # ax.set_facecolor('white')
    # ax.axvline(x=0, color='grey', linestyle='--', linewidth=0.8)
    # ax.axvline(x=0.2, color='grey', linestyle='--', linewidth=0.8)
    # ax.axvline(x=0.4, color='grey', linestyle='--', linewidth=0.8)
    # ax.axvline(x=0.6, color='grey', linestyle='--', linewidth=0.8)
    # ax.axvline(x=0.8, color='grey', linestyle='--', linewidth=0.8)
    # ax.axvline(x=1, color='grey', linestyle='--', linewidth=0.8)
    # # ax.set_xlim(left=-0.1, right=1.1)
    # sns.set_context("paper")
    # sns.violinplot(x="y",
    #                y="extension",
    #                hue=None,
    #                data=df_log,
    #                order=None,
    #                hue_order=None,
    #                bw='scott',
    #                cut=0,
    #                scale='width',
    #                scale_hue=True,
    #                gridsize=100,
    #                width=0.8,
    #                inner="quartile",
    #                split=False,
    #                orient="h",
    #                linewidth=None,
    #                color=None,
    #                palette=None,
    #                saturation=0.8,
    #                ax=ax)
    # ax.set_xlabel("Number of columns", fontsize=15)
    # ax.yaxis.label.set_visible(False)
    # plt.tight_layout()
    #
    # # save figures
    # path = os.path.join(path_jpeg, "size columns.jpeg")
    # plt.savefig(path)
    # path = os.path.join(path_pdf, "size columns.pdf")
    # plt.savefig(path)
    # path = os.path.join(path_png, "size columns.png")
    # plt.savefig(path)
    # path = os.path.join(path_svg, "size columns.svg")
    # plt.savefig(path)
    #
    # plt.close("all")

    return


def main(result_directory, reset):
    # paths
    path_log = os.path.join(result_directory, "log_final_reduced")

    # get log data
    df_log = pd.read_csv(path_log, header=0, encoding="utf-8", sep=";",
                         index_col=False)
    print("log length :", df_log.shape[0], "\n")

    # check folders
    check_graph_folders(result_directory, reset=reset)

    # plot
    plot_extension_cleaned(result_directory, df_log)
    plot_size_cleaned(result_directory, df_log)

    return


if __name__ == "__main__":

    # paths
    result_directory = get_config_tag("result", "cleaning")

    # run
    main(result_directory,
         reset=True)
