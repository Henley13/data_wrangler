# -*- coding: utf-8 -*-

""" Metric learning algorithms. """

# libraries
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import seaborn as sns
print("\n")


def interval(array, percentile_low, percentile_up):
    up = np.nanpercentile(array, percentile_up)
    low = np.nanpercentile(array, percentile_low)
    return up, low


def load_array(l_auc, l_auc_up, l_auc_low, filename, model_directory,
               percentile_low, percentile_up, std):

    path_auc = os.path.join(model_directory, filename)
    l = np.load(path_auc)
    m = np.nanmean(l)
    l_auc.append(m)
    if std:
        s = np.nanstd(l, ddof=1)
        up, low = m + s, m - s
        l_auc_up.append(up)
        l_auc_low.append(low)
    else:
        up, low = interval(l, percentile_low, percentile_up)
        l_auc_up.append(up)
        l_auc_low.append(low)

    return l_auc, l_auc_up, l_auc_low


def load_array_boxplot(filename, model_directory):
    path_auc = os.path.join(model_directory, filename)
    l = np.load(path_auc)
    return list(l)


def plot_axe(ax, x, y, color, linestyle, up, low, marker):
    x = x[:len(y)]
    ax.plot(x, y, linewidth=2, c=color, linestyle=linestyle, alpha=0.8)
    ax.fill_between(x, y1=up, y2=low, where=None, color=color, alpha=0.2)
    aa = ax.scatter(x, y, s=25, c=color, marker=marker)
    return aa


def graph_auc(result_directory, page, norm, percentile_low, percentile_up, std):

    # paths
    path_png = os.path.join(result_directory, "graphs", "png")
    path_pdf = os.path.join(result_directory, "graphs", "pdf")
    path_jpeg = os.path.join(result_directory, "graphs", "jpeg")
    path_svg = os.path.join(result_directory, "graphs", "svg")
    model_directory = os.path.join(result_directory, "model_result")

    # get data
    l_auc_nmf = []
    l_auc_nmf_up = []
    l_auc_nmf_low = []

    l_auc_learned_nmf = []
    l_auc_learned_nmf_up = []
    l_auc_learned_nmf_low = []

    l_auc_svd = []
    l_auc_svd_up = []
    l_auc_svd_low = []

    l_auc_learned_svd = []
    l_auc_learned_svd_up = []
    l_auc_learned_svd_low = []

    l_auc_random_nmf = []
    l_auc_random_nmf_up = []
    l_auc_random_nmf_low = []

    l_auc_random_svd = []
    l_auc_random_svd_up = []
    l_auc_random_svd_low = []

    topics = [i for i in range(5, 101, 5)]
    for n_topic in topics:

        # auc nmf
        filename = "auc_%s_%i_%s_%s_%s_%s.npy" % ("nmf", n_topic, page, "noml",
                                                  norm, "False")
        l_auc_nmf, l_auc_nmf_up, l_auc_nmf_low = load_array(
            l_auc_nmf, l_auc_nmf_up, l_auc_nmf_low, filename, model_directory,
            percentile_low, percentile_up, std)

        # auc learned nmf
        filename = "auc_%s_%i_%s_%s_%s_%s.npy" % ("nmf", n_topic, page, "ml",
                                                  norm, "False")
        (l_auc_learned_nmf, l_auc_learned_nmf_up,
         l_auc_learned_nmf_low) = load_array(
            l_auc_learned_nmf, l_auc_learned_nmf_up, l_auc_learned_nmf_low,
            filename, model_directory, percentile_low, percentile_up, std)

        # random auc nmf
        filename = "auc_%s_%i_%s_%s_%s_%s.npy" % ("nmf", n_topic, page, "noml",
                                                  norm, "True")
        (l_auc_random_nmf, l_auc_random_nmf_up,
         l_auc_random_nmf_low) = load_array(
            l_auc_random_nmf, l_auc_random_nmf_up, l_auc_random_nmf_low,
            filename, model_directory, percentile_low, percentile_up, std)

        # auc svd
        filename = "auc_%s_%i_%s_%s_%s_%s.npy" % ("svd", n_topic, page, "noml",
                                                  norm, "False")
        l_auc_svd, l_auc_svd_up, l_auc_svd_low = load_array(
            l_auc_svd, l_auc_svd_up, l_auc_svd_low, filename, model_directory,
            percentile_low, percentile_up, std)

        # auc learned svd
        filename = "auc_%s_%i_%s_%s_%s_%s.npy" % ("svd", n_topic, page, "ml",
                                                  norm, "False")
        (l_auc_learned_svd, l_auc_learned_svd_up,
         l_auc_learned_svd_low) = load_array(
            l_auc_learned_svd, l_auc_learned_svd_up, l_auc_learned_svd_low,
            filename, model_directory, percentile_low, percentile_up, std)

        # random auc svd
        filename = "auc_%s_%i_%s_%s_%s_%s.npy" % ("svd", n_topic, page, "noml",
                                                  norm, "True")
        (l_auc_random_svd, l_auc_random_svd_up,
         l_auc_random_svd_low) = load_array(
            l_auc_random_svd, l_auc_random_svd_up, l_auc_random_svd_low,
            filename, model_directory, percentile_low, percentile_up, std)

    # plot data
    x = topics
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_facecolor('white')
    ax.set_ylim([0.25, 1])

    # auc nmf
    aa = plot_axe(ax, x, l_auc_nmf, "steelblue", "-", l_auc_nmf_up,
                  l_auc_nmf_low, "^")

    # auc learned nmf
    bb = plot_axe(ax, x, l_auc_learned_nmf, "forestgreen", "--",
                  l_auc_learned_nmf_up, l_auc_learned_nmf_low, "*")

    # auc svd
    cc = plot_axe(ax, x, l_auc_svd, "firebrick", "-", l_auc_svd_up,
                  l_auc_svd_low, "x")

    # auc learned svd
    dd = plot_axe(ax, x, l_auc_learned_svd, "darkorchid", "--",
                  l_auc_learned_svd_up, l_auc_learned_svd_low, "D")

    # auc random nmf
    ee = plot_axe(ax, x, l_auc_random_nmf, "black", ":", l_auc_random_nmf_up,
                  l_auc_random_nmf_low, ".")

    # auc random svd
    ff = plot_axe(ax, x, l_auc_random_svd, "darkorange", ":",
                  l_auc_random_svd_up, l_auc_random_svd_low, "o")

    ax.set_xlabel("Number of topics", fontsize=15)
    ax.set_ylabel("Mean AUC (precision recall)", fontsize=15)

    plt.legend((aa, bb, cc, dd, ee, ff),
               ('NMF', 'NMF & metric learned', 'SVD', 'SVD & metric learned',
                'NMF & label randomized', 'SVD & label randomized'),
               scatterpoints=3,
               loc='upper center',
               ncol=2,
               fontsize=8)
    plt.tight_layout()

    # save figures
    path = os.path.join(path_jpeg, "auc summary (%s %s).jpeg" % (page, norm))
    plt.savefig(path)
    path = os.path.join(path_pdf, "auc summary (%s %s).pdf" % (page, norm))
    plt.savefig(path)
    path = os.path.join(path_png, "auc summary (%s %s).png" % (page, norm))
    plt.savefig(path)
    path = os.path.join(path_svg, "auc summary (%s %s).svg" % (page, norm))
    plt.savefig(path)

    plt.close("all")

    return


def graph_boxplot(result_directory, n_topic, page, norm):
    # TODO improve this function
    # paths
    path_png = os.path.join(result_directory, "graphs", "png")
    path_pdf = os.path.join(result_directory, "graphs", "pdf")
    path_jpeg = os.path.join(result_directory, "graphs", "jpeg")
    path_svg = os.path.join(result_directory, "graphs", "svg")
    model_directory = os.path.join(result_directory, "model_result")

    # auc nmf
    filename = "auc_%s_%i_%s_%s_%s_%s.npy" % ("nmf", n_topic, page, "noml",
                                              norm, "False")
    l_auc_nmf = load_array_boxplot(filename, model_directory)

    # auc learned nmf
    filename = "auc_%s_%i_%s_%s_%s_%s.npy" % ("nmf", n_topic, page, "ml",
                                              norm, "False")
    l_auc_learned_nmf = load_array_boxplot(filename, model_directory)

    # auc svd
    filename = "auc_%s_%i_%s_%s_%s_%s.npy" % ("svd", n_topic, page, "noml",
                                              norm, "False")
    l_auc_svd = load_array_boxplot(filename, model_directory)

    # auc learned svd
    filename = "auc_%s_%i_%s_%s_%s_%s.npy" % ("svd", n_topic, page, "ml",
                                              norm, "False")
    l_auc_learned_svd = load_array_boxplot(filename, model_directory)

    # plot with violinplot
    data = pd.DataFrame()
    auc = []
    model = []
    values = [l_auc_nmf, l_auc_learned_nmf, l_auc_svd, l_auc_learned_svd]
    names = ["NMF", "NMF metric learning", "SVD", "SVD metric learning"]
    for i in range(len(values)):
        auc += values[i]
        l = [names[i]] * len(values[i])
        model += l
    data["auc"] = auc
    data["model"] = model
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_facecolor('white')
    ax.axvline(x=0, color='grey', linestyle='--', linewidth=0.8)
    ax.axvline(x=0.2, color='grey', linestyle='--', linewidth=0.8)
    ax.axvline(x=0.4, color='grey', linestyle='--', linewidth=0.8)
    ax.axvline(x=0.6, color='grey', linestyle='--', linewidth=0.8)
    ax.axvline(x=0.8, color='grey', linestyle='--', linewidth=0.8)
    ax.axvline(x=1, color='grey', linestyle='--', linewidth=0.8)
    ax.set_xlim(left=-0.1, right=1.1)
    sns.set_context("paper")
    sns.violinplot(x="auc",
                   y="model",
                   hue=None,
                   data=data,
                   order=reversed(names),
                   hue_order=None,
                   bw='scott',
                   cut=0,
                   scale='width',
                   scale_hue=True,
                   gridsize=100,
                   width=0.8,
                   inner=None,
                   split=False,
                   orient="h",
                   linewidth=None,
                   color=None,
                   palette=None,
                   saturation=0.8,
                   ax=ax)
    ax.set_xlabel("AUC", fontsize=15)
    ax.yaxis.label.set_visible(False)
    plt.text(0, 3.9, "bad reuse \nprediction", fontsize=9,
             multialignment='center')
    plt.text(0.825, 3.9, "good reuse \nprediction", fontsize=9,
             multialignment='center')
    plt.axvline(x=0.3, linewidth=2, color="red")
    plt.tight_layout()

    # save figures
    path = os.path.join(path_jpeg, "boxplot auc (%s %s %i).jpeg" % (page, norm,
                                                                    n_topic))
    plt.savefig(path)
    path = os.path.join(path_pdf, "boxplot auc (%s %s %i).pdf" % (page, norm,
                                                                  n_topic))
    plt.savefig(path)
    path = os.path.join(path_png, "boxplot auc (%s %s %i).png" % (page, norm,
                                                                  n_topic))
    plt.savefig(path)
    path = os.path.join(path_svg, "boxplot auc (%s %s %i).svg" % (page, norm,
                                                                  n_topic))
    plt.savefig(path)

    # compare n_topics
    values = []
    names = []
    for n_topic in [5, 10, 15, 20, 25, 30, 40, 50]:
        # get data
        filename = "auc_%s_%i_%s_%s_%s_%s.npy" % ("svd", n_topic, page, "ml",
                                                  norm, "False")
        values.append(load_array_boxplot(filename, model_directory))
        names.append("%i topics" % n_topic)
    data = pd.DataFrame()
    auc = []
    model = []
    for i in range(len(values)):
        auc += values[i]
        l = [names[i]] * len(values[i])
        model += l
    data["auc"] = auc
    data["model"] = model

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_facecolor('white')
    ax.axvline(x=0, color='grey', linestyle='--', linewidth=0.8)
    ax.axvline(x=0.2, color='grey', linestyle='--', linewidth=0.8)
    ax.axvline(x=0.4, color='grey', linestyle='--', linewidth=0.8)
    ax.axvline(x=0.6, color='grey', linestyle='--', linewidth=0.8)
    ax.axvline(x=0.8, color='grey', linestyle='--', linewidth=0.8)
    ax.axvline(x=1, color='grey', linestyle='--', linewidth=0.8)
    ax.set_xlim(left=-0.1, right=1.1)
    sns.set_context("paper")
    sns.violinplot(x="auc",
                   y="model",
                   hue=None,
                   data=data,
                   order=reversed(names),
                   hue_order=None,
                   bw='scott',
                   cut=0,
                   scale='width',
                   scale_hue=True,
                   gridsize=100,
                   width=0.8,
                   inner=None,
                   split=False,
                   orient="h",
                   linewidth=None,
                   color=None,
                   palette=None,
                   saturation=0.8,
                   ax=ax)
    ax.set_xlabel("AUC", fontsize=15)
    ax.yaxis.label.set_visible(False)
    plt.text(0, 8.3, "bad reuse \nprediction", fontsize=9,
             multialignment='center')
    plt.text(0.825, 8.3, "good reuse \nprediction", fontsize=9,
             multialignment='center')
    plt.axvline(x=0.3, linewidth=2, color="red")
    plt.tight_layout()

    # save figures
    path = os.path.join(path_jpeg, "boxplot auc topics(%s %s SVD).jpeg" % (
        page, norm))
    plt.savefig(path)
    path = os.path.join(path_pdf, "boxplot auc topics(%s %s SVD).pdf" % (
        page, norm))
    plt.savefig(path)
    path = os.path.join(path_png, "boxplot auc topics(%s %s SVD).png" % (
        page, norm))
    plt.savefig(path)
    path = os.path.join(path_svg, "boxplot auc topics(%s %s SVD).svg" % (
        page, norm))
    plt.savefig(path)

    plt.close("all")

    return


def graph_auc_page(result_directory, model, norm, percentile_low,
                   percentile_up, std):

    # paths
    path_png = os.path.join(result_directory, "graphs", "png")
    path_pdf = os.path.join(result_directory, "graphs", "pdf")
    path_jpeg = os.path.join(result_directory, "graphs", "jpeg")
    path_svg = os.path.join(result_directory, "graphs", "svg")
    model_directory = os.path.join(result_directory, "model_result")

    # get data
    l_auc_nopage = []
    l_auc_nopage_up = []
    l_auc_nopage_low = []

    l_auc_nopage_lm = []
    l_auc_nopage_lm_up = []
    l_auc_nopage_lm_low = []

    l_auc_page = []
    l_auc_page_up = []
    l_auc_page_low = []

    l_auc_page_lm = []
    l_auc_page_lm_up = []
    l_auc_page_lm_low = []

    l_auc_nopage_random = []
    l_auc_nopage_random_up = []
    l_auc_nopage_random_low = []

    l_auc_page_random = []
    l_auc_page_random_up = []
    l_auc_page_random_low = []

    topics = [i for i in range(5, 101, 5)]
    for n_topic in topics:

        # auc nopage
        filename = "auc_%s_%i_%s_%s_%s_%s.npy" % (model, n_topic, "nopage",
                                                  "noml", norm, "False")
        l_auc_nopage, l_auc_nopage_up, l_auc_nopage_low = load_array(
            l_auc_nopage, l_auc_nopage_up, l_auc_nopage_low, filename,
            model_directory, percentile_low, percentile_up, std)

        # auc nopage learned
        filename = "auc_%s_%i_%s_%s_%s_%s.npy" % (model, n_topic, "nopage",
                                                  "ml", norm, "False")
        l_auc_nopage_lm, l_auc_nopage_lm_up, l_auc_nopage_lm_low = load_array(
            l_auc_nopage_lm, l_auc_nopage_lm_up, l_auc_nopage_lm_low, filename,
            model_directory, percentile_low, percentile_up, std)

        # auc page
        filename = "auc_%s_%i_%s_%s_%s_%s.npy" % (model, n_topic, "page",
                                                  "noml", norm, "False")
        l_auc_page, l_auc_page_up, l_auc_page_low = load_array(
            l_auc_page, l_auc_page_up, l_auc_page_low, filename,
            model_directory, percentile_low, percentile_up, std)

        # auc page learned
        filename = "auc_%s_%i_%s_%s_%s_%s.npy" % (model, n_topic, "page",
                                                  "ml", norm, "False")
        l_auc_page_lm, l_auc_page_lm_up, l_auc_page_lm_low = load_array(
            l_auc_page_lm, l_auc_page_lm_up, l_auc_page_lm_low, filename,
            model_directory, percentile_low, percentile_up, std)

        # auc nopage random
        filename = "auc_%s_%i_%s_%s_%s_%s.npy" % (model, n_topic, "nopage",
                                                  "noml", norm, "True")
        (l_auc_nopage_random, l_auc_nopage_random_up,
         l_auc_nopage_random_low) = load_array(
            l_auc_nopage_random, l_auc_nopage_random_up,
            l_auc_nopage_random_low, filename, model_directory, percentile_low,
            percentile_up, std)

        # auc page random
        filename = "auc_%s_%i_%s_%s_%s_%s.npy" % (model, n_topic, "page",
                                                  "noml", norm, "True")
        (l_auc_page_random, l_auc_page_random_up,
         l_auc_page_random_low) = load_array(
            l_auc_page_random, l_auc_page_random_up, l_auc_page_random_low,
            filename, model_directory, percentile_low, percentile_up, std)

    # plot data
    x = topics
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_facecolor('white')
    ax.set_ylim([0.25, 1])

    # auc nopage
    aa = plot_axe(ax, x, l_auc_nopage, "steelblue", "-", l_auc_nopage_up,
                  l_auc_nopage_low, "^")

    # auc learned nopage
    bb = plot_axe(ax, x, l_auc_nopage_lm, "forestgreen", "--",
                  l_auc_nopage_lm_up, l_auc_nopage_lm_low, "*")

    # auc page
    cc = plot_axe(ax, x, l_auc_page, "firebrick", "-", l_auc_page_up,
                  l_auc_page_low, "x")

    # auc learned page
    dd = plot_axe(ax, x, l_auc_page_lm, "darkorchid", "--", l_auc_page_lm_up,
                  l_auc_page_lm_low, "D")

    # auc random nopage
    ee = plot_axe(ax, x, l_auc_nopage_random, "black", ":",
                  l_auc_nopage_random_up, l_auc_nopage_random_low, ".")

    # auc random page
    ff = plot_axe(ax, x, l_auc_page_random, "darkorange", ":",
                  l_auc_page_random_up, l_auc_page_random_low, "o")

    ax.set_xlabel("Number of topics", fontsize=15)
    ax.set_ylabel("Mean AUC (precision recall)", fontsize=15)

    plt.legend((aa, bb, cc, dd, ee, ff),
               ('%s & %s norm' % (model.upper(), norm.upper()),
                'metric learned',
                'same page included',
                'metric learned & same page included',
                'label randomized',
                'label randomized & same page included'),
               scatterpoints=3,
               loc='upper center',
               ncol=2,
               fontsize=8)
    plt.tight_layout()

    # save figures
    path = os.path.join(path_jpeg, "auc summary sampling (%s %s).jpeg" %
                        (model, norm))
    plt.savefig(path)
    path = os.path.join(path_pdf, "auc summary sampling (%s %s).pdf" %
                        (model, norm))
    plt.savefig(path)
    path = os.path.join(path_png, "auc summary sampling (%s %s).png" %
                        (model, norm))
    plt.savefig(path)
    path = os.path.join(path_svg, "auc summary sampling (%s %s).svg" %
                        (model, norm))
    plt.savefig(path)

    plt.close("all")

    return


def graph_auc_norm(result_directory, model, page, percentile_low,
                   percentile_up, std):

    # paths
    path_png = os.path.join(result_directory, "graphs", "png")
    path_pdf = os.path.join(result_directory, "graphs", "pdf")
    path_jpeg = os.path.join(result_directory, "graphs", "jpeg")
    path_svg = os.path.join(result_directory, "graphs", "svg")
    model_directory = os.path.join(result_directory, "model_result")

    # get data
    l_auc_l1 = []
    l_auc_l1_up = []
    l_auc_l1_low = []

    l_auc_l1_ml = []
    l_auc_l1_ml_up = []
    l_auc_l1_ml_low = []

    l_auc_l2 = []
    l_auc_l2_up = []
    l_auc_l2_low = []

    l_auc_l2_ml = []
    l_auc_l2_ml_up = []
    l_auc_l2_ml_low = []

    l_auc_inf = []
    l_auc_inf_up = []
    l_auc_inf_low = []

    l_auc_inf_ml = []
    l_auc_inf_ml_up = []
    l_auc_inf_ml_low = []

    topics = [i for i in range(5, 101, 5)]
    for n_topic in topics:

        # auc l1
        filename = "auc_%s_%i_%s_%s_%s_%s.npy" % (model, n_topic, page,
                                                  "noml", "l1", "False")
        l_auc_l1, l_auc_l1_up, l_auc_l1_low = load_array(
            l_auc_l1, l_auc_l1_up, l_auc_l1_low, filename,
            model_directory, percentile_low, percentile_up, std)

        # auc l1 learned
        filename = "auc_%s_%i_%s_%s_%s_%s.npy" % (model, n_topic, page,
                                                  "ml", "l1", "False")
        l_auc_l1_ml, l_auc_l1_ml_up, l_auc_l1_ml_low = load_array(
            l_auc_l1_ml, l_auc_l1_ml_up, l_auc_l1_ml_low, filename,
            model_directory, percentile_low, percentile_up, std)

        # auc l2
        filename = "auc_%s_%i_%s_%s_%s_%s.npy" % (model, n_topic, page,
                                                  "noml", "l2", "False")
        l_auc_l2, l_auc_l2_up, l_auc_l2_low = load_array(
            l_auc_l2, l_auc_l2_up, l_auc_l2_low, filename,
            model_directory, percentile_low, percentile_up, std)

        # auc l2 learned
        filename = "auc_%s_%i_%s_%s_%s_%s.npy" % (model, n_topic, page,
                                                  "ml", "l2", "False")
        l_auc_l2_ml, l_auc_l2_ml_up, l_auc_l2_ml_low = load_array(
            l_auc_l2_ml, l_auc_l2_ml_up, l_auc_l2_ml_low, filename,
            model_directory, percentile_low, percentile_up, std)

        # auc inf
        filename = "auc_%s_%i_%s_%s_%s_%s.npy" % (model, n_topic, page,
                                                  "noml", "inf", "False")
        l_auc_inf, l_auc_inf_up, l_auc_inf_low = load_array(
            l_auc_inf, l_auc_inf_up, l_auc_inf_low, filename,
            model_directory, percentile_low, percentile_up, std)

        # auc inf learned
        filename = "auc_%s_%i_%s_%s_%s_%s.npy" % (model, n_topic, page,
                                                  "ml", "inf", "False")
        l_auc_inf_ml, l_auc_inf_ml_up, l_auc_inf_ml_low = load_array(
            l_auc_inf_ml, l_auc_inf_ml_up, l_auc_inf_ml_low, filename,
            model_directory, percentile_low, percentile_up, std)

    # plot data
    x = topics
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_facecolor('white')
    ax.set_ylim([0.25, 1])

    # auc l1
    aa = plot_axe(ax, x, l_auc_l1, "steelblue", "-", l_auc_l1_up,
                  l_auc_l1_low, "^")

    # auc l1 learned
    bb = plot_axe(ax, x, l_auc_l1_ml, "forestgreen", "--", l_auc_l1_ml_up,
                  l_auc_l1_ml_low, "*")

    # auc l2
    cc = plot_axe(ax, x, l_auc_l2, "firebrick", "-", l_auc_l2_up,
                  l_auc_l2_low, "x")

    # auc l2 learned
    dd = plot_axe(ax, x, l_auc_l2_ml, "darkorchid", "--", l_auc_l2_ml_up,
                  l_auc_l2_ml_low, "D")

    # auc inf
    ee = plot_axe(ax, x, l_auc_inf, "black", ":", l_auc_inf_up,
                  l_auc_inf_low, ".")

    # auc inf learned
    ff = plot_axe(ax, x, l_auc_inf_ml, "darkorange", ":", l_auc_inf_ml_up,
                  l_auc_inf_ml_low, "o")

    ax.set_xlabel("Number of topics", fontsize=15)
    ax.set_ylabel("Mean AUC (precision recall)", fontsize=15)

    plt.legend((aa, bb, cc, dd, ee, ff),
               ('norm L1', 'norm L1 & metric learned',
                'norm L2', 'norm L2 & metric learned',
                'norm INF', 'norm INF & metric learned'),
               scatterpoints=3,
               loc='upper center',
               ncol=2,
               fontsize=8)
    plt.tight_layout()

    # save figures
    path = os.path.join(path_jpeg, "auc summary norm (%s %s).jpeg" %
                        (model, page))
    plt.savefig(path)
    path = os.path.join(path_pdf, "auc summary norm (%s %s).pdf" %
                        (model, page))
    plt.savefig(path)
    path = os.path.join(path_png, "auc summary norm (%s %s).png" %
                        (model, page))
    plt.savefig(path)
    path = os.path.join(path_svg, "auc summary norm (%s %s).svg" %
                        (model, page))
    plt.savefig(path)

    plt.close("all")

    return


def main(result_directory, up, low, std):

    # summary graph
    for page in ["page", "nopage"]:
        for norm in ["l1", "l2", "inf"]:
            graph_auc(result_directory, page, norm, low, up, std)

    # sampling graph
    for model in ["nmf", "svd"]:
        for norm in ["l1", "l2", "inf"]:
            graph_auc_page(result_directory, model, norm, low, up, std)

    # norm graph
    for model in ["nmf", "svd"]:
        for page in ["page", "nopage"]:
            graph_auc_norm(result_directory, model, page, low, up, std)

    # boxplot graph
    graph_boxplot(result_directory, 20, "nopage", "l2")

    return


if __name__ == "__main__":

    # paths
    # result_directory = get_config_tag("result", "cleaning")
    result_directory = "../data/res3"

    # run
    main(result_directory=result_directory,
         up=95,
         low=5,
         std=False)
