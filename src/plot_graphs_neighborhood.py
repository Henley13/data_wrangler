# -*- coding: utf-8 -*-

""" Different functions to plot """

# libraries
import os
import pandas as pd
import numpy as np
import joblib
import shutil
from tqdm import tqdm
from collections import defaultdict
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import precision_recall_curve
from toolbox.utils import (get_config_tag, load_sparse_csr,
                           split_str, get_path_cachedir)
from sklearn.decomposition import PCA
from metric_transformation import (compute_topic_space_nmf,
                                   compute_topic_space_svd,
                                   get_all_reused_pairs, get_x_y_balanced,
                                   get_auc, get_x_score, learn_metric,
                                   transform_space)
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

print("\n")

# memory cache
general_directory = get_config_tag("data", "general")
memory = joblib.Memory(cachedir=get_path_cachedir(general_directory), verbose=0)


@memory.cache()
def compute_best_topic_space(result_directory, tfidf, df_log, d_best):
    # parameters
    model = d_best["model"]
    n_topics = d_best["n_topics"]
    balance_reuse = d_best["balance_reuse"]
    same_page = d_best["same_page"]
    max_reuse = d_best["max_reuse"]
    learning = d_best["learning"]
    norm = d_best["norm"]

    # reshape data
    df_log_reset = df_log.reset_index(drop=False, inplace=False)

    # get topic space
    if model == "nmf":
        w, model_fitted = compute_topic_space_nmf(tfidf, n_topics)
    else:
        w, model_fitted = compute_topic_space_svd(tfidf, n_topics)

    # get all reused pairs
    df_pairs_reused = get_all_reused_pairs(df_log_reset)

    #  build a dataset Xy
    x, y = get_x_y_balanced(df_pairs_reused, df_log_reset, w, same_page,
                            balance_reuse, max_reuse)

    # compute auc, precision, recall and threshold
    auc = get_auc(x, y, norm)
    x_score = get_x_score(x, norm)
    precision, recall, threshold = precision_recall_curve(y, x_score)

    # transform topic space
    if learning:
        l = learn_metric(x, y)
        w = transform_space(w, l)

    # save topic space
    path_w = os.path.join(result_directory, "best_w.npy")
    np.save(path_w, w)

    return w, auc, precision, recall, threshold


###############################################################################


def find_radius(precision, recall, threshold, min_precision):
    top_i = []
    for i in range(len(precision)):
        if precision[i] > min_precision:
            top_i.append(i)
    top_recall = [recall[i] for i in top_i]
    i_max = top_i[np.argmax(top_recall)]
    radius = - threshold[i_max]
    return radius, i_max


@memory.cache()
def define_neighbors(result_directory, w, radius, metric):
    # fit model
    neigh = NearestNeighbors(radius=radius, metric=metric)
    neigh.fit(w, )

    # save model
    model_directory = os.path.join(result_directory, "model_result")
    path = os.path.join(model_directory, "neighbors.pkl")
    joblib.dump(neigh, path)

    return neigh


def graph_precision_recall(auc, recall, precision, i_radius):
    # path
    path_png = os.path.join(result_directory, "graphs", "png")
    path_pdf = os.path.join(result_directory, "graphs", "pdf")
    path_jpeg = os.path.join(result_directory, "graphs", "jpeg")
    path_svg = os.path.join(result_directory, "graphs", "svg")

    # example
    recall_example = recall[i_radius]
    precision_example = precision[i_radius]

    # graph
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_facecolor('white')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])

    ax.step(recall, precision, color='steelblue', alpha=0.2, where='post')
    ax.fill_between(recall, precision, step='post', alpha=0.2,
                    color='steelblue')

    if isinstance(i_radius, int):
        aa = ax.scatter([recall_example], [precision_example], s=35,
                        c="firebrick", marker="x")

    ax.set_xlabel("Recall", fontsize=15)
    ax.set_ylabel("Precision", fontsize=15)
    ax.set_title("Precision-Recall curve: AUC={0:0.2f}".format(auc),
                 fontsize=15)
    plt.tight_layout()

    # save figures
    path = os.path.join(path_jpeg, "precision recall curve.jpeg")
    plt.savefig(path)
    path = os.path.join(path_pdf, "precision recall curve.pdf")
    plt.savefig(path)
    path = os.path.join(path_png, "precision recall curve.png")
    plt.savefig(path)
    path = os.path.join(path_svg, "precision recall curve.svg")
    plt.savefig(path)

    plt.close("all")

    return


def find_neighbors(neigh, target):
    rng = neigh.radius_neighbors(target, return_distance=True)
    distances = rng[0][0]
    indices = rng[1][0]
    return distances, indices


@memory.cache()
def all_neighbors(result_directory, df_log, w, neigh):
    # find neighbors for each file and save information in a dataframe
    names = ["indice_target", "id_page", "page", "producer", "extension",
             "mean_distance", "indices_neighbors",
             "n_pages", "n_pages_clone",
             "n_reuses_target", "n_reuses_target_same_page",
             "n_reuses_target_clone",
             "n_reuses", "n_reuse_same_page", "n_reuses_clone",
             "n_neighbors", "n_neighbors_same_page", "n_neighbors_clone"]
    df = pd.DataFrame(columns=names)

    for i in tqdm(range(df_log.shape[0]), desc="best neighbors"):
        id_page = df_log.at[i, "id_page"]
        target = w[i].reshape(1, -1)
        target_reuse = split_str(df_log.at[i, "reuses"])
        distances, indices = find_neighbors(neigh, target)

        # compute statistics with clone files included
        n_pages_clone = len(set(list(df_log.loc[indices, "id_page"])))
        n_neighbors_clone = len(indices)

        # compute n_reuses_clone
        n_reuses_clone = 0
        for a in range(len(indices) - 1):
            a_neighbor = indices[a]
            a_reuse = split_str(df_log.at[a_neighbor, "reuses"])
            for b_neighbor in indices[a + 1:]:
                b_reuse = split_str(df_log.at[b_neighbor, "reuses"])
                if len(set(a_reuse).intersection(b_reuse)) > 0:
                    n_reuses_clone += 1

        # compute n_reuses_target_clone
        n_reuses_target_clone = 0
        for j_neighbor in indices:
            j_reuse = split_str(df_log.at[j_neighbor, "reuses"])
            if len(set(target_reuse).intersection(j_reuse)) > 0:
                n_reuses_target_clone += 1

        # filter the clone files
        l = []
        k = []
        for i_distance, distance in enumerate(distances):
            if distance != 0:
                l.append(distance)
                k.append(indices[i_distance])
        distances = l
        indices = k

        # compute statistics with files from targeted page included
        n_neighbors_same_page = len(indices)

        # compute n_reuses_clone
        n_reuse_same_page = 0
        for a in range(len(indices) - 1):
            a_neighbor = indices[a]
            a_reuse = split_str(df_log.at[a_neighbor, "reuses"])
            for b_neighbor in indices[a + 1:]:
                b_reuse = split_str(df_log.at[b_neighbor, "reuses"])
                if len(set(a_reuse).intersection(b_reuse)) > 0:
                    n_reuse_same_page += 1

        # compute n_reuses_target_clone
        n_reuses_target_same_page = 0
        for j_neighbor in indices:
            j_reuse = split_str(df_log.at[j_neighbor, "reuses"])
            if len(set(target_reuse).intersection(j_reuse)) > 0:
                n_reuses_target_same_page += 1

        # filter the files from the targeted page
        l = []
        k = []
        for i_indice, indice in enumerate(indices):
            if df_log.at[indice, "id_page"] != id_page:
                l.append(distances[i_indice])
                k.append(indice)
        distances = l
        indices = k

        # compute statistics with a restricted neighborhood
        n_pages = len(set(list(df_log.loc[indices, "id_page"])))
        n_neighbors = len(indices)
        mean_distance = np.mean(distances) if n_neighbors > 0 else np.nan

        # compute n_reuses
        n_reuses = 0
        for a in range(len(indices) - 1):
            a_neighbor = indices[a]
            a_reuse = split_str(df_log.at[a_neighbor, "reuses"])
            for b_neighbor in indices[a + 1:]:
                b_reuse = split_str(df_log.at[b_neighbor, "reuses"])
                if len(set(a_reuse).intersection(b_reuse)) > 0:
                    n_reuses += 1

        # compute n_reuses_target
        n_reuses_target = 0
        for j_neighbor in indices:
            j_reuse = split_str(df_log.at[j_neighbor, "reuses"])
            if len(set(target_reuse).intersection(j_reuse)) > 0:
                n_reuses_target += 1

        # save results
        l = [i, id_page, df_log.at[i, "title_page"],
             df_log.at[i, "title_producer"], df_log.at[i, "extension"],
             mean_distance, indices,
             n_pages, n_pages_clone,
             n_reuses_target, n_reuses_target_same_page,
             n_reuses_target_clone,
             n_reuses, n_reuse_same_page, n_reuses_clone,
             n_neighbors, n_neighbors_same_page, n_neighbors_clone]
        df_row = pd.DataFrame([l], columns=names)
        df = pd.concat([df, df_row])

    path = os.path.join(result_directory, "graphs", "neighbors_information")
    df.to_csv(path, sep=";", encoding="utf-8", index=False, header=True)

    return df


def information_neighbors(i_target, df_log, distances, indices, paths):
    # target information
    print("#################################")
    print("target :", df_log.at[i_target, "matrix_name"])
    print("--- source file :", df_log.at[i_target, "source_file"])
    print("--- page :", df_log.at[i_target, "title_page"])
    print("--- producer :", df_log.at[i_target, "title_producer"])
    print("--- extension :", df_log.at[i_target, "extension"])
    print("--- tags :", df_log.at[i_target, "tags_page"])
    target_reuse = split_str(df_log.at[i_target, "reuses"])
    print("#################################", "\n")

    # neighbors information
    for i in range(len(indices)):
        indice = indices[i]
        if indice != i_target:
            print("neighbors :", df_log.at[indice, "matrix_name"])
            print("--- source file :", df_log.at[indice, "source_file"])
            print("--- distance :", distances[i])
            print("--- page :", df_log.at[indice, "title_page"])
            print("--- producer :", df_log.at[indice, "title_producer"])
            print("--- extension :", df_log.at[indice, "extension"])
            print("--- tags :", df_log.at[indice, "tags_page"])
            neighbors_reuse = split_str(df_log.at[indice, "reuses"])
            if len(set(target_reuse).intersection(neighbors_reuse)) > 0:
                print("--- common reuse :", True)
            else:
                print("--- common reuse :", False)
            print("-----------------------------------------------", "\n")

    for path in paths:
        filename = os.path.join(path, "log %s" % str(i_target))
        with open(filename, mode="at", encoding="utf-8") as f:
            f.write("#################################" + "\n")
            f.write("target : " +
                    str(df_log.at[i_target, "matrix_name"]) + "\n")
            f.write("--- source file : " +
                    str(df_log.at[i_target, "source_file"]) + "\n")
            f.write("--- page : " +
                    str(df_log.at[i_target, "title_page"]) + "\n")
            f.write("--- producer : " +
                    str(df_log.at[i_target, "title_producer"]) + "\n")
            f.write("--- extension : " +
                    str(df_log.at[i_target, "extension"]) + "\n")
            f.write("--- tags : " +
                    str(df_log.at[i_target, "tags_page"]) + "\n")
            f.write("#################################" + "\n")
            f.write("\n")
            for i in range(len(indices)):
                indice = indices[i]
                if indice != i_target:
                    f.write("neighbors : " +
                            str(df_log.at[indice, "matrix_name"]) + "\n")
                    f.write("--- source file : " +
                            str(df_log.at[indice, "source_file"]) + "\n")
                    f.write("--- distance : " +
                            str(distances[i]) + "\n")
                    f.write("--- page : " +
                            str(df_log.at[indice, "title_page"]) + "\n")
                    f.write("--- producer : " +
                            str(df_log.at[indice, "title_producer"]) + "\n")
                    f.write("--- extension : " +
                            str(df_log.at[indice, "extension"]) + "\n")
                    f.write("--- tags : " +
                            str(df_log.at[indice, "tags_page"]) + "\n")
                    neighbors_reuse = split_str(df_log.at[indice, "reuses"])
                    if len(set(target_reuse).intersection(neighbors_reuse)) > 0:
                        f.write("--- common reuse : True" + "\n")
                    else:
                        f.write("--- common reuse : False" + "\n")
                    f.write("------------------------------------------" + "\n")
                    f.write("\n")

    return


def describe_one_file(result_directory, df_log, w, neigh, i_target,
                      null_distance):
    # paths
    path_png = os.path.join(result_directory, "graphs", "png", "plot_neighbors")
    path_pdf = os.path.join(result_directory, "graphs", "pdf", "plot_neighbors")
    path_jpeg = os.path.join(result_directory, "graphs", "jpeg",
                             "plot_neighbors")
    path_svg = os.path.join(result_directory, "graphs", "svg", "plot_neighbors")
    paths = [path_png, path_pdf, path_jpeg, path_svg]

    # get information for a specific file
    target = w[i_target].reshape(1, -1)
    distances, indices = find_neighbors(neigh, target)
    if not null_distance:
        l = []
        k = []
        for i_distance, distance in enumerate(distances):
            if distance != 0:
                l.append(distance)
                k.append(indices[i_distance])
        distances = l
        indices = k
    all_indices = indices + [i_target]

    # log file
    for path in paths:
        filename = os.path.join(path, "log %s" % str(i_target))
        with open(filename, mode="wt", encoding="utf-8") as f:
            f.write("i_target : " + str(i_target))
            f.write("\n")
            f.write("number of neighbors : " + str(len(indices)))
            f.write("\n")

    print("i_target :", i_target)
    print("number of neighbors :", len(indices), "\n")
    information_neighbors(i_target, df_log, distances, indices, paths)
    print()
    print("############################################################")
    print("############################################################", "\n")

    return indices, distances


@memory.cache()
def dimensionality_reduction(result_directory, w):
    # 2D
    pca = PCA(n_components=2, whiten=False, random_state=13)
    w_reduced_2d = pca.fit_transform(w)
    variance_explained_2d = pca.explained_variance_ratio_

    # save topic space
    path = os.path.join(result_directory, "best_w_reduced_2d.npy")
    np.save(path, w_reduced_2d)

    # 3D
    pca = PCA(n_components=3, whiten=False, random_state=13)
    w_reduced_3d = pca.fit_transform(w)
    variance_explained_3d = pca.explained_variance_ratio_
    # save topic space
    path = os.path.join(result_directory, "best_w_reduced_3d.npy")
    np.save(path, w_reduced_3d)

    return (w_reduced_2d, variance_explained_2d, w_reduced_3d,
            variance_explained_3d)


def plot_template_target(result_directory, x, y, xlabel, ylabel, filename,
                         color, marker, x_neighbors=None, y_neighbors=None,
                         radius=None, limits=None, i_target=None):
    # paths
    path_png = os.path.join(result_directory, "graphs", "png")
    path_pdf = os.path.join(result_directory, "graphs", "pdf")
    path_jpeg = os.path.join(result_directory, "graphs", "jpeg")
    path_svg = os.path.join(result_directory, "graphs", "svg")

    # plot neighborhood
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_facecolor('white')
    if limits is not None:
        ax.set_xlim(limits[0], limits[1])
        ax.set_ylim(limits[2], limits[3])

    ax.scatter(x, y, s=20, c=color, marker=marker)

    if i_target is not None:
        ax.scatter(x_neighbors, y_neighbors, s=20, c="firebrick", marker="D")
        ax.scatter(x[i_target], y[i_target], s=25, c="forestgreen", marker="o")

    if radius is not None and i_target is not None:
        circle_1 = plt.Circle((x[i_target], y[i_target]),
                              radius, color='darkorange', alpha=0.3)
        ax.add_patch(circle_1)

    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)

    plt.tight_layout()

    # save figures
    path = os.path.join(path_jpeg, filename + ".jpeg")
    plt.savefig(path)
    path = os.path.join(path_pdf, filename + ".pdf")
    plt.savefig(path)
    path = os.path.join(path_png, filename + ".png")
    plt.savefig(path)
    path = os.path.join(path_svg, filename + ".svg")
    plt.savefig(path)

    plt.close("all")

    return


def plot_topic_space_reduced(result_directory, w_3d, variance_3d):
    # plot 1st and 2nd component
    plot_template_target(result_directory,
                         x=w_3d[:, 0],
                         y=w_3d[:, 1],
                         xlabel="1st component (%s%% variance explained)" %
                                str(round(variance_3d[0] * 100, 3)),
                         ylabel="2nd component(%s%% variance explained)" %
                                str(round(variance_3d[1] * 100, 3)),
                         filename="acp_1_2",
                         color="steelblue",
                         marker="x")

    # plot 2nd and 3rd component
    plot_template_target(result_directory,
                         x=w_3d[:, 1],
                         y=w_3d[:, 2],
                         xlabel="2nd component (%s%% variance explained)" %
                                str(round(variance_3d[1] * 100, 3)),
                         ylabel="3rd component(%s%% variance explained)" %
                                str(round(variance_3d[2] * 100, 3)),
                         filename="acp_2_3",
                         color="steelblue",
                         marker="x")

    # plot 1st and 3rd component
    plot_template_target(result_directory,
                         x=w_3d[:, 0],
                         y=w_3d[:, 2],
                         xlabel="1st component (%s%% variance explained)" %
                                str(round(variance_3d[0] * 100, 3)),
                         ylabel="3rd component(%s%% variance explained)" %
                                str(round(variance_3d[2] * 100, 3)),
                         filename="acp_1_3",
                         color="steelblue",
                         marker="x")
    return


def plot_pca_neighborhood(w_reduced, reused_pairs, dict_page, indices, i_target,
                          radius, id_page_target, ax, xlabel, ylabel, x, y):
    all_indices = indices + [i_target]

    # draw straight lines between reused files
    for pair in reused_pairs:
        loc_i, loc_j = pair[0], pair[1]
        ax.plot([loc_i[x], loc_j[x]], [loc_i[y], loc_j[y]], linestyle='-',
                linewidth=1, alpha=1, c="darkorange")

    for cluster_page in dict_page:
        indices_cluster = dict_page[cluster_page]
        w_cluster_page = w_reduced[indices_cluster]
        if cluster_page == id_page_target:
            ax.scatter(w_cluster_page[:, x], w_cluster_page[:, y], s=60,
                       c="firebrick", marker="D")
        else:
            ax.scatter(w_cluster_page[:, x], w_cluster_page[:, y], s=50,
                       marker="x")

    i_w_target = len(all_indices) - 1
    ax.scatter(w_reduced[i_w_target, x], w_reduced[i_w_target, y], s=70,
               c="firebrick", marker="o")
    circle_1 = plt.Circle((w_reduced[i_w_target, x],
                           w_reduced[i_w_target, y]),
                          radius * (2 / 3), color='darkorange', alpha=0.1)
    circle_2 = plt.Circle((w_reduced[i_w_target, x],
                           w_reduced[i_w_target, y]),
                          radius / 3, color='darkorange', alpha=0.1)
    circle_3 = plt.Circle((w_reduced[i_w_target, x],
                           w_reduced[i_w_target, y]),
                          radius, color='darkorange', alpha=0.1)
    ax.add_patch(circle_1)
    ax.add_patch(circle_2)
    ax.add_patch(circle_3)

    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=15)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=15)

    return


def create_specific_neighbors_directory(result_directory):
    # paths
    path_png = os.path.join(result_directory, "graphs", "png")
    path_pdf = os.path.join(result_directory, "graphs", "pdf")
    path_jpeg = os.path.join(result_directory, "graphs", "jpeg")
    path_svg = os.path.join(result_directory, "graphs", "svg")

    # create a distinct directory
    for path in [path_png, path_pdf, path_jpeg, path_svg]:
        new_path = os.path.join(path, "plot_neighbors")
        if os.path.exists(new_path):
            shutil.rmtree(new_path)
        os.mkdir(new_path)

    return


def graph_neighbors_3d_local(result_directory, df_log, w, indices,
                             i_target, radius, filename):
    # paths
    path_png = os.path.join(result_directory, "graphs", "png")
    path_pdf = os.path.join(result_directory, "graphs", "pdf")
    path_jpeg = os.path.join(result_directory, "graphs", "jpeg")
    path_svg = os.path.join(result_directory, "graphs", "svg")

    # get data
    id_page_target = df_log.at[i_target, "id_page"]
    title_page_target = df_log.at[i_target, "title_page"]
    all_indices = indices + [i_target]

    # apply local pca in three dimensions
    # TODO centered?
    w_neighborhood = w[all_indices]
    pca = PCA(n_components=3, whiten=False, random_state=13)
    w_reduced_3d = pca.fit_transform(w_neighborhood)

    # gather the files reused together
    pairs = []
    for i_w in range(len(all_indices) - 1):
        indice_i = all_indices[i_w]
        i_reuse = split_str(df_log.at[indice_i, "reuses"])
        loc_i = [w_reduced_3d[i_w, 0], w_reduced_3d[i_w, 1],
                 w_reduced_3d[i_w, 2]]
        for j_w in range(i_w + 1, len(all_indices)):
            indice_j = all_indices[j_w]
            j_reuse = split_str(df_log.at[indice_j, "reuses"])
            if len(set(i_reuse).intersection(j_reuse)) > 0:
                loc_j = [w_reduced_3d[j_w, 0], w_reduced_3d[j_w, 1],
                         w_reduced_3d[j_w, 2]]
                pairs.append([loc_i, loc_j])

    # gather the neighbors by page
    d_indices_page = defaultdict(lambda: [])
    for i_w in range(len(indices)):
        indice_i = indices[i_w]
        id_page_neighbor = df_log.at[indice_i, "id_page"]
        d_indices_page[id_page_neighbor].append(i_w)

    # plot data
    fig = plt.figure(figsize=(10, 10))
    ax1 = plt.subplot2grid((2, 2), (0, 0))
    ax1.set_facecolor('white')
    ax2 = plt.subplot2grid((2, 2), (0, 1))
    ax2.set_facecolor('white')
    ax3 = plt.subplot2grid((2, 2), (1, 0), sharex=ax1)
    ax3.set_facecolor('white')
    ax4 = plt.subplot2grid((2, 2), (1, 1), sharey=ax3)
    ax4.set_facecolor('white')

    # first and third component
    plot_pca_neighborhood(w_reduced=w_reduced_3d,
                          reused_pairs=pairs,
                          dict_page=d_indices_page,
                          indices=indices,
                          i_target=i_target,
                          radius=radius,
                          id_page_target=id_page_target,
                          ax=ax1,
                          xlabel="Component 1",
                          ylabel="Component 3",
                          x=0,
                          y=2)

    # first and second component
    plot_pca_neighborhood(w_reduced=w_reduced_3d,
                          reused_pairs=pairs,
                          dict_page=d_indices_page,
                          indices=indices,
                          i_target=i_target,
                          radius=radius,
                          id_page_target=id_page_target,
                          ax=ax3,
                          xlabel="Component 1",
                          ylabel="Component 2",
                          x=0,
                          y=1)

    # second and third component
    plot_pca_neighborhood(w_reduced=w_reduced_3d,
                          reused_pairs=pairs,
                          dict_page=d_indices_page,
                          indices=indices,
                          i_target=i_target,
                          radius=radius,
                          id_page_target=id_page_target,
                          ax=ax4,
                          xlabel="Component 3",
                          ylabel="Component 2",
                          x=2,
                          y=1)

    a = ax2.scatter(0, 0, s=60, c="firebrick", marker="D",
                    label="target page")
    b = ax2.scatter(0, 0, s=70, c="firebrick", marker="o",
                    label="target file")
    c = ax2.scatter(0, 0, s=50, marker="x", c="black", label="other neighbors")
    title = title_page_target.split(" ")
    n = 0
    for i in range(len(title)):
        if i % 4 == 0 and i != 0:
            title.insert(i + n, "\n")
    title = " ".join(title)
    ax2.legend(loc="center", borderpad=2, fontsize=15).set_title(
        title, prop={'size': 'large'})
    for obj in [a, b, c]:
        obj.set_visible(False)
    ax2.axes.get_xaxis().set_visible(False)
    ax2.axes.get_yaxis().set_visible(False)
    ax2.axis('off')
    plt.tight_layout()

    # save figures
    path = os.path.join(path_jpeg, "plot_neighbors", "kneighbors pca 3d %s.jpeg"
                        % filename)
    plt.savefig(path)
    path = os.path.join(path_pdf, "plot_neighbors", "kneighbors pca 3d %s.pdf"
                        % filename)
    plt.savefig(path)
    path = os.path.join(path_png, "plot_neighbors", "kneighbors pca 3d %s.png"
                        % filename)
    plt.savefig(path)
    path = os.path.join(path_svg, "plot_neighbors", "kneighbors pca 3d %s.svg"
                        % filename)
    plt.savefig(path)

    plt.close("all")

    return

###############################################################################


def main(result_directory, d_best, min_precision, targeted_files,
         null_distance):
    """
    Function to run all the script
    :param result_directory: string
    :param d_best: dict
    :param min_precision: float
    :param targeted_files: list of int
    :param null_distance: bool
    :return:
    """

    # paths
    path_log = os.path.join(result_directory, "log_final_reduced")

    # get data
    df_log = pd.read_csv(path_log, header=0, encoding="utf-8", sep=";",
                         index_col=False)
    path_tfidf = os.path.join(result_directory, "tfidf.npz")
    tfidf = load_sparse_csr(path_tfidf)
    print("df_log shape :", df_log.shape)
    print("tfidf shape :", tfidf.shape, "\n")

    # compute topic space
    w, auc, precision, recall, threshold = compute_best_topic_space(
        result_directory, tfidf, df_log, d_best)
    print("--------------------------------------------")
    print("w shape :", w.shape)
    print("best auc :", auc, "(%i topics)" % d_best["n_topics"], "\n")

    # fit the neighborhood
    radius, i_radius = find_radius(precision, recall, threshold, min_precision)
    neigh = define_neighbors(result_directory, w, radius, d_best["norm"])
    print("recommendation radius :", radius, "\n")

    # plot precision recall curve
    graph_precision_recall(auc, recall, precision, i_radius)

    # find all neighborhoods
    # all_neighbors(result_directory, df_log, w, neigh)

    # get a 2D plan from the topic space
    (w_reduced_2d, variance_explained_2d, w_reduced_3d,
     variance_explained_3d) = dimensionality_reduction(result_directory, w)

    # plot reduced topic space
    plot_topic_space_reduced(result_directory, w_reduced_3d,
                             variance_explained_3d)

    # reset new directories for neighbors plots
    create_specific_neighbors_directory(result_directory)

    # get information for a specific file
    for i_target in targeted_files:
        indices, distances = describe_one_file(result_directory, df_log, w,
                                               neigh, i_target, null_distance)

        # plot neighbors 3D
        graph_neighbors_3d_local(result_directory, df_log, w, indices,
                                 i_target, radius, str(i_target))

    return


if __name__ == "__main__":
    # paths
    result_directory = get_config_tag("result", "cleaning")

    # parameters
    d_best = dict()
    d_best["model"] = "nmf"
    d_best["balance_reuse"] = 0.3
    d_best["same_page"] = False
    d_best["max_reuse"] = 30000
    d_best["learning"] = True
    d_best["norm"] = "l2"
    d_best["n_topics"] = 25
    min_precision = 0.5
    i_target = [3855, 3874, 18351, 18344, 18350, 20032, 5309, 20283, 20223,
                20182, 19074, 19090, 19091, 19075, 7901, 22566, 15356, 15005,
                14360, 5556]
    null_distance = False

    # run
    main(result_directory=result_directory,
         d_best=d_best,
         min_precision=min_precision,
         targeted_files=i_target,
         null_distance=null_distance)
