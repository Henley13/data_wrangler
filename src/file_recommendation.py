# -*- coding: utf-8 -*-

""" Algorithm to recommend pertinent files to cross. """

# libraries
import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import precision_recall_curve
from metric_transformation import (compute_topic_space_nmf,
                                   compute_topic_space_svd,
                                   get_x_score,
                                   get_x_y_balanced,
                                   learn_metric,
                                   transform_space,
                                   get_all_reused_pairs,
                                   get_auc)
from toolbox.utils import load_sparse_csr, get_config_tag, get_path_cachedir
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.externals import joblib
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
random.seed(13)
print("\n")

# memory cache
general_directory = get_config_tag("data", "general")
memory = joblib.Memory(cachedir=get_path_cachedir(general_directory), verbose=0)


@memory.cache()
def compute_best_topic_space(result_directory, tfidf, df_log, d_best):

    # paths
    model_directory = os.path.join(result_directory, "model_result")

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
    path_w = os.path.join(model_directory, "best_w.npy")
    np.save(path_w, w)

    return w, auc, precision, recall, threshold


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

        # TODO annotate text

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


def find_neighbors(neigh, target):
    rng = neigh.radius_neighbors(target, return_distance=True)
    distances = rng[0][0]
    indices = rng[1][0]
    return distances, indices


def _split_str(s):
    """
    Function to split a string as a list of strings.

    Parameters
    ----------
    s : str
        String with the format "word1 word2 word3"

    Returns
    -------
    l : list
        List of strings with the format ["word1", "word2", "word3"]
    """
    stopwords = ["passerelle_inspire", "donnees_ouvertes",
                 "geoscientific_information", "grand_public"]
    if isinstance(s, str):
        s = s.split(" ")
        for word in stopwords:
            if word in s:
                s.remove(word)
        return s
    elif isinstance(s, float) and np.isnan(s):
        return []
    else:
        print("s :", type(s), s)
        raise ValueError("wrong type : a string is expected")


def information_neighbors(i_target, df_log, distances, indices):
    # target information
    print("#################################")
    print("target :", df_log.at[i_target, "matrix_name"])
    print("--- source file :", df_log.at[i_target, "source_file"])
    print("--- page :", df_log.at[i_target, "title_page"])
    print("--- producer :", df_log.at[i_target, "title_producer"])
    print("--- extension :", df_log.at[i_target, "extension"])
    print("--- tags :", df_log.at[i_target, "tags_page"])
    target_reuse = _split_str(df_log.at[i_target, "reuse"])
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
            neighbors_reuse = _split_str(df_log.at[indice, "reuse"])
            if len(set(target_reuse).intersection(neighbors_reuse)) > 0:
                print("--- common reuse :", True)
            else:
                print("--- common reuse :", False)
            print("-----------------------------------------------", "\n")

    return


@memory.cache()
def dimensionality_reduction(result_directory, w):
    pca = PCA(n_components=2, whiten=False, random_state=13)
    w_reduced = pca.fit_transform(w)
    # save topic space
    model_directory = os.path.join(result_directory, "model_result")
    path = os.path.join(model_directory, "best_w_reduced.npy")
    np.save(path, w_reduced)
    return w_reduced


def graph_neighbors(result_directory, df_log, w_reduced, indices, i_target):
    # paths
    path_png = os.path.join(result_directory, "graphs", "png")
    path_pdf = os.path.join(result_directory, "graphs", "pdf")
    path_jpeg = os.path.join(result_directory, "graphs", "jpeg")
    path_svg = os.path.join(result_directory, "graphs", "svg")

    # get data
    id_page_target = df_log.at[i_target, "id_page"]

    # gather the files reused together
    pairs = []
    all_indices = indices + [i_target]
    for i in range(len(all_indices) - 1):
        indice_i = all_indices[i]
        i_reuse = _split_str(df_log.at[indice_i, "reuse"])
        loc_i = [w_reduced[indice_i, 0], w_reduced[indice_i, 1]]
        for indice_j in all_indices[i + 1:]:
            j_reuse = _split_str(df_log.at[indice_j, "reuse"])
            if len(set(i_reuse).intersection(j_reuse)) > 0:
                loc_j = [w_reduced[indice_j, 0], w_reduced[indice_j, 1]]
                pairs.append([loc_i, loc_j])

    # gather the neighbors by page
    d_indices_page = defaultdict(lambda: [])
    for indice_i in indices:
        id_page_neighbor = df_log.at[indice_i, "id_page"]
        d_indices_page[id_page_neighbor].append(indice_i)

    # plot data
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_facecolor('white')

    # draw straight lines between reused files
    for pair in pairs:
        loc_i, loc_j = pair[0], pair[1]
        plt.plot([loc_i[0], loc_j[0]], [loc_i[1], loc_j[1]], linestyle='-',
                 linewidth=1, alpha=0.2, c="darkorange")

    axes = []
    axes_names = []
    aaa = None
    name_target = None
    for cluster_page in d_indices_page:
        indices_cluster = d_indices_page[cluster_page]
        name_cluster = df_log.at[indices_cluster[0], "title_page"]
        w_cluster_page = w_reduced[indices_cluster]
        if cluster_page == id_page_target:
            aaa = ax.scatter(w_reduced[i_target, 0], w_reduced[i_target, 1],
                             s=45, c="firebrick", marker="D")
            name_target = name_cluster
        else:
            axe = ax.scatter(w_cluster_page[:, 0], w_cluster_page[:, 1], s=40,
                             marker="x")
            axes.append(axe)
            axes_names.append(name_cluster)

    aa = ax.scatter(w_reduced[i_target, 0], w_reduced[i_target, 1], s=50,
                    c="firebrick", marker="o")

    ax.set_xlabel("First ACP component", fontsize=15)
    ax.set_ylabel("Second ACP component", fontsize=15)

    # if aaa is not None:
    #     axes_legend = tuple([aa, aaa] + axes)
    #     names_legend = tuple(["targeted file", name_target] + axes_names)
    #     plt.legend(axes_legend,
    #                names_legend,
    #                scatterpoints=3,
    #                loc='upper center',
    #                ncol=2,
    #                fontsize=10)
    # else:
    #     axes_legend = tuple([aa] + axes)
    #     names_legend = tuple(["targeted file"] + axes_names)
    #     plt.legend(axes_legend,
    #                names_legend,
    #                scatterpoints=3,
    #                loc='upper center',
    #                ncol=2,
    #                fontsize=10)
    plt.tight_layout()

    # save figures
    path = os.path.join(path_jpeg, "kneighbors pca.jpeg")
    plt.savefig(path)
    path = os.path.join(path_pdf, "kneighbors pca.pdf")
    plt.savefig(path)
    path = os.path.join(path_png, "kneighbors pca.png")
    plt.savefig(path)
    path = os.path.join(path_svg, "kneighbors pca.svg")
    plt.savefig(path)

    plt.close("all")

    return


@memory.cache()
def best_neighbors(result_directory, df_log, w, neigh):
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
        target_reuse = _split_str(df_log.at[i, "reuse"])
        distances, indices = find_neighbors(neigh, target)

        # compute statistics with clone files included
        n_pages_clone = len(set(list(df_log.loc[indices, "id_page"])))
        n_neighbors_clone = len(indices)

        # compute n_reuses_clone
        n_reuses_clone = 0
        for a in range(len(indices) - 1):
            a_neighbor = indices[a]
            a_reuse = _split_str(df_log.at[a_neighbor, "reuse"])
            for b_neighbor in indices[a + 1:]:
                b_reuse = _split_str(df_log.at[b_neighbor, "reuse"])
                if len(set(a_reuse).intersection(b_reuse)) > 0:
                    n_reuses_clone += 1

        # compute n_reuses_target_clone
        n_reuses_target_clone = 0
        for j_neighbor in indices:
            j_reuse = _split_str(df_log.at[j_neighbor, "reuse"])
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
            a_reuse = _split_str(df_log.at[a_neighbor, "reuse"])
            for b_neighbor in indices[a + 1:]:
                b_reuse = _split_str(df_log.at[b_neighbor, "reuse"])
                if len(set(a_reuse).intersection(b_reuse)) > 0:
                    n_reuse_same_page += 1

        # compute n_reuses_target_clone
        n_reuses_target_same_page = 0
        for j_neighbor in indices:
            j_reuse = _split_str(df_log.at[j_neighbor, "reuse"])
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
            a_reuse = _split_str(df_log.at[a_neighbor, "reuse"])
            for b_neighbor in indices[a + 1:]:
                b_reuse = _split_str(df_log.at[b_neighbor, "reuse"])
                if len(set(a_reuse).intersection(b_reuse)) > 0:
                    n_reuses += 1

        # compute n_reuses_target
        n_reuses_target = 0
        for j_neighbor in indices:
            j_reuse = _split_str(df_log.at[j_neighbor, "reuse"])
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


def graph_template(result_directory, x, y, xlabel, ylabel, filename, color,
                   marker):
    # paths
    path_png = os.path.join(result_directory, "graphs", "png")
    path_pdf = os.path.join(result_directory, "graphs", "pdf")
    path_jpeg = os.path.join(result_directory, "graphs", "jpeg")
    path_svg = os.path.join(result_directory, "graphs", "svg")

    # plot neighborhood
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_facecolor('white')

    ax.scatter(x, y, s=20, c=color, marker=marker)

    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)

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


def graph_neighborhood(result_directory, df, w_reduced):

    # plot neighborhood
    graph_template(result_directory,
                   x=w_reduced[0],
                   y=w_reduced[1],
                   xlabel="First PCA component",
                   ylabel="Second PCA component",
                   filename="neighborhood",
                   color="steelblue",
                   marker="x")

    # plot reuse neighborhood
    graph_template(result_directory,
                   x=df["n_reuses"],
                   y=df["n_reuses_target"],
                   xlabel="Number of reuses between the neighbors",
                   ylabel="Number of reuses with the targeted file",
                   filename="reuse neighborhood",
                   color="darkorange",
                   marker="D")
    graph_template(result_directory,
                   x=df["n_reuse_same_page"],
                   y=df["n_reuses_target_same_page"],
                   xlabel="Number of reuses between the neighbors",
                   ylabel="Number of reuses with the targeted file",
                   filename="reuse neighborhood (target page)",
                   color="darkorange",
                   marker="D")
    graph_template(result_directory,
                   x=df["n_reuses_clone"],
                   y=df["n_reuses_target_clone"],
                   xlabel="Number of reuses between the neighbors",
                   ylabel="Number of reuses with the targeted file",
                   filename="reuse neighborhood (clone)",
                   color="darkorange",
                   marker="D")

    # plot page neighborhood
    graph_template(result_directory,
                   x=df["n_pages"],
                   y=df["n_neighbors"],
                   xlabel="Number of neighbors",
                   ylabel="Number of pages in the neighborhood",
                   filename="page neighborhood",
                   color="firebrick",
                   marker="o")
    graph_template(result_directory,
                   x=df["n_pages"],
                   y=df["n_neighbors_same_page"],
                   xlabel="Number of neighbors",
                   ylabel="Number of pages in the neighborhood",
                   filename="page neighborhood (target page)",
                   color="firebrick",
                   marker="o")
    graph_template(result_directory,
                   x=df["n_pages_clone"],
                   y=df["n_neighbors_clone"],
                   xlabel="Number of neighbors",
                   ylabel="Number of pages in the neighborhood",
                   filename="page neighborhood (clone)",
                   color="firebrick",
                   marker="o")

    return


def find_radius(precision, recall, threshold):
    p = np.percentile(precision, q=95)
    top_i = []
    for i in range(len(precision)):
        if precision[i] > p:
            top_i.append(i)
    top_recall = [recall[i] for i in top_i]
    i_max = top_i[np.argmax(top_recall)]
    radius = - threshold[i_max]
    return radius, i_max


def main(result_directory, d_best, null_distance, i_target):

    # get data
    path_log = os.path.join(result_directory, "log_final_reduced_with_reuse")
    df_log = pd.read_csv(path_log, header=0, encoding="utf-8", sep=";",
                         index_col=False)
    path_tfidf = os.path.join(result_directory, "tfidf.npz")
    tfidf = load_sparse_csr(path_tfidf)
    print("df_log shape :", df_log.shape)
    print("tfidf shape :", tfidf.shape, "\n")

    # find best topic space
    max_auc = 0
    best_n_topics = 0
    for n_topics in range(5, 101, 5):
        d_best["n_topics"] = n_topics
        w, auc, precision, recall, threshold = compute_best_topic_space(
            result_directory, tfidf, df_log, d_best)
        print("--- auc :", auc, "(%i topics)" % n_topics)
        if auc > max_auc:
            best_n_topics = n_topics
            max_auc = auc

    # compute best topic space
    d_best["n_topics"] = best_n_topics
    w, auc, precision, recall, threshold = compute_best_topic_space(
        result_directory, tfidf, df_log, d_best)
    print("--------------------------------------------")
    print("w shape :", w.shape)
    print(" best auc :", auc, "(%i topics)" % best_n_topics)

    # fit the neighborhood
    radius, i_radius = find_radius(precision, recall, threshold)
    neigh = define_neighbors(result_directory, w, radius, d_best["norm"])
    print("recommendation radius :", radius, "\n")

    # graph precision recall curve
    graph_precision_recall(auc, recall, precision, i_radius)

    # find neighbors
    df = best_neighbors(result_directory, df_log, w, neigh)
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
    print("i_target :", i_target)
    print("number of neighbors :", len(indices), "\n")
    information_neighbors(i_target, df_log, distances, indices)

    # get a 2D plan from the topic space
    w_reduced = dimensionality_reduction(result_directory, w)

    # plot neighbors
    graph_neighbors(result_directory, df_log, w_reduced, indices, i_target)

    # plot neighborhood statistics
    graph_neighborhood(result_directory, df, w_reduced)

    return


if __name__ == "__main__":

    # paths
    # result_directory = get_config_tag("result", "cleaning")
    result_directory = "../data/res3"

    # parameters
    d_best = dict()
    d_best["model"] = "svd"
    d_best["balance_reuse"] = 0.3
    d_best["same_page"] = False
    d_best["max_reuse"] = 30000
    d_best["learning"] = True
    d_best["norm"] = "l2"

    # run
    main(result_directory=result_directory,
         d_best=d_best,
         null_distance=False,
         i_target=13816)
