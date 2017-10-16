# -*- coding: utf-8 -*-

""" Run metric learning algorithms with different parametrization to find the
best setup. """

# libraries
import os
import pandas as pd
import numpy as np
import joblib
import random
from collections import defaultdict
from joblib import Parallel, delayed
from metric_learn import lsml
from toolbox.utils import (get_config_tag, get_path_cachedir, save_array,
                           check_directory, load_sparse_csr)
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.decomposition import NMF, TruncatedSVD
random.seed(13)
print("\n")

# memory cache
general_directory = get_config_tag("data", "general")
memory = joblib.Memory(cachedir=get_path_cachedir(general_directory), verbose=0)


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


@memory.cache()
def get_all_reused_pairs(df_log):
    """
    Function to collect all the pairs of files reused together.

    Parameters
    ----------
    df_log : pandas Dataframe
        Dataframe of the metadata

    Returns
    -------
    df : pandas Dataframe
        Dataframe of pairs reused together with original index
    """
    # reduce the dataframe to the reused files
    df_reuse = df_log.query("n_reuses > 0")

    # for each reused file...
    same_page = []
    same_reuse = []
    x1 = []
    x2 = []
    for n, i in enumerate(df_reuse.index.values.tolist()):
        # for i in range(df_reuse.shape[0] - 1):

        i_reuse = _split_str(df_reuse.at[i, "reuses"])
        i_page = df_reuse.at[i, "id_page"]

        # ...get all the potential pairs...
        for j in df_reuse.index.values.tolist()[n + 1:]:
            j_reuse = _split_str(df_reuse.at[j, "reuses"])
            j_page = df_reuse.at[j, "id_page"]

            # ...and test them
            same_reuse.append(int(len(set(i_reuse).intersection(j_reuse)) > 0))
            same_page.append(int(i_page == j_page))
            x1.append(i)
            x2.append(j)

    # create a dataframe with all the pairs reused together
    df = pd.DataFrame({"x1": x1,
                       "x2": x2,
                       "same_page": same_page,
                       "same_reuse": same_reuse})
    df = df.query("same_reuse == 1")

    return df


def get_test_page(df_log, ratio_test):
    """
    Function to randomly select a bunch of pages for the test sample.

    Parameters
    ----------
    df_log : pandas Dataframe
        Dataframe of the metadata

    ratio_test : float
        Proportion of the test sample compare to the train one

    Returns
    -------
    test_title_page : list
        list of pages for the test sample
    """
    # get a sample of files (belonging to the same pages) for the test
    title_page = df_log["title_page"]
    test_title_page = list(set(title_page))
    k = int(len(test_title_page) * ratio_test)
    test_title_page = random.sample(test_title_page, k)
    return test_title_page


def get_train_sample(tfidf, df_log, test_title_page):
    """
    Function to extract a train sample from the data.

    Parameters
    ----------
    tfidf : scipy csr matrix
        Sparse row matrix [n_samples, n_words]

    df_log : pandas Dataframe
        Dataframe of the metadata

    test_title_page : list

    Returns
    -------
    df_log_train : pandas Dataframe
        Dataframe of the train metadata with reset index

    tfidf_train : scipy csr matrix
        Sparse row matrix [n_samples, n_words]
    """
    # reduce the dataframe
    df_log_train = df_log.query("title_page not in @test_title_page")

    # extract a sample of the tfidf matrix
    tfidf_train = tfidf[df_log_train.index.values.tolist()]

    # reset the index
    df_log_train.reset_index(drop=False, inplace=True)

    return df_log_train, tfidf_train


def get_test_sample(tfidf, df_log, test_title_page):
    """
    Function to extract a test sample from the data.

    Parameters
    ----------
    tfidf : scipy csr matrix
        Sparse row matrix [n_samples, n_words]

    df_log : pandas Dataframe
        Dataframe of the metadata

    test_title_page : list

    Returns
    -------
    df_log_test : pandas Dataframe
        Dataframe of the train metadata with reset index

    tfidf_test : scipy csr matrix
        Sparse row matrix [n_samples, n_words]
    """
    # reduce the dataframe
    df_log_test = df_log.query("title_page in @test_title_page")

    # extract a sample of the tfidf matrix
    tfidf_test = tfidf[df_log_test.index.values.tolist()]

    # reset the index
    df_log_test.reset_index(drop=False, inplace=True)

    return df_log_test, tfidf_test


def compute_topic_space_nmf(tfidf, n_topics):
    """
    Function to compute a topic extraction from a tfidf matrix.

    Parameters
    ----------
    tfidf : scipy csr matrix
        Sparse row matrix [n_samples, n_words]

    n_topics : int
         Number of topics to compute

    Returns
    -------
    w : numpy matrix
        Topic space matrix [n_samples, n_topics]

    nmf : fitted model
    """
    nmf = NMF(n_components=n_topics, random_state=1, alpha=.1, l1_ratio=.5)
    nmf.fit(tfidf)
    w = nmf.transform(tfidf)
    return w, nmf


def compute_topic_space_svd(tfidf, n_topics):
    """
    Function to compute a topic extraction from a tfidf matrix.

    Parameters
    ----------
    tfidf : scipy csr matrix
        Sparse row matrix [n_samples, n_words]

    n_topics : int
        Number of topics to compute

    Returns
    -------
    w : numpy matrix
        Topic space matrix [n_samples, n_topics]

    svd : fitted model
    """
    svd = TruncatedSVD(n_components=n_topics, algorithm='randomized', n_iter=5,
                       random_state=1, tol=0.0)
    svd.fit(tfidf)
    w = svd.transform(tfidf)
    return w, svd


def compute_topic_space_train(tfidf_train, n_topics, model):
    """
    Function to fit a topic extraction model on the train sample and get the
    fitted model.

    Parameters
    ----------
    tfidf_train : scipy csr matrix
        Sparse row matrix [n_samples, n_words]

    n_topics : int
        Number of topics to compute

    model : str
        String to specify the model used to generate topic space ('nmf' or
        'svd')

    Returns
    -------
    w_train : numpy matrix
        Topic space matrix [n_samples, n_topics]

    fitted_model : fitted model
    """
    if model == "nmf":
        w_train, fitted_model = compute_topic_space_nmf(tfidf_train, n_topics)
    elif model == "svd":
        w_train, fitted_model = compute_topic_space_svd(tfidf_train, n_topics)
    else:
        print("model :", model)
        raise ValueError("Model isn't recognized : 'nmf' or 'svd' are expected")

    return w_train, fitted_model


def get_x_y_balanced(df_pairs_reused, df_log_sample, w, same_page,
                     balance_reuse, max_reuse):
    """
    Function to extract a ndarray with the distances between pairs and the
    boolean to know if the pair has been present in a common reuse.

    Parameters
    ----------
    df_pairs_reused : pandas Dataframe
        Dataframe of all the pairs reused together

    df_log_sample : pandas Dataframe
        Dataframe of the sample metadata

    w : numpy matrix
        Topic space matrix [n_samples, n_topics]

    same_page : bool
        Boolean to decide if files from the same page are accepted as a proper
        pair

    balance_reuse : float
        Proportion of reused pairs in the final dataset

    max_reuse : int
        Maximum number or reused pairs to add in the dataset

    Returns
    -------
    x : numpy matrix

    y : numpy array
    """

    if not same_page:
        df_pairs_reused = df_pairs_reused.query("same_page == 0")

    all_original_index = list(df_log_sample["index"])
    all_new_index = df_log_sample.index.values.tolist()
    df = df_pairs_reused.query("x1 in @all_original_index & x2 in "
                               "@all_original_index")
    n = min(max_reuse, df.shape[0])
    df = df.sample(n=n, replace=False)

    # for each pairs reused together
    x_reuse = []
    for i in df.index.values.tolist():
        # get the original index of both files
        x_1 = df.at[i, "x1"]
        x_2 = df.at[i, "x2"]

        # get their new index
        new_x_1 = int(df_log_sample.query("index == @x_1").index.values)
        new_x_2 = int(df_log_sample.query("index == @x_2").index.values)

        # get their vector in the topic space and compute their distance
        topic_1 = w[new_x_1, :]
        topic_2 = w[new_x_2, :]
        distance_12 = np.absolute(topic_1 - topic_2)
        if np.any(np.isnan(distance_12)):
            continue
        if not np.all(np.isfinite(distance_12)):
            continue
        x_reuse.append(distance_12)

    # balance the data
    n_reused_pairs = len(x_reuse)
    n_noreused_pairs = int((1 - balance_reuse) * n_reused_pairs / balance_reuse)

    # for each non-reused files
    n = 0
    x_no_reuse = []
    while n < n_noreused_pairs:
        l = random.sample(all_new_index, 2)

        # test them
        reuse_1 = _split_str(df_log_sample.at[l[0], "reuses"])
        reuse_2 = _split_str(df_log_sample.at[l[1], "reuses"])
        if len(set(reuse_1).intersection(reuse_2)) > 0:
            continue
        page_1 = df_log_sample.at[l[0], "title_page"]
        page_2 = df_log_sample.at[l[1], "title_page"]
        if not same_page and page_1 == page_2:
            continue

        # and compute their distance
        topic_1 = w[l[0], :]
        topic_2 = w[l[1], :]
        distance_12 = np.absolute(topic_1 - topic_2)
        if np.any(np.isnan(distance_12)):
            continue
        if not np.all(np.isfinite(distance_12)):
            continue
        x_no_reuse.append(distance_12)

        n += 1

    # get the target vectors
    y_reuse = [1] * len(x_reuse)
    y_no_reuse = [0] * len(x_no_reuse)

    # create the future dataset
    x = x_reuse + x_no_reuse
    y = y_reuse + y_no_reuse
    df = pd.DataFrame(np.asarray(x))
    names = ["X_%i" % i for i in range(df.shape[1])]
    df.columns = names
    df["y"] = y
    x_names = list(df.columns)
    x_names.remove("y")
    x = df.as_matrix(columns=x_names)
    y = np.asarray(df["y"])

    return x, y


def learn_metric(x, y):
    """
    Function to learn a metric from distance and boolean vectors.

    Parameters
    ----------
    x : numpy matrix
        Distances between pairs of files (n_samples, n_topics)

    y : numpy array
        Boolean to know if the pairs has been reused together (n_samples,)

    Returns
    -------
    l : numpy matrix
        Transformer learned (n_topics, n_topics)
    """
    model = lsml.LSML_Supervised(verbose=False)
    model.fit(x, y)
    l = model.transformer()
    return l


def transform_space(space, l):
    """
    Function to apply a transformation to a vector space.

    Parameters
    ----------
    space : numpy matrix
        Topic space

    l : numpy matrix
        Learned transformer

    Returns
    -------
    space_transformed : numpy matrix
        Transformed space
    """
    space_transformed = np.dot(space, l.T)
    return space_transformed


def get_x_y_test(df_pairs_reused, df_log_train, w_train, df_log_test, w_test,
                 balance_reuse, max_reuse):
    """
    Function to extract a balanced testing dataset X, y.

    Parameters
    ----------
    df_pairs_reused : pandas Dataframe
        Dataframe with all reused pairs

    df_log_train : pandas Dataframe
        Metadata dataframe

    w_train : sparse matrix csr
        Shape [n_sample, n_topic]

    df_log_test : pandas Dataframe
        Metadata dataframe

    w_test : sparse matrix csr
        Shape [n_sample, n_topic]

    balance_reuse : float
        Ratio of reused pairs to keep in the X, y dataset

    max_reuse : int
        Maximum number of reused pairs to keep in the X, y dataset. It also
        limits the dataset.
    Returns
    -------
    x_test : numpy matrix

    x_test_transformed : numpy matrix

    y_test : numpy array

    same_page : bool
    """
    for same_page in [True, False]:

        # build a dataset X, y with the train data
        x_train, y_train = get_x_y_balanced(df_pairs_reused,
                                            df_log_train,
                                            w_train,
                                            same_page,
                                            balance_reuse=balance_reuse,
                                            max_reuse=max_reuse)

        # build a dataset X, y with the test data untransformed
        x_test, y_test = get_x_y_balanced(df_pairs_reused,
                                          df_log_test,
                                          w_test,
                                          same_page,
                                          balance_reuse=balance_reuse,
                                          max_reuse=max_reuse)

        try:
            # learn a metric
            l = learn_metric(x_train, y_train)

            # apply transformation on test data
            x_test_transformed = transform_space(x_test, l)
        except Exception:
            x_test_transformed = None

        yield x_test, x_test_transformed, y_test, same_page


def get_x_score(x, norm):
    """
    Function to compute a score from distances between pairs.

    Parameters
    ----------
    x : numpy matrix
        Distance between pairs

    norm : str
        Norm used to compute X_score ('l1', 'l2', 'inf')

    Returns
    -------
    x_score : numpy array
    """
    ord = None
    if norm == "l1":
        ord = 1
    elif norm == "l2":
        ord = 2
    elif norm == "inf":
        ord = np.inf
    x_score = np.linalg.norm(x, axis=1, ord=ord)
    x_score = - x_score
    return x_score


def get_auc(x, y, norm):
    """
    Function to compute auc.

    Parameters
    ----------
    x : numpy matrix
        Distances between pairs

    y : numpy array
        Boolean to know if the pairs have been reused together

    norm : str
        Norm used to compute X_score ('l1', 'l2', 'inf')

    Returns
    -------
    auc : float
    """
    ord = None
    if norm == "l1":
        ord = 1
    elif norm == "l2":
        ord = 2
    elif norm == "inf":
        ord = np.inf
    x_score = np.linalg.norm(x, axis=1, ord=ord)
    x_score = - x_score
    auc = average_precision_score(y, x_score)

    return auc


def get_precision_recall_curve(x, y, norm):
    """
    Function to compute precision recall curve.

    Parameters
    ----------
    x : numpy matrix

    y : numpy array

    norm : str
        Norm used to compute the distance

    Returns
    -------
    precision : numpy array

    recall : numpy array

    threshold : numpy array
    """
    ord = None
    if norm == "l1":
        ord = 1
    elif norm == "l2":
        ord = 2
    elif norm == "inf":
        ord = np.inf
    x_score = np.linalg.norm(x, axis=1, ord=ord)
    x_score = - x_score
    precision, recall, threshold = precision_recall_curve(y, x_score)

    return precision, recall, threshold


def run_fold(tfidf, df_log, df_pairs_reused, test_title_page, model, n_topics,
             balance_reuse, max_reuse):
    """
    Function to compute results for one specific fold.

    Parameters
    ----------
    tfidf : sparse csr matrix
        Shape [n_sample, n_words]

    df_log : pandas Dataframe
        Metadata dataframe

    df_pairs_reused : pandas Dataframe
        Dataframe with all reused pairs

    test_title_page :

    model : str
        Model to use for topic extraction

    n_topics : int
        Number of topics to extract

    balance_reuse : float
        Ratio of reused pairs to keep in the X, y dataset

    max_reuse : int
        Maximum number of reused pairs to keep in the X, y dataset. It also
        limits the size of the dataset.

    Returns
    -------
    d_auc : dict
        Dictionary of floats (AUC value for different parametrization)

    d_auc_ml : dict
        Dictionary of floats (AUC value for different metric learned
        parametrization)
    d_prc : dict
        Dictionary of tuple with shape (numpy array, numpy array, numpy array)

    d_prc_ml : dict
        Dictionary of tuple with shape (numpy array, numpy array, numpy array)

    d_size : tuple
        Shape (number of test observations, number of reused pairs in the test
        set)
    """
    # extract train data
    df_log_train, tfidf_train = get_train_sample(tfidf, df_log, test_title_page)

    # compute a topic space and keep the fitted dictionary matrix H
    w_train, fitted_model = compute_topic_space_train(tfidf_train, n_topics,
                                                      model)

    # extract test data
    df_log_test, tfidf_test = get_test_sample(tfidf, df_log, test_title_page)

    # compute a topic space with the test data and the fitted dictionary
    w_test = fitted_model.transform(tfidf_test)

    d_auc = {}
    d_auc_ml = {}
    d_prc = {}
    d_prc_ml = {}
    d_size = {}
    for x_test, x_test_transformed, y_test, same_page in get_x_y_test(
            df_pairs_reused,
            df_log_train,
            w_train,
            df_log_test,
            w_test,
            balance_reuse,
            max_reuse):

        page = "page" if same_page else "nopage"

        # store some values
        d_size[page] = (len(y_test), np.count_nonzero(y_test))

        # shuffle labels
        y_test_random = np.random.permutation(y_test)

        # compute auc for different norms
        for norm in ["l1", "l2", "inf"]:
            name = "_".join([page, norm])

            # without learning
            auc = get_auc(x_test, y_test, norm)
            d_auc[name + "_False"] = auc
            precision, recall, threshold = get_precision_recall_curve(
                x_test, y_test, norm)
            d_prc[name + "_False"] = (precision, recall, threshold)

            # with learning
            if x_test_transformed is not None:
                auc_ml = get_auc(x_test_transformed, y_test, norm)
                precision, recall, threshold = get_precision_recall_curve(
                                x_test_transformed, y_test, norm)
            else:
                auc_ml = np.nan
                precision, recall, threshold = np.nan, np.nan, np.nan
            d_auc_ml[name + "_False"] = auc_ml
            d_prc_ml[name + "_False"] = (precision, recall, threshold)

            # random auc
            auc_random = get_auc(x_test, y_test_random, norm)
            d_auc[name + "_True"] = auc_random
            if x_test_transformed is not None:
                auc_random_ml = get_auc(x_test_transformed, y_test_random, norm)
            else:
                auc_random_ml = np.nan
            d_auc_ml[name + "_True"] = auc_random_ml

    return d_auc, d_auc_ml, d_prc, d_prc_ml, d_size


def initialize_log_auc(model_directory, reset):
    """
    Function to initialize the log file with AUC results.

    Parameters
    ----------
    model_directory : str
        Path of the directory with the model results

    reset : bool
        Boolean to remove the previous results

    Returns
    -------
    """
    # path
    path_log_auc = os.path.join(model_directory, "log_auc")

    # reset
    if os.path.isfile(path_log_auc) and reset:
        os.remove(path_log_auc)

    # initialize
    with open(path_log_auc, mode="wt", encoding="utf-8") as f:
        f.write("fold;model;n_topics;page;metric_learning;norm;random;auc;size;"
                "n_reuses")
        f.write("\n")

    return


def add_row(path_output, i_fold, model, n_topics, page, ml, norm, random, auc,
            size, reuse):
    """
    Function to add a row in the result log.

    Parameters
    ----------
    path_output : str
        Path of the log file

    i_fold : int
        Indice of the fold

    model : str
        Model used for topic extraction

    n_topics : int
        Number of topics extracted

    page : str
        'page' or 'nopage' to determine if we allow a pair from the same page
        dataset

    ml : str
        'ml' or 'noml' to determine if we apply metric learning

    norm : str
        Norm used to compute the distance

    random : str
        'True' or 'False' to determine if we shuffle results or not

    auc : float
        AUC value

    size : int
        Number of test observations

    reuse : int
        Number of reused pairs in the test set

    Returns
    -------
    """
    l = [str(i_fold), model, str(n_topics), page, ml, norm, random, str(auc),
         str(size), str(reuse)]
    with open(path_output, mode="at", encoding="utf-8") as f:
        f.write(";".join(l))
        f.write("\n")
    return


def run_folds(tfidf, df_log, df_pairs_reused, folds, model, n_topics,
              balance_reuse, max_reuse):
    """
    Function to yield results for all folds.

    Parameters
    ----------
    tfidf : sparse csr matrix
        Shape [n_sample, n_words]

    df_log : pandas Dataframe
        Metadata dataframe

    df_pairs_reused : pandas Dataframe
        Dataframe with all reused pairs

    folds : list of int
        List of ids

    model : str
        Name of the model to use for topic extraction

    n_topics : int
        Number of topics to extract

    balance_reuse : float
        Ratio of reused pairs to keep in the X, y dataset

    max_reuse : int
        Maximum number of reused pairs to keep in the X, y dataset. It also
        limits the size of the dataset.

    Returns
    -------
    i : int
        Indice of the fold

    d_auc : dict
        Dictionary of floats (AUC value for different parametrization)

    d_auc_ml : dict
        Dictionary of floats (AUC value for different metric learned
        parametrization)

    d_prc : dict
        Dictionary of tuples with shape (numpy array, numpy array, numpy array)

    d_prc_ml : dict
        Dictionary of tuples with shape (numpy array, numpy array, numpy array)

    d_size : tuple
        Shape (number of test observations, number of reused pairs in the test
        set)
    """
    # compute auc for each fold
    for i, test_title_page in enumerate(folds):
        d_auc, d_auc_ml, d_prc, d_prc_ml, d_size = run_fold(tfidf,
                                                            df_log,
                                                            df_pairs_reused,
                                                            test_title_page,
                                                            model,
                                                            n_topics,
                                                            balance_reuse,
                                                            max_reuse)

        yield i, d_auc, d_auc_ml, d_prc, d_prc_ml, d_size


def worker(model_directory, tfidf, df_log, df_pairs_reused, folds, model,
           n_topics, balance_reuse, max_reuse):
    """
    Function to multiprocess.

    Parameters
    ----------
    model_directory : str
        Path to the directory with model's results

    tfidf : sparse csr matrix
        Shape [n_sample, n_words]

    df_log : pandas Dataframe
        Metadata dataframe

    df_pairs_reused : pandas Dataframe
        Dataframe with all the possible reused pairs

    folds : list of int
        List of ids for the files contained in the random fold

    model : str
        Model used for topic extraction

    n_topics : int
        Number of topics to compute

    balance_reuse : float
        Ratio of reused pairs to ensure in the X, y dataset

    max_reuse : int
        Maximum number of reused pairs to keep in the dataset. This also limits
        the size of the dataset.

    Returns
    -------

    """
    # path
    path_log_auc = os.path.join(model_directory, "log_auc")

    # compute auc for each fold
    d_all = defaultdict(lambda: [])

    for i, d_auc, d_auc_ml, d_prc, d_prc_ml, d_size in run_folds(
            tfidf, df_log, df_pairs_reused, folds, model, n_topics,
            balance_reuse, max_reuse):

        (size_page, reuse_page) = d_size["page"]
        (size_nopage, reuse_nopage) = d_size["nopage"]

        # auc value
        for key in d_auc:
            auc = d_auc[key]
            l = key.split("_")
            page, norm, random = l[0], l[1], l[2]
            name = "auc_%s_%i_%s_%s_%s_%s.npy" % (model, n_topics, page, "noml",
                                                  norm, random)
            d_all[name].append(auc)

            if page == "page":
                size = size_page
                reuse = reuse_page
            else:
                size = size_nopage
                reuse = reuse_nopage
            add_row(path_log_auc, i, model, n_topics, page, "noml", norm,
                    random, auc, size, reuse)

        # precision recall curve
        for key in d_prc:
            l = key.split("_")
            page, norm, random = l[0], l[1], l[2]
            name_prc = "prc_%s_%i_%s_%s_%s_%s_%i.npz" % (model, n_topics, page,
                                                         "noml", norm, random,
                                                         i)
            (precision, recall, threshold) = d_prc[key]
            path_prc = os.path.join(model_directory, name_prc)
            np.savez(path_prc, precision=precision, recall=recall,
                     threshold=threshold)

        # auc value
        for key in d_auc_ml:
            auc = d_auc_ml[key]
            l = key.split("_")
            page, norm, random = l[0], l[1], l[2]
            name = "auc_%s_%i_%s_%s_%s_%s.npy" % (model, n_topics, page, "ml",
                                                  norm, random)
            d_all[name].append(auc)
            if page == "page":
                size = size_page
                reuse = reuse_page
            else:
                size = size_nopage
                reuse = reuse_nopage
            add_row(path_log_auc, i, model, n_topics, page, "ml", norm, random,
                    auc, size, reuse)

        # precision recall curve
        for key in d_prc_ml:
            l = key.split("_")
            page, norm, random = l[0], l[1], l[2]
            name_prc = "prc_%s_%i_%s_%s_%s_%s_%i.npz" % (model, n_topics, page,
                                                         "ml", norm, random, i)
            (precision, recall, threshold) = d_prc_ml[key]
            path_prc = os.path.join(model_directory, name_prc)
            np.savez(path_prc, precision=precision, recall=recall,
                     threshold=threshold)

    # save the results
    for name in d_all:
        array = np.asarray(d_all[name])
        path = os.path.join(model_directory, name)
        save_array(path, array)

    return


def main(result_directory, ratio_test, n_folds, n_jobs, balance_reuse,
         max_reuse, reset):
    """
    Function to run all the script and multiprocess the grid search by
    configuration.

    Parameters
    ----------
    result_directory : str
        Path of the result directory

    ratio_test : float
        Ratio test/train observations

    n_folds : int
        Number of folds to compute per setup

    n_jobs : Number of worker to use simultaneously

    balance_reuse : float
        Ratio of reused pairs to ensure

    max_reuse : int
        Maximum number of reused pairs to keep. Fixing a ratio of reused pairs,
        thus, the size of our X, y dataset is limited.

    reset : bool
        Boolean to remove the previous results

    Returns
    -------
    """
    # paths
    path_log = os.path.join(result_directory, "log_final_reduced")
    path_tfidf = os.path.join(result_directory, "tfidf.npz")
    model_directory = os.path.join(result_directory, "model_result")

    # initialization
    check_directory(model_directory, reset)
    initialize_log_auc(model_directory, reset)

    # get data
    df_log = pd.read_csv(path_log, header=0, encoding="utf-8", sep=";",
                         index_col=False)
    print("log length :", df_log.shape[0])
    tfidf = load_sparse_csr(path_tfidf)
    print("tfidf shape :", tfidf.shape)

    # compute fold ids
    folds = [get_test_page(df_log, ratio_test) for i in range(n_folds)]
    print("number of folds :", len(folds))

    # find all reused pairs
    df_pairs_reused = get_all_reused_pairs(df_log)
    print("df_pairs_reused shape :", df_pairs_reused.shape, "\n")

    topics = [i for i in range(5, 101, 5)]
    Parallel(n_jobs=n_jobs, verbose=30)(delayed(worker)
                                        (model_directory=model_directory,
                                         tfidf=tfidf,
                                         df_log=df_log,
                                         df_pairs_reused=df_pairs_reused,
                                         folds=folds,
                                         model=model,
                                         n_topics=n_topics,
                                         balance_reuse=balance_reuse,
                                         max_reuse=max_reuse)
                                        for model in ["nmf", "svd"]
                                        for n_topics in topics)

    return


if __name__ == "__main__":

    # paths
    result_directory = get_config_tag("result", "cleaning")

    # run
    main(result_directory=result_directory,
         ratio_test=0.5,
         n_folds=50,
         n_jobs=20,
         balance_reuse=0.3,
         max_reuse=30000,
         reset=True)
