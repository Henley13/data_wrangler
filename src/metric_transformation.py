# -*- coding: utf-8 -*-

""" Metric learning algorithms. """

# libraries
import os
import pandas as pd
import numpy as np
import joblib
import random
from metric_learn import lsml
from collections import defaultdict
from tqdm import tqdm
from toolbox.utils import get_config_tag, get_path_cachedir
from scipy.spatial.distance import cosine
print("\n")

# memory cache
general_directory = get_config_tag("data", "general")
memory = joblib.Memory(cachedir=get_path_cachedir(general_directory), verbose=0)


def _split_str(s):
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
        print(type(s), s)
        raise ValueError("wrong type")


def _get_X_y(df_log, w):
    """
    Function to randomly compute distances between files
    :param df_log: pandas Dataframe
    :param w: matrix [n_samples, n_topics]
    :return: list of floats, list of integers (boolean)
    """
    # collect several random couples of files
    X = []
    y = []
    for i in tqdm(range(df_log.shape[0])):
        i_reuse = _split_str(df_log.at[i, "reuse"])
        i_topic = w[i, :]
        partners = random.sample(
            [j for j in range(df_log.shape[0]) if j != i],
            k=500)
        for j in partners:
            j_reuse = _split_str(df_log.at[j, "reuse"])
            j_topic = w[j, :]
            distance_ij = cosine(i_topic, j_topic)

            if not np.isnan(distance_ij) and np.isfinite(distance_ij):
                X.append(distance_ij)
                if len(set(i_reuse).intersection(j_reuse)) > 0:
                    y.append(1)

    return X, y

@memory.cache()
def _get_auc(X_train, y_train, X_test, y_test, binary_clf):
    return


def main(result_directory):
    # paths
    path_log = os.path.join(result_directory, "log_final_reduced_with_reuse")
    path_w = os.path.join(result_directory, "w.npy")

    # get data
    df_log = pd.read_csv(path_log, header=0, encoding="utf-8", sep=";",
                         index_col=False)
    print("log length :", df_log.shape[0], "\n")
    w = np.load(path_w)

    # shape data
    X, y = _get_X_y(df_log, w)
    model = lsml.LSML_Supervised(verbose=True)
    model.fit(X, y)

    return


if __name__ == "__main__":

    # paths
    result_directory = get_config_tag("result", "cleaning")

    # run
    main(result_directory)
