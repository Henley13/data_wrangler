# -*- coding: utf-8 -*-

""" Some useful functions and decorators. """

# libraries
import sys
import os
import time
import shutil
import tempfile
import numpy as np
import scipy.sparse as sp
from joblib.format_stack import format_exc
from configobj import ConfigObj, ConfigObjError
from validate import Validator
from contextlib import contextmanager


@contextmanager
def tmp_dir(host_dir):
    name = tempfile.mkdtemp(dir=host_dir)
    try:
        yield name
    finally:
        shutil.rmtree(name, ignore_errors=True)


def check_directory(path_directory, reset):
    """
    Function to check the directory.

    Parameters
    ----------
    path_directory : str
        Path of the output directory

    reset : bool
        Boolean to decide if the directory should be reset or not

    Returns
    -------
    """
    if not os.path.isdir(path_directory):
        os.mkdir(path_directory)
    elif reset:
        shutil.rmtree(path_directory)
        os.mkdir(path_directory)
    else:
        pass
    return


def get_path_cachedir(directory):
    """
    Function to check the cache directory.

    Parameters
    ----------
    directory : str
        Path of the cache directory

    Returns
    -------
    """
    path_cache = os.path.join(directory, "cache")
    if os.path.isdir(path_cache):
        pass
    else:
        os.mkdir(path_cache)
    return path_cache


def check_graph_folders(result_directory, reset=False):
    """
    Function to check if the folders exist
    :param result_directory: string
    :param reset: boolean
    :return:
    """
    # paths
    path_graph = os.path.join(result_directory, "graphs")
    path_png = os.path.join(result_directory, "graphs", "png")
    path_pdf = os.path.join(result_directory, "graphs", "pdf")
    path_jpeg = os.path.join(result_directory, "graphs", "jpeg")
    path_svg = os.path.join(result_directory, "graphs", "svg")

    if not os.path.isdir(path_graph):
        os.mkdir(path_graph)

    for path in [path_pdf, path_svg, path_png, path_jpeg]:
        if not os.path.isdir(path):
            os.mkdir(path)
        elif reset:
            shutil.rmtree(path)
            os.mkdir(path)
        else:
            pass

    return


def print_top_words(model, feature_names, n_top_words):
    """
        Function to print the most important words per topic
        :param model: NMF fitted model (sklearn)
        :param feature_names: list of words with the right index
        :param n_top_words: integer
        :return:
        """
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()
    return


def save_dictionary(dictionary, path, header):
    """
    Function to save a dictionary in a csv format with two columns
    (key and value)
    :param dictionary: dictionary
    :param path: string
    :param header: list of strings
    :return:
    """
    with open(path, mode="wt", encoding="utf-8") as f:
        f.write(";".join(header))
        f.write("\n")
        for key in dictionary:
            s = ";".join([str(key), str(dictionary[key])])
            f.write(s)
            f.write("\n")
    return


def dict_to_list(dictionary, reversed=False):
    """
    Function to convert a dictionary to a list of keys, ordering by value
    :param dictionary: dictionary
    :param reversed: boolean
    :return: list of keys
    """
    return sorted(dictionary, key=dictionary.get, reverse=reversed)


def save_array(path_output, array):
    if isinstance(array, list):
        array = np.asarray(array)
    np.save(path_output, array)
    return


def load_sparse_csr(path):
    """
    Function to load a saved sparse matrix (scipy object)
    :param path: string
    :return: sparse row matrix
    """
    loader = np.load(path)
    return sp.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                         shape=loader['shape'])


def save_sparse_csr(path, array):
    """
    Function to save a sparse row matrix "
    :param path: string
    :param array: sparse matrix (scipy object)
    :return:
    """
    np.savez(path, data=array.data, indices=array.indices, indptr=array.indptr,
             shape=array.shape)
    return


def load_numpy_matrix(path):
    """
    Function to load a numpy matrix previously saved in a .npy format
    :param path: string
    :return: numpy matrix
    """
    return np.load(path)


def reset_log_error(path_error, reset):
    """
    Function to reset the error directory
    :param path_error: string
    :param reset: boolean
    :return:
    """
    # initialize files and directories
    if not os.path.isdir(os.path.dirname(path_error)):
        os.mkdir(os.path.dirname(path_error))
    if not os.path.isdir(path_error):
        os.mkdir(path_error)
    elif reset:
        shutil.rmtree(path_error)
        os.mkdir(path_error)
    else:
        pass
    return


def _get_config_spec():
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'config_spec.txt')


def get_config_trace():
    """
    Function to keep a trace of the latest configuration ran
    :return:
    """
    path_input = "config.txt"
    path_output = get_config_tag("path", "general")
    with open(path_input, mode='rt', encoding='utf-8') as f:
        text = f.read()
    with open(path_output, mode='wt', encoding='utf-8') as f:
        f.write(text)
    return


def get_config_tag(tag, section):
    """
    Function to get parameters values and path from a configuration file
    :param tag: string
    :param section: string
    :return:
    """
    config = ConfigObj("config.txt",
                       encoding="utf-8",
                       configspec=_get_config_spec())
    test = config.validate(Validator())
    if test is not True:
        raise ConfigObjError("Config file validation failed.")
    return config[section][tag]


def log_error(path_error, source):
    """
    Function to write errors in a text file
    :param path_error: string
    :param source: list of strings
    :return:
    """
    exc_type, exc_value, exc_traceback = sys.exc_info()
    text = format_exc(exc_type, exc_value, exc_traceback, context=5,
                      tb_offset=0)
    with open(path_error, mode='wt', encoding='utf-8') as f:
        f.write("##########################################################"
                "########### \n")
        for i in source:
            f.write(i)
            f.write("\n")
        f.write(text)
        f.write("\n")
        f.write("\n")
    return exc_type


def log_error_without_except(path_error, source, exception):
    with open(path_error, mode='wt', encoding='utf-8') as f:
        f.write("##########################################################"
                "########### \n")
        for i in source:
            f.write(i)
            f.write("\n")
        f.write(exception)
        f.write("\n")
        f.write("\n")
    return


def split_str(s):
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


#############################################################################


def _do_nothing():
    pass


def _raise_again(exception):
    raise exception


def _pause(duration=5):
    time.sleep(duration)


class TryMultipleTimes(object):
    """
    Class used as decorator
    """

    def __init__(self, action=_pause, on_fail=_raise_again, n_tries=5):
        self.action_ = action
        self.n_tries_ = n_tries
        self.on_fail_ = on_fail

    def __call__(self, fun):
        def decorate(*args, **kwargs):
            n_tries = kwargs.get('n_tries', self.n_tries_)
            action = kwargs.get('action', self.action_)
            on_fail = kwargs.get('on_fail', self.on_fail_)
            tries = 0
            while True:
                try:
                    return fun(*args, **kwargs)
                except Exception as e:
                    error = e
                    if tries == n_tries:
                        break
                tries += 1
                action()
            on_fail(error)
        return decorate
