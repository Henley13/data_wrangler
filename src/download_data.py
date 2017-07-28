# -*- coding: utf-8 -*-

""" We download data from their url."""

# libraries
import os
import shutil
import pandas as pd
import joblib
from tqdm import tqdm
from contextlib import closing
from joblib import Parallel, delayed
from urllib.request import urlopen
from toolbox.utils import (log_error, get_config_tag, get_config_trace,
                           reset_log_error, TryMultipleTimes, check_directory,
                           get_path_cachedir)
print("\n")

# memory cache
general_directory = get_config_tag("data", "general")
memory = joblib.Memory(cachedir=get_path_cachedir(general_directory), verbose=0)


@memory.cache()
def _get_format_urls(metadata_dataset):
    """
    Function to get the urls to download the files.

    Parameters
    ----------
    metadata_dataset : pandas Dataframe
        Dataframe with the metadata of the datasets

    Returns
    -------
    l_urls : list of tuples(str, str, str)
        List of tuples (url, filename, format)
    """
    # get the lists
    df = metadata_dataset.query("url_destination_file == 'file'")
    l_url = list(df["url_file"].astype(str))
    l_filename = list(df["id_file"].astype(str))
    l_format = list(df["format_file"].astype(str))
    print("number of files to download :", df.shape[0], "\n")

    # create a list of tuples
    res = zip(l_url, l_filename, l_format)

    return res


@TryMultipleTimes()
def download(url, local_file_name):
    """
    Function to download datasets from their url.

    Parameters
    ----------
    url : str
        Url to download the file

    local_file_name : str
        Path to save the file

    Returns
    -------
    """
    if not os.path.isfile(local_file_name):
        with open(local_file_name, 'wb') as local_istream:
            with closing(urlopen(url)) as remote_file:
                shutil.copyfileobj(remote_file, local_istream)
    elif os.path.getsize(local_file_name) == 0:
        with open(local_file_name, 'wb') as local_istream:
            with closing(urlopen(url)) as remote_file:
                shutil.copyfileobj(remote_file, local_istream)
    return


def worker_activity(tuple_file, output_directory, error_directory):
    """
    Function to encapsulate the downloading and allow a multiprocessing process.

    Parameters
    ----------
    tuple_file : tuple(str, str, str)
        Tuple (url, filename, format)

    output_directory : str
        Path of the output directory

    error_directory : str
        Path of the error directory

    Returns
    -------
    """
    (url, filename, format) = tuple_file
    path_file = os.path.join(output_directory, filename)
    path_error = os.path.join(error_directory, filename)
    try:
        download(url, path_file)
        if os.path.isfile(path_error):
            os.remove(path_error)
    except Exception:
        log_error(path_error, [url, filename, format])
        if os.path.isfile(path_file):
            os.remove(path_file)
    return


def main(general_directory, output_directory, error_directory, n_jobs, reset,
         multi):
    """
    Function to run all the script and handle the multiprocessing.

    Parameters
    ----------
    general_directory : str
        Path of the general data directory

    output_directory : str
        Path of the output directory

    error_directory : str
        Path of the error directory

    n_jobs : int
        Number of workers to use

    reset : bool
        Boolean to determine if the output and the error directories have
        to be cleaned

    multi : bool
        Boolean to determine if the multiprocessing has to be used

    Returns
    -------
    """
    # check the output directory
    check_directory(output_directory, reset)

    # check the error directory
    reset_log_error(error_directory, reset)

    # get urls list
    filepath = os.path.join(general_directory, "metadata_dataset.csv")
    df_dataset = pd.read_csv(filepath, sep=';', encoding='utf-8',
                             index_col=False)
    data_files = _get_format_urls(df_dataset)

    if multi:
        # multiprocessing
        Parallel(n_jobs=n_jobs, verbose=20)(delayed(worker_activity)
                                            (tuple_file=tuple_file,
                                             output_directory=output_directory,
                                             error_directory=error_directory)
                                            for tuple_file in data_files)
    else:
        for tuple_file in tqdm(data_files):
            worker_activity(tuple_file=tuple_file,
                            output_directory=output_directory,
                            error_directory=error_directory)

    return


if __name__ == "__main__":

    get_config_trace()

    # paths
    general_directory = get_config_tag("data", "general")
    output_directory = get_config_tag("output", "download")
    error_directory = get_config_tag("error", "download")

    # parameters
    n_jobs = get_config_tag("n_jobs", "download")
    reset = get_config_tag("reset", "download")
    multi = get_config_tag("multi", "download")

    # run
    main(general_directory=general_directory,
         output_directory=output_directory,
         error_directory=error_directory,
         n_jobs=n_jobs,
         reset=reset,
         multi=multi)
