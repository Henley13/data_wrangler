# -*- coding: utf-8 -*-

""" We download data from their url."""

# libraries
import os
import shutil
from tqdm import tqdm
from contextlib import closing
from joblib import Parallel, delayed
from lxml import etree
from urllib.request import urlopen
from toolbox.utils import (log_error, get_config_tag, get_config_trace,
                           reset_log_error, TryMultipleTimes)
print("\n")


def _check_output_directory(output_directory, reset):
    """
    Function to check if the output directory exists.

    Parameters
    ----------
    output_directory : str
        Path of the output directory

    reset : bool
        boolean to determine if the directory needs to be cleaned up before
        starting the download

    Returns
    -------

    """
    # check if the output directory exists
    if not os.path.isdir(output_directory):
        os.mkdir(output_directory)
    else:
        if reset:
            shutil.rmtree(output_directory)
        else:
            pass
    return


def get_format_urls(input_directory, format='all'):
    """
    Function to get the urls for the requested format.

    Parameters
    ----------
    input_directory : str
        Path of the urls directory
    format : str
        Requested format('all', 'csv', 'geojson', 'html', 'json', 'kml', 'pdf',
        'shp', 'text','xls''xml', 'zip')

    Returns
    -------
    l_urls : list of tuples(str, str, str)
        List of tuples (url, filename, format)
    """
    # get the path for the requested format
    filepath = os.path.join(input_directory, "url_" + format + ".xml")

    # get the list
    l_urls = []
    tree = etree.parse(filepath)
    for table in tree.xpath("/results/table"):
        url, filename, format = table[0].text, table[1].text, table[2].text
        if url is not None and filename is not None:
            l_urls.append((url, filename, format))
    print("number of urls :", len(l_urls), "\n")

    return l_urls


@TryMultipleTimes(n_tries=2)
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
    url = tuple_file[0]
    local_file_name = os.path.join(output_directory, tuple_file[1])
    try:
        download(url, local_file_name)
    except Exception:
        path_error = os.path.join(error_directory, tuple_file[1])
        log_error(path_error, [tuple_file[0], tuple_file[1], tuple_file[2]])
        if os.path.isfile(local_file_name):
            os.remove(local_file_name)
    return


def main(input_directory, output_directory, error_directory, n_jobs, reset,
         format, multi):
    """
    Function to run all the script and handle the multiprocessing.

    Parameters
    ----------
    input_directory : str
        Path of the BaseX results directory

    output_directory : str
        Path of the output directory

    error_directory : str
        Path of the error directory

    n_jobs : int
        Number of workers to use

    reset : bool
        Boolean to determine if the output and the error directories have
        to be cleaned

    format : str
        Format requested to download

    multi : bool
        Boolean to determine if the multiprocessing has to be used

    Returns
    -------
    """
    # check the output directory
    _check_output_directory(output_directory, reset)

    # check the error directory
    reset_log_error(error_directory, reset)

    # get urls list
    data_files = get_format_urls(input_directory, format)

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
    input_directory = get_config_tag("output", "basex")
    output_directory = get_config_tag("output", "download")
    error_directory = get_config_tag("error", "download")

    # parameters
    n_jobs = get_config_tag("n_jobs", "download")
    reset = get_config_tag("reset", "download")
    format = get_config_tag("format", "download")
    multi = get_config_tag("multi", "download")

    # run
    main(input_directory, output_directory, error_directory, n_jobs, reset,
         format, multi)
