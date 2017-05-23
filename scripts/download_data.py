# -*- coding: utf-8 -*-

""" We download data from their url."""

# libraries
import os
import shutil
from contextlib import closing
from joblib import Parallel, delayed
from lxml import etree
from urllib.request import urlopen
from src.toolbox.utils import log_error, get_config_tag, reset_log_error
print("\n")


def download(url, title):
    """
    Download data.
    :param url: str
    :param title: str
    :return:
    """
    local_file_name = os.path.join(directory, title)
    try:
        if not os.path.isfile(local_file_name):
            with open(local_file_name, 'wb') as local_istream:
                with closing(urlopen(url)) as remote_file:
                    shutil.copyfileobj(remote_file, local_istream)
        elif os.path.getsize(local_file_name) == 0:
            with open(local_file_name, 'wb') as local_istream:
                with closing(urlopen(url)) as remote_file:
                    shutil.copyfileobj(remote_file, local_istream)
        else:
            pass
    except:
        path = os.path.join(path_error, title)
        log_error(path, [url, title])
        if os.path.isfile(local_file_name):
            os.remove(local_file_name)
    return


def directory_size(path):
    """
    Compute the size of a directory.
    :param path: str
    :return: int
    """
    size = 0
    for (path, dirs, files) in os.walk(path):
        for file in files:
            filename = os.path.join(path, file)
            size += os.path.getsize(filename)
    return size


def worker_activity(url_title):
    """
    Function to encapsulate the process and use multiprocessing.
    :param url_title: [url, title]
    :return:
    """
    download(url_title[0], url_title[1])

# path
filename = get_config_tag("input", "download")
directory = get_config_tag("output", "download")
path_error = get_config_tag("error", "download")
n_jobs = get_config_tag("n_jobs", "download")

# reset the log
reset_log_error(path_error)

# check if the output directory exists
if not os.path.isdir(directory):
    os.mkdir(directory)

# get the url list
l_tables = []
tree = etree.parse(filename)
url_list = [url.text for url in tree.xpath("/results/table/url")]
print("number of urls :", len(url_list), "\n")
for table in tree.xpath("/results/table"):
    url, id = table[0].text, table[1].text
    title = str(id)
    if url is not None and title is not None:
        l_tables.append([url, title])

# multiprocessing
Parallel(n_jobs=n_jobs, verbose=20)(delayed(worker_activity)(url_title=l)
                                    for l in l_tables)

print()
print("total number of files :", len(os.listdir(directory)))
directory_size(directory)
