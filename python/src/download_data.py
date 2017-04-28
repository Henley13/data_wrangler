#!/bin/python3
# coding: utf-8

""" We download data from their url."""

# libraries
import os
import sys
import shutil
from contextlib import closing
from urllib.request import urlopen
from lxml import etree
import functions
from joblib import Parallel, delayed
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
        functions.log_error(path, [url, title])
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

# variables
directory = "../data/data_collected_xls"
filename = "../url_xls.xml"
path_error = "../data/download_errors"

# reset the log
if os.path.isdir(path_error):
    for file in os.listdir(path_error):
        os.remove(os.path.join(path_error, file))
else:
    os.mkdir(path_error)

# check if the output directory exists
if not os.path.isdir(directory):
    os.mkdir(directory)
    print(directory, "created")
    print("\n")

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
Parallel(n_jobs=4, verbose=20)(delayed(worker_activity)(url_title=l) for l in l_tables)

print("\n")
print("total number of files :", len(os.listdir(directory)))
directory_size(directory)
