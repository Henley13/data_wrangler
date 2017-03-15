#!/bin/python3
# coding: utf-8

""" We download data from their url."""

# libraries
import os
import sys
import shutil
import time
from contextlib import closing
from urllib.request import urlopen
from urllib import error
from lxml import etree
print("\n")

# variables
directory = "../data/data_collected"
filename = "../url_csv.xml"
fileformat = "csv"
list_errors = []
s = 0

# functions


def download(url, title):
    """
    Download data.
    :param url: str
    :param title: str
    :return:
    """
    print(url)
    global list_errors
    global s
    # time.sleep(20)
    try:
        local_file_name = os.path.join(directory, title)
        with open(local_file_name, 'wb') as local_istream:
            with closing(urlopen(url)) as remote_file:
                shutil.copyfileobj(remote_file, local_istream)
    except error.HTTPError as err:
        s -= 1
        print(err.code)
        list_errors.append(url)
    except UnicodeEncodeError:
        s -= 1
        print("UnicodeEncodeError")
        list_errors.append(url)
    except error.URLError:
        s -= 1
        print("URLError")
        list_errors.append(url)
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
    print("\n")
    print("taille fichiers collectés :", size)
    print("\n")
    return size

# check if the output directory exists
if not os.path.isdir(directory):
    os.mkdir(directory)
    print(directory, "created")
    print("\n")

# get the url list
tree = etree.parse("../url_csv.xml")
url_list = [url.text for url in tree.xpath("/results/table/url")]
print("nombre de liens :", len(url_list), "\n")
for table in tree.xpath("/results/table"):
    url, id = table[0].text, table[1].text
    title = str(id) + "." + fileformat
    download(url, title)
    s += 1
    if s % 100 == 0:
        x = directory_size(directory)
        if x > 400000000000:
            sys.exit("Fichiers trop gros!")

print("\n")
print("errors :", len(list_errors))
with open('../data/HTTPError&UnicodeEncodeError.txt', mode='wt', encoding='utf-8') as f:
    f.write('\n'.join(list_errors))
print("nombre de table téléchargées :", str(s))
directory_size(directory)
