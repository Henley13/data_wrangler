#!/bin/python3
# coding: utf-8

""" Download data from their url."""

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
list_errors = []
s = 0

# function


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
    time.sleep(20)
    try:
        local_file_name = os.path.join(directory, title)
        with open(local_file_name, 'wb') as local_istream:
            with closing(urlopen(url)) as remote_file:
                shutil.copyfileobj(remote_file, local_istream)
    except error.HTTPError as err:
        s -= 1
        print(err.code)
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


# TODO vérifier la concordance entre les fichiers et les id.

# get url list - csv
tree = etree.parse(os.path.join("..", "url_csv.xml"))
url_list = [url.text for url in tree.xpath("/urls/result/url")]
print("nombre de liens :", len(url_list))
for table in tree.xpath("/urls/result"):
    url, id = table[0].text, table[1].text
    title = str(id) + ".csv"
    download(url, title)
    s += 1
    if s % 100 == 0:
        x = directory_size(directory)
        if x > 400000000000:
            sys.exit("Fichiers trop gros!")

# get url list - json
tree = etree.parse(os.path.join("..", "url_json.xml"))
url_list = [url.text for url in tree.xpath("/urls/result/url")]
print("nombre de liens :", len(url_list))
for table in tree.xpath("/urls/result"):
    url, id = table[0].text, table[1].text
    title = str(id) + ".json"
    download(url, title)
    s += 1
    if s % 100 == 0:
        x = directory_size(directory)
        if x > 400000000000:
            sys.exit("Fichiers trop gros!")

print("\n")
print("errors :", len(list_errors))
with open('../data/HTTPErrors.txt', mode='wt', encoding='utf-8') as f:
    f.write('\n'.join(list_errors))
print("nombre de table téléchargées :", str(s))
directory_size(directory)

# test csv
#url_csv = "https://www.data.gouv.fr/s/resources/compte-administratif-2013-annexes-elements-du-bilan-detail-fonctionnement/20170308-170753/Onglet_25_CA2013_Annexes.csv"
#download(url_csv, "test.csv")

# test json
#url_json = "https://inspire.data.gouv.fr/api/geogw/services/556c5d51330f1fcd48335c41/feature-types/Geoportail_WMS_Preview:OUVERTURE_VISUELLE/download?format=GeoJSON&projection=WGS84"
#download(url_json, "test.json")