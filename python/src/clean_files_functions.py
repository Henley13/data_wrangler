#!/bin/python3
# coding: utf-8

""" Different functions used to detect the file extension and reshape it. """

# libraries
import pandas as pd
import os
import magic
import chardet
import random
import time
from _csv import reader, Error
from csv import Sniffer
from io import StringIO
from collections import Counter
random.seed(123)
print("\n")


def cleaner(filename, input_directory, output_directory):
    """
    Function to clean the files.
    :param filename: string
    :param input_directory: string
    :param output_directory: string
    :return:
    """
    path = os.path.join(input_directory, filename)
    size = os.path.getsize(path)
    if size > 0 and not os.path.isfile(os.path.join(output_directory, filename)):
        extension = magic.Magic(mime=True).from_file(path)
        if extension == "text/plain":
            return plain(filename, input_directory, output_directory)
        elif extension == "application/octet-stream":
            return octet_stream(filename, input_directory, output_directory)
        elif extension == "application/x-dbf":
            pass
        elif extension == "application/pdf":
            pass
        elif extension == "application/msword":
            pass
        elif extension == "application/zip":
            pass
        elif extension == "image/jpeg":
            pass
        elif extension == "text/html":
            pass
        elif extension == "text/xml":
            pass
        elif extension == "application/CDFV2-unknown":
            pass
        elif extension == "application/vnd.ms-excel":
            pass
        elif extension == "application/vnd.oasis.opendocument.text":
            pass
        elif extension == "application/vnd.oasis.opendocument.spreadsheet":
            pass
        elif extension == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
            pass
        elif extension == "application/x-dosexec":
            pass
        elif extension == "image/png":
            pass
        elif extension == "image/tiff":
            pass
        elif extension == "application/x-iso9660-image":
            pass
        else:
            pass
    return


# TODO save results (metadata, df, x, y, d)
# TODO check useless columns
# TODO check useless rows
# TODO erreurs d'encoding
# TODO simplifier par des fonctions
def plain(filename, input_directory, output_directory, mode="rb", nbytes=20000, threshold_n_row=50, ratio_sample=20, max_sample=1000, threshold_n_col=0.8, check_header=30):
    """
    Function to clean a file with a text/plain extension.
    :param filename: string
    :param input_directory: string
    :param output_directory: string
    :param mode: "rb", "r" or "rt"
    :param nbytes: integer
    :param threshold_n_row: integer
    :param ratio_sample: integer
    :param max_sample: integer
    :param threshold_n_col: integer
    :param check_header: integer
    :return:
    """
    path = os.path.join(input_directory, filename)
    # number of rows
    nrow = file_len(path, mode=mode)
    # encoding
    encoding = detect_encoding(path, nbytes=nbytes)
    # sample
    sample = []
    if nrow >= threshold_n_row:
        size_sample = min(int(round(nrow * ratio_sample / 100, 0)), max_sample)
        sample = get_sample(path, encoding, nrow, size_sample, mode="rb")
    else:
        with open(path, mode="rb") as f:
            for row in f:
                sample.append(row.decode(encoding))
        threshold_n_col = 0.60
    full_sample = "".join(sample)
    # separator and quotechar
    try:
        sep = Sniffer().sniff(full_sample).delimiter
    except Error:
        return
    # number of columns
    frequency = Counter(list(map(lambda x: x.count(sep), sample)))
    n_col = None
    for i in frequency:
        if frequency[i] / len(sample) > threshold_n_col:
            n_col = i + 1
            break
    if n_col is None:
        return
    # header
    list_content = []
    with open(path, mode="rb") as f:
        for row in f:
            list_content.append(row.decode(encoding))
    check_header = min(check_header, int(len(list_content) / 2))
    row_header = 0
    n_blank_lines = 0
    for row in range(check_header):
        if is_empty(list_content[row]):
            n_blank_lines += 1
        content = "".join(list_content[row:])
        if has_header(content):
            row_header = row
            break
    # empty columns
    df = pd.read_csv(path, encoding=encoding, sep=sep, header=row_header - n_blank_lines, skip_blank_lines=True)
    l = []
    for col in df.columns:
        if df[col].count() > 0:
            l.append(col)
    df = df[l]
    # statistics and distribution
    x = df.shape[0]
    y = df.shape[1]
    if x * y == 0:
        return
    pd.to_numeric(df.columns, errors="ignore")
    d = {}
    for i in df.dtypes:
        if i not in d:
            d[i] = 1
        else:
            d[i] += 1
    # metadata
    metadata = ""
    metadata += "".join(list_content[:row_header])
    # save results
    path = os.path.join(output_directory, filename)
    print("saving...", path)
    time.sleep(10)
    df.to_csv(path, sep=";", index=False, encoding="utf-8")
    if metadata != "":
        with open(path + ".txt", "wt", encoding="utf-8") as f:
            f.write(metadata)
    return x, y, d


def octet_stream(filename, input_directory, output_directory):
    """
    Function to clean a file with a octet-stream extension.
    :param filename: string
    :param input_directory: string
    :param output_directory: string
    :return:
    """
    return plain(filename, input_directory, output_directory)


def detect_encoding(path, nbytes=20000):
    """
    Function to detect the file encoding.
    :param path: string
    :param nbytes: integer
    :return: string
    """
    f = open(path, 'rb')
    detection = chardet.detect(f.read(nbytes))
    return detection["encoding"]


def file_len(filename, mode="rb"):
    """
    Function to count the number of lines from a file
    :param filename: string
    :param mode: "rb", "r" or "rt"
    :return: integer
    """
    i = 0
    with open(filename, mode=mode) as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def get_sample(filename, encoding, len_file, len_sample, mode="rb"):
    """
    Functions to get a random sample of rows from a text file.
    :param filename: string
    :param encoding: string
    :param len_file: integer
    :param len_sample: integer
    :param mode: "rb", "r" or "rt"
    :return: list of string
    """
    sample = []
    x = random.sample(range(0, len_file), len_sample)
    with open(filename, mode=mode) as f:
        for i, l in enumerate(f):
            if i in x:
                sample.append(l.decode(encoding))
    return sample


# TODO rendre la fonction plus rapide pour les fichiers volumineux.
def has_header(sample):
    """
    Function to determine if the first row of the given sample is a potential header.
    :param sample: string
    :return: boolean
    """
    # Creates a dictionary of types of data in each column. If any
    # column is of a single type (say, integers), *except* for the first
    # row, then the first row is presumed to be labels. If the type
    # can't be determined, it is assumed to be a string in which case
    # the length of the string is the determining factor: if all of the
    # rows except for the first are the same length, it's a header.
    # Finally, a 'vote' is taken at the end for each column, adding or
    # subtracting from the likelihood of the first row being a header.
    rdr = reader(StringIO(sample), Sniffer().sniff(sample))
    header = next(rdr)  # assume first row is header
    columns = len(header)
    columnTypes = {}
    for i in range(columns):
        columnTypes[i] = None

    checked = 0
    row_analized = 0
    for row in rdr:
        # arbitrary number of rows to check, to keep it sane
        if checked > 20:
            break
        checked += 1

        if len(row) != columns:
            continue  # skip rows that have irregular number of columns
        else:
            row_analized += 1

        for col in list(columnTypes.keys()):
            for thisType in [int, float, complex]:
                try:
                    thisType(row[col])
                    break
                except (ValueError, OverflowError):
                    pass

            else:
                # fallback to length of string
                thisType = len(row[col])

            if thisType != columnTypes[col]:
                if columnTypes[col] is None:  # add new column type
                    columnTypes[col] = thisType
                else:
                    # type is inconsistent, remove column from
                    # consideration
                    del columnTypes[col]

    if row_analized == 0:
        return False
    if len(columnTypes) == 0:
        return False

    # finally, compare results against first row and "vote"
    # on whether it's a header
    hasHeader = 0
    for col, colType in columnTypes.items():
        if type(colType) == type(0):  # it's a length
            if len(header[col]) != colType:
                hasHeader += 1
            else:
                hasHeader -= 1
        else:  # attempt typecast
            try:
                colType(header[col])
            except (ValueError, TypeError):
                hasHeader += 1
            else:
                hasHeader -= 1

    return hasHeader > 0


def is_empty(row):
    """
    Function to check if a row is empty.
    :param row: string
    :return: boolean
    """
    return len(row.strip()) == 0

