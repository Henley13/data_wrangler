# -*- coding: utf-8 -*-

""" Different functions used to detect the file extension and clean it. """

# libraries
import os
import magic
import chardet
import random
import json
import zipfile
import xmljson
import re
import numpy as np
import pandas as pd
import geopandas as gpd
from xlrd import open_workbook
from xlutils.copy import copy
from pandas.io.json import json_normalize
from _csv import Error
from csv import Sniffer
from lxml import etree
from collections import Counter
from .utils import log_error, log_error_without_except, tmp_dir
print("\n")


def cleaner(filename, input_directory, output_directory, log_directory,
            extradata_directory, dict_param, error_directory):
    """
    Function to clean a file (unzipped or not).

    Parameters
    ----------
    filename : str
        Name of the file (most of the time its Id)

    input_directory : str
        Path of the collected data directory

    output_directory : str
        Path of the cleaned files directory

    log_directory : str
        Path of the log directory

    extradata_directory : str
        Path of the directory to save the data that is not included in the
        cleaned matrix

    dict_param : dict
        Dictionary with several parameters stored

    error_directory : str
        Path of the error log directory

    Returns
    -------
    """

    # parameters, path and file size
    path = os.path.join(input_directory, filename)
    dict_result = dict()
    dict_result["source_file"] = filename

    # test file consistency
    try:
        size_file = os.path.getsize(path)
    except Exception:
        extension = "unknown size"
        log_error(os.path.join(error_directory, filename),
                  [filename, filename, filename, "non-zipped file", extension])
        return
    if size_file <= 0:
        extension = "zerobyte"
        log_error(os.path.join(error_directory, filename),
                  [filename, filename, filename, "non-zipped file", extension])
        return

    # detect the MIME type
    try:
        extension = magic.Magic(mime=True).from_file(path)
    except Exception:
        extension = "unknown extension"
        log_error(os.path.join(error_directory, filename),
                  [filename, filename, filename, "non-zipped file", extension])
        return

    # check if the it's a zipfile
    if extension != "application/zip":

        # test if the file has already been cleaned...
        if os.path.isfile(os.path.join(output_directory, filename)):
            return

        # ... and clean it
        dict_result["size_file"] = size_file
        dict_result["extension"] = extension
        try:
            cleaner_file(filename=filename,
                         input_directory=input_directory,
                         output_directory=output_directory,
                         log_directory=log_directory,
                         extradata_directory=extradata_directory,
                         error_directory=error_directory,
                         zip_file=False,
                         dict_param=dict_param,
                         dict_result=dict_result)
        except Exception:
            if extension in ["text/plain", "application/octet-stream"]:
                x = is_json_geojson(path)
                if x != "text":
                    extension = x
            log_error(os.path.join(error_directory, filename),
                      [filename, filename, filename, "non-zipped file",
                       extension])

    # extract files from a zipfile...
    else:
        try:
            z = zipfile.ZipFile(path)
        except Exception:
            log_error(os.path.join(error_directory, filename),
                      [filename, filename, filename, "zip file", extension])
            return

        with tmp_dir(host_dir=output_directory) as tempdir:
            try:
                z.extractall(tempdir)
            except Exception:
                log_error(os.path.join(error_directory, filename),
                          [filename, filename, filename, "zip file", extension])
                return

            for file in z.namelist():
                temp_path = os.path.join(tempdir, file)
                flatten_file = file.replace("/", "--")

                # ... test their consistency...
                if not os.path.isfile(temp_path):
                    continue

                try:
                    size_file_temp = os.path.getsize(temp_path)
                except Exception:
                    extension = "unknown size"
                    log_error(os.path.join(error_directory, flatten_file),
                              [filename, flatten_file, flatten_file,
                               "zipped file", extension])
                    continue

                if size_file_temp <= 0:
                    extension = "zerobyte"
                    log_error(os.path.join(error_directory, flatten_file),
                              [filename, flatten_file, flatten_file,
                               "zipped file", extension])
                    continue

                if os.path.isfile(
                        os.path.join(output_directory, flatten_file)):
                    continue

                try:
                    extension_temp = magic.Magic(mime=True).from_file(temp_path)
                except Exception:
                    extension = "unknown extension"
                    log_error(os.path.join(error_directory, flatten_file),
                              [filename, flatten_file, flatten_file,
                               "zipped file", extension])
                    continue

                dict_result["size_file"] = size_file_temp
                dict_result["extension"] = extension_temp

                # ... and clean it
                try:
                    cleaner_file(filename=file,
                                 input_directory=tempdir,
                                 output_directory=output_directory,
                                 log_directory=log_directory,
                                 extradata_directory=extradata_directory,
                                 error_directory=error_directory,
                                 zip_file=True,
                                 dict_param=dict_param,
                                 dict_result=dict_result)
                except Exception:
                    if extension_temp in ["text/plain",
                                          "application/octet-stream"]:
                        x = is_json_geojson(temp_path)
                        if x != "text":
                            extension_temp = x
                    log_error(os.path.join(error_directory, flatten_file),
                              [filename, flatten_file, flatten_file,
                               "zipped file", extension_temp])

    return


def cleaner_file(filename, input_directory, output_directory, log_directory,
                 extradata_directory, error_directory, zip_file, dict_param,
                 dict_result):
    """
    Function to clean unzipped files.

    Parameters
    ----------
    filename : str
        Name of the file (most of the time its Id)

    input_directory : str
        Path of the collected data directory or a temporary directory

    output_directory : str
        Path of the cleaned files directory

    log_directory : str
        Path of the log directory

    extradata_directory : str
        Path of the directory to save the data that is not included in the
        cleaned matrix

    error_directory : str
        Path of the error log directory

    zip_file : bool
        Boolean to specify if the file comes from a zipfile or not

    dict_param : dict
        Dictionary with several parameters stored

    dict_result : dict
        Dictionary with several results stored

    Returns
    -------
    """
    extension = dict_result["extension"]

    # store first results
    dict_result["matrix_name"] = filename.replace("/", "--")
    dict_result["file_name"] = filename.replace("/", "--")
    dict_result["zip_file"] = zip_file

    # clean the file
    if extension == "text/plain":
        return plain(filename, input_directory, output_directory, log_directory,
                     extradata_directory, dict_param, dict_result)
    elif extension == "application/octet-stream":
        return octet_stream(filename, input_directory, output_directory,
                            log_directory, extradata_directory, dict_param,
                            dict_result)
    elif extension == "application/x-dbf":
        pass
    elif extension == "application/pdf":
        pass
    elif extension == "application/msword":
        pass
    elif extension == "image/jpeg":
        pass
    elif extension == "text/html":
        pass
    elif extension == "text/xml":
        return xml(filename, input_directory, output_directory, log_directory,
                   extradata_directory, dict_result)
    elif extension == "application/vnd.ms-excel":
        return excel(filename, input_directory, output_directory, log_directory,
                     extradata_directory, error_directory, dict_param,
                     dict_result)
    elif extension == "application/CDFV2-unknown":
        return cdfv2(filename, input_directory, output_directory, log_directory,
                     extradata_directory, error_directory, dict_param,
                     dict_result)
    elif extension == "application/vnd.oasis.opendocument.text":
        pass
    elif extension == "application/vnd.oasis.opendocument.spreadsheet":
        pass
    elif extension == "application/vnd.openxmlformats-officedocument." \
                      "spreadsheetml.sheet":
        return office_document(filename, input_directory, output_directory,
                               log_directory, extradata_directory,
                               error_directory, dict_param, dict_result)
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


##############################################################################


def plain(filename, input_directory, output_directory, log_directory,
          extradata_directory, dict_param, dict_result):
    """
    Function to clean a file with a text/plain extension.

    Parameters
    ----------
    filename : str
        Name of the file

    input_directory : str
        Path of the collected data directory or a temporary directory

    output_directory : str
        Path of the output directory

    log_directory : str
        Path of the log directory

    extradata_directory : str
        Path of the extradata directory

    dict_param : dict
        Dictionary with several parameters stored

    dict_result : dict
        Dictionary with several results stored

    Returns
    -------
    """
    # path of the file
    path = os.path.join(input_directory, filename)

    # test if it's ...
    x = is_json_geojson(path)

    # ... a json...
    if x == "json":
        dict_result["extension"] = "json"
        df, extradata, no_header = clean_json(path)

    # ... a geojson...
    elif x == "geojson":
        dict_result["extension"] = "geojson"
        df, extradata, no_header = clean_geojson(path)

    # ... or a text file
    else:
        # number of rows
        nrow = file_len(path)

        # encoding
        encoding = detect_encoding(path)
        if encoding is None:
            encoding = "utf-8"

        # sample
        sample, full_sample, threshold_n_col = get_sample(
            path, encoding, nrow, dict_param["threshold_n_row"],
            dict_param["ratio_sample"], dict_param["max_sample"],
            dict_param["threshold_n_col"])

        # get a matrix
        # TODO rentrer les lignes précédentes dans clean_csv
        df, extradata, no_header = clean_csv(path, encoding, sample,
                                             full_sample, threshold_n_col,
                                             dict_param["check_header"])

    if df is None:
        raise Exception("df extracted if None")

    # empty columns
    df = clean_empty_col(df)

    # statistics and distribution
    x, y, d = distribution_df(df)
    if x is None:
        raise Exception("No row extracted")

    # save results
    dict_result["integer"] = d["integer"]
    dict_result["float"] = d["float"]
    dict_result["object"] = d["object"]
    dict_result["x"] = x
    dict_result["y"] = y
    dict_result["extradata"] = extradata
    dict_result["no_header"] = no_header
    save_results(df, output_directory, extradata_directory, log_directory,
                 dict_result)

    return


def octet_stream(filename, input_directory, output_directory, log_directory,
                 extradata_directory, dict_param, dict_result):
    return plain(filename, input_directory, output_directory, log_directory,
                 extradata_directory, dict_param, dict_result)


def excel(filename, input_directory, output_directory, log_directory,
          extradata_directory, error_directory, dict_param, dict_result):
    """
    Function to clean an excel file in order to extract a matrix.

    Parameters
    ----------
    filename : str
        Name of the file

    input_directory : str
        Path of the collected data directory or a temporary directory

    output_directory : str
        Path of the output directory

    log_directory : str
        Path of the log directory

    extradata_directory : str
        Path of the extradata directory

    error_directory : str
        Path of the error log directory

    dict_param : dict
        Dictionary with several parameters stored

    dict_result : dict
        Dictionary with several results stored

    Returns
    -------
    """
    # path
    path = os.path.join(input_directory, filename)

    # read data
    rb, rb_info = read_excel(path, info=True)

    # edit excel
    rb_edited = edit_excel(rb, rb_info, output_directory)
    sheets = rb_edited.sheet_names()

    # clean sheets
    with tmp_dir(host_dir=output_directory) as temptempdir:
        temp_path = os.path.join(temptempdir, filename)
        wb = copy(rb_edited)
        wb.save(temp_path)
        for sheet in sheets:
            out_filename = "__".join([filename,
                                      sheet.replace("/", "--")])
            if os.path.isfile(os.path.join(output_directory, out_filename)):
                continue

            # extract matrix
            df, metadata, no_header = clean_sheet_excel(
                rb_edited, sheet, temp_path, dict_param["check_header"])
            if df is None:
                path_error = os.path.join(error_directory, out_filename)
                log_error_without_except(
                    path_error,
                    [dict_result["source_file"], filename, out_filename,
                     "excel sheet", dict_result["extension"]],
                    "df extracted if None")

            # empty columns
            df = clean_empty_col(df)

            # statistics and distribution
            x, y, d = distribution_df(df)
            if x is None:
                path_error = os.path.join(error_directory, out_filename)
                log_error_without_except(
                    path_error,
                    [dict_result["source_file"], filename, out_filename,
                     "excel sheet", dict_result["extension"]],
                    "No row extracted")

            # save results
            dict_result["matrix_name"] = out_filename
            dict_result["integer"] = d["integer"]
            dict_result["float"] = d["float"]
            dict_result["object"] = d["object"]
            dict_result["x"] = x
            dict_result["y"] = y
            dict_result["extradata"] = metadata
            dict_result["no_header"] = no_header
            save_results(df, output_directory, extradata_directory,
                         log_directory, dict_result)

    return


def cdfv2(filename, input_directory, output_directory, log_directory,
          extradata_directory, error_directory, dict_param, dict_result):
    return excel(filename, input_directory, output_directory, log_directory,
                 extradata_directory, error_directory, dict_param, dict_result)


def office_document(filename, input_directory, output_directory, log_directory,
                    extradata_directory, error_directory, dict_param,
                    dict_result):
    return excel(filename, input_directory, output_directory, log_directory,
                 extradata_directory, error_directory, dict_param, dict_result)


def xml(filename, input_directory, output_directory, log_directory,
        extradata_directory, dict_result):
    """
    Function to clean a file with a text/plain extension.

    Parameters
    ----------
    filename
    input_directory
    output_directory
    log_directory
    extradata_directory
    dict_result

    Returns
    -------
    """
    path = os.path.join(input_directory, filename)

    # get a matrix
    df, metadata, no_header = clean_xml(path, deepness=20)
    if df is None:
        raise Exception("df extracted if None")

    # statistics and distribution
    x, y, d = distribution_df(df)
    if x is None:
        raise Exception("No row extracted")

    # save results
    dict_result["integer"] = d["integer"]
    dict_result["float"] = d["float"]
    dict_result["object"] = d["object"]
    dict_result["x"] = x
    dict_result["y"] = y
    dict_result["extradata"] = metadata
    dict_result["no_header"] = no_header
    save_results(df, output_directory, extradata_directory, log_directory,
                 dict_result)

    return


##############################################################################


def detect_encoding(path):
    """
    Function to detect the file encoding.
    :param path: string
    :return: string
    """
    with open(path, 'rb') as f:
        detection = chardet.detect(f.read(20000))
    return detection["encoding"]


def file_len(filename):
    """
    Function to count the number of lines from a file
    :param filename: string
    :return: integer
    """
    i = 0
    with open(filename, mode="rb") as f:
        for i, l in enumerate(f):
            pass
    return i + 1


def get_sample(path, encoding, nrow, threshold_n_row, ratio_sample, max_sample,
               threshold_n_col):
    """
    Functions to get a random sample of rows from a text file.
    :param path: string
    :param encoding: string
    :param nrow: integer
    :param threshold_n_row: integer
    :param ratio_sample: integer
    :param max_sample: integer
    :param threshold_n_col: float
    :return: list of strings, string, float
    """
    sample = []
    if nrow >= threshold_n_row:
        size_sample = min(int(round(nrow * ratio_sample / 100, 0)), max_sample)
        x = random.sample(range(0, nrow), size_sample)
        with open(path, mode="rb") as f:
            for i, l in enumerate(f):
                if i in x:
                    sample.append(l.decode(encoding))
    else:
        with open(path, mode="rb") as f:
            for row in f:
                sample.append(row.decode(encoding))
        threshold_n_col = 0.60
    full_sample = "".join(sample)
    return sample, full_sample, threshold_n_col


def has_header(content, separator, n_col):
    """
    Function to determine if the first row is a potential header.
    :param content: list of strings
    :param separator: string
    :param n_col: integer
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

    # assume first row is header
    header = content[0].strip().split(separator)
    if len(header) != n_col:
        return -1
    # initialize dictionary of types
    col_types = {}
    for i in range(n_col):
        col_types[i] = None
    # analyze the consistency of 10 rows
    for row in content[1:10]:
        row = row.strip().split(separator)
        # test the type of each column
        for col in list(col_types.keys()):
            for this_type in [int, float]:
                try:
                    this_type(row[col])
                    break
                except (ValueError, OverflowError):
                    pass
            # if it's neither a integer or a float, it's assumed to be a string
            # (and we keep it length)
            else:
                this_type = len(row[col])
            if this_type != col_types[col]:
                # add new column type
                if col_types[col] is None:
                    col_types[col] = this_type
                # or test the lack of consistency
                else:
                    del col_types[col]

    if len(col_types) == 0:
        return -1

    # finally, compare results against first row and "vote" on whether
    # it's a header
    results = 0
    for col, x in col_types.items():
        # penalize invalid column names
        if len(header[col]) == 1:
            results -= 1 / n_col
        # test if it's a length
        elif isinstance(x, int):
            if len(header[col]) != x:
                results += 1
            else:
                results -= 1 / n_col
        # attempt typecast
        else:
            try:
                x(header[col])
                results -= 1 / n_col
            except (ValueError, TypeError):
                results += 1
    return results


def search_header(list_content, separator, n_col, row_to_check_header):
    """
    Function to determine the row header and the number of blank lines before.
    :param list_content: list of strings
    :param separator: string
    :param n_col: integer
    :param row_to_check_header: integer
    :return: integer, boolean
    """
    h = [has_header(list_content[row:], separator, n_col)
         for row in range(row_to_check_header)]
    for row in range(row_to_check_header):
        if h[row] > 0:
            return row, False
    for row in range(row_to_check_header):
        if h[row] > -1:
            return row, True
    return None, None


def file_is_json(filename, input_directory, dict_param):
    """
    Function to determine is the file is a json file
    (directly from the filename)
    :param filename: string
    :param input_directory: string
    :param dict_param: dictionary
    :return: boolean
    """
    path = os.path.join(input_directory, filename)
    # number of rows
    nrow = file_len(path)
    # encoding
    encoding = detect_encoding(path)
    # sample
    sample, full_sample, threshold_n_col = \
        get_sample(path, encoding, nrow, dict_param["threshold_n_row"],
                   dict_param["ratio_sample"], dict_param["max_sample"],
                   dict_param["threshold_n_col"])
    return is_json(full_sample, dict_param["threshold_json"])


def is_json(sample, threshold):
    """
    Function to determine if the file is a json file.
    :param sample: string
    :param threshold: float
    :return: boolean
    """
    m = 0
    n = 0
    for i in sample:
        n += 1
        if i in ["{", "}", "[", "]", ":"]:
            m += 1
    ratio = m / n
    return ratio > threshold


def count_col(sample, separator, threshold_n_col):
    """
    Function to count the number of columns from a file.
    :param sample: list of strings
    :param separator: string
    :param threshold_n_col: float
    :return: None object or integer
    """
    # number of columns
    frequency = Counter(list(map(lambda x: x.count(separator), sample)))
    n_col = None
    for i in frequency:
        if frequency[i] / len(sample) > threshold_n_col:
            n_col = i + 1
            break
    return n_col


def get_content(path, encoding, dialect, n_col):
    """
    Function to distinguish the metadata from the matrix data
    :param path: string
    :param encoding: string
    :param dialect: dialect object
    :param n_col: integer
    :return: list of strings, list of integers, string
    """
    list_content = []
    bad_rows = []
    metadata = []
    i = 0
    with open(path, mode="rb") as f:
        for row in f:
            row = row.decode(encoding)
            if len(row.split(dialect.delimiter)) != n_col:
                bad_rows.append(i)
                metadata.append(row)
            else:
                list_content.append(row)
            i += 1
    return list_content, bad_rows, " ".join(metadata)


def clean_empty_col(df):
    """
    Function to clean empty columns from a dataframe.
    :param df: dataframe
    :return: dataframe
    """
    l = []
    for col in df.columns:
        if df[col].count() > 0:
            l.append(col)
    return df[l]


def distribution_df(df):
    """
    Function to compute some statistics on a dataframe.
    :param df: dataframe
    :return: integer, integer, dictionary
    """
    x = df.shape[0]
    y = df.shape[1]
    if x * y == 0:
        return None, None, None
    d = {"integer": 0, "float": 0, "object": 0}
    for i in df.dtypes:
        if i in [int, np.int, np.int_, np.intc, np.intp, np.int8, np.int16,
                 np.int32, np.int64]:
            i = "integer"
        elif i in [float, np.float, np.float_, np.float16, np.float32,
                   np.float64]:
            i = "float"
        else:
            i = "object"
        if i not in d:
            d[i] = 1
        else:
            d[i] += 1
    return x, y, d


def save_results(df, output_directory, extradata_directory, log_directory,
                 dict_result):
    """
    Function to save results (dataframe and metadata)
    """
    dict_result["matrix_name"] = dict_result["matrix_name"].replace(" ", "_")
    dict_result["file_name"] = dict_result["file_name"].replace(" ", "_")

    path = os.path.join(output_directory, dict_result["matrix_name"])

    if os.path.isfile(path):
        return

    df.to_csv(path, sep=";", index=False, encoding="utf-8", header=True)

    if dict_result["extradata"] != "":
        path_metadata = os.path.join(extradata_directory,
                                     dict_result["matrix_name"])
        with open(path_metadata, "wt", encoding="utf-8") as f:
            f.write(dict_result["extradata"])
        dict_result["extradata"] = True
    else:
        dict_result["extradata"] = False

    if not isinstance(list(df.columns)[0], str):
        dict_result["multiheader"] = True
    else:
        dict_result["multiheader"] = False

    l = [dict_result["matrix_name"], dict_result["file_name"],
         dict_result["source_file"], dict_result["x"], dict_result["y"],
         dict_result["integer"], dict_result["float"],
         dict_result["object"], dict_result["extradata"],
         not dict_result["no_header"],
         dict_result["multiheader"], list(df.columns),
         dict_result["extension"], dict_result["zip_file"]]
    l = [str(i).replace(";", ",") for i in l]

    path_log = os.path.join(log_directory, dict_result["matrix_name"])
    with open(path_log, mode='wt', encoding='utf-8') as f:
        f.write("##########################################################"
                "########### \n")
        for i in l:
            f.write(i)
            f.write("\n")

    return


def get_df(path, list_content, bad_rows, encoding, sep, check_header, n_col):
    """
    Function to get a dataframe from the data
    :param path: string
    :param list_content: list of strings
    :param bad_rows: list of integers
    :param encoding: string
    :param sep: string
    :param check_header: integer
    :param n_col: integer
    :return: dataframe, boolean
    """
    # TODO exploit bold and color cells
    row_to_check_header = min(check_header, int(len(list_content) / 2))
    row_header, no_header = search_header(list_content, sep, n_col,
                                          row_to_check_header)
    if row_header is None:
        return None, None
    if no_header:
        # TODO gérer le cas où row_header est supérieur à 0
        df = pd.read_csv(path, encoding=encoding, sep=sep, header=None,
                         skiprows=bad_rows, skip_blank_lines=False,
                         low_memory=False, index_col=False,)
    else:
        if row_header > 0:
            row_header = [i for i in range(row_header + 1)]
        df = pd.read_csv(path, encoding=encoding, sep=sep, header=row_header,
                         skiprows=bad_rows,
                         skip_blank_lines=False, low_memory=False,
                         index_col=None)
    if df is None:
        return None, None
    pd.to_numeric(df.columns, errors="ignore")
    return df, no_header


def clean_csv(path, encoding, sample, full_sample, threshold_n_col,
              check_header):
    """
    Function to clean csv files
    :param path: string
    :param encoding: string
    :param sample: list of strings
    :param full_sample: string
    :param threshold_n_col: integer
    :param check_header: integer
    :return: dataframe, string, boolean
    """
    # separator
    try:
        dialect = Sniffer().sniff(full_sample)
        sep = dialect.delimiter
    except Error:
        return None, None, None
    # number of columns
    n_col = count_col(sample, sep, threshold_n_col)
    if n_col is None:
        return None, None, None
    # content and metadata
    list_content, bad_rows, metadata = get_content(path, encoding, dialect,
                                                   n_col)
    # header
    df, no_header = get_df(path, list_content, bad_rows, encoding, sep,
                           check_header, n_col)
    return df, metadata, no_header


def is_consistent_list(json):
    """
    Function to test the pattern of a json file.
    The function determine if a part of the json can be read as a list of
    dictionaries

    Parameters
    ----------
    json : opened json file
        The json file to test

    Returns
    -------
    is_consistent : bool
        The test result
    """
    # check if the json is a first order consistent list (list of dictionaries)
    l = []
    if isinstance(json, list) and len(json) > 0:
        for i in json:
            l.append(type(i))
        if len(set(l)) == 1 and l[0] == dict:
            return True
    return False


def explore_json(json):
    """
    Function to explore the pattern of a json file.
    The function is looking for a dictionary-like pattern or a list of
    dictionaries

    Parameters
    ----------
    json : opened json file
        The json file to explore

    Returns
    -------
    contents : list
        A part of the json file as a list of dictionaries

    extradata : str
        The rest of the json file
    """
    # first order json
    if is_consistent_list(json):
        return json, ""

    # second order json
    if isinstance(json, dict):
        contents = []
        key_contents = None
        extradata = []
        for key in json:
            if is_consistent_list(json[key]) and len(json[key]) > len(contents):
                extradata.append(str(contents))
                key_contents = key
                contents = json[key]
            else:
                extradata.append(str(json[key]))
        if key_contents is not None:
            return contents, " ".join(extradata)

    return None, ""


def explore_json_deep(json, deepness):
    """
    Function to recursively explore the pattern of a json file.

    Parameters
    ----------
    json : opened json file
        The json file to explore

    deepness : int
        The  number of json layers we need to test to find the proper pattern

    Returns
    -------
    contents : list
        A part of the json file as a list of dictionaries

    extradata : str
        The rest of the json file
    """
    data, extradata = explore_json(json)
    if data is not None:
        return data, extradata
    if deepness == 0:
        return None, None
    l_extradata = []
    if isinstance(json, dict):
        for key in json:
            data, metadata = explore_json_deep(json[key], deepness=deepness-1)
            if data is not None:
                return data, extradata + " " + " ".join(l_extradata)
            else:
                l_extradata.append(str(json[key]))

    return None, ""


def clean_json(path):
    """
    Function to extract a matrix from a json file.

    Parameters
    ----------
    path : str
        Path of the json file

    Returns
    -------
    df : pandas Dataframe
        The matrix extracted

    extradata : str
        Extradata extracted from the json file

    no_header : bool
        Boolean to specify if a header has been detected from the json file
    """
    # read file
    with open(path, mode="rb") as file:
        json_data = json.load(file)

    # flatten the json
    data, metadata = explore_json_deep(json_data, deepness=20)
    if data is not None:
        df = json_normalize(data, record_path=None, meta=None)
        return df, metadata, False
    else:
        return None, None, None


def clean_geojson(path):
    return clean_json(path)


def clean_xml(path, deepness):
    """
    Function to explore the pattern of a xml file and clean it.
    The function is looking for a dictionary-like pattern or a list of
    dictionaries from which a matrix is extracted

    Parameters
    ----------
    path : str
        Path of the xml file

    deepness : int
        The  number of xml layers we need to test to find the proper pattern

    Returns
    -------
    df : pandas Dataframe
        The matrix extracted

    extradata : str
        Extradata extracted from the xml file

    no_header : bool
        Boolean to specify if a header has been detected from the xml file
    """
    # read data
    tree = etree.iterparse(path, huge_tree=True, events=("start", "end"))

    # TODO improve this function
    # extract matrix
    results = None
    results_extradata = None
    max_row = 0
    max_col = 0
    for event, element in tree:
        x = xmljson.badgerfish.data(element)
        json_data = json.loads(json.dumps(x))
        data, extradata = explore_json_deep(json_data, deepness=deepness)
        if data is None:
            continue
        df = json_normalize(data, record_path=None, meta=None)
        df = clean_empty_col(df)
        if df.shape[1] > max_col:
            max_row = df.shape[0]
            max_col = df.shape[1]
        elif df.shape[1] == max_col and df.shape[0] > max_row:
            max_row == df.shape[0]
        else:
            continue
        results = df
        results_extradata = extradata
    names = []
    for i in results.columns:
        m = re.findall('{(.+?)}', i)
        for j in m:
            i = i.replace("{" + j + "}", "")
        i = i.replace("@", "").replace("$", "").replace(".", "_")
        if re.search("_$", i):
            i = i[:len(i) - 1]
        names.append(i)
    results.columns = names

    return results, results_extradata, False


def read_excel(path, info=False):
    """
    Function to read an excel file and edit it
    :param path: string
    :param info: boolean
    :return: xlrd.book.Book object, xlrd.book.Book object
    """
    rb = open_workbook(path, ragged_rows=True, on_demand=True,
                       formatting_info=False)
    if info:
        try:
            rb_info = open_workbook(path, ragged_rows=True, on_demand=True,
                                    formatting_info=True)
        except:
            rb_info = rb
        return rb, rb_info
    else:
        return rb


def edit_excel(read_book, info_book, output_directory):
    """
    Function to edit and correct an excel file
    :param read_book: xlrd.book.Book object
    :param info_book: xlrd.book.Book object
    :param output_directory: str
    :return: xlrd.book.Book object
    """
    wb = copy(read_book)
    sheets = read_book.sheet_names()
    with tmp_dir(host_dir=output_directory)as temp_directory:
        for sheet in sheets:
            # load sheet
            s = read_book.sheet_by_name(sheet)
            sheet_index = sheets.index(sheet)
            ws = wb.get_sheet(sheet_index)
            # clean data from "\n", "\r" and "\r\n"
            for rowx in range(s.nrows):
                for colx in range(len(s.row(rowx))):
                    old_value = str(s.cell_value(rowx, colx))
                    if isinstance(old_value, str):
                        new_value = old_value.replace("\n", " ")\
                            .replace("\r", " ").replace(";", " ")
                        ws.write(rowx, colx, new_value)
            # fill merged cells
            merged_cells = info_book.sheet_by_name(sheet).merged_cells
            if len(merged_cells) > 0:
                for crange in merged_cells:
                    rlo, rhi, clo, chi = crange
                    if len(s.row(rlo)) >= clo + 1:
                        information = str(s.cell_value(rlo, clo))\
                            .replace("\n", " ").replace("\r", " ")\
                            .replace(";", " ")
                        for rowx in range(rlo, rhi):
                            for colx in range(clo, chi):
                                ws.write(rowx, colx, information)
        # save temporary results
        temp_path = os.path.join(temp_directory, "temporary_file")
        wb.save(temp_path)
        rb_temp = read_excel(temp_path)
    return rb_temp


def filter_excel(read_sheet):
    """
    Function to filter the good rows only
    :param read_sheet: xlrd.sheet.Sheet object
    :return: list of integers, list of integers
    """
    bad_rows = []
    good_rows = []
    for i in range(read_sheet.nrows):
        if read_sheet.row_len(i) == read_sheet.ncols:
            # distinguish the newly filled merged cells
            l = [read_sheet.cell_value(i, j) for j in range(read_sheet.ncols)]
            if len(set(l)) == 1:
                bad_rows.append(i)
            else:
                good_rows.append(i)
        else:
            bad_rows.append(i)
    return good_rows, bad_rows


def excel_to_text(path, sheet_name):
    """
    Function to convert an excel file in a text file (with a csv format)
    :param path: string
    :param sheet_name: string
    :return: string
    """
    excel = pd.ExcelFile(path)
    df = excel.parse(sheet_name, header=None)
    new_path = "_".join([path, sheet_name])
    df.to_csv(new_path, sep=";", index=False, encoding="utf-8", header=False)
    return new_path


def get_content_excel(path, good_rows):
    """
    Function to filter the matrix-like content
    :param path: string
    :param good_rows: list of integers
    :return: list of strings, string
    """
    list_content = []
    metadata = []
    with open(path, mode="rt", encoding="utf-8") as f:
        i = 0
        for row in f:
            if i in good_rows:
                list_content.append(row)
            else:
                metadata.append(row)
            i += 1
    metadata = " ".join(metadata)
    return list_content, metadata


def clean_sheet_excel(read_book, sheet_name, temporary_path, check_header):
    """
    Function to clean an excel sheet
    :param read_book: xlrd.book.Book object
    :param sheet_name: string
    :param temporary_path: string
    :param check_header: integer
    :return: dataframe, string, boolean
    """
    # load sheet
    s = read_book.sheet_by_name(sheet_name)
    # good rows and bad rows
    good_rows, bad_rows = filter_excel(s)
    # convert to text file
    temp_sheet_path = excel_to_text(temporary_path, sheet_name)
    # get content and metadata
    list_content, metadata = get_content_excel(temp_sheet_path, good_rows)
    # header
    df, no_header = get_df(temp_sheet_path, list_content, bad_rows, "utf-8",
                           ";", check_header, s.ncols)
    return df, metadata, no_header


def is_json_geojson(path):
    """
    Function to test if a text file is a json, a geojson or a casual text file.

    Parameters
    ----------
    path : str
        Path of the text file to test

    Returns
    -------
    x : str
        Either 'text', 'json' or 'geojson'
    """
    with open(path, mode="rb") as file:
        try:
            json.load(file)
            try:
                gpd.read_file(path)
                return "geojson"
            except Exception:
                return "json"
        except Exception:
            return "text"
