# -*- coding: utf-8 -*-

""" Count empty files and check download results. """

# libraries
import os
import magic
import pandas as pd
import hashlib
from lxml import etree
from tqdm import tqdm
from toolbox.utils import get_config_tag, get_config_trace
from download_data import get_format_urls
print("\n")


def _initialize_log():
    """
    Function to initialize the csv file with a summary of the download process.
    Returns
    -------
    summary_download_path : str
        Path of the csv file
    """
    summary_download_path = "../data/summary_download"
    if os.path.isfile(summary_download_path):
        os.remove(summary_download_path)
    with open(summary_download_path, mode="wt", encoding="utf-8") as f:
        f.write("filename;id_page;format_declared;extension;size;downloaded;"
                "id_producer;title_producer")
        f.write("\n")
    return summary_download_path


def _initialize_log_error():
    """
    Function to initialize the csv file with a summary of the download errors.
    Returns
    -------
    summary_download_path : str
        Path of the csv file
    """
    summary_error_path = "../data/summary_download_error"
    if os.path.isfile(summary_error_path):
        os.remove(summary_error_path)
    with open(summary_error_path, mode="wt", encoding="utf-8") as f:
        f.write("filename;error;content;url")
        f.write("\n")
    return summary_error_path


def get_id_page(basex_directory):
    """
    Function to link each table to its table id.

    Parameters
    ----------
    basex_directory : str
        Path of the BaseX results directory

    Returns
    -------
    df : pandas Dataframe
        (n_tables, [id_table, id_page, title_page, id_producer, title_producer,
                    url])
    """
    print("get id page", "\n")

    # get the path for the xml file
    filepath = os.path.join(basex_directory, "id_page_table.xml")

    # read the xml and fill in a dataframe with it
    l_id_table = []
    l_id_page = []
    l_title_page = []
    l_id_producer = []
    l_title_producer = []
    l_url = []
    tree = etree.parse(filepath)
    for table in tqdm(tree.xpath("/results/page"), desc="id page"):
        (id_page, title_page, id_table, id_producer,
         title_producer, url) = (table[0].text, table[1].text, table[2].text,
                                 table[3].text, table[4].text, table[5].text)
        if id_table is not None and url is not None:
            if id_table in l_id_table:
                id_table = hashlib.sha224(bytes(url, 'utf-8')).hexdigest()
            l_id_table.append(id_table)
            l_id_page.append(id_page)
            l_title_page.append(title_page)
            l_id_producer.append(id_producer)
            l_title_producer.append(title_producer)
            l_url.append(url)

    df = pd.DataFrame({"id_table": l_id_table,
                       "id_page": l_id_page,
                       "title_page": l_title_page,
                       "id_producer": l_id_producer,
                       "title_producer": l_title_producer,
                       "url": l_url})

    print("df shape :", df.shape, "\n")

    return df


def _log_file(input_directory, tuple_file, df_id_table):
    """
    Function to gather the results to save about every files.

    Parameters
    ----------
    input_directory : str
        Path of the input directory (collected data)

    tuple_file : tuple(str, str, str)
        Tuple (url, filename, format)

    df_id_table : pandas Dataframe
        (n_tables, [id_table, id_page, title_page, id_producer, title_producer])

    Returns
    -------
    row : list
        Row to write in a text file (list of str)
    """
    # path
    (_, filename, format_declared) = tuple_file
    path_file = os.path.join(input_directory, filename)
    # check if the file has been downloaded
    if os.path.isfile(path_file):
        downloaded = True
        size = os.path.getsize(path_file)
        if size == 0:
            extension = None
        else:
            extension = magic.Magic(mime=True).from_file(path_file)
    else:
        downloaded = False
        size = -1
        extension = None

    # results to save
    id_page = df_id_table.loc[filename, "id_page"]
    id_producer = df_id_table.loc[filename, "id_producer"]
    title_producer = df_id_table.loc[filename, "title_producer"]
    row = [filename, id_page, format_declared, extension, size, downloaded,
           id_producer, title_producer]
    row = [str(item) for item in row]

    return row


def _count_empty_files(input_directory, df_id_table, summary_download_path,
                       error_directory, basex_directory):
    """
    Function to count empty files downloaded and summarize the results.

    Parameters
    ----------
    input_directory : str
        Path of the input directory (collected data)

    df_id_table : pandas Dataframe
        Dataframe to link each id table to its id page

    summary_download_path : str
        Path to the summary file

    error_directory : str
        Path of the download errors directory

    basex_directory : str
        Path of the BaseX results directory

    Returns
    -------
    """
    print("count empty files", "\n")

    # set the table id as index
    df_id_table.set_index(keys="id_table", drop=True, inplace=True)

    # count the empty files and store the extensions
    tuple_files = get_format_urls(basex_directory, format_requested='all')
    for tuple_file in tqdm(tuple_files, desc="count empty files"):
        row = _log_file(tuple_file, df_id_table)

        # save results
        with open(summary_download_path, mode="at", encoding="utf-8") as f:
            f.write(";".join(row))
            f.write("\n")

    df = pd.read_csv(summary_download_path, sep=';', encoding='utf-8',
                     index_col=False)

    # check consistency of the results
    n_downloaded = df.query("size > 0").shape[0]
    n_null = df.query("size == 0").count().shape[0]
    n_errors = df.query("size == -1").count().shape[0]

    print("number of files downloaded (zipped files excluded) :", n_downloaded)
    print("number of files in the 'collected data' directory :",
          len(os.listdir(input_directory)))
    print("number of null files downloaded :", n_null)
    print("number of errors :", n_errors)
    print("number of error logs :", len(os.listdir(error_directory)))
    print("number of urls :", len(tuple_files), "\n")

    return


def _analyze_error(error_directory, summary_error_path, summary_download_path):
    """
    Function to count the different errors that occurred during the downloading.

    Parameters
    ----------
    error_directory : str
        Path of the error directory

    summary_error_path : str
        Path of the error summary file

    summary_download_path : str
        Path of download summary file

    Returns
    -------
    """

    print("analyze download errors", "\n")

    # get data
    df_download_summary = pd.read_csv(summary_download_path, sep=";",
                                      encoding="utf-8", index_col=False)
    for filename in tqdm(os.listdir(error_directory),
                         desc="analyze download errors"):
        path = os.path.join(error_directory, filename)
        with open(path, mode="rt", encoding="utf-8") as f:
            c = f.readlines()
            url = c[1].strip()
            error = c[3].split(" ")[0]
            content = c[-2].strip()
        with open(summary_error_path, mode="at", encoding="utf-8") as f:
            f.write(";".join([str(filename), str(error), str(content),
                              str(url)]))
            f.write("\n")

    # merge data
    df_error_summary = pd.read_csv(summary_error_path, sep=";",
                                   encoding="utf-8", index_col=False)

    df = df_error_summary.merge(df_download_summary,
                                how='left',
                                left_on='filename',
                                right_on='filename',
                                left_index=True,
                                right_index=False,
                                copy=False)
    df.to_csv(summary_error_path, sep=';', encoding='utf-8', index=False,
              header=True)

    print("df download shape :", df_download_summary.shape)
    print("df error shape :", df_error_summary.shape)
    print("df shape :", df.shape, "\n")

    print(df.columns, "\n")

    print(df_download_summary.head(), "\n")
    print(df_error_summary.head(), "\n")
    print(df.head(), "\n")

    # analyze data
    print("analyze errors", "\n")
    print(list(df.columns))
    print(df.shape, "\n")
    print(df['extension'].value_counts(), "\n")
    print(df['error'].value_counts(), "\n")

    print("--------------------", "\n")

    extensions = list(set(list(df["extension"])))
    for ext in extensions:
        print("extension :", ext, "\n")
        query = "extension == '%s'" % ext
        df_error_ext = df.query(query)
        print(df_error_ext["error"].value_counts(), "\n")
        max_e = df_error_ext["error"].value_counts().index.tolist()[0]
        print(df_error_ext.query("error == '%s'" % max_e)["content"].
              value_counts(), "\n")
        print("---", "\n")

    return


def main(input_directory, basex_directory, error_directory):
    """
    Function to run all the script.

    Parameters
    ----------
    input_directory : str
        Path of the collected data directory

    basex_directory : str
        Path of the BaseX results directory

    error_directory : str
        Path of the download errors directory
    Returns
    -------
    """
    # initialize the log files
    summary_download_path = _initialize_log()
    summary_error_path = _initialize_log_error()

    # get the id page for each table
    df_id_table = get_id_page(basex_directory)

    # count the empty files and save the results
    _count_empty_files(input_directory, df_id_table, summary_download_path,
                       error_directory, basex_directory)

    # analyze the errors
    _analyze_error(error_directory, summary_error_path, summary_download_path)

    return


if __name__ == "__main__":

    get_config_trace()

    # paths
    input_directory = "../data/sample"
    basex_directory = get_config_tag("output", "basex")
    error_directory = get_config_tag("error", "download")
    # input_directory = get_config_tag("output", "download")
    # basex_directory = get_config_tag("output", "basex")
    # error_directory = get_config_tag("error", "download")

    # run
    main(input_directory, basex_directory, error_directory)
