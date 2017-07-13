# -*- coding: utf-8 -*-

""" Remove empty files and check download results. """

# libraries
import os
import magic
import pandas as pd
from warnings import warn
from lxml import etree
from tqdm import tqdm
from toolbox.utils import get_config_tag, get_config_trace
from download_data import get_format_urls
print("\n")


def _initialize_log():
    summary_download_path = "../data/summary_download"
    if os.path.isfile(summary_download_path):
        os.remove(summary_download_path)
    with open(summary_download_path, mode="wt", encoding="utf-8") as f:
        f.write("filename;id_page;format_declared;extension;size;downloaded;"
                "id_producer;title_producer")
        f.write("\n")
    return summary_download_path


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
        (n_tables, [id_table, id_page, title_page, id_producer, title_producer])
    """
    # get the path for the xml file
    filepath = os.path.join(basex_directory, "id_page_table.xml")

    # read the xml and fill in a dataframe with it
    l_id_table = []
    l_id_page = []
    l_title_page = []
    l_id_producer = []
    l_title_producer = []
    tree = etree.parse(filepath)
    for table in tree.xpath("/results/page"):
        (id_page, title_page, id_table, id_producer,
         title_producer) = (table[0].text, table[1].text, table[2].text,
                            table[3].text, table[4].text)
        if id_table is not None:
            l_id_table.append(id_table)
            l_id_page.append(id_page)
            l_title_page.append(title_page)
            l_id_producer.append(id_producer)
            l_title_producer.append(title_producer)
    df = pd.DataFrame({"id_table": l_id_table,
                       "id_page": l_id_page,
                       "title_page": l_title_page,
                       "id_producer": l_id_producer,
                       "title_producer": l_title_producer})

    return df


def _remove_empty_files(input_directory, df_id_table, summary_download_path,
                        error_directory):
    """
    Function to remove empty files downloaded and summarize the results.

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

    Returns
    -------
    """
    print("remove empty files", "\n")

    # set the table id as index
    df_id_table.set_index(keys="id_table", drop=True, inplace=True)

    # remove the empty files and store the extensions
    n_downloaded = 0
    n_null = 0
    n_errors = 0
    tuple_files = get_format_urls(input_directory, format='all')
    for tuple_file in tqdm(tuple_files):
        (_, filename, format_declared) = tuple_file
        path_file = os.path.join(input_directory, filename)
        if os.path.isfile(path_file):
            downloaded = True
            size = os.path.getsize(path_file)
            if size == 0:
                extension = None
                os.remove(path_file)
                n_null += 1
            else:
                extension = magic.Magic(mime=True).from_file(path_file)
                n_downloaded += 1
        else:
            downloaded = False
            size = 0
            extension = None
            n_errors += 1
        # save results
        id_page = df_id_table.loc[filename, "id_page"]
        id_producer = df_id_table.loc[filename, "id_producer"]
        title_producer = df_id_table.loc[filename, "title_producer"]
        row = [filename, id_page, format_declared, extension, str(size),
               downloaded, id_producer, title_producer]
        with open(summary_download_path, mode="at", encoding="utf-8") as f:
            f.write(";".join(row))
            f.write("\n")

    # check consistency of the results
    print("number of files properly downloaded (zipped files excluded) :",
          n_downloaded)
    print("number of files in the 'collected data' directory :",
          len(os.listdir(input_directory)))
    if n_downloaded != len(os.listdir(input_directory)):
        warn("The number of downloads counted is not consistent with the "
             "number of files saved.")
    print("number of null files downloaded :", n_null)
    print("number of errors :", n_errors)
    print("number of error logs :", len(os.listdir(error_directory)))
    if n_errors != len(os.listdir(error_directory)):
        warn("The number of errors counted is not consistent with the error "
             "logs saved.")
    if n_downloaded + n_null + n_errors != len(tuple_files):
        warn("The total number of available urls is not consistent with the "
             "number of treated files (downloaded, null and errors).")
    print()

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
    # initialize the log file
    summary_download_path = _initialize_log()

    # get the id page for each table
    df_id_table = get_id_page(basex_directory)

    # remove the empty files and save the results
    _remove_empty_files(input_directory, df_id_table, summary_download_path,
                        error_directory)

    return


if __name__ == "__main__":

    get_config_trace()

    # paths
    input_directory = get_config_tag("output", "download")
    basex_directory = get_config_tag("output", "basex")
    error_directory = get_config_tag("error", "download")

    # run
    main(input_directory, basex_directory, error_directory)
