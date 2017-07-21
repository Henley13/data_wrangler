# -*- coding: utf-8 -*-

""" Check xml files and metadata collected. """

# libraries
import os
import pandas as pd
import hashlib as hash
import shutil
import glob
from tqdm import tqdm
from lxml import etree
from toolbox.utils import get_config_tag
print("\n")


def _check_output_directory(path_directory, reset):
    """
    Function to check the output directory.

    Parameters
    ----------
    path_directory : str
        Path of the output directory

    reset : bool
        Boolean to decide if the directory should be reset or not

    Returns
    -------
    """
    if not os.path.isdir(path_directory):
        os.mkdir(path_directory)
    elif reset:
        shutil.rmtree(path_directory)
        os.mkdir(path_directory)
    else:
        pass


def _load_xml_dataset(metadata_directory):
    """
    Function to load the xml files with the metadata on datasets.

    Parameters
    ----------
    metadata_directory : str
        Path of the datasets metadata directory

    Returns
    -------
    d : dict{'id_table': (tree_table, tree_page)}
    """
    d = {}
    for xml_file in tqdm(os.listdir(metadata_directory),
                         desc="load xml dataset"):
        path_xml_file = os.path.join(metadata_directory, xml_file)
        tree = etree.parse(path_xml_file)
        for table in tree.xpath("/metadata/tables/table"):
            d[table.findtext("id")] = (table, tree)
    return d


def _load_xml_reuse(metadata_directory):
    """
    Function to load the xml files with the metadata on reuses.

    Parameters
    ----------
    metadata_directory : str
        Path of the reuses metadata directory

    Returns
    -------
    d : dict{'id_table': (tree_table, tree_reuse)}
    """
    d = {}
    for xml_file in tqdm(os.listdir(metadata_directory),
                         desc="load xml reuse"):
        path_xml_file = os.path.join(metadata_directory, xml_file)
        tree = etree.parse(path_xml_file)
        for table in tree.xpath("/metadata/datasets/dataset"):
            d[table.findtext("id")] = (table, tree)
    return d


def _metadata_datasets(loader, output_directory):

    # paths
    path_output = os.path.join(output_directory, "metadata_dataset")

    # get metadata...
    l_table = []
    l_url = []
    for (table, tree) in tqdm(loader):

        # ... specific to the file
        title_file = table.findtext("table/title")
        id_file = table.findtext("table/id")
        url_file = table.findtext("table/url")
        url_destination_file = table.findtext("table/url_destination")
        description_file = table.findtext("table/description")
        creation_file = table.findtext("table/creation")
        publication_file = table.findtext("table/publication")
        last_modification_file = table.findtext("table/last_modification")
        availability_file = table.findtext("table/availability")

        # ... and more general
        title_page = tree.findtext(".//title")
        id_page = tree.findtext(".//id")
        url_page = tree.findtext(".//url")
        url_api_page = tree.findtext(".//url_api")
        license_page = tree.findtext(".//license")
        description_page = tree.findtext(".//description")
        creation_page = tree.findtext(".//creation")
        last_modification_page = tree.findtext(".//last_modification")
        last_resources_update_page = tree.findtext(".//last_resources_update")
        frequency_page = tree.findtext(".//frequency")
        start_coverage_page = tree.findtext(".//start_coverage")
        end_coverage_page = tree.findtext(".//end_coverage")
        granularity_page = tree.findtext(".//granularity")
        zones_page = tree.findtext(".//zones")
        geojson_page = tree.findtext(".//geojson")
        title_producer = tree.findtext(".//organization/title")
        url_producer = tree.findtext(".//organization/url")
        url_api_producer = tree.findtext(".//organization/url_api")
        tags = []
        for tag in tree.xpath("/metadata/tags/tag"):
            if tag.text is None:
                tags.append("")
            else:
                tags.append(tag.text)
        tags_page = " ".join(tags).replace("-", "_")
        # TODO use hash method
        # save a new row
        row = [id_file, title_file, url_file, url_destination_file, description_file, creation_file, publication_file, last_modification_file, availability_file, title_page, id_page, url_page, url_api_page, license_page, description_page, creation_page, last_modification_page, last_resources_update_page, frequency_page, start_coverage_page, end_coverage_page, granularity_page, zones_page, geojson_page, title_producer, url_producer, url_api_producer, tags_page]
        row = [str(item).replace(";", ",").replace('\n', '').replace('\r', '')
                        .replace('"', "'") for item in row]
        with open(path_output, mode='at', encoding='utf-8') as f:
            f.write(";".join(row))
            f.write("\n")

        l_table.append(id_file)
        l_url.append(url_file)

    # keep the original id and url
    df = pd.read_csv(path_output, sep=";", encoding="utf-8", index_col=False)
    df["id_table"] = l_table
    df["url"] = l_url
    print("df shape :", df.shape, "\n")

    return


def initialization_file(result_directory):
    """
    Function to initialize a text file with specific headers
    :param result_directory: string
    :return:
    """

    path_metadata_edited = os.path.join(result_directory, "log_final")
    if os.path.isfile(path_metadata_edited):
        os.remove(path_metadata_edited)
    with open(path_metadata_edited, mode='wt', encoding='utf-8') as f:
        f.write("matrix_name;file_name;source_file;n_row;n_col;integer;float;"
                "object;metadata;time;header;multiheader;header_name;extension;"
                "zipfile;size;commentary;title_file;id_file;url_file;"
                "url_destination_file;description_file;creation_file;"
                "publication_file;last_modification_file;availability_file;"
                "title_page;id_page;url_page;url_api_page;license_page;"
                "description_page;creation_page;last_modification_page;"
                "last_resources_update_page;frequency_page;start_coverage_page;"
                "end_coverage_page;granularity_page;zones_pages;geojson_page;"
                "title_producer;url_producer;url_api_producer;tags_page")
        f.write("\n")
    return


def edit(result_directory, original_metadata):
    """
    Function to edit metadata and add new features extracted from the cleaning
    process
    :param result_directory: string
    :param original_metadata: dictionary
    :return:
    """

    # paths
    data_directory = os.path.join(result_directory, "data_fitted")
    path_log = os.path.join(result_directory, "log_cleaning")
    path_metadata_edited = os.path.join(result_directory, "log_final")
    metadata_clean_directory = os.path.join(result_directory,
                                            "metadata_cleaning")

    # gather metadata
    df_log = pd.read_csv(path_log, sep=";", encoding="utf-8", index_col=False)
    for i in tqdm(range(df_log.shape[0])):

        matrix = df_log.at[i, "matrix_name"]
        # get metadata extracted from the matrix
        path_matrix = os.path.join(data_directory, matrix)
        size_matrix = os.path.getsize(path_matrix)

        # get metadata extracted from the cleaning process
        if df_log.at[i, "metadata"]:
            path_metadata_matrix = os.path.join(metadata_clean_directory,
                                                matrix)
            with open(path_metadata_matrix, mode='rt', encoding='utf-8') as f:
                commentary_matrix = f.read()
        else:
            commentary_matrix = ""

        # get metadata extracted from data.gouv.fr
        try:
            table, tree = original_metadata[df_log.at[i, "source_file"]]
        except KeyError:
            print("KeyError :", df_log.at[i, "source_file"])
            continue

        # ... specific to the file
        title_file = table.findtext("table/title")
        id_file = table.findtext("table/id")
        url_file = table.findtext("table/url")
        url_destination_file = table.findtext("table/url_destination")
        description_file = table.findtext("table/description")
        creation_file = table.findtext("table/creation")
        publication_file = table.findtext("table/publication")
        last_modification_file = table.findtext("table/last_modification")
        availability_file = table.findtext("table/availability")

        # ... and more general
        title_page = tree.findtext(".//title")
        id_page = tree.findtext(".//id")
        url_page = tree.findtext(".//url")
        url_api_page = tree.findtext(".//url_api")
        license_page = tree.findtext(".//license")
        description_page = tree.findtext(".//description")
        creation_page = tree.findtext(".//creation")
        last_modification_page = tree.findtext(".//last_modification")
        last_resources_update_page = tree.findtext(".//last_resources_update")
        frequency_page = tree.findtext(".//frequency")
        start_coverage_page = tree.findtext(".//start_coverage")
        end_coverage_page = tree.findtext(".//end_coverage")
        granularity_page = tree.findtext(".//granularity")
        zones_page = tree.findtext(".//zones")
        geojson_page = tree.findtext(".//geojson")
        title_producer = tree.findtext(".//organization/title")
        url_producer = tree.findtext(".//organization/url")
        url_api_producer = tree.findtext(".//organization/url_api")

        tags = []
        for tag in tree.xpath("/metadata/tags/tag"):
            if tag.text is None:
                tags.append("")
            else:
                tags.append(tag.text)
        tags_page = " ".join(tags).replace("-", "_")

        l = [matrix, df_log.at[i, "file_name"], df_log.at[i, "source_file"],
             df_log.at[i, "n_row"], df_log.at[i, "n_col"],
             df_log.at[i, "integer"],
             df_log.at[i, "float"], df_log.at[i, "object"],
             df_log.at[i, "metadata"], df_log.at[i, "time"],
             df_log.at[i, "header"], df_log.at[i, "multiheader"],
             df_log.at[i, "header_name"], df_log.at[i, "extension"],
             df_log.at[i, "zipfile"], size_matrix, commentary_matrix,
             title_file, id_file, url_file, url_destination_file,
             description_file,
             creation_file, publication_file, last_modification_file,
             availability_file, title_page, id_page, url_page, url_api_page,
             license_page, description_page, creation_page,
             last_modification_page,
             last_resources_update_page, frequency_page, start_coverage_page,
             end_coverage_page, granularity_page, zones_page, geojson_page,
             title_producer, url_producer, url_api_producer, tags_page]
        l = [str(j).replace(";", ",").replace('\n', '').replace('\r', '')
                   .replace('"', "'") for j in l]
        with open(path_metadata_edited, mode='at', encoding='utf-8') as f:
            f.write(";".join(l))
            f.write("\n")
    print()
    return


def get_format_url(basex_directory, output_directory):
    """
    Function to get the urls for the requested format.

    Parameters
    ----------
    basex_directory : str
        Path of the BaseX directory

    output_directory : str
        Path of the output directory

    Returns
    -------
    l_urls : list of tuples(str, str, str)
        List of tuples (url, filename, format)
    """
    print("get url and declared format", "\n")

    # get the path and the format for each xml file
    for file_xml in os.listdir(basex_directory):
        if "url_" in file_xml:
            format = file_xml.split("_")[1].split(".")[0]
            path_xml = os.path.join(basex_directory, file_xml)
            print("format requested :", format)

            # get the url and the declared format
            l_url = []
            l_filename = []
            l_format = []
            duplicated_id = 0
            duplicated_id_removed = 0
            tree = etree.parse(path_xml)
            for table in tree.xpath("/results/table"):
                url, filename, format = (table[0].text, table[1].text,
                                         table[2].text)
                if url is not None and filename is not None:
                    if filename not in l_filename:
                        pass
                    else:
                        duplicated_id += 1
                        filename = hash.sha224(bytes(url, 'utf-8')).hexdigest()
                        if filename not in l_filename:
                            pass
                        else:
                            duplicated_id_removed += 1
                            continue
                l_url.append(url)
                l_filename.append(filename)
                l_format.append(format)

            # save results
            df = pd.DataFrame({"table": l_filename,
                               "format_declared": l_format,
                               "url": l_url})
            path_output = os.path.join(output_directory, "url_" + format)
            df.to_csv(path_output,
                      sep=';',
                      encoding='utf-8',
                      index=False,
                      header=True)

            print("number of urls with duplicated ids : %i (%i of them removed)"
                  % (duplicated_id, duplicated_id_removed))
            print("number of urls :", df.shape[0], "\n")

    return


def get_page(basex_directory, output_directory):
    """
    Function to link each table to its table id.

    Parameters
    ----------
    basex_directory : str
        Path of the BaseX results directory

    output_directory : str
        Path of the output directory

    Returns
    -------
    """
    print("get page", "\n")

    # get the path for the xml file
    file_xml = os.path.join(basex_directory, "id_page_table.xml")

    # read the xml and fill in a dataframe with it
    l_id_table = []
    l_id_page = []
    l_title_page = []
    l_id_producer = []
    l_title_producer = []
    l_url = []
    duplicated_id = 0
    duplicated_id_removed = 0
    tree = etree.parse(file_xml)
    for table in tree.xpath("/results/page"):
        (id_page, title_page, id_table, id_producer,
         title_producer, url) = (table[0].text, table[1].text, table[2].text,
                                 table[3].text, table[4].text, table[5].text)
        if id_table is not None and url is not None:
            if id_table not in l_id_table:
                pass
            else:
                duplicated_id += 1
                id_table = hash.sha224(bytes(url, 'utf-8')).hexdigest()
                if id_table not in l_id_table:
                    pass
                else:
                    duplicated_id_removed += 1
                    continue

        l_id_table.append(id_table)
        l_id_page.append(id_page)
        l_title_page.append(title_page)
        l_id_producer.append(id_producer)
        l_title_producer.append(title_producer)
        l_url.append(url)

    # save results
    df = pd.DataFrame({"table": l_id_table,
                       "id_page": l_id_page,
                       "title_page": l_title_page,
                       "id_producer": l_id_producer,
                       "title_producer": l_title_producer,
                       "url": l_url})
    path_output = os.path.join(output_directory, "page")
    df.to_csv(path_output,
              sep=';',
              encoding='utf-8',
              index=False,
              header=True)

    print("number of urls with duplicated ids : %i (%i of them removed)"
          % (duplicated_id, duplicated_id_removed))
    print("number of tables :", df.shape[0], "\n")

    return


def get_metadata_table(output_directory):
    print("get metadata", "\n")

    # get page
    path_page = os.path.join(output_directory, "page")
    df_page = pd.read_csv(path_page,
                          sep=";",
                          encoding="utf-8",
                          index_col=False)
    print("df page shape :", df_page.shape, "\n")

    # get the url
    template_path = os.path.join(output_directory, "url_*")
    for path_csv_url in glob.glob(template_path):
        df_url = pd.read_csv(path_csv_url,
                             sep=";",
                             encoding="utf-8",
                             index_col=False)

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

    return




def main():
    return

if __name__ == "__main__":

    # paths
    metadata_directory = get_config_tag("output", "metadata")
    error_metadata_directory = get_config_tag("error", "metadata")
    organization_directory = get_config_tag("output_organization", "metadata")
    error_organization_directory = get_config_tag("error_organization",
                                                  "metadata")
    reuse_directory = get_config_tag("output_reuse", "metadata")
    error_reuse_directory = get_config_tag("error_reuse", "metadata")
    query_directory = get_config_tag("query", "basex")
    basex_directory = get_config_tag("output", "basex")

    # parameters
    reset_metadata = get_config_tag("reset", "basex")
    reset_basex = get_config_tag("reset", "basex")
    reset = reset_metadata or reset_basex

    main()


