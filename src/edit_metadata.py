#!/bin/python3
# coding: utf-8

""" We edit metadata with new information inferred from the file. """

# libraries
import os
import pandas as pd
from tqdm import tqdm
from lxml import etree
from toolbox.utils import get_config_tag
import nltk
from nltk import word_tokenize
nltk_path = get_config_tag("nltk", "text_extraction")
nltk.data.path.append(nltk_path)
print("\n")


def _get_original_metadata(metadata_directory):
    """
    Function to get the original metadata, collected from data.gouv.fr
    :param metadata_directory: string
    :return: dictionary
    """
    d = {}
    for file in os.listdir(metadata_directory):
        tree = etree.parse(os.path.join(metadata_directory, file))
        for table in tree.xpath("/metadata/tables/table"):
            d[table.findtext("id")] = (table, tree)
    return d


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


def _refactor_extension(row):
    """
    Function to rename extension values
    :param row: row from a pandas Dataframe
    :return: string
    """
    if row["extension"] == "text/plain":
        return "text"
    elif row["extension"] == "application/CDFV2-unknown":
        return "cdfv2"
    elif row["extension"] == "application/octet-stream":
        return "binary"
    elif row["extension"] == "application/vnd.ms-excel":
        return "excel"
    elif row["extension"] == "text/xml":
        return "xml"
    elif row["extension"] == "json":
        if row["geojson"]:
            return "geojson"
        else:
            return "json"

    else:
        return row["extension"]


def _is_geojson(header):
    """
    Function to detect if a file is a geojson or a json
    :param header: string
    :return: boolean
    """
    m = 0
    n = 0
    for i in word_tokenize(header):
        n += 1
        if ("type" in i or "properties" in i or "geometry" in i
            or "coordinates" in i):
            m += 1
    ratio = m / n
    return ratio >= 0.2


def edit_reduced(result_directory):
    """
    Function to extract smaller dataframe from the edited one
    :param result_directory: string
    :return: pandas Dataframe
    """

    # paths
    path_log = os.path.join(result_directory, "log_final")
    path_reduced = os.path.join(result_directory, "log_final_reduced")

    # get data
    df_log = pd.read_csv(path_log, header=0, encoding="utf-8", sep=";",
                         index_col=False)

    # reduced data log
    df = df_log[["matrix_name", "file_name", "source_file", "n_row", "n_col",
                 "integer", "float", "object", "header", "header_name",
                 "zipfile", "id_page", "title_page", "title_producer",
                 "tags_page"]]
    df_log["geojson"] = df_log["header_name"].apply(func=_is_geojson)
    df["extension"] = df_log.apply(func=_refactor_extension, axis=1)

    print("reduced shape :", df.shape, "\n")
    print(df["extension"].value_counts(), "\n")
    df.to_csv(path_reduced, sep=";", encoding="utf-8", index=False, header=True)

    return


def main(result_directory, metadata_directory):
    """
    Function to run all the script
    :param result_directory: string
    :param metadata_directory: string
    :return:
    """

    print("metadata edition...", "\n")

    # check current metadata
    path_log = os.path.join(result_directory, "log_cleaning")
    df = pd.read_csv(path_log, sep=";", encoding="utf-8", index_col=False)
    print("metadata shape :", df.shape)

    # get original metadata
    d = _get_original_metadata(metadata_directory)
    print("length d :", len(d), "\n")

    # initialize a text file
    initialization_file(result_directory)

    # edit metadata
    edit(result_directory, d)

    # reduce edit
    edit_reduced(result_directory)

    # check changes
    path_metadata_edited = os.path.join(result_directory, "log_final")
    df = pd.read_csv(path_metadata_edited, sep=";", encoding="utf-8",
                     index_col=False)
    print("metadata edited shape :", df.shape)
    path_metadata_edited_reduced = os.path.join(result_directory,
                                                "log_final_reduced")
    df = pd.read_csv(path_metadata_edited_reduced, sep=";", encoding="utf-8",
                     index_col=False)
    print("metadata edited_reduced shape :", df.shape)

    return


if __name__ == "__main__":

    # path
    result_directory = get_config_tag("result", "cleaning")
    metadata_directory = get_config_tag("output", "metadata")

    # run code
    main(result_directory=result_directory,
         metadata_directory=metadata_directory)
