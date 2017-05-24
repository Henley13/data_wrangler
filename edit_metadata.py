#!/bin/python3
# coding: utf-8

""" We edit metadata with new information inferred from the file. """

# libraries
import os
import pandas as pd
from lxml import etree
from toolbox.utils import get_config_tag
print("\n")

# path
result_directory = get_config_tag("result", "cleaning")
metadata_directory = get_config_tag("output", "metadata")
data_directory = os.path.join(result_directory, "data_fitted")
path_log = os.path.join(result_directory, "log_cleaning")
path_metadata_edited = os.path.join(result_directory, "log_final")
metadata_clean_directory = os.path.join(result_directory, "metadata_cleaning")


def get_original_metadata(metadata_directory):
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

# get metadata
d = get_original_metadata(metadata_directory)
print("length d :", len(d))

# initialize a text file
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

# gather metadata
df_log = pd.read_csv(path_log, sep=";", encoding="utf-8", index_col=False)
for i in range(df_log.shape[0]):

    if i % 100 == 0:
        print(i)

    matrix = df_log.at[i, "matrix_name"]
    # get metadata extracted from the matrix
    path_matrix = os.path.join(data_directory, matrix)
    size_matrix = os.path.getsize(path_matrix)

    # get metadata extracted from the cleaning process
    if df_log.at[i, "metadata"]:
        path_metadata_matrix = os.path.join(metadata_clean_directory, matrix)
        with open(path_metadata_matrix, mode='rt', encoding='utf-8') as f:
            commentary_matrix = f.read()
    else:
        commentary_matrix = ""

    # get metadata extracted from data.gouv.fr
    try:
        table, tree = d[df_log.at[i, "source_file"]]
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
         df_log.at[i, "n_row"], df_log.at[i, "n_col"], df_log.at[i, "integer"],
         df_log.at[i, "float"], df_log.at[i, "object"],
         df_log.at[i, "metadata"], df_log.at[i, "time"],
         df_log.at[i, "header"], df_log.at[i, "multiheader"],
         df_log.at[i, "header_name"], df_log.at[i, "extension"],
         df_log.at[i, "zipfile"], size_matrix, commentary_matrix,
         title_file, id_file, url_file, url_destination_file, description_file,
         creation_file, publication_file, last_modification_file,
         availability_file, title_page, id_page, url_page, url_api_page,
         license_page, description_page, creation_page, last_modification_page,
         last_resources_update_page, frequency_page, start_coverage_page,
         end_coverage_page, granularity_page, zones_page, geojson_page,
         title_producer, url_producer, url_api_producer, tags_page]
    l = [str(j).replace(";", ",").replace('\n', '').replace('\r', '')
               .replace('"', "'") for j in l]
    with open(path_metadata_edited, mode='at', encoding='utf-8') as f:
        f.write(";".join(l))
        f.write("\n")

# load data
df = pd.read_csv(path_metadata_edited, sep=";", encoding="utf-8",
                 index_col=False)
print("df shape :", df.shape)
