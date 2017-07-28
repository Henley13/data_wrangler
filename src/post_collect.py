# -*- coding: utf-8 -*-

""" Check xml files and metadata collected then store them in a csv file. """

# libraries
import os
import pandas as pd
import hashlib as hash
import joblib
from collections import defaultdict
from warnings import warn
from tqdm import tqdm
from lxml import etree
from toolbox.utils import get_config_tag, get_path_cachedir
print("\n")

# memory cache
general_directory = get_config_tag("data", "general")
memory = joblib.Memory(cachedir=get_path_cachedir(general_directory), verbose=0)


def _initialization_files(output_directory):
    """
    Function to initialize the metadata csv files.

    Parameters
    ----------
    output_directory : str
        Path of the output directory

    Returns
    -------
    """
    # metadata dataset
    path_output = os.path.join(output_directory, "metadata_dataset.csv")
    if os.path.isfile(path_output):
        os.remove(path_output)
    with open(path_output, mode='wt', encoding='utf-8') as f:
        f.write("title_file;id_file;url_file;creation_file;publication_file;"
                "last_modification_file;format_file;url_destination_file;"
                "availability_file;description_file;id_page;title_page;"
                "url_page;url_api_page;license_page;creation_page;"
                "last_modification_page;last_resources_update_page;"
                "description_page;frequency_page;start_coverage_page;"
                "end_coverage_page;granularity_page;zones_page;geojson_page;"
                "title_producer;id_producer;url_producer;url_api_producer;"
                "tags_page;discussions_page;followers_page;views_page;"
                "nb_uniq_visitors_page;nb_visits_page;nb_hits_page;reuses_page")
        f.write("\n")

    # metadata organization
    path_output = os.path.join(output_directory, "metadata_organization.csv")
    if os.path.isfile(path_output):
        os.remove(path_output)
    with open(path_output, mode='wt', encoding='utf-8') as f:
        f.write("id_org;title_org;url_org;url_api_org;description_org;"
                "dataset_views_org;n_datasets_org;followers_org;members_org;"
                "resource_downloads_org;reuse_views_org;n_reuses_org;views_org;"
                "nb_uniq_visitors_org;nb_visits_org;permitted_reuses_org;"
                "nb_hits_org")
        f.write("\n")

    # metadata reuse
    path_output = os.path.join(output_directory, "metadata_page_reuse.csv")
    if os.path.isfile(path_output):
        os.remove(path_output)
    with open(path_output, mode='wt', encoding='utf-8') as f:
        f.write("title_page;id_page;url_page;url_api_page;id_reuse;title_reuse;"
                "url_reuse;url_api_reuse;creation_reuse;"
                "last_modification_reuse;type_reuse;description_reuse;"
                "tags_reuse;n_datasets_reuse;followers_reuse;views_reuse;"
                "nb_uniq_visitors_reuse;nb_visits_reuse;nb_hits_reuse")
        f.write("\n")

    return


def _get_row_dataset(table, tree):
    """
    Function to create a string row to save in a csv file from xml metadata.

    Parameters
    ----------
    table : lxml.etree object
        Relative to each file

    tree : lxml.etree object
        Relative to the entire page dataset

    Returns
    -------
    row : list of str
    """
    # metadata specific to the file...
    title_file = table.findtext("./title")
    id_file = table.findtext("./id")
    url_file = table.findtext("./url")
    creation_file = table.findtext("./creation")
    publication_file = table.findtext("./publication")
    last_modification_file = table.findtext("./last_modification")
    format_file = table.findtext("./format")
    url_destination_file = table.findtext("./url_destination")
    availability_file = table.findtext("./availability")
    description_file = table.findtext("./description")

    # ... and more general
    id_page = tree.findtext(".//id")
    title_page = tree.findtext(".//title")
    url_page = tree.findtext(".//url")
    url_api_page = tree.findtext(".//url_api")
    license_page = tree.findtext(".//license")
    creation_page = tree.findtext(".//creation")
    last_modification_page = tree.findtext(".//last_modification")
    last_resources_update_page = tree.findtext(".//last_resources_update")
    description_page = tree.findtext(".//description")
    frequency_page = tree.findtext(".//frequency")
    start_coverage_page = tree.findtext(".//start_coverage")
    end_coverage_page = tree.findtext(".//end_coverage")
    granularity_page = tree.findtext(".//granularity")
    zones_page = tree.findtext(".//zones")
    geojson_page = tree.findtext(".//geojson")
    title_producer = tree.findtext(".//organization/title")
    id_producer = tree.findtext(".//organization/id")
    url_producer = tree.findtext(".//organization/url")
    url_api_producer = tree.findtext(".//organization/url_api")
    tags = []
    for tag in tree.xpath("/metadata/tags/tag"):
        if tag.text is not None:
            tags.append(tag.text)
    tags_page = " ".join(tags).replace("-", "_")
    discussions_page = tree.findtext(".//discussions")
    followers_page = tree.findtext(".//followers")
    views_page = tree.findtext(".//views")
    nb_uniq_visitors_page = tree.findtext(".//nb_uniq_visitors")
    nb_visits_page = tree.findtext(".//nb_visits")
    nb_hits_page = tree.findtext(".//nb_hits")
    reuses_page = tree.findtext(".//reuses")

    # create a new row
    row = [title_file, id_file, url_file, creation_file, publication_file,
           last_modification_file, format_file, url_destination_file,
           availability_file, description_file,
           id_page, title_page, url_page, url_api_page, license_page,
           creation_page, last_modification_page,
           last_resources_update_page, description_page, frequency_page,
           start_coverage_page, end_coverage_page, granularity_page,
           zones_page, geojson_page, title_producer, id_producer,
           url_producer, url_api_producer, tags_page, discussions_page,
           followers_page, views_page, nb_uniq_visitors_page,
           nb_visits_page, nb_hits_page, reuses_page]
    row = [str(item).replace(";", ",").replace('\n', '')
                    .replace('\r', '').replace('"', "'")
           for item in row]

    return row


def _get_row_organization(tree):
    """
    Function to create a string row to save in a csv file from xml metadata.

    Parameters
    ----------
    tree : lxml.etree object
        Relative to the organization

    Returns
    -------
    row : list of str
    """
    # general metadata about the organization
    id = tree.findtext(".//id")
    title = tree.findtext(".//title")
    url = tree.findtext(".//url")
    url_api = tree.findtext(".//url_api")
    description = tree.findtext(".//description")
    dataset_views = tree.findtext(".//dataset_views")
    n_datasets = tree.findtext(".//datasets")
    followers = tree.findtext(".//followers")
    members = tree.findtext(".//members")
    resource_downloads = tree.findtext(".//resource_downloads")
    reuse_views = tree.findtext(".//reuse_views")
    reuses = tree.findtext(".//reuses")
    views = tree.findtext(".//views")
    nb_uniq_visitors = tree.findtext(".//nb_uniq_visitors")
    nb_visits = tree.findtext(".//nb_visits")
    permitted_reuses = tree.findtext(".//permitted_reuses")
    nb_hits = tree.findtext(".//nb_hits")

    # create a new row
    row = [id, title, url, url_api, description, dataset_views, n_datasets,
           followers, members, resource_downloads, reuse_views, reuses,
           views, nb_uniq_visitors, nb_visits, permitted_reuses, nb_hits]
    row = [str(item).replace(";", ",").replace('\n', '').replace('\r', '')
                    .replace('"', "'") for item in row]

    return row


def _get_row_reuse(page, tree):
    """
    Function to create a string row to save in a csv file from xml metadata.

    Parameters
    ----------
    page : lxml.etree object
        Relative to the page dataset

    tree : lxml.etree object
        relative to the entire reuse

    Returns
    -------
    row : list of str
    """
    # ... specific to the file
    title_page = page.findtext("./title")
    id_page = page.findtext("./id")
    url_page = page.findtext("./url")
    url_api_page = page.findtext("./url_api")

    # ... and more general
    id_reuse = tree.findtext(".//id")
    title_reuse = tree.findtext(".//title")
    url_reuse = tree.findtext(".//url")
    url_api_reuse = tree.findtext(".//url_api")
    creation_reuse = tree.findtext(".//creation")
    last_modification_reuse = tree.findtext(".//last_modification")
    type_reuse = tree.findtext(".//type")
    description_reuse = tree.findtext(".//description")
    tags = []
    for tag in tree.xpath("/metadata/tags/tag"):
        if tag.text is not None:
            tags.append(tag.text)
    tags_reuse = " ".join(tags).replace("-", "_")
    n_datasets_reuse = tree.findtext(".//datasets")
    followers_reuse = tree.findtext(".//followers")
    views_reuse = tree.findtext(".//views")
    nb_uniq_visitors_reuse = tree.findtext(".//nb_uniq_visitors")
    nb_visits_reuse = tree.findtext(".//nb_visits")
    nb_hits_reuse = tree.findtext(".//nb_hits")

    # create a new row
    row = [title_page, id_page, url_page, url_api_page, id_reuse, title_reuse,
           url_reuse, url_api_reuse, creation_reuse, last_modification_reuse,
           type_reuse, description_reuse, tags_reuse, n_datasets_reuse,
           followers_reuse, views_reuse, nb_uniq_visitors_reuse,
           nb_visits_reuse, nb_hits_reuse]
    row = [str(item).replace(";", ",").replace('\n', '')
                    .replace('\r', '').replace('"', "'")
           for item in row]

    return row


def _metadata_dataset(metadata_directory, output_directory):
    """
    Function to summarize the dataset metadata within a csv file.

    Parameters
    ----------
    metadata_directory : str
        Path of the metadata directory

    output_directory : str
        Path of the output directory

    Returns
    -------
    """
    # paths
    path_output = os.path.join(output_directory, "metadata_dataset.csv")

    # get metadata...
    for xml_file in tqdm(os.listdir(metadata_directory),
                         desc="load xml dataset"):
        path_xml_file = os.path.join(metadata_directory, xml_file)
        tree = etree.parse(path_xml_file)
        for table in tree.xpath("/metadata/tables/table"):
            row = _get_row_dataset(table, tree)

            # save the row
            with open(path_output, mode='at', encoding='utf-8') as f:
                f.write(";".join(row))
                f.write("\n")

    return


def _metadata_organization(metadata_directory, output_directory):
    """
    Function to summarize the organization metadata within a csv file.

    Parameters
    ----------
    metadata_directory : str
        Path of the metadata directory

    output_directory : str
        Path of the output directory

    Returns
    -------
    """
    # paths
    path_output = os.path.join(output_directory, "metadata_organization.csv")

    # get metadata...
    for xml_file in tqdm(os.listdir(metadata_directory),
                         desc="load xml organization"):
        path_xml_file = os.path.join(metadata_directory, xml_file)
        tree = etree.parse(path_xml_file)
        row = _get_row_organization(tree)

        # save the row
        with open(path_output, mode='at', encoding='utf-8') as f:
            f.write(";".join(row))
            f.write("\n")

    return


def _metadata_reuse(metadata_directory, output_directory):
    """
    Function to summarize the reuse metadata within a csv file.

    Parameters
    ----------
    metadata_directory : str
        Path of the metadata directory

    output_directory : str
        Path of the output directory

    Returns
    -------
    """
    # paths
    path_output = os.path.join(output_directory, "metadata_page_reuse.csv")

    # get metadata...
    for xml_file in tqdm(os.listdir(metadata_directory),
                         desc="load xml reuse"):
        path_xml_file = os.path.join(metadata_directory, xml_file)
        tree = etree.parse(path_xml_file)
        for page in tree.xpath("/metadata/datasets/dataset"):
            row = _get_row_reuse(page, tree)

            # save the row
            with open(path_output, mode='at', encoding='utf-8') as f:
                f.write(";".join(row))
                f.write("\n")

    return


@memory.cache()
def _check_uniqueness(df_dataset, df_organization, df_page_reuse):
    """
    Function to check the uniqueness of ids in the metadata and correct them.

    Parameters
    ----------
    df_dataset : pandas Dataframe
        Dataframe with the datasets metadata

    df_organization : pandas Dataframe
        Dataframe with the organization metadata

    df_page_reuse : pandas Dataframe
        Dataframe with the reuse metadata (a page per row)

    Returns
    -------
    df_dataset : pandas Dataframe
        Dataframe with the datasets metadata

    df_organization : pandas Dataframe
        Dataframe with the organization metadata

    df_page_reuse : pandas Dataframe
        Dataframe with the reuse metadata (a page per row)
    """

    # check page id uniqueness in reuse metadata
    if (len(df_page_reuse["id_page"].unique()) !=
            len(df_page_reuse["url_page"].unique())):
        warn("Page ids are not unique in reuse metadata")

    # check reuse id uniqueness in reuse metadata
    if (len(df_page_reuse["id_reuse"].unique()) !=
            len(df_page_reuse["url_reuse"].unique())):
        warn("Reuse ids are not unique in reuse metadata")

    # check organization id uniqueness in organization metadata
    if len(df_organization["id_org"].unique()) != df_organization.shape[0]:
        warn("Organization ids are not unique in organization metadata")

    # check file id uniqueness in dataset metadata
    if len(df_dataset["id_file"].unique()) != df_dataset.shape[0]:
        l_id_file = []
        for i in tqdm(range(df_dataset.shape[0]), desc="check uniqueness"):
            id_file = df_dataset.at[i, "id_file"]
            if id_file in l_id_file:
                url_file = df_dataset.at[i, "url_file"]
                id_file = hash.sha224(bytes(url_file, 'utf-8')).hexdigest()
            l_id_file.append(id_file)
            df_dataset.loc[i, "id_file"] = id_file
    if len(df_dataset["id_file"].unique()) != df_dataset.shape[0]:
        warn("File ids are still not unique in dataset metadata")

    # check page id uniqueness in dataset metadata
    if (len(df_dataset["id_page"].unique()) !=
            len(df_dataset["url_page"].unique())):
        warn("Page ids are not unique in dataset metadata")

    return df_dataset, df_organization, df_page_reuse


@memory.cache()
def _edit_reuses(df_dataset, df_page_reuse):

    # get metadata for reuse only
    reuse_columns = ["id_reuse", "title_reuse", "url_reuse", "url_api_reuse",
                     "creation_reuse", "last_modification_reuse", "type_reuse",
                     "description_reuse", "tags_reuse", "n_datasets_reuse",
                     "followers_reuse", "views_reuse", "nb_uniq_visitors_reuse",
                     "nb_visits_reuse", "nb_hits_reuse"]
    df_reuse = df_page_reuse[reuse_columns]
    df_reuse = df_reuse.drop_duplicates(subset=None, keep='first',
                                        inplace=False)

    # remove useless features from df_page_reuse
    df_page_reuse = df_page_reuse[["title_page", "id_page", "url_page",
                                   "id_reuse", "title_reuse", "url_reuse"]]

    # get reuses for each page
    d_reuses = defaultdict(lambda: "")
    d_nb_reuses = defaultdict(lambda: 0)
    for id_page in tqdm(df_page_reuse["id_page"].unique(), desc="edit reuse"):
        df = df_page_reuse.query("id_page == '%s'" %id_page)
        reuses = " ".join(list(df["id_reuse"]))
        d_reuses[id_page] = reuses
        d_nb_reuses[id_page] = len(list(df["id_reuse"]))

    # add reuses to the datasets metadata
    l_id_reuse = []
    l_n_reuse = []
    for i in tqdm(range(df_dataset.shape[0]), desc="edit reuse"):
        id_page = df_dataset.at[i, "id_page"]
        l_id_reuse.append(d_reuses[id_page])
        l_n_reuse.append(d_nb_reuses[id_page])
    df_dataset["reuses"] = l_id_reuse
    df_dataset["n_reuses"] = l_n_reuse

    return df_dataset, df_page_reuse, df_reuse


def main(metadata_directory, organization_directory, reuse_directory,
         output_directory):
    """
    Function to run all the script.

    Parameters
    ----------
    metadata_directory : str
        Path of the dataset metadata directory

    organization_directory : str
        Path of the organization metadata directory

    reuse_directory : str
        Path of the reuse metadata directory

    output_directory : str
        Path of the general data directory

    Returns
    -------
    """
    print("edit metadata", "\n")

    # collect metadata
    _initialization_files(output_directory)
    _metadata_dataset(metadata_directory, output_directory)
    _metadata_organization(organization_directory, output_directory)
    _metadata_reuse(reuse_directory, output_directory)
    print()

    # get data
    path_output = os.path.join(output_directory, "metadata_dataset.csv")
    df_dataset = pd.read_csv(path_output, sep=";", encoding="utf-8",
                             index_col=False)
    path_output = os.path.join(output_directory, "metadata_organization.csv")
    df_organization = pd.read_csv(path_output, sep=";", encoding="utf-8",
                                  index_col=False)
    path_output = os.path.join(output_directory, "metadata_page_reuse.csv")
    df_page_reuse = pd.read_csv(path_output, sep=";", encoding="utf-8",
                                index_col=False)

    # merge metadata
    df_dataset, df_organization, df_page_reuse = _check_uniqueness(
        df_dataset, df_organization, df_page_reuse)
    df_dataset, df_page_reuse, df_reuse = _edit_reuses(df_dataset,
                                                       df_page_reuse)

    # save metadata
    path_output = os.path.join(output_directory, "metadata_dataset.csv")
    df_dataset.to_csv(path_output, sep=';', encoding='utf-8', index=False,
                      header=True)
    path_output = os.path.join(output_directory, "metadata_organization.csv")
    df_organization.to_csv(path_output, sep=';', encoding='utf-8', index=False,
                           header=True)
    path_output = os.path.join(output_directory, "metadata_page_reuse.csv")
    df_page_reuse.to_csv(path_output, sep=';', encoding='utf-8', index=False,
                         header=True)
    path_output = os.path.join(output_directory, "metadata_reuse.csv")
    df_reuse.to_csv(path_output, sep=';', encoding='utf-8', index=False,
                    header=True)

    print("df dataset shape :", df_dataset.shape, "\n")
    print(df_dataset.columns, "\n")
    print("------------", "\n")
    print("df organization shape :", df_organization.shape, "\n")
    print(df_organization.columns, "\n")
    print("------------", "\n")
    print("df page reuse shape :", df_page_reuse.shape, "\n")
    print(df_page_reuse.columns, "\n")
    print("------------", "\n")
    print("df reuse shape :", df_reuse.shape, "\n")
    print(df_reuse.columns, "\n")

    return

if __name__ == "__main__":

    # paths
    general_directory = get_config_tag("data", "general")
    metadata_directory = get_config_tag("output", "metadata")
    organization_directory = get_config_tag("output_organization", "metadata")
    reuse_directory = get_config_tag("output_reuse", "metadata")

    # run the script
    main(metadata_directory=metadata_directory,
         organization_directory=organization_directory,
         reuse_directory=reuse_directory,
         output_directory=general_directory)
