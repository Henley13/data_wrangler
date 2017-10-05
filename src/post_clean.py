# -*- coding: utf-8 -*-

""" Check cleaned files then store a summary in a csv file. """

# libraries
import os
import shutil
import pandas as pd
from collections import defaultdict
from toolbox.utils import get_config_tag
print("\n")


def remove_temporary(directory):
    """
    Function to remove the temporary directories from the file directory.

    Parameters
    ----------
    directory: str
        Path of the file directory

    Returns
    -------
    """
    print("closed temporary directories", "\n")

    for file in os.listdir(directory):
        path = os.path.join(directory, file)
        if os.path.isfile(path):
            pass
        else:
            if file[0:3] == "tmp":
                shutil.rmtree(path)
    return


def refactor_extension(row):
    """
    Function to rename extension values
    :param row: row from a pandas Dataframe
    :return: string
    """
    if row["extension"] == "text/plain":
        return "csv"
    elif row["extension"] == "application/CDFV2-unknown":
        return "cdfv2"
    elif row["extension"] == "application/octet-stream":
        return "other"
    elif row["extension"] in ["application/vnd.ms-excel",
                              "application/vnd.openxmlformats-officedocument."
                              "spreadsheetml.sheet"]:
        return "excel"
    elif row["extension"] == "text/xml":
        return "xml"
    elif row["extension"] == "application/zip":
        return "zipfile"
    else:
        return row["extension"]


def collect_cleaning_metadata(log_directory):
    """
    Function to collect the metadata extracted from the cleaning process and
    store them in a dataframe.

    Parameters
    ----------
    log_directory: str
        Path of the log directory

    Returns
    -------
    df_log: pandas Dataframe
        Dataframe with the metadata of the cleaning process
    """
    print("collect cleaning log", "\n")

    d = defaultdict(lambda: [])

    # for each file, we read its log
    for log in os.listdir(log_directory):
        path = os.path.join(log_directory, log)
        with open(path, mode='rt', encoding="utf-8") as f:
            c = f.readlines()
            d["matrix_name"].append(c[1].strip())
            d["file_name"].append(c[2].strip())
            d["source_file"].append(c[3].strip())
            d["x"].append(c[4].strip())
            d["y"].append(c[5].strip())
            d["integer"].append(c[6].strip())
            d["float"].append(c[7].strip())
            d["object"].append(c[8].strip())
            d["extradata"].append(c[9].strip())
            d["header"].append(c[10].strip())
            d["multiheader"].append(c[11].strip())
            d["header_name"].append(c[12].strip())
            d["extension"].append(c[13].strip())
            d["zipfile"].append(c[14].strip())

    # we store everything in a dataframe
    names = ["matrix_name", "file_name", "source_file", "x", "y", "integer",
             "float", "object", "extradata",
             "header", "multiheader", "header_name", "extension", "zipfile"]
    df_log = pd.DataFrame(d, columns=names)

    return df_log


def merge_metadata_cleaned(df_log, df_log_download, path_log, path_log_reduced):
    """
    Function to merge the metadata collected about the cleaned files.

    Parameters
    ----------
    df_log: pandas Dataframe
        Dataframe with the metadata from the cleaning process

    df_log_download: pandas Dataframe
        Dataframe with the previous metadata

    path_log: str
        Path of the output file

    path_log_reduced: str
        Path of the reduced output file

    Returns
    -------
    df = pandas Dataframe
        Dataframe with all the metadata

    df_reduced = pandas Dataframe
        A reduced version of df
    """
    print("merge metadata", "\n")

    # merge data
    df = df_log.merge(df_log_download,
                      how='left',
                      left_on='source_file',
                      right_on='id_file',
                      left_index=True,
                      right_index=False,
                      copy=False)

    # we keep the wanted columns
    names = ['matrix_name', 'file_name', 'source_file',
             'x', 'y', 'integer', 'float', 'object',
             'extradata', 'header', 'multiheader', 'header_name',
             'extension', 'zipfile',
             'title_file', 'url_file', 'creation_file', 'publication_file',
             'last_modification_file', 'format_file',
             'description_file', 'id_page', 'title_page', 'url_page',
             'url_api_page', 'license_page', 'creation_page',
             'last_modification_page', 'last_resources_update_page',
             'description_page', 'frequency_page', 'start_coverage_page',
             'end_coverage_page', 'granularity_page', 'zones_page',
             'geojson_page',
             'title_producer', 'id_producer', 'url_producer',
             'url_api_producer',
             'tags_page', 'discussions_page', 'followers_page', 'views_page',
             'nb_uniq_visitors_page', 'nb_visits_page', 'nb_hits_page',
             'reuses_page', 'reuses', 'n_reuses']
    df = df[names]
    df["extension"] = df.apply(func=refactor_extension, axis=1)

    # we shape a reduced dataframe
    names_reduced = ['matrix_name', 'file_name', 'source_file',
                     'x', 'y', 'integer', 'float', 'object',
                     'extradata', 'header', 'multiheader', 'header_name',
                     'extension', 'zipfile', 'format_file',
                     'id_page', 'title_page', 'url_page', 'frequency_page',
                     'start_coverage_page', 'end_coverage_page',
                     'granularity_page', 'zones_page', 'geojson_page',
                     'title_producer', 'id_producer', 'url_producer',
                     'tags_page', 'reuses_page', 'reuses', 'n_reuses']
    df_reduced = df[names_reduced]

    df.to_csv(path_log, sep=";", encoding="utf-8", index=False, header=True)
    df_reduced.to_csv(path_log_reduced, sep=";", encoding="utf-8", index=False,
                      header=True)

    print("df shape :", df.shape, "\n")
    print("df reduced shape:", df_reduced, "\n")
    print(df_reduced.columns, "\n")

    return df, df_reduced


def collect_error(error_directory):
    """
    Function to collect the errors from the cleaning process

    Parameters
    ----------
    error_directory: str
        Path of the error directory

    Returns
    -------
    df_error = pandas Dataframe
        Dataframe with the error from the cleaning process
    """
    print("collect errors", "\n")

    # for each error log we collect it...
    source_file = []
    file_name = []
    matrix_name = []
    zipfile = []
    extension = []
    error_type = []
    content = []
    for error_file in os.listdir(error_directory):
        path = os.path.join(error_directory, error_file)
        with open(path, mode='rt', encoding="utf-8") as f:
            c = f.readlines()
            source_file.append(c[1].strip())
            file_name.append(c[2].strip())
            matrix_name.append(c[3].strip())
            zipfile.append(c[4].strip())
            extension.append(c[5].strip())
            error_type.append(c[6].strip().split(" ")[0])
            content.append(c[-2].strip())

    # ... then we store everything in a dataframe
    df_error = pd.DataFrame({"source_file": source_file,
                             "file_name": file_name,
                             "matrix_name": matrix_name,
                             "zipfile": zipfile,
                             "extension": extension,
                             "error_type": error_type,
                             "content": content})

    return df_error


def merge_metadata_error(df_log_download, df_error, path_error):
    """
    Function to merge the previous metadata and the error logs collected during
    the cleaning process.

    Parameters
    ----------
    df_log_download: pandas Dataframe
        Dataframe with the previous metadata

    df_error: pandas Dataframe
        Dataframe with the error logs

    path_error: str
        Path for the output file

    Returns
    -------
    df: pandas Dataframe
        Output from the merge
    """
    print("merge errors metadata", "\n")

    # merge data
    df = df_error.merge(df_log_download,
                        how='left',
                        left_on='source_file',
                        right_on='id_file',
                        left_index=True,
                        right_index=False,
                        copy=False)

    names = ['content', 'error_type', 'extension', 'file_name', 'matrix_name',
             'source_file', 'zipfile',
             'title_file', 'url_file',
             'creation_file', 'publication_file', 'last_modification_file',
             'format_file',
             'description_file', 'id_page', 'title_page', 'url_page',
             'url_api_page',
             'license_page', 'creation_page', 'last_modification_page',
             'last_resources_update_page', 'description_page', 'frequency_page',
             'start_coverage_page', 'end_coverage_page', 'granularity_page',
             'zones_page', 'geojson_page', 'title_producer', 'id_producer',
             'url_producer', 'url_api_producer', 'tags_page',
             'discussions_page',
             'followers_page', 'views_page', 'nb_uniq_visitors_page',
             'nb_visits_page', 'nb_hits_page', 'reuses_page', 'reuses',
             'n_reuses',
             'size_file']
    df = df[names]
    df["extension"] = df.apply(func=refactor_extension, axis=1)

    df.to_csv(path_error, sep=";", encoding="utf-8", index=False, header=True)

    print("df_error shape:", df.shape, "\n")

    return df


def main(general_directory, result_directory):

    # paths
    cleaned_data_directory = os.path.join(result_directory, "data_fitted")
    error_log_directory = os.path.join(result_directory, "error_cleaning")
    log_directory = os.path.join(result_directory, "log_cleaning")
    extradata_directory = os.path.join(result_directory, "extradata_cleaning")

    # remove the temporary directories
    remove_temporary(cleaned_data_directory)
    print("total number of files cleaned (excel sheets included) :",
          len(os.listdir(cleaned_data_directory)))
    print("total number of files saved in the log :",
          len(os.listdir(log_directory)))
    print("total number of extra data :", len(os.listdir(extradata_directory)))
    print("total number of failures :", len(os.listdir(error_log_directory)),
          "\n")

    # we collect the metadata from the cleaning process
    df_log = collect_cleaning_metadata(log_directory)

    # we get the previous metadata
    summary_download_path = os.path.join(general_directory, "summary_download")
    df_log_download = pd.read_csv(summary_download_path, sep=";",
                                  encoding="utf-8", index_col=False)

    # merge the metadata for the cleaned files
    path_log = os.path.join(result_directory, "log_final")
    path_log_reduced = os.path.join(result_directory, "log_final_reduced")
    df, df_reduced = merge_metadata_cleaned(df_log, df_log_download, path_log,
                                            path_log_reduced)

    # collect error data and merge them with the previous metadata
    path_error = os.path.join(result_directory, "log_error")
    df_error_cleaning = collect_error(error_log_directory)
    df_error = merge_metadata_error(df_log_download, df_error_cleaning,
                                    path_error)

    # print statistics
    print("distribution extension cleaned :")
    print(df_reduced["extension"].value_counts(), "\n")
    print("supposed distribution extension failed :")
    print(df_error["extension"].value_counts(), "\n")

    return


if __name__ == "__main__":

    # paths
    general_directory = get_config_tag("data", "general")
    result_directory = get_config_tag("result", "cleaning")

    # run code
    main(general_directory=general_directory,
         result_directory=result_directory)
