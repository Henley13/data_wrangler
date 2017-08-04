# -*- coding: utf-8 -*-

""" Count empty files and check download results. """

# libraries
import os
import magic
import pandas as pd
import joblib
from tqdm import tqdm
from toolbox.utils import get_config_tag, get_config_trace, get_path_cachedir
print("\n")

# memory cache
general_directory = get_config_tag("data", "general")
memory = joblib.Memory(cachedir=get_path_cachedir(general_directory), verbose=0)


def _analyze_download(download_directory):
    """
    Function to check the different files successfully downloaded.

    Parameters
    ----------
    download_directory : str
        Path of the downloaded files directory

    Returns
    -------
    df_download_summary : pandas Dataframe
        Dataframe of the download logs
    """
    print("analyze downloaded files", "\n")

    # check downloaded files
    l_filename = []
    l_size = []
    l_extension = []
    for filename in tqdm(os.listdir(download_directory)):
        path_file = os.path.join(download_directory, filename)
        size_file = os.path.getsize(path_file)
        if size_file == 0:
            extension_file = None
        else:
            extension_file = magic.Magic(mime=True).from_file(path_file)

        l_filename.append(filename)
        l_size.append(size_file)
        l_extension.append(extension_file)

    # store results in a Dataframe
    df_download_summary = pd.DataFrame({"id_file": l_filename,
                                        "size_file": l_size,
                                        "extension_file": l_extension})
    df_download_summary["downloaded"] = [True] * df_download_summary.shape[0]

    print()

    return df_download_summary


def _initialize_log_error(general_directory):
    """
    Function to initialize the csv file with a summary of the download errors.

    Parameters
    ----------
    general_directory : str
        Path to the general data directory

    Returns
    -------
    summary_download_path : str
        Path of the csv file
    """
    summary_error_path = os.path.join(general_directory,
                                      "summary_download_error")
    if os.path.isfile(summary_error_path):
        os.remove(summary_error_path)
    with open(summary_error_path, mode="wt", encoding="utf-8") as f:
        f.write("id_file;error;content_error;url_file;format_file")
        f.write("\n")
    return summary_error_path


def _collect_error(error_directory, summary_error_path):
    """
    Function to count the different errors that occurred during the downloading.

    Parameters
    ----------
    error_directory : str
        Path of the error directory

    summary_error_path : str
        Path of the error summary file

    Returns
    -------
    df_error_summary : pandas Dataframe
        Dataframe of the error logs
    """

    print("collect download errors", "\n")

    # check errors
    for filename in tqdm(os.listdir(error_directory)):
        path = os.path.join(error_directory, filename)
        with open(path, mode="rt", encoding="utf-8") as f:
            c = f.readlines()
            url = c[1].strip()
            filename_saved = c[2].strip()
            format = c[3].strip()
            error = c[4].split(" ")[0]
            content = c[-2].strip()
            if filename != filename_saved:
                raise ValueError("A wrong filename has been saved!")
        with open(summary_error_path, mode="at", encoding="utf-8") as f:
            f.write(";".join([str(filename), str(error), str(content),
                              str(url), str(format)]))
            f.write("\n")

    df_error_summary = pd.read_csv(summary_error_path, sep=";",
                                   encoding="utf-8", index_col=False)

    print()

    return df_error_summary


@memory.cache()
def _edit_metadata(df_dataset, df_download_summary, df_error_summary):
    """
    Function to merge the different dataframes created.

    Parameters
    ----------
    df_dataset : pandas Dataframe
        Dataframe of the metadata collected

    df_download_summary : pandas Dataframe
        Dataframe of the download logs

    df_error_summary : pandas Dataframe
        Dataframe of the error logs

    Returns
    -------
    df : pandas Dataframe
        Dataframe merged
    """
    print("merge dataframes", "\n")

    # merge data
    df = df_dataset.merge(df_download_summary,
                          how='left',
                          left_on='id_file',
                          right_on='id_file',
                          left_index=True,
                          right_index=False,
                          copy=False)
    df_error = df_error_summary[["id_file", "error", "content_error"]]
    df = df.merge(df_error,
                  how='left',
                  left_on='id_file',
                  right_on='id_file',
                  left_index=True,
                  right_index=False,
                  copy=False)

    # fill nan value
    df["downloaded"].fillna(False, inplace=True)

    # check consistency between downloaded files and errors
    if df.query("downloaded == True & error == error").shape[0] > 0:
        pass
    elif df.query("downloaded == False & error != error").shape[0] > 0:
        pass
    else:
        print(df.query("downloaded == True & error == error"))
        raise IndexError("some files are both in the download and error "
                         "directories")

    # save results
    summary_download_path = "../data/summary_download"
    df.to_csv(summary_download_path, sep=";", encoding="utf-8",
              index=False, header=True)

    n_files = df.shape[0]
    n_remote = df.query("url_destination_file != 'file'").shape[0]
    n_downloads = df.query("downloaded == True").shape[0]
    n_errors = df.query("error == error").shape[0]
    n_empty = df.query("size_file == 0").shape[0]
    print("total number of file registered :", n_files)
    print("number of remote files :", n_remote)
    print("number of downloaded files :", n_downloads)
    print("number of empty downloaded files :", n_empty)
    print("number of error :", n_errors)
    print("number of ignored files :",
          n_files - n_remote - n_downloads - n_errors, "\n")

    return df


def _analyze_error(df_download):
    """
    Function to analyze errors that happened during the download step.

    Parameters
    ----------
    df_download : pandas Dataframe
        Dataframe with the edited metadata (post download)

    Returns
    -------
    """
    print("analyze results", "\n")

    # keep only the data about the errors
    query = "url_destination_file == 'file'"
    df_download_attempt = df_download.query(query)
    query = "url_destination_file == 'file' & error == error"
    df_download_error = df_download.query(query)

    # general information
    print("download attempts", "\n")
    print(df_download_attempt['format_file'].value_counts(), "\n")

    print("--------------------", "\n")

    # general information
    print("errors", "\n")
    print(df_download_error['format_file'].value_counts(), "\n")
    print(df_download_error['error'].value_counts(), "\n")

    print("--------------------", "\n")

    # specific information per inferred extension
    extensions = list(set(list(df_download_error["format_file"])))
    print(extensions)
    for ext in extensions:
        print("declared extension :", ext, "\n")
        if str(ext) == "nan":
            query = "format_file != format_file"
        else:
            query = "format_file == '%s'" % ext
        df_error_ext = df_download_error.query(query)
        print(df_error_ext["error"].value_counts(), "\n")
        max_e = df_error_ext["error"].value_counts().index.tolist()[0]
        print(df_error_ext.query("error == '%s'" % max_e)["content_error"].
              value_counts(), "\n")
        print("---", "\n")

    # efficiency rate per extension
    efficiency = {}
    extensions = list(set(list(df_download_attempt["format_file"])))
    for ext in extensions:
        if str(ext) == "nan":
            query_success = "format_file != format_file & error != error"
            query_fail = "format_file != format_file & error == error"
            ext = "nan"
        else:
            query_success = "format_file == '%s' & error != error" % ext
            query_fail = "format_file == '%s' & error == error" % ext
        n_success = df_download_attempt.query(query_success).shape[0]
        n_fail = df_download_attempt.query(query_fail).shape[0]
        if n_success + n_fail == 0:
            efficiency[ext] = 0.0
        else:
            efficiency[ext] = round(n_success / (n_success + n_fail) * 100, 2)
    for ext in sorted(efficiency.items(), reverse=True, key=lambda x: x[1]):
        print(efficiency[ext], "% ==>", ext)
    print("\n")

    # efficiency rate per producer
    efficiency = {}
    producers = list(set(list(df_download_attempt["title_producer"])))
    for producer in producers:
        if str(producer) == "nan":
            query_success = "title_producer != title_producer & error != error"
            query_fail = "title_producer != title_producer & error == error"
            producer = "nan"
        else:
            query_success = "title_producer == '%s' & error != error" % producer
            query_fail = "title_producer == '%s' & error == error" % producer
        n_success = df_download_attempt.query(query_success).shape[0]
        n_fail = df_download_attempt.query(query_fail).shape[0]
        if n_success + n_fail == 0:
            efficiency[producer] = 0.0
        else:
            efficiency[producer] = round(
                n_success / (n_success + n_fail) * 100, 2)
    for producer in sorted(efficiency.items(), reverse=True,
                           key=lambda x: x[1]):
        print(efficiency[producer], "% ==>", producer)
    print("\n")

    return


def main(general_directory, input_directory, error_directory):
    """
    Function to run all the script.

    Parameters
    ----------
    general_directory : str
        Path of the general data directory

    input_directory : str
        Path of the collected data directory

    error_directory : str
        Path of the download errors directory

    Returns
    -------
    """
    # analyze downloaded files
    df_download_summary = _analyze_download(input_directory)

    # collect errors
    summary_error_path = _initialize_log_error(general_directory)
    df_error_summary = _collect_error(error_directory, summary_error_path)

    # merge results
    path_df_dataset = os.path.join(general_directory, "metadata_dataset.csv")
    df_dataset = pd.read_csv(path_df_dataset, sep=";", encoding="utf-8",
                             index_col=False)
    df = _edit_metadata(df_dataset, df_download_summary, df_error_summary)

    # analyze errors
    _analyze_error(df)

    return


if __name__ == "__main__":

    get_config_trace()

    # paths
    general_directory = get_config_tag("data", "general")
    input_directory = get_config_tag("output", "download")
    error_directory = get_config_tag("error", "download")

    # run
    main(general_directory=general_directory,
         input_directory=input_directory,
         error_directory=error_directory)
