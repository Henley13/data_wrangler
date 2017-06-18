# -*- coding: utf-8 -*-

""" Remove empty files """

# libraries
import os
import zipfile
import magic
import shutil
from tqdm import tqdm
from toolbox.utils import get_config_tag, get_config_trace
print("\n")


def remove_empty_files(input_directory):
    """
    Function to remove empty files from the input directory and store extensions
    :param input_directory: string
    :return:
    """
    print("remove empty files", "\n")

    # paths
    main_dir = os.path.dirname(input_directory)
    sum_extension_path = os.path.join(main_dir, "sum_extension")
    temp_path = os.path.join(main_dir, "temporary")

    # initialize temporary directory
    if os.path.isdir(temp_path):
        shutil.rmtree(temp_path)
        os.mkdir(temp_path)
    else:
        os.mkdir(temp_path)

    # initialize sum extension path
    if os.path.isfile(sum_extension_path):
        os.remove(sum_extension_path)
    with open(sum_extension_path, mode="wt", encoding="utf-8") as f:
        f.write("extension;zipfile")
        f.write("\n")

    # remove the empty files and store the extension
    n = 0
    for filename in tqdm(os.listdir(input_directory)):
        path = os.path.join(input_directory, filename)
        if os.path.getsize(path) == 0:
            os.remove(path)
            continue
        else:
            extension = magic.Magic(mime=True).from_file(path)
        if extension == "application/zip":
            try:
                z = zipfile.ZipFile(path)
                z.extractall(temp_path)
                for z_filename in tqdm(z.namelist(), desc="zipfile"):
                    z_path = os.path.join(temp_path, z_filename)
                    if os.path.isfile(z_path):
                        z_extension = magic.Magic(mime=True).from_file(z_path)
                        n += 1
                        with open(sum_extension_path, mode="at",
                                  encoding="utf-8") as f:
                            f.write(";".join([z_extension, "True"]))
                            f.write("\n")
            except zipfile.BadZipFile:
                continue
            shutil.rmtree(temp_path)
            os.mkdir(temp_path)
        else:
            n += 1
            with open(sum_extension_path, mode="at", encoding="utf-8") as f:
                f.write(";".join([extension, "False"]))
                f.write("\n")

    print("number of files to clean (zipfiles included) :", n, "\n")

    return n


def main(input_directory):
    """
    Function to run all the script
    :param input_directory: string
    :return:
    """
    return remove_empty_files(input_directory)

if __name__ == "__main__":

    get_config_trace()

    # parameters
    input_directory = get_config_tag("input", "cleaning")

    # run
    main(input_directory)
