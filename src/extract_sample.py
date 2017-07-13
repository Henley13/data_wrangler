# -*- coding: utf-8 -*-

""" Extract a sample from the collected data in order to easily profile the
    code """

# libraries
import os
import shutil
import random
from tqdm import tqdm
from toolbox.utils import get_config_tag
random.seed(13)
print("\n")


def initialize_folder(output_directory):
    """
    Function to initialize the output folder
    :param output_directory: string
    :return:
    """
    if os.path.isdir(output_directory):
        shutil.rmtree(output_directory)
        os.mkdir(output_directory)
    else:
        os.mkdir(output_directory)
    return


def extraction(data_directory, output_directory, n_sample):
    """
    Function to randomly copy some files in a new folder
    :param data_directory: string
    :param output_directory: string
    :param n_sample: integer
    :return:
    """
    print("sample extraction", "\n")

    # sample
    files = os.listdir(data_directory)
    sample = random.sample(files, n_sample)
    print("number of files extracted :", n_sample, "\n")

    # copy
    for filename in tqdm(sample):
        path_in = os.path.join(data_directory, filename)
        path_out = os.path.join(output_directory, filename)
        shutil.copyfile(path_in, path_out)

    return


def main(data_directory, output_directory, n_sample=500):
    """
    Function to run all the script
    :param data_directory: string
    :param output_directory: string
    :param n_sample: integer
    :return:
    """
    initialize_folder(output_directory)
    extraction(data_directory, output_directory, n_sample)
    return

if __name__ == "__main__":

    # paths
    data_directory = get_config_tag("output", "download")
    output_directory = os.path.join(os.path.dirname(data_directory), "sample")

    # run
    main(data_directory, output_directory)








