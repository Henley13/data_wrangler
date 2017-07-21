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


def _check_directory(output_directory):
    """
    Function to initialize the output directory
    :param output_directory: string
    :return:
    """
    if os.path.isdir(output_directory):
        shutil.rmtree(output_directory)
        os.mkdir(output_directory)
    else:
        os.mkdir(output_directory)
    return


def extraction(input_directory, output_directory, n_sample):
    """
    Function to randomly copy some files in a new directory
    :param input_directory: string
    :param output_directory: string
    :param n_sample: integer
    :return:
    """
    print("sample extraction", "\n")

    # sample
    files = os.listdir(input_directory)
    sample = random.sample(files, n_sample)
    print("number of files extracted :", n_sample, "\n")

    # copy
    for filename in tqdm(sample):
        path_in = os.path.join(input_directory, filename)
        path_out = os.path.join(output_directory, filename)
        shutil.copyfile(path_in, path_out)

    return


def main(input_directory, output_directory, n_sample=200):
    """
    Function to run all the script
    :param input_directory: string
    :param output_directory: string
    :param n_sample: integer
    :return:
    """
    _check_directory(output_directory)
    extraction(input_directory, output_directory, n_sample)
    return

if __name__ == "__main__":

    # paths
    input_directory = get_config_tag("output", "download")
    output_directory = os.path.join(os.path.dirname(input_directory),
                                    "sample_2")

    # run
    main(input_directory, output_directory)








