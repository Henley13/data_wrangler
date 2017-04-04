#!/bin/python3
# coding: utf-8

""" Detect the file extension and reshape it. """

# libraries
import os
from clean_files_functions import cleaner
import matplotlib.pyplot as plt
from functions import log_error
from numpy import percentile
print("\n")

input_directory = "../data/data_collected_csv"
output_directory = "../data/data_fitted"
path_error = "../data/fit_errors"
n_files = len(os.listdir(input_directory))
print("number of files :", n_files, "\n")

# reset the log
if os.path.isdir(path_error):
    for file in os.listdir(path_error):
        os.remove(os.path.join(path_error, file))
else:
    os.mkdir(path_error)

# check output directory exists
if not os.path.isdir(output_directory):
    os.mkdir(output_directory)

# clean files
n_rows = []
n_cols = []
dico_types = {}
n = 0
for filename in os.listdir(input_directory):
    print("file number", n, ":", filename)
    try:
        res = cleaner(filename, input_directory, output_directory)
    except:
        res = None
        log_error(os.path.join(path_error, filename), [filename])
    if res is not None:
        x, y, d = res
        n_rows.append(x)
        n_cols.append(y)
        for i in d:
            if i not in dico_types:
                dico_types[i] = d[i]
            else:
                dico_types[i] += d[i]

# print results
print("\n")
for i in dico_types:
    print(i, dico_types[i])

print("\n")
print("row distribution")
for i in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
    print(i, percentile(n_rows, i))

print("\n")
print("col distribution")
for i in [10, 20, 30, 40, 50, 60, 70, 80, 90]:
    print(i, percentile(n_cols, i))

# graphs
plt.scatter(x=n_cols, y=n_rows, alpha=0.5, c="b")
plt.title("Files shape (n = " + str(len(n_rows)) + ")")
plt.xlabel("Columns")
plt.ylabel("Rows")
plt.savefig('fig_1')

plt.hist(x=n_rows, bins=20000, alpha=0.5)
plt.xlim(0, 3000)
plt.xlabel("Number of rows")
plt.ylabel("Frequency")
plt.title("Rows histogram (n = " + str(len(n_rows)) + ")")
plt.savefig('fig_2')

plt.hist(x=n_cols, bins=200, alpha=0.5)
plt.xlim(0, 100)
plt.xlabel("Number of columns")
plt.ylabel("Frequency")
plt.title("Columns histogram (n = " + str(len(n_rows)) + ")")
plt.savefig('fig_3')
