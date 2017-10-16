# -*- coding: utf-8 -*-

""" Sandbox file to try, test and explore potential solutions """

# libraries
import os
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from toolbox.utils import (get_config_tag, load_sparse_csr, save_sparse_csr,
                           save_dictionary)
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("french")
print("\n")

# paths
result_directory = get_config_tag("result", "cleaning")
path_vocabulary = os.path.join(result_directory, "token_vocabulary")
path_count = os.path.join(result_directory, "count_normalized.npz")

# get data
matrix = load_sparse_csr(path_count)
df = pd.read_csv(path_vocabulary, header=0, encoding="utf-8", sep=";",
                 index_col=False)
df.sort_values(by="index", axis=0, ascending=True, inplace=True)
feature_names = list(df["word"])
print("matrix shape :", matrix.shape)
print(type(matrix))
print("features length :", len(feature_names))

df_count = pd.DataFrame(data=matrix.todense(), columns=feature_names)
print("df shape :", df_count.shape)

weight_features = (df_count.sum(axis=0, skipna=True))
print("weight length :", len(weight_features), "\n")

# stem
d_stem = defaultdict(lambda: [])
d_score = defaultdict(lambda: [])
features_stem = []
for i, feature in tqdm(enumerate(feature_names), desc="stemming"):
    if str(feature) == "nan":
        feature = "nan"
    stem = stemmer.stem(feature)
    d_stem[stem].append(feature)
    d_score[stem].append(weight_features[i])
    if stem in features_stem:
        continue
    else:
        features_stem.append(stem)

# unstem
features_unstem = []
for stem in tqdm(features_stem, desc="unstemming"):
    i = d_score[stem].index(max(d_score[stem]))
    unstem = d_stem[stem][i]
    features_unstem.append(unstem)

print("length features :", len(feature_names))
print("length stem features :", len(features_stem))
print("length unstem features :", len(features_unstem), "\n")

d_features = {}
for i, word in enumerate(features_unstem):
    d_features[word] = i
path_vocabulary = os.path.join(result_directory, "token_vocabulary_unstem_bis")
save_dictionary(d_features, path_vocabulary, ["word", "index"])
