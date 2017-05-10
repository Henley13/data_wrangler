#!/bin/python3
# coding: utf-8

""" Analyze text elements, extract topics and make a wordcloud."""

# libraries
import time
import os
import pandas as pd
from sklearn.decomposition import NMF
from wordcloud.wordcloud import WordCloud
import matplotlib.pyplot as plt
print("\n")


def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()
    return


def load_sparse_csr(path):
    import scipy.sparse as sp
    import numpy as np
    loader = np.load(path)
    return sp.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                         shape=loader['shape'])

# parameters
n_topics = 5
n_top_words = 20

# path
path_vocabulary = "../data/token_vocabulary"
path_tfidf = "../data/tfidf.npz"
path_graph = "../data/graphs"

start = time.clock()

# check graph directory exists
if os.path.isdir(path_graph):
    pass
else:
    os.mkdir(path_graph)

# load data
tfidf = load_sparse_csr(path_tfidf)
print("tfidf shape :", tfidf.shape)
df = pd.read_csv(path_vocabulary, header=0, encoding="utf-8", sep=";",
                 index_col=False)
print("number of unique word :", df.shape[0])
feature_names = list(df["word"])
print(feature_names)

print("\n", "#######################", "\n")

# NMF
print("NMF", "\n")
nmf = NMF(n_components=n_topics, random_state=1, alpha=.1, l1_ratio=.5)
nmf = nmf.fit(tfidf)
print_top_words(nmf, feature_names, n_top_words)

print("\n", "#######################", "\n")

# wordcloud
for topic_ind, topic in enumerate(nmf.components_):
    print("topic #%i" % topic_ind)
    print(" ".join([feature_names[i]
                    for i in topic.argsort()[:-n_top_words - 1:-1]]))
    d = {}
    for i in range(len(feature_names)):
        word = feature_names[i]
        weight = topic[i]
        if weight > 0:
            d[word] = weight
    wc = WordCloud(width=1000, height=500, margin=2, prefer_horizontal=0.9,
                   background_color='white', colormap="viridis")
    wc = wc.fit_words(d)
    plt.figure()
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title("Wordcloud topic %i" % topic_ind, fontweight="bold")
    ax = plt.gca()
    ttl = ax.title
    ttl.set_position([.5, 1.06])
    plt.show()
    path = os.path.join(path_graph, "topic %i.png" % topic_ind)
    wc.to_file(path)

print("\n", "#######################", "\n")
