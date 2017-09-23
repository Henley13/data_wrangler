# -*- coding: utf-8 -*-

""" Extract metadata from reuses. """

# libraries
import os
import pandas as pd
from lxml import etree
from tqdm import tqdm
from collections import defaultdict, Counter
from toolbox.utils import get_config_tag
print("\n")

# paths
result_directory = get_config_tag("result", "cleaning")
reuse_directory = get_config_tag("output_reuse", "metadata")
path_log = os.path.join(result_directory, "log_final_reduced")
path_output = os.path.join(result_directory, "log_final_reduced_with_reuse")
path_reuse = os.path.join(result_directory, "log_reuse")

# get reuse metadata
d = defaultdict(lambda: [])
for xml_file in tqdm(os.listdir(reuse_directory)):
    tree = etree.parse(os.path.join(reuse_directory, xml_file))
    for page in tree.xpath("/metadata/datasets/dataset"):
        d[page.findtext("id")].append((page, tree))

print("number of unique pages :", len(d), "\n")

# TODO check unicity id
# get id and title for each reuse
l_id_reuse = []
l_title_reuse = []
l_type_reuse = []
for xml_file in tqdm(os.listdir(reuse_directory)):
    tree = etree.parse(os.path.join(reuse_directory, xml_file))
    id_reuse = tree.findtext(".//id")
    title_reuse = tree.findtext(".//title").replace(";", "")
    type_reuse = tree.findtext(".//type")
    l_id_reuse.append(id_reuse)
    i = 0
    while title_reuse in l_title_reuse:
        n = Counter(l_title_reuse)[title_reuse]
        title_reuse = title_reuse + " v" + str(n + i + 1)
        i += 1
    l_title_reuse.append(title_reuse)
    l_type_reuse.append(type_reuse)
df_reuse = pd.DataFrame({"id_reuse": l_id_reuse,
                         "title_reuse": l_title_reuse,
                         "type_reuse": l_type_reuse})
df_reuse.to_csv(path_reuse, sep=";", encoding="utf-8", index=False, header=True)

# get previous metadata
df_log = pd.read_csv(path_log, sep=";", encoding="utf-8",
                     index_col=False)

# merge metadata
l_id_reuse = []
l_n_reuse = []
for i in tqdm(range(df_log.shape[0])):
    reuses = []
    page_id = df_log.at[i, "id_page"]
    if page_id in d:
        reuse_tuples = d[page_id]
        l_n_reuse.append(len(reuse_tuples))
        for reuse_tuple in reuse_tuples:
            _, tree = reuse_tuple
            reuses.append(tree.findtext(".//id"))
        l_id_reuse.append(" ".join(reuses))
    else:
        l_n_reuse.append(0)
        l_id_reuse.append("")

print("reuses consistently collected :", len(l_id_reuse) == df_log.shape[0],
      "\n")

print("df_log shape :", df_log.shape)
df_log["n_reuse"] = l_n_reuse
df_log["reuse"] = l_id_reuse
print("df_log (with reuse) shape :", df_log.shape, "\n")

# save result
df_log.to_csv(path_output, sep=";", encoding="utf-8", index=False, header=True)
