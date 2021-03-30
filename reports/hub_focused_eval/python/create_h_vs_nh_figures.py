import os
import pandas as pd
import statistics
import math
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme()

def read_file(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
    d = {}
    d["map_l"] = float(lines[4].strip().split(": ")[1])
    d["map_h"] = float(lines[5].strip().split(": ")[1])
    d["mwu"] = float(lines[6].strip().split(": ")[1])
    d["prob_l"] = float(lines[7].strip().split(": ")[1])
    d["prob_h"] = float(lines[8].strip().split(": ")[1])
    return d

directory = "native_hubness_map_stats"
datasets = ["cora", "cora_ml", "karate_club_graph", "dblp", "les_miserables_graph", "citeseer", "amazon_electronics_photo", "amazon_electronics_computers", "pubmed", "florentine_families_graph"]

data_map_prob = []
data_mwu = []
for dataset in datasets:
    stats = read_file(os.path.join(directory, dataset+"_d10_N2V.txt"))
    data_map_prob.append([dataset, "map", "low-hubness points", stats["map_l"]])
    data_map_prob.append([dataset, "map", "high-hubness points", stats["map_h"]])
    data_map_prob.append([dataset, "probability of having higher map", "low-hubness points", stats["prob_l"]])
    data_map_prob.append([dataset, "probability of having higher map", "high-hubness points", stats["prob_h"]])
    data_mwu.append([dataset, "mwu", stats["mwu"]])

f_data = pd.DataFrame(data=data_map_prob, columns=["dataset", "type", "sub-type", "value"])
g = sns.FacetGrid(f_data, row="type", height=2, aspect=4, sharey=False)
g.map_dataframe(sns.barplot, x="dataset", y="value", hue="sub-type", palette=sns.color_palette("tab10"), order=["cora", "cora_ml", "karate_club_graph", "dblp", "les_miserables_graph", "citeseer", "amazon_electronics_photo", "amazon_electronics_computers", "pubmed", "florentine_families_graph"])
plt.xticks( 
    fontweight='light',
    rotation=30,
    horizontalalignment='right',
)
plt.tight_layout()
g.add_legend()
plt.savefig("low-high_hubness-map-prob.png")
plt.close()

fig, ax = plt.subplots(figsize=[6, 3])
f_data = pd.DataFrame(data=data_mwu, columns=["dataset", "mwu", "p-value"])
f = sns.barplot(ax=ax, x="dataset", y="p-value", hue="mwu", data=f_data)
f.get_legend().remove()
f.set_xlabel("")
plt.xticks( 
    fontweight='light',
    rotation=30,
    horizontalalignment='right',
)
plt.tight_layout()
plt.savefig("low-high_hubness-mwu.png")
plt.close()