import os
import pandas as pd
import statistics
import math
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats

sns.set_theme()


def read_file(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
    return float(lines[3].strip().split(": ")[1])


directory = "native_hubness_map_stats"
datasets_order = [
    "cora",
    "cora_ml",
    "karate_club_graph",
    "dblp",
    "les_miserables_graph",
    "citeseer",
    "amazon_electronics_photo",
    "amazon_electronics_computers",
    "pubmed",
    "florentine_families_graph",
]

datasets = {
    "amazon_electronics_computers": {
        "nodes_cnt": 13752,
        "edges_cnt": 491722,
        "edges_density": 0.005200552551701883,
        "avg_hubness": 35.7563990692263,
        "max_hubness": 2992,
        "max_hubness_norm": 83.67732987338373,
    },
    "amazon_electronics_photo": {
        "nodes_cnt": 7650,
        "edges_cnt": 238163,
        "edges_density": 0.008140258413035324,
        "avg_hubness": 31.132418300653594,
        "max_hubness": 1434,
        "max_hubness_norm": 46.061310950903376,
    },
    "citeseer": {
        "nodes_cnt": 4230,
        "edges_cnt": 10674,
        "edges_density": 0.0011933810618676515,
        "avg_hubness": 2.523404255319149,
        "max_hubness": 85,
        "max_hubness_norm": 33.68465430016863,
    },
    "cora": {
        "nodes_cnt": 19793,
        "edges_cnt": 126842,
        "edges_density": 0.0006475775284706004,
        "avg_hubness": 6.408427221745061,
        "max_hubness": 297,
        "max_hubness_norm": 46.345224767821385,
    },
    "cora_ml": {
        "nodes_cnt": 2995,
        "edges_cnt": 16316,
        "edges_density": 0.003639109047254219,
        "avg_hubness": 5.447746243739566,
        "max_hubness": 246,
        "max_hubness_norm": 45.15628830595735,
    },
    "dblp": {
        "nodes_cnt": 17716,
        "edges_cnt": 105734,
        "edges_density": 0.0006738105857737093,
        "avg_hubness": 5.96827726349063,
        "max_hubness": 339,
        "max_hubness_norm": 56.800310212419845,
    },
    "pubmed": {
        "nodes_cnt": 19717,
        "edges_cnt": 88648,
        "edges_density": 0.00045607817651622764,
        "avg_hubness": 4.496018664096972,
        "max_hubness": 171,
        "max_hubness_norm": 38.03364994134103,
    },
    "karate_club_graph": {
        "nodes_cnt": 34,
        "edges_cnt": 156,
        "edges_density": 0.27807486631016043,
        "avg_hubness": 4.588235294117647,
        "max_hubness": 17,
        "max_hubness_norm": 3.7051282051282053,
    },
    "florentine_families_graph": {
        "nodes_cnt": 15,
        "edges_cnt": 40,
        "edges_density": 0.38095238095238093,
        "avg_hubness": 2.6666666666666665,
        "max_hubness": 6,
        "max_hubness_norm": 2.25,
    },
    "les_miserables_graph": {
        "nodes_cnt": 77,
        "edges_cnt": 508,
        "edges_density": 0.17361585782638414,
        "avg_hubness": 6.597402597402597,
        "max_hubness": 36,
        "max_hubness_norm": 5.456692913385827,
    },
}

data = []
nodes_cnts = []
densities = []
hubnesses = []
maps = []
for dataset in datasets:
    map_val = read_file(os.path.join(directory, dataset + "_d10_N2V.txt"))
    data.append(["nodes count", datasets[dataset]["nodes_cnt"], map_val])
    data.append(["density", datasets[dataset]["edges_density"], map_val])
    data.append(
        ["normalized max hubness", datasets[dataset]["max_hubness_norm"], map_val]
    )
    nodes_cnts.append(datasets[dataset]["nodes_cnt"])
    densities.append(datasets[dataset]["edges_density"])
    hubnesses.append(datasets[dataset]["max_hubness_norm"])
    maps.append(map_val)

pearson_nodes_cnts = scipy.stats.pearsonr(nodes_cnts, maps)[0]
spearman_nodes_cnts = scipy.stats.spearmanr(nodes_cnts, maps)[0]
kendall_nodes_cnts = scipy.stats.kendalltau(nodes_cnts, maps)[0]

pearson_densities = scipy.stats.pearsonr(densities, maps)[0]
spearman_densities = scipy.stats.spearmanr(densities, maps)[0]
kendall_densities = scipy.stats.kendalltau(densities, maps)[0]

pearson_hubnesses = scipy.stats.pearsonr(hubnesses, maps)[0]
spearman_hubnesses = scipy.stats.spearmanr(hubnesses, maps)[0]
kendall_hubnesses = scipy.stats.kendalltau(hubnesses, maps)[0]

print(
    "node count vs. map: {}, {}, {}".format(
        pearson_nodes_cnts, spearman_nodes_cnts, kendall_nodes_cnts
    )
)
print(
    "density vs. map: {}, {}, {}".format(
        pearson_densities, spearman_densities, kendall_densities
    )
)
print(
    "hubness vs. map: {}, {}, {}".format(
        pearson_hubnesses, spearman_hubnesses, kendall_hubnesses
    )
)

f_data = pd.DataFrame(data=data, columns=["type", "value", "map"])
g = sns.FacetGrid(f_data, row="type", height=2, aspect=4, sharex=False)
g.map_dataframe(sns.regplot, x="value", y="map", truncate=False)
plt.tight_layout()
plt.savefig("dataset_properties_influence.png")
plt.close()