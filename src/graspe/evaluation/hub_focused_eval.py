from common.dataset_pool import DatasetPool
from embeddings.embedding_node2vec import Node2VecEmbedding
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import scipy.stats
import numpy as np


def eval(d, out, g=None):
    datasets = DatasetPool.get_datasets() if g == None else [g]
    for g_name in datasets:
        g = DatasetPool.load(g_name)
        e = Node2VecEmbedding(g, d)
        e.embed()
        gr = e.reconstruct(len(g.edges()))

        h = g.get_hubness()
        g.map(gr)
        map = g.get_map_per_node()

        data = []
        h_vals = []
        map_vals = []
        for n, n_h in h.items():
            n_map = map[n]
            h_vals.append(n_h)
            map_vals.append(n_map)
            data.append([n_h, n_map])

        pearson = scipy.stats.pearsonr(h_vals, map_vals)[0]
        spearman = scipy.stats.spearmanr(h_vals, map_vals)[0]
        kendall = scipy.stats.kendalltau(h_vals, map_vals)[0]

        df = pd.DataFrame(data, columns=["hubness", "map"])
        f = sns.scatterplot(data=df, x="hubness", y="map")
        f.set_title(
            "pearson {}; spearman {}; kandall {}".format(
                round(pearson, 2), round(spearman, 2), round(kendall, 2)
            )
        )
        plt.savefig(os.path.join(out, g_name + ".png"))
