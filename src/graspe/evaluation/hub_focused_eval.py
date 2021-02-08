from common.dataset_pool import DatasetPool
from embeddings.embedding_node2vec import Node2VecEmbedding
from embeddings.embfactory import LazyEmbFactory
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
        g_edges_cnt = len(g.edges())
        g_hubness = g.get_hubness()

        emb_fact = LazyEmbFactory(g, d, preset="N2V")
        for i in range(emb_fact.num_methods()):
            e = emb_fact.get_embedding(i)
            e_name = emb_fact.get_name(i)
            gr = e.reconstruct(g_edges_cnt)

            g.map(gr)
            map = g.get_map_per_node()

            data = []
            h_vals = []
            map_vals = []
            for n, n_h in g_hubness.items():
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
            plt.savefig(os.path.join(out, "{}_{}.png".format(g_name, e_name)))
            plt.close()
