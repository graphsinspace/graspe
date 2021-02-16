from common.dataset_pool import DatasetPool
from embeddings.embedding_node2vec import Node2VecEmbedding
from embeddings.embfactory import FileEmbFactory
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import scipy.stats
import numpy as np


def native_hubness_map_correlation(emb_directory, d, out, g_name=None):
    hubness_func = lambda g, e: g.get_hubness()
    hubness_map_correlation(emb_directory, d, out, hubness_func, g_name)


def knng_hubness_map_correlation(emb_directory, d, k, out, g_name=None):
    hubness_func = lambda g, e: e.get_knng(k).get_hubness()
    hubness_map_correlation(emb_directory, d, out, hubness_func, g_name)


def hubness_map_correlation(emb_directory, d, out, hubness_func, g_name=None):
    datasets = DatasetPool.get_datasets() if g_name == None else [g_name]
    for g_name in datasets:
        g = DatasetPool.load(g_name)
        g_edges_cnt = len(g.edges())
        emb_fact = FileEmbFactory(g_name, emb_directory, d, preset="N2V")
        for i in range(emb_fact.num_methods()):
            e = emb_fact.get_embedding(i)
            e_name = emb_fact.get_full_name(g_name, i)
            gr = e.reconstruct(g_edges_cnt)

            g_hubness = hubness_func(g, e)

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
            plt.savefig(os.path.join(out, "{}.png".format(e_name)))
            plt.close()
