from common.dataset_pool import DatasetPool
from embeddings.embedding_node2vec import Node2VecEmbedding
from embeddings.embfactory import FileEmbFactory
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import scipy.stats
import statistics
import numpy as np


def native_hubness_map_stats(emb_directory, d, out, g_name=None):
    hubness_func = lambda g, e: g.get_hubness()
    hubness_map_stats(emb_directory, d, out, hubness_func, g_name)


def knng_hubness_map_stats(emb_directory, d, k, out, g_name=None):
    hubness_func = lambda g, e: e.get_knng(k).get_hubness()
    hubness_map_stats(emb_directory, d, out, hubness_func, g_name)


def hubness_map_stats(emb_directory, d, out, hubness_func, g_name=None):
    datasets = DatasetPool.get_datasets() if g_name == None else [g_name]
    for g_name in datasets:
        g = DatasetPool.load(g_name)
        g_undirected = g.to_undirected()
        g_nodes_cnt = len(g.nodes())
        g_edges_cnt = len(g_undirected.edges())
        emb_fact = FileEmbFactory(g_name, emb_directory, d, algs=["N2V"])
        for i in range(emb_fact.num_methods()):
            e = emb_fact.get_embedding(i)
            e_name = emb_fact.get_full_name(g_name, i)
            base_filename = os.path.join(out, e_name)
            gr = e.reconstruct(g_edges_cnt)
            map_avg = g_undirected.map(gr)
            map = g_undirected.get_map_per_node()
            hubness = hubness_func(g, e)
            avg_hubness = statistics.mean(hubness.values())
            pearson, spearman, kendall = create_correlation_figure(
                hubness,
                map,
                "hubness",
                "map",
                base_filename + ".png",
            )
            map_lo, map_hi = [], []
            for n in hubness:
                n_h = hubness[n]
                n_map = map[n]
                if n_h <= avg_hubness:
                    map_lo.append(n_map)
                else:
                    map_hi.append(n_map)
            map_lo_avg = statistics.mean(map_lo)
            map_hi_avg = statistics.mean(map_hi)
            U, pU = scipy.stats.mannwhitneyu(map_lo, map_hi)
            ps1 = prob_sup(map_lo, map_hi)
            ps2 = prob_sup(map_hi, map_lo)
            frac_hi = len(map_hi) / g_nodes_cnt
            frac_lo = len(map_lo) / g_nodes_cnt
            f = open(base_filename + ".txt", "w")
            output = "Average hubness: {}\nFraction of nodes with low hubness: {}\nFraction of nodes with high hubness: {}\nAverage map: {}\nAverage map for low hubness: {}\nAverage map for high hubness: {}\nMann Whitney U: {}\nProbability that low hubness points have greater map: {}\nProbability that high hubness points have greater map: {}\nPearson: {}\nSpearman: {}\nKendall: {}".format(
                avg_hubness,
                frac_lo,
                frac_hi,
                map_avg,
                map_lo_avg,
                map_hi_avg,
                pU,
                ps1,
                ps2,
                pearson,
                spearman,
                kendall,
            )
            f.write(output)
            f.close()


def rec_hubness_hubness_stats(emb_directory, d, out, g_name=None):
    hubness_func = lambda g, e: e.reconstruct(len(g.edges())).get_hubness()
    hubness_hubness_stats(emb_directory, d, out, hubness_func, g_name)


def knng_hubness_hubness_stats(emb_directory, d, k, out, g_name=None):
    hubness_func = lambda g, e: e.get_knng(k).get_hubness()
    hubness_hubness_stats(emb_directory, d, out, hubness_func, g_name)


def hubness_hubness_stats(emb_directory, d, out, hubness_func, g_name=None):
    datasets = DatasetPool.get_datasets() if g_name == None else [g_name]
    for g_name in datasets:
        g = DatasetPool.load(g_name)
        g_undirected = g.to_undirected()
        g_edges_cnt = len(g_undirected.edges())
        emb_fact = FileEmbFactory(g_name, emb_directory, d, preset="N2V")
        for i in range(emb_fact.num_methods()):
            e = emb_fact.get_embedding(i)
            e_name = emb_fact.get_full_name(g_name, i)
            gr = e.reconstruct(g_edges_cnt)
            g_hubness = g_undirected.get_hubness()
            r_hubness = hubness_func(g, e)
            create_correlation_figure(
                g_hubness,
                r_hubness,
                "native hubness",
                "reconstructed hubness",
                os.path.join(out, "{}.png".format(e_name)),
            )


def create_correlation_figure(x_data, y_data, x_label, y_label, out):
    data = []
    x_vals = []
    y_vals = []
    for key, x_val in x_data.items():
        y_val = y_data[key]
        x_vals.append(x_val)
        y_vals.append(y_val)
        data.append([x_val, y_val])

    pearson = scipy.stats.pearsonr(x_vals, y_vals)[0]
    spearman = scipy.stats.spearmanr(x_vals, y_vals)[0]
    kendall = scipy.stats.kendalltau(x_vals, y_vals)[0]

    df = pd.DataFrame(data, columns=[x_label, y_label])
    f = sns.scatterplot(data=df, x=x_label, y=y_label)
    f.set_title(
        "pearson {}; spearman {}; kandall {}".format(
            round(pearson, 2), round(spearman, 2), round(kendall, 2)
        )
    )
    plt.savefig(out)
    plt.close()
    return pearson, spearman, kendall


def prob_sup(X, Y):
    h = 0
    for x in X:
        for y in Y:
            if x > y:
                h += 1

    total = len(X) * len(Y)
    return h / total
