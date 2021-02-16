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
            g.map(gr)
            map = g.get_map_per_node()
            create_correlation_figure(hubness_func(g, e), map, "hubness", "map", os.path.join(out, "{}.png".format(e_name)))

def rec_hubness_hubness_correlation(emb_directory, d, out, g_name=None):
    hubness_func = lambda g, e: e.reconstruct(len(g.edges())).get_hubness()
    hubness_hubness_correlation(emb_directory, d, out, hubness_func, g_name)

def knng_hubness_hubness_correlation(emb_directory, d, k, out, g_name=None):
    hubness_func = lambda g, e: e.get_knng(k).get_hubness()
    hubness_hubness_correlation(emb_directory, d, out, hubness_func, g_name)

def hubness_hubness_correlation(emb_directory, d, out, hubness_func, g_name=None):
    datasets = DatasetPool.get_datasets() if g_name == None else [g_name]
    for g_name in datasets:
        g = DatasetPool.load(g_name)
        g_edges_cnt = len(g.edges())
        emb_fact = FileEmbFactory(g_name, emb_directory, d, preset="N2V")
        for i in range(emb_fact.num_methods()):
            e = emb_fact.get_embedding(i)
            e_name = emb_fact.get_full_name(g_name, i)
            gr = e.reconstruct(g_edges_cnt)            
            g_hubness = g.get_hubness()
            r_hubness = hubness_func(g, e)
            create_correlation_figure(g_hubness, r_hubness, "native hubness", "reconstructed hubness", os.path.join(out, "{}.png".format(e_name)))

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