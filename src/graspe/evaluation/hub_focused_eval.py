import os
import statistics

import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats
import seaborn as sns

from common.dataset_pool import DatasetPool
from embeddings.emb_factory import FileEmbFactory


def hubness_stats(emb_directory, d, out, g_name=None, preset="_", algs=None, k=5):
    datasets = DatasetPool.get_datasets() if g_name == None else [g_name]
    for g_name in datasets:
        g = DatasetPool.load(g_name)
        g_undirected = g.to_undirected()
        emb_fact = FileEmbFactory(g_name, emb_directory, d, preset=preset, algs=algs)

        for i in range(emb_fact.num_methods()):
            e = emb_fact.get_embedding(i)
            e_name = emb_fact.get_full_name(g_name, i)
            base_filename = os.path.join(out, e_name)

            recg = e.reconstruct(g_undirected.edges_cnt())
            knng_k = e.get_knng(k)
            knng_auto = e.get_knng(g.edges_cnt() / g.nodes_cnt())
            if knng_auto % 2 == 1:
                knng_auto -= 1

            recg_avg_map, recg_maps = g_undirected.map_value(recg)
            recg_avg_recall, recg_recalls = g_undirected.recall(recg)
            recg_f1 = {
                node: (
                    (2 * recg_maps[node] * recg_recalls[node])
                    / (recg_maps[node] + recg_recalls[node])
                    if recg_maps[node] + recg_recalls[node] != 0
                    else 0
                )
                for node in recg_maps
            }

            knng_k_avg_map, knng_k_maps = g.map_value(knng_k)
            knng_k_avg_recall, knng_k_recalls = g.recall(knng_k)

            knng_auto_avg_map, knng_auto_maps = g.map_value(knng_auto)
            knng_auto_avg_recall, knng_auto_recalls = g.recall(knng_auto)

            native_hubness = g.get_hubness()
            native_undirected_hubness = g_undirected.get_hubness()
            recg_hubness = recg.get_hubness()
            knng_k_hubness = knng_k.get_hubness()
            knng_auto_hubness = knng_auto.get_hubness()

            map_lo, map_hi = [], []
            recall_lo, recall_hi = [], []
            f1_lo, f1_hi = [], []
            avg_hubness = statistics.mean(native_hubness.values())
            for n in native_hubness:
                n_h = native_hubness[n]
                n_map = recg_maps[n]
                n_recall = recg_recalls[n]
                n_f1 = recg_f1[n]
                if n_h <= avg_hubness:
                    map_lo.append(n_map)
                    recall_lo.append(n_recall)
                    f1_lo.append(n_f1)
                else:
                    map_hi.append(n_map)
                    recall_hi.append(n_recall)
                    f1_hi.append(n_f1)

            output = []
            output.append("Average hubness: {}".format(avg_hubness))
            output.append(
                "Fraction of nodes with low hubness: {}".format(
                    len(map_lo) / g.nodes_cnt()
                )
            )
            output.append(
                "Fraction of nodes with high hubness: {}".format(
                    len(map_hi) / g.nodes_cnt()
                )
            )
            output.append("Average map: {}".format(recg_avg_map))
            nonparametric_tests("map", map_lo, map_hi, output)
            output.append("Average recall: {}".format(recg_avg_recall))
            nonparametric_tests("recall", recall_lo, recall_hi, output)
            nonparametric_tests("f1", f1_lo, f1_hi, output)
            output.append("Average knng map: {}".format(knng_auto_avg_map))
            output.append("Average knng recall: {}".format(knng_auto_avg_recall))
            sets = {
                "native hubness": native_hubness,
                "reconstructed hubness": recg_hubness,
                "knng hubness (k=" + str(k) + ")": knng_k_hubness,
                "knng hubness (k=auto)": knng_auto_hubness,
            }
            calc_correlations_pairwise(sets, output, base_filename)
            calc_correlations_one_to_many(
                sets, ("map", recg_maps), output, base_filename
            )
            calc_correlations_one_to_many(
                sets, ("recall", recg_recalls), output, base_filename
            )
            calc_correlations_one_to_many(sets, ("f1", recg_f1), output, base_filename)

            f = open(base_filename + ".txt", "w")
            f.write("\n".join(output))
            f.close()


def calc_correlations_pairwise(sets, output, base_filename):
    names = list(sets.keys())
    for i in range(len(names)):
        name1 = names[i]
        for j in range(i + 1, len(names)):
            name2 = names[j]
            calc_correlations(
                (name1, sets[name1]), (name2, sets[name2]), output, base_filename
            )


def calc_correlations_one_to_many(multiple_sets, single_set, output, base_filename):
    for name in multiple_sets:
        calc_correlations(
            (name, multiple_sets[name]), single_set, output, base_filename
        )


def calc_correlations(set1, set2, output, base_filename):
    name1, name2 = set1[0], set2[0]
    pearson, spearman, kendall = create_correlation_figure(
        set1[1],
        set2[1],
        name1,
        name2,
        base_filename + "_" + name1 + "_" + name2 + ".png",
    )
    output.append(
        "Correlation between {} and {} (pearson, spearman, kendall): {}, {}, {}".format(
            name1, name2, pearson, spearman, kendall
        )
    )


def nonparametric_tests(name, group_lo, group_hi, output):
    group_lo_avg = statistics.mean(group_lo)
    group_hi_avg = statistics.mean(group_hi)
    try:
        U, pU = scipy.stats.mannwhitneyu(group_lo, group_hi)
    except:
        pU = -1
    group_lo_ps = prob_sup(group_lo, group_hi)
    group_hi_ps = prob_sup(group_hi, group_lo)

    output.append("Average {} for low hubness: {}".format(name, group_lo_avg))
    output.append("Average {} for high hubness: {}".format(name, group_hi_avg))
    output.append(
        "Mann Whitney U for {} values of two groups of nodes 1) nodes with low hubness, 2) nodes with high hubness: {}".format(
            name, pU
        )
    )
    output.append(
        "Probability that low hubness points have greater {}: {}".format(
            name, group_lo_ps
        )
    )
    output.append(
        "Probability that high hubness points have greater {}: {}".format(
            name, group_hi_ps
        )
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
