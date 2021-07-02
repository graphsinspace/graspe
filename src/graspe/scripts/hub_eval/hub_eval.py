
from embeddings.base.embedding import Embedding
from common.dataset_pool import DatasetPool

import sys
from os import listdir
from os.path import isfile, join

from operator import itemgetter

from datetime import datetime

from scipy.stats import spearmanr, kendalltau, pearsonr
from scipy.stats import mannwhitneyu
from statistics import mean, stdev
import scipy

from multiprocessing import Process



class HubEval:
    def __init__(self, dataset_name, graph, emb_dir):
        files = [f for f in listdir(emb_dir) if isfile(join(emb_dir, f))]
        self.base = []
        for file_name in files:
            file_name, dataset, dim, p, q = self.parse_file_name(file_name)
            if dataset == dataset_name:
                self.base.append((file_name, dataset, dim, p, q))

        self.graph = graph
        self.dataset_name = dataset_name
        self.emb_dir = emb_dir

        self.base.sort(key=lambda tup: tup[2])


    def parse_file_name(self, file_name):
        toks = file_name.split(".")[0].split("-")
        dataset = "-".join(toks[1:-3])
        dim = int(toks[-3])
        p = float(toks[-2].replace("_", "."))
        q = float(toks[-1].replace("_", "."))
        #print("---", dataset, dim, p, q)
        return (file_name, dataset, dim, p, q)


    def eval(self, k=5):
        mwu_results = []
        outf = open("hub-eval-" + self.dataset_name + ".csv", "w")
        outf.write("\nDATASET,DIM,KC-DEG-F1,KC-DEG-RCDEG,KC-DEG-5NNDEG,KC-DEG-AUTODEG\n")
        print("DATASET,DIM,KC-DEG-F1,KC-DEG-RCDEG,KC-DEG-5NNDEG,KC-DEG-AUTODEG")
        
        for b in self.base:
            emb_file, dataset, dim, p, q = b
            emb_file_path = join(self.emb_dir, emb_file)
            
            emb = Embedding.from_file(emb_file_path)
            numl = self.graph.edges_cnt()
            rg = emb.reconstruct(numl)

            knng_k = emb.get_knng(k)
            knng_auto = emb.get_knng(self.graph.edges_cnt() / self.graph.nodes_cnt())

            recg_avg_map, recg_maps = self.graph.map_value(rg)
            recg_avg_recall, recg_recalls = self.graph.recall(rg)
            recg_f1 = {
                node: (
                    (2 * recg_maps[node] * recg_recalls[node])
                    / (recg_maps[node] + recg_recalls[node])
                    if recg_maps[node] + recg_recalls[node] != 0
                    else 0
                )
                for node in recg_maps
            }

            native_hubness = self.graph.get_hubness()
            recg_hubness = rg.get_hubness()
            knng_k_hubness = knng_k.get_hubness()
            knng_auto_hubness = knng_auto.get_hubness()

            f1_lo, f1_hi = [], []
            avg_hubness = mean(native_hubness.values())
            for n in native_hubness:
                n_h = native_hubness[n]
                if n_h <= avg_hubness:
                    f1_lo.append(n)
                else:
                    f1_hi.append(n)

            c1 = compute_corrs(native_hubness, recg_f1)
            c2 = compute_corrs(native_hubness, recg_hubness)
            c3 = compute_corrs(native_hubness, knng_k_hubness)
            c4 = compute_corrs(native_hubness, knng_auto_hubness)

            cmsg = dataset + "," + str(dim) + "," + str(c1) + "," + str(c2) + "," + str(c3) + "," + str(c4)
            outf.write(cmsg + "\n")
            print(cmsg)

            mwu_res = dataset + "," + str(dim) + "," + mwu_test(f1_hi, f1_lo, recg_f1)
            mwu_results.append(mwu_res)

        outf.write("DATASET,DIM,AVG-HIGH,AVG-LOW,FRAC-HIGH,FRAC-LOW,U,p,H0-ACC,PS-HIGH,PS-LOW\n")
        for m in mwu_results:
            outf.write(m + "\n")
            print(m)

        outf.close()


def compute_corrs(x_data, y_data):
    x_vals = []
    y_vals = []
    for key, x_val in x_data.items():
        y_val = y_data[key]
        x_vals.append(x_val)
        y_vals.append(y_val)

    #pearson = scipy.stats.pearsonr(x_vals, y_vals)[0]
    #spearman = scipy.stats.spearmanr(x_vals, y_vals)[0]
    kendall = scipy.stats.kendalltau(x_vals, y_vals)[0]   

    return kendall 

def mwu_test(high, low, f1_map):
    x, y = [], []
    for n in high:
        x.append(f1_map[n])
    for n in low:
        y.append(f1_map[n])

    avg_x = mean(x)
    avg_y = mean(y)

    U, pU = mannwhitneyu(x, y)
    ps1 = prob_sup(x, y)
    ps2 = prob_sup(y, x)
    total = len(x) + len(y)
    x_frac = len(x) / total
    y_frac = len(y) / total
    acc = "yes" if pU >= 0.05 else "no"

    return str(avg_x) + "," + str(avg_y) + "," + str(x_frac) + "," + str(y_frac) + "," +\
            str(U) + "," + str(pU) + "," + acc + "," + str(ps1) + "," + str(ps2)


def prob_sup(X, Y):
    h = 0
    for x in X:
        for y in Y:
            if x > y:
                h += 1

    total = len(X) * len(Y)
    return h / total


datasets = [
    "karate_club_graph",
    "les_miserables_graph",
    "florentine_families_graph",
    "cora_ml",  
    "citeseer",
    "amazon_electronics_photo",
    "amazon_electronics_computers",
    "pubmed",
    "cora",
    "dblp"
]

def hub_eval_function(d, folder):
    graph = DatasetPool.load(d)
    graph.remove_selfloop_edges()
    graph = graph.to_undirected()

    he = HubEval(d, graph, folder)
    he.eval()

if __name__ == "__main__":
    folder = sys.argv[1]

    procs = []
    for d in datasets:
        proc = Process(target=hub_eval_function, args=(d, folder))
        procs.append(proc)

    for p in procs:
        p.start()

    for p in procs:
        p.join()