from embeddings.base.embedding import Embedding
from common.dataset_pool import DatasetPool

from evaluation.lid_eval import EmbLIDMLEEstimator, NCLIDEstimator

import sys
from os import listdir
from os.path import isfile, join

from operator import itemgetter

from datetime import datetime

from scipy.stats import spearmanr, kendalltau, pearsonr
from scipy.stats import mannwhitneyu
from statistics import mean, stdev

def prob_sup(X, Y):
    h = 0
    for x in X:
        for y in Y:
            if x > y:
                h += 1

    total = len(X) * len(Y)
    return h / total

class LidEval:
    def __init__(self, dataset, graph, best_embs_dir, alpha_values):
        self.best_embs_dir = best_embs_dir
        self.dataset = dataset
        self.graph = graph
        self.alpha_values = alpha_values

        files = [f for f in listdir(best_embs_dir) if isfile(join(best_embs_dir, f))]
        self.base = []
        for file_name in files:
            if dataset in file_name and not dataset + "_" in file_name:
                self.base.append(self.parse_file_name(file_name))

        if len(self.base) != 5:
            print("[WARNINIG!!!!!] More than 5 embeddings in base...")
        
        self.base.sort(key=lambda tup: tup[2])
        self.dims = []
        for b in self.base:
            print(b)
            self.dims.append(b[2])
        
        self.rec_emblid_eval()

    def parse_file_name(self, file_name):
        toks = file_name.split(".")[0].split("-")
        dataset = toks[1]
        dim = int(toks[2])
        p = float(toks[3].replace("_", "."))
        q = float(toks[4].replace("_", "."))
        return (file_name, dataset, dim, p, q)

    def rec_emblid_eval(self):
        self.rec_map, self.map_map, self.emblid_map = dict(), dict(), dict()
        for b in self.base:
            dim = b[2]
            emb_file_name = self.best_embs_dir + "/" + b[0]
            emb = Embedding.from_file(emb_file_name)
            numl = self.graph.edges_cnt()
            numn = self.graph.nodes_cnt()
            rg = emb.reconstruct(numl)

            _, recValues = self.graph.recall(rg)
            _, mapValues = self.graph.map_value(rg)
            
            self.rec_map[dim] = recValues
            self.map_map[dim] = mapValues

            estsize = 100 if numn >= 100 else numn
            emblid = EmbLIDMLEEstimator(graph, emb, estsize)
            emblid.estimate_lids()
            self.emblid_map[dim] = emblid.get_lid_values()
            

    def lideval(self):
        ncfile = open("LIDEVAL_LOGS/NC_SIZE_DISTR_" + self.dataset + ".txt", "w")
        f = open("LIDEVAL_LOGS/" + self.dataset + "_lideval_log.csv", "w")

        for alpha in self.alpha_values:
            print("LID for alpha", alpha, "--", datetime.now())
            nclid = NCLIDEstimator(self.graph, alpha=alpha)
            nclid.estimate_lids()
            nclid_values = nclid.get_lid_values()
            avg_nclid = mean(nclid_values.values())
            
            ncfile.write("ALPHA = " + str(alpha) + "\n")
            ncfile.write(nclid.nc_size_distr_str())
            ncfile.write("\n")

            f.write("ALPHA," + str(alpha) + "\n")
            f.write("DATASET,DIM,\
                PCC-NCLID-EMBLID,SCC-NCLID-EMBLID,KCC-NCLID-EMBLID,\
                PCC-NCLID-f1,SCC-NCLID-f1,KCC-NCLID-f1,\
                PCC-EMBLID-f1,SCC-EMBLID-f1,KCC-EMBLID-f1\n")
            
            nclid_f1, emblid_f1 = [], []

            for dim in self.dims:
                emblid_values = self.emblid_map[dim]
                avg_emblid = mean(emblid_values.values())
                map_values = self.map_map[dim]
                rec_values = self.rec_map[dim]

                cors, high_nclid, low_nclid, high_emblid, low_emblid, f1_map =\
                    self.compute_lidcorrs_and_separation(\
                        nclid_values, emblid_values, map_values, rec_values, avg_nclid, avg_emblid)

                f.write(dataset + "," + str(dim) + "," + ",".join(str(c) for c in cors) + "\n")
                
                nclid_f1.append(self.mwu_test(high_nclid, low_nclid, f1_map))
                emblid_f1.append(self.mwu_test(high_emblid, low_emblid, f1_map))

            f.write("\nNCLID-F1 MWU TEST\n")
            f.write("DIM,AVG-HIGH,AVG-LOW,FRAC-HIGH,FRAC-LOW,U,p,H0-ACC,PS-HIGH,PS-LOW\n")
            for i in range(len(self.dims)):
                f.write(str(self.dims[i]) + "," + nclid_f1[i] + "\n")

            f.write("\nEMBLID-F1 MWU TEST\n")
            f.write("DIM,AVG-HIGH,AVG-LOW,FRAC-HIGH,FRAC-LOW,U,p,H0-ACC,PS-HIGH,PS-LOW\n")
            for i in range(len(self.dims)):
                f.write(str(self.dims[i]) + "," + emblid_f1[i] + "\n")

            f.write("\n\n")

        ncfile.close()
        f.close()

    def mwu_test(self, high, low, f1_map):
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


    def compute_lidcorrs_and_separation(self, nclid_values, emblid_values, map_values, rec_values, avg_nclid, avg_emblid):
        x, y, z = [], [], []
        high, low = [], []
        high2, low2 = [], []
        f1_map = dict()
        for node in self.graph.nodes():
            nid = node[0]
            nclid = nclid_values[nid]
            emblid = emblid_values[nid]
            mapv = map_values[nid]
            rec = rec_values[nid] 
            f1 = 0
            if mapv + rec > 0:
                f1 = 2 * mapv * rec / (mapv + rec)

            f1_map[nid] = f1

            x.append(nclid)
            y.append(emblid)
            z.append(f1)

            if nclid >= avg_nclid:
                high.append(nid)
            else:
                low.append(nid)

            if emblid >= avg_emblid:
                high2.append(nid)
            else:
                low2.append(nid)


        pcc1 = pearsonr(x, y)[0]
        scc1 = spearmanr(x, y)[0]
        kcc1 = kendalltau(x, y)[0]

        pcc2 = pearsonr(x, z)[0]
        scc2 = spearmanr(x, z)[0]
        kcc2 = kendalltau(x, z)[0]

        pcc3 = pearsonr(y, z)[0]
        scc3 = spearmanr(y, z)[0]
        kcc3 = kendalltau(y, z)[0]

        corrs = [pcc1, scc1, kcc1, pcc2, scc2, kcc2, pcc3, scc3, kcc3]

        return corrs, high, low, high2, low2, f1_map


if __name__ == "__main__":
    dataset = sys.argv[1]
    print("Evaluating embeddings for ", dataset)
    graph = DatasetPool.load(dataset)
    graph.remove_selfloop_edges()
    graph = graph.to_undirected()
    alpha_values = [1]
    le = LidEval(dataset, graph, "genembeddings/n2v_BEST_EMBEDDINGS", alpha_values)
    le.lideval()
