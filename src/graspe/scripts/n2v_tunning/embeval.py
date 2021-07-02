
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

class EmbEval:
    def __init__(self, graph, emb_dir):
        print("#nodes = ", graph.nodes_cnt())
        print("#links = ", graph.edges_cnt())
        self.emb_dir = emb_dir
        files = [f for f in listdir(emb_dir) if isfile(join(emb_dir, f))]
        self.base = []
        for file_name in files:
            self.base.append(self.parse_file_name(file_name))

        self.graph = graph
        self.dataset = self.base[0][1]

    def parse_file_name(self, file_name):
        toks = file_name.split(".")[0].split("-")
        dataset = toks[1]
        dim = int(toks[2])
        p = float(toks[3].replace("_", "."))
        q = float(toks[4].replace("_", "."))
        return (file_name, dataset, dim, p, q)

    def receval(self, receval_report_file):
        print("Starting receval", datetime.now())
        f = open(receval_report_file, "w")
        f.write("DATASET,DIM,p,q,PREC,REC,MAP,F1\n")

        dimres = dict()
        for b in self.base:
            emb_file, dataset, dim, p, q = b
            emb_file_path = join(self.emb_dir, emb_file)
            
            emb = Embedding.from_file(emb_file_path)
            numl = self.graph.edges_cnt()
            rg = emb.reconstruct(numl)

            prec = self.graph.link_precision(rg)
            rec, _ = self.graph.recall(rg)
            mapv, _ = self.graph.map_value(rg)

            f1 = 2 * rec * mapv / (rec + mapv)
            
            f.write(dataset + "," + str(dim) + "," + str(p) + "," + str(q) + "," + 
                str(prec) + "," + str(rec) + "," + str(mapv) + "," + str(f1) + "\n")
            print(dataset, "dim = ", dim, "p = ", p, "q = ", q, "prec = ", prec, "rec = ", rec, "map = ", mapv, "f1 = ", f1)

            if dim in dimres:
                dimres[dim].append((p, q, prec, rec, mapv, f1, emb_file_path))
            else:
                dimres[dim] = [(p, q, prec, rec, mapv, f1, emb_file_path)]

        f.write("\n\n")
        f.write("MAX F1 per dim\n")
        f.write("dim,p,q,prec,rec,map,f1\n")
        print("\nMax f1 per dim (p, q, prec, rec, map, f1, emb_file)")
        self.maxf1 = dict()
        for d in dimres:
            l = dimres[d]
            max_f1_tuple = max(l, key=itemgetter(5))
            print(d, max_f1_tuple)
            s = str(d) + "," + str(max_f1_tuple[0]) + "," + str(max_f1_tuple[1]) + "," +\
                str(max_f1_tuple[2]) + "," + str(max_f1_tuple[3]) + "," + str(max_f1_tuple[4]) + "," + str(max_f1_tuple[5])
            f.write(s + "\n")
            self.maxf1[d] = max_f1_tuple[6]

        f.close()

    def lideval(self, lideval_report_file):
        print("\nStarting LID-eval", datetime.now())

        f = open(lideval_report_file, "w")
        
        nclid = NCLIDEstimator(self.graph)
        nclid.estimate_lids()
        avg_nclid = nclid.get_avg_lid()
        f.write("AVG_NCLID," + str(avg_nclid) + "\n")
        f.write("MIN_NCLID," + str(nclid.get_min_lid()) + "\n")
        f.write("MAX_NCLID," + str(nclid.get_max_lid()) + "\n")
        f.write("STD_NCLID," + str(nclid.get_stdev_lid()) + "\n\n")
        nclid.nc_size_distr("NC_SIZE_DISTR_" + self.dataset + ".txt")
        
        f.write("DATASET,DIM,p,q,AVG-EMBLID,STD-EMBLID,MIN-EMBLID,MAX-EMBLID,PCC-NCLID-EMBLID,SCC-NCLID-EMBLID,KCC-NCLID-EMBLID\n")
        map_results = []
        recall_results = []
        for d in sorted(self.maxf1.keys()):
            print(d, self.maxf1[d])
            emb_file, dataset, dim, p, q = self.parse_file_name(self.maxf1[d])
            #print(emb_file, dataset, dim, p, q)
            
            emb = Embedding.from_file(emb_file)
            numn = self.graph.nodes_cnt()
            numl = self.graph.edges_cnt()
            rg = emb.reconstruct(numl)

            _, recdict = self.graph.recall(rg)
            _, mapdict = self.graph.map_value(rg)
            
            estsize = 100 if numn >= 100 else numn
            emblid = EmbLIDMLEEstimator(graph, emb, estsize)
            emblid.estimate_lids()
            avg_emblid = emblid.get_avg_lid()
            max_emblid = emblid.get_max_lid()
            min_emblid = emblid.get_min_lid()
            std_emblid = emblid.get_stdev_lid()

            pcc, scc, kcc, high_nclid, low_nclid = self.compute_lidcorrs_and_separation(nclid, emblid, avg_nclid)

            map_results.append(self.mwu_test(high_nclid, low_nclid, mapdict))
            recall_results.append(self.mwu_test(high_nclid, low_nclid, recdict))

            f.write(dataset + "," + str(dim) + "," + str(p) + "," + str(q) + "," +
                str(avg_emblid) + "," + str(std_emblid) + "," + str(min_emblid) + "," + str(max_emblid) + "," + 
                str(pcc) + "," + str(scc) + "," + str(kcc) + "\n"   
            )

        dims = [10, 25, 50, 100, 200]
        f.write("\n\nMAP MWU TEST\n")
        f.write("DIM,AVG-HIGH,AVG-LOW,FRAC-HIGH,FRAC-LOW,U,p,H0-ACC,PS-HIGH,PS-LOW\n")
        for i in range(len(dims)):
            f.write(str(dims[i]) + "," + map_results[i] + "\n")

        f.write("\n\nRECALL MWU TEST\n")
        f.write("DIM,AVG-HIGH,AVG-LOW,FRAC-HIGH,FRAC-LOW,U,p,H0-ACC,PS-HIGH,PS-LOW\n")
        for i in range(len(dims)):
            f.write(str(dims[i]) + "," + recall_results[i] + "\n")   

        f.close()  



    def mwu_test(self, high_nclid, low_nclid, metric_dict):
        x, y = [], []
        for n in high_nclid:
            x.append(metric_dict[n])
        for n in low_nclid:
            y.append(metric_dict[n])

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


    def compute_lidcorrs_and_separation(self, nclid, emblid, avg_nclid):
        x, y = [], []
        high, low = [], []
        for node in self.graph.nodes():
            nid = node[0]
            nid_nclid = nclid.get_lid(nid)
            x.append(nclid.get_lid(nid))
            y.append(emblid.get_lid(nid))

            if nid_nclid >= avg_nclid:
                high.append(nid)
            else:
                low.append(nid)


        pcc = pearsonr(x, y)[0]
        scc = spearmanr(x, y)[0]
        kcc = kendalltau(x, y)[0]

        return pcc, scc, kcc, high, low



if __name__ == "__main__":
    dataset = sys.argv[1]
    print("Evaluating embeddings for ", dataset)
    graph = DatasetPool.load(dataset)
    graph.remove_selfloop_edges()
    graph = graph.to_undirected()
    ee = EmbEval(graph, "genembeddings/" + dataset + "_embeddings")
    ee.receval(dataset + "_receval_log.csv")
    ee.lideval(dataset + "_lideval_log.csv")

