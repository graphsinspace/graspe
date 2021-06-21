#import sys
#import os
#sys.path.append('/home/lucy/grasp/gitrepo/graspe/src')

from os import listdir
from os.path import isfile, join
from multiprocessing import Process

import evaluation.clustering as evaluator
from embeddings.base.embedding import Embedding
from common.dataset_pool import DatasetPool

from cdlib import algorithms
from networkx.algorithms.community.quality import modularity, coverage
from sklearn.metrics.cluster import normalized_mutual_info_score


def get_method(name):
    name_sm = name.lower()

    if(name_sm == "louvain"):
        return algorithms.louvain
    elif(name_sm == "walktrap"):
        return algorithms.walktrap
    elif(name_sm == "girvan_newman"):
        return algorithms.girvan_newman
    else: 
        return algorithms.greedy_modularity

class EmbClusterer:
    def __init__(self, graph, embedding, num_clusters):
        cl_eval = evaluator.ClusteringEval(graph, embedding, "kmeans")
        cl = cl_eval.evaluate_extern_labels("kmeans", range(num_clusters))
        nv = [embedding[n[0]] for n in graph.nodes()]
        self.clusters = cl.fit_predict(nv)
        self.sil_score = cl_eval.get_silhouette_score()

    def get_clusters(self):
        return self.clusters

    def get_sil_score(self):
        return self.sil_score


class GraphClusterer:
    def __init__(self, graph, method):
        self.graph = graph
        G = self.graph.to_networkx().to_undirected()
        fun = get_method(method)
        self.communities = fun(G)
        self.numcoms = len(self.communities)
        self.q = modularity(G, self.communities)

    def get_num_communities(self):
        return self.numcoms

    def get_modularity(self):
        return self.q

    def get_community_labels(self):
        cdict = dict()
        for i in range(len(self.communities)):
            for n in self.communities[i]:
                cdict[n] = i

        clabels = []
        for n in self.graph.nodes():
            clabels.append(cdict[n[0]])

        return clabels

class ClusteringEval:
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


    def eval(self):
        outf = open("clustering-eval-" + self.dataset_name + ".csv", "w")
        outf.write("DATASET,DIM,NUM_GT_LABELS,NUM_COMMUNITIES,MODULARITY,NMI_GT_COMMUNITIES,SIL1_GT,NMI1_GT,SIL2_COMMS,NMI2_COMMS\n")
        
        gt_labels =  [n[1]["label"] for n in self.graph.nodes()]
        num_labels = len(set(gt_labels))

        # perform community detection
        grc = GraphClusterer(self.graph)
        numcoms = grc.get_num_communities()
        modularity = grc.get_modularity()
        community_labels = grc.get_community_labels()
        nmi_gt_community = normalized_mutual_info_score(gt_labels, community_labels)
        

        for b in self.base:
            emb_file, dataset, dim, p, q = b
            emb_file_path = join(self.emb_dir, emb_file)
            
            emb = Embedding.from_file(emb_file_path)
            
            # K-means za onoliko klastera koliko ima labela
            embc1 = EmbClusterer(self.graph, emb, num_labels)
            sil1 = embc1.get_sil_score()
            nmi1 = normalized_mutual_info_score(gt_labels, embc1.get_clusters())

            # K-means za onoliko klastera koliko ima zajednica
            embc2 = EmbClusterer(self.graph, emb, numcoms)
            sil2 = embc2.get_sil_score()
            nmi2 = normalized_mutual_info_score(community_labels, embc2.get_clusters())
            
            msg = dataset + "," + str(dim) + "," + str(num_labels) + "," + str(numcoms) + "," +\
                str(modularity) + "," + str(nmi_gt_community) + "," + str(sil1) + "," + str(nmi1) + "," +\
                str(sil2) + "," + str(nmi2)

            outf.write(msg + "\n")

        outf.close()


def pecentage_equal_labels(name, graph, embedding):
    method = get_method(name)
    graph_mat = graph.to_adj_matrix()
    labels = method.fit_transform(graph_mat) #niz labela za svaki cvor
    cl_eval = ClusteringEval(graph, embedding, 'kmeans')
    clusters_eval = cl_eval.evaluate_extern_labels(labels) #niz labela za svaki cvor
    num_equal = 0
    for i in range(len(labels)):
        for j in range(len(labels)):
            if(labels[i] == labels[j] and clusters_eval[i] == clusters_eval[j]):
                num_equal = num_equal + 1

    percentage = num_equal/(len(labels) * len(labels))     
    return percentage


datasets = [
    "karate_club_graph",
    "cora_ml",  
    "citeseer",
    "amazon_electronics_photo",
    "amazon_electronics_computers",
    "pubmed",
    "cora",
    "dblp"
]

def clustering_eval_function(d, folder):
    graph = DatasetPool.load(d)
    graph.remove_selfloop_edges()
    graph = graph.to_undirected()

    ce = ClusteringEval(d, graph, folder)
    ce.eval()


if __name__ == "__main__":
    folder = sys.argv[1]

    procs = []
    for d in datasets:
        proc = Process(target=clustering_eval_function, args=(d, folder))
        procs.append(proc)

    for p in procs:
        p.start()

    for p in procs:
        p.join()