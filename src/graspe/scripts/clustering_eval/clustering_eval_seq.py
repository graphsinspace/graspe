from embeddings.base.embedding import Embedding
from common.dataset_pool import DatasetPool

import sys
from os import listdir
from os.path import isfile, join

from multiprocessing import Process

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.cluster import normalized_mutual_info_score

from networkx.algorithms.community.modularity_max import greedy_modularity_communities
from networkx.algorithms.community.quality import modularity
from networkx.algorithms.components import number_connected_components, connected_components


class EmbClusterer:
    def __init__(self, graph, embedding, num_clusters):
        cl = KMeans(num_clusters)
        nv = [embedding[n[0]] for n in graph.nodes()]
        self.clusters = cl.fit_predict(nv)
        self.sil_score = silhouette_score(nv, self.clusters)

    def get_clusters(self):
        return self.clusters

    def get_sil_score(self):
        return self.sil_score



### Za detekciju zajednica pored networkx ima super biblioteka koja se zove
### cdlib biblioteka -- https://github.com/GiulioRossetti/cdlib

class GraphClusterer:
    def __init__(self, graph):
        self.graph = graph
        G = self.graph.to_networkx().to_undirected()
        self.communities = greedy_modularity_communities(G)
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


def detect_connected_components(G):
    num_nodes = G.number_of_nodes()
    ncc = number_connected_components(G)
    comps = connected_components(G)
    lcc = max(comps, key=len)
    lcc_frac = len(lcc) / num_nodes
    return ncc, lcc_frac


class ClusteringEval:
    def __init__(self, dataset_name, graph, emb_dir, method):
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
        self.method = method


    def parse_file_name(self, file_name):
        toks = file_name.split(".")[0].split("-")
        dataset = "-".join(toks[1:-3])
        dim = int(toks[-3])
        p = float(toks[-2].replace("_", "."))
        q = float(toks[-1].replace("_", "."))
        #print("---", dataset, dim, p, q)
        return (file_name, dataset, dim, p, q)


    def eval(self):
        outf = open("clustering-eval-" + self.method + "-" + self.dataset_name + ".csv", "w")
        outf.write("DATASET,DIM,NUM_GT_LABELS,NUM_COMPS,LCC_FRAC,NUM_COMMUNITIES,MODULARITY,NMI_GT_COMMUNITIES,")
        outf.write("SIL1_GT,NMI1_GT,SIL2_COMMS,NMI2_COMMS,RG_NUMCOMPS,RG_LCC_FRAC,RG_NUMCOMS,RG_MODULARITY,RG_NMI_GT\n")
        
        gt_labels =  [n[1]["label"] for n in self.graph.nodes()]
        num_labels = len(set(gt_labels))

        # perform community detection
        grc = GraphClusterer(self.graph)
        numcoms = grc.get_num_communities()
        modularity = grc.get_modularity()
        community_labels = grc.get_community_labels()
        nmi_gt_community = normalized_mutual_info_score(gt_labels, community_labels)
        
        # detect connected components
        nkxG = self.graph.to_networkx().to_undirected()
        comps, lcc_frac = detect_connected_components(nkxG)

        for b in self.base:
            emb_file, dataset, dim, p, q = b
            emb_file_path = join(self.emb_dir, emb_file)
            
            emb = Embedding.from_file(emb_file_path)
            
            # rekonstrukcija grafa
            numl = self.graph.edges_cnt()
            rg = emb.reconstruct(numl)
            
            # detekcija komponenti u rekonstruisanom grafu
            nkxG_rg = rg.to_networkx().to_undirected()
            comps_rg, lcc_frac_rg = detect_connected_components(nkxG_rg)

            # community detection na rekonstruisanom grafu
            grcrg = GraphClusterer(rg)
            modularity_rg = grcrg.get_modularity()
            numcoms_rg = grcrg.get_num_communities()
            community_labels_rg = grcrg.get_community_labels()
            nmi_gt_community_rg = normalized_mutual_info_score(community_labels, community_labels_rg)

            # K-means za onoliko klastera koliko ima labela
            embc1 = EmbClusterer(self.graph, emb, num_labels)
            sil1 = embc1.get_sil_score()
            nmi1 = normalized_mutual_info_score(gt_labels, embc1.get_clusters())

            # K-means za onoliko klastera koliko ima zajednica
            embc2 = EmbClusterer(self.graph, emb, numcoms)
            sil2 = embc2.get_sil_score()
            nmi2 = normalized_mutual_info_score(community_labels, embc2.get_clusters())
            
            msg = dataset + "," + str(dim) + "," + str(num_labels) + "," + str(comps) + "," + str(lcc_frac) + "," + str(numcoms) + "," +\
                str(modularity) + "," + str(nmi_gt_community) + "," + str(sil1) + "," + str(nmi1) + "," +\
                str(sil2) + "," + str(nmi2) + "," +\
                str(comps_rg) + "," + str(lcc_frac_rg) + "," +\
                str(numcoms_rg) + "," + str(modularity_rg) + "," + str(nmi_gt_community_rg)

            outf.write(msg + "\n")

        outf.close()



datasets = [
    "karate_club_graph",
    "cora_ml",  
    "citeseer",
    "amazon_electronics_photo",
    "amazon_electronics_computers",
    "pubmed",
    "cora",
    "dblp"
    #"blog-catalog-undirected",
    #"ca-AstroPh-undirected",
    #"ca-CondMat-undirected",
    #"ca-GrQc-undirected",
    #"ca-HepPh-undirected",
    #"cit-HepPh",
    #"cit-HepTh",
    #"facebook-ego-undirected"
]

def clustering_eval_function(d, folder, method):
    graph = DatasetPool.load(d)
    graph.remove_selfloop_edges()
    graph = graph.to_undirected()

    ce = ClusteringEval(d, graph, folder, method)
    ce.eval()


if __name__ == "__main__":
    folder = sys.argv[1]
    method = sys.argv[2]

    for d in datasets:
        print("clustering eval for ", d, " -- ", folder, " -- ", method)
        clustering_eval_function(d, folder, method)
