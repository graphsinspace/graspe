"""
GRASPE -- Graphs in Space: Graph Embeddings for Machine Learning on Complex Data

LID-based evaluation

author: svc@dmi.uns.ac.rs
"""

import numpy as np
import networkx as nx
import heapq
from statistics import mean
from scipy.stats import kendalltau

from abc import ABC, abstractmethod  


class LID_MLE_Estimator(ABC):
    """
    Base class for node LID MLE estimators
    """
    def __init__(self, estimator_name, graph, k):
        self.estimator_name = estimator_name
        self.k = k
        self.graph = graph
        self.nodes = graph.nodes()
        self.lid_values = dict()
        self.estimate_lids()

    
    def get_lid(self, node_id):
        return self.lid_values[node_id]

    
    def get_avg_lid(self):
        return mean(self.lid_values.values())

    
    def print_lid_values(self):
        print("\nLID estimates by", self.estimator_name, " k = ", self.k)
        for d in self.lid_values:
            print("Node ", d, "LID =", self.lid_values[d])

    
    @abstractmethod
    def compute_distance(self, src, dst):
        pass

    
    def estimate_lids(self):
        for i in range(len(self.nodes)):
            src = self.nodes[i][0]
            src_dists = []
            for j in range(len(self.nodes)):
                dst = self.nodes[j][0]
                if src != dst:
                    d = self.compute_distance(src, dst)
                    src_dists.append(d)

            src_lid = self.estimate_lid(src_dists)
            self.lid_values[src] = src_lid

    
    def estimate_lid(self, distances):
        heapq.heapify(distances)
        k_smallest = heapq.nsmallest(self.k, distances)
        s = 0
        for j in range(0, len(k_smallest)):
            s += np.log(k_smallest[j] / k_smallest[self.k - 1])

        s /= self.k
        lid = -1.0 / s if s < 0 else 1
        return lid




class EmbLID_MLE_Estimator(LID_MLE_Estimator):
    """
    MLE estimator for node LIDs in the embedded space
    """
    def __init__(self, graph, embedding, k):
        self.node_vectors = [embedding[n[0]] for n in graph.nodes()]
        super().__init__("EMB-LID", graph, k)
        

    def compute_distance(self, src, dst):
        return np.linalg.norm(self.node_vectors[src] - self.node_vectors[dst])   
    



class GLID_ShortestPath_MLE_Estimator(LID_MLE_Estimator):
    """
    MLE estimator for node LIDs based on shortest path distance
    """
    def __init__(self, graph, k):
        self.nx_graph = graph.to_networkx()
        super().__init__("G-SP-LID", graph, k)


    def compute_distance(self, src, dst):
        return nx.shortest_path_length(self.nx_graph, source=src, target=dst) 




class GLID_SimRank_MLE_Estimator(LID_MLE_Estimator):
    """
    MLE estimator for node LIDs based on SimRank node similarity
    """
    def __init__(self, graph, k):
        self.simrank = nx.simrank_similarity_numpy(graph.to_networkx())
        super().__init__("G-SimRank-LID", graph, k)


    def compute_distance(self, src, dst):
        return 1 / self.simrank[src][dst]   




def jaccard_similarity(nx_graph, src, dst):
    src_nei = set(nx.all_neighbors(nx_graph, src))
    dst_nei = set(nx.all_neighbors(nx_graph, dst))
    jaccard = len(src_nei.intersection(dst_nei)) / len(src_nei.union(dst_nei))
    return jaccard


def jaccard_distance(nx_graph, src, dst):
    j = jaccard_similarity(nx_graph, src, dst)
    return 1 / j if j > 0 else 1


def adamic_adar_similarity(nx_graph, src, dst):
    src_nei = set(nx.all_neighbors(nx_graph, src))
    dst_nei = set(nx.all_neighbors(nx_graph, dst))
    common = src_nei.intersection(dst_nei)
    aa = 0
    for c in common:
        aa += 1 / np.log(nx_graph.degree(c))
    
    return aa


def adamic_adar_distance(nx_graph, src, dst):
    aa = adamic_adar_similarity(nx_graph, src, dst)
    return 1 / aa if aa > 0 else 1


class GLID_JaccardSim_MLE_Estimator(LID_MLE_Estimator):
    """
    MLE estimator for node LIDs based on Jaccard node similarity
    """
    def __init__(self, graph, k):
        self.nx_graph = graph.to_networkx()
        super().__init__("G-Jaccard-LID", graph, k)


    def compute_distance(self, src, dst):
        return jaccard_distance(self.nx_graph, src, dst)




class GLID_AdamicAdar_MLE_Estimator(LID_MLE_Estimator):
    """
    MLE estimator for node LIDs based on Adamic-Adar node similarity
    """
    def __init__(self, graph, k):
        self.nx_graph = graph.to_networkx()
        super().__init__("G-AdamicAdar-LID", graph, k)

    def compute_distance(self, src, dst):
        return adamic_adar_distance(self.nx_graph, src, dst)
    

class DistanceAnalyzer:
    def __init__(self, graph, embedding):
        self.graph = graph
        self.embedding = embedding
        self.nx_graph = graph.to_networkx()
        self.node_vectors = [embedding[n[0]] for n in graph.nodes()]
        self.simrank = nx.simrank_similarity_numpy(graph.to_networkx())
        self.compute_distances()


    def compute_distances(self):
        self.emb_dist = []
        self.sp_dist = []
        self.sr_dist = []
        self.jaccard_dist = []
        self.aa_dist = []

        nodes = self.graph.nodes()
        for i in range(len(nodes)):
            src = nodes[i][0]
            for j in range(len(nodes)):
                dst = nodes[j][0]
                if src != dst:
                    self.emb_dist.append(np.linalg.norm(self.node_vectors[src] - self.node_vectors[dst]))
                    self.sp_dist.append(nx.shortest_path_length(self.nx_graph, source=src, target=dst))
                    self.sr_dist.append(1 / self.simrank[src][dst]) 
                    self.jaccard_dist.append(jaccard_distance(self.nx_graph, src, dst))
                    self.aa_dist.append(adamic_adar_distance(self.nx_graph, src, dst))


    def distance_correlations(self, to_print=True):
        corrs = [
            kendalltau(self.emb_dist, self.sp_dist),
            kendalltau(self.emb_dist, self.sr_dist),
            kendalltau(self.emb_dist, self.jaccard_dist),
            kendalltau(self.emb_dist, self.aa_dist)
        ]

        if to_print:
            print("\nKendallTau distance correlations (Embedding -- Graph)")
            print("KendallTau(EMB, ShortestPath) = ", corrs[0])
            print("KendallTau(EMB, SimRank) = ", corrs[1])
            print("KendallTau(EMB, Jaccard) = ", corrs[2])
            print("KendallTau(EMB, AdamicAdar) = ", corrs[3])
