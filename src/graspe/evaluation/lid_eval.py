"""
GRASPE -- Graphs in Space: Graph Embeddings for Machine Learning on Complex Data

LID-based evaluation

author: svc@dmi.uns.ac.rs
"""

import heapq
import random
from abc import ABC, abstractmethod
from statistics import mean

import networkx as nx
import numpy as np
from scipy.stats import kendalltau


class LIDEstimator(ABC):
    """
    Base class for node LID estimators
    """

    def __init__(self, estimator_name, graph):
        self.estimator_name = estimator_name
        self.graph = graph
        self.nx_graph = graph.to_networkx()
        self.nodes = graph.nodes()
        self.lid_values = dict()

    @abstractmethod
    def estimate_lids(self):
        pass

    def get_lid(self, node_id):
        return self.lid_values[node_id]

    def get_avg_lid(self):
        return mean(self.lid_values.values())

    def print_lid_values(self):
        print("\nLID estimates by", self.estimator_name)
        for d in self.lid_values:
            print("Node ", d, "LID =", self.lid_values[d])


class LIDMLEEstimator(LIDEstimator):
    """
    Base class for node LID MLE estimators
    """

    def __init__(self, estimator_name, graph, k):
        self.estimator_name = estimator_name
        self.k = k
        super().__init__(estimator_name, graph)

    @abstractmethod
    def compute_distance(self, src, dst):
        pass

    def estimate_lids(self):
        numn = len(self.nodes)
        for i in range(numn):
            if i % 100 == 0:
                print(i, "/", numn, " nodes finished")
            src = self.nodes[i][0]
            src_dists = []
            for j in range(len(self.nodes)):
                dst = self.nodes[j][0]
                if src != dst:
                    d = self.compute_distance(src, dst)
                    src_dists.append(d)

            src_lid = self.estimate_lid(src_dists)
            self.lid_values[src] = src_lid

    def estimate_lids_bfs(self, stop_at_k=False, max_depth=None):
        numn = len(self.nodes)
        g = self.graph.to_networkx()

        for i in range(numn):
            if i % 100 == 0:
                print(i, "/", numn, " nodes finished")
            src = self.nodes[i][0]
            src_dists = []

            queue = [(src, 0)]
            visited = set([src])
            while len(queue) > 0:
                if stop_at_k and len(src_dists) > self.k:
                    break

                curr = queue.pop(0)
                dst = curr[0]
                depth = curr[1]

                if max_depth != None and depth > max_depth:
                    break

                if dst != src:
                    d = self.compute_distance(src, dst)
                    src_dists.append(d)

                cneis = nx.all_neighbors(g, dst)

                for c in cneis:
                    if not c in visited:
                        visited.add(c)
                        queue.append((c, depth + 1))

            src_lid = self.estimate_lid(src_dists)
            self.lid_values[src] = src_lid

    def estimate_lid(self, distances):
        heapq.heapify(distances)
        k_smallest = heapq.nsmallest(self.k, distances)
        kss = len(k_smallest)

        s = 0
        for j in range(0, kss):
            s += np.log(k_smallest[j] / k_smallest[kss - 1])

        s /= kss
        lid = -1.0 / s if s < 0 else 1
        return lid


class EmbLIDMLEEstimator(LIDMLEEstimator):
    """
    MLE estimator for node LIDs in the embedded space
    """

    def __init__(self, graph, embedding, k):
        self.node_vectors = [embedding[n[0]] for n in graph.nodes()]
        super().__init__("EMB-LID", graph, k)

    def compute_distance(self, src, dst):
        return np.linalg.norm(self.node_vectors[src] - self.node_vectors[dst])


def shortest_path_distance(nx_graph, src, dst):
    try:
        d = nx.shortest_path_length(nx_graph, source=src, target=dst)
        return d
    except nx.NetworkXNoPath:
        return nx_graph.number_of_nodes()


class GLIDShortestPathMLEEstimator(LIDMLEEstimator):
    """
    MLE estimator for node LIDs based on shortest path distance
    """

    def __init__(self, graph, k):
        super().__init__("G-SP-LID", graph, k)

    def compute_distance(self, src, dst):
        return shortest_path_distance(self.nx_graph, src, dst)


class GLIDSimRankMLEEstimator(LIDMLEEstimator):
    """
    MLE estimator for node LIDs based on SimRank node similarity
    """

    def __init__(self, graph, k):
        self.simrank = nx.simrank_similarity_numpy(graph.to_networkx())
        super().__init__("G-SimRank-LID", graph, k)

    def compute_distance(self, src, dst):
        sr = self.simrank[src][dst]
        return 1 if sr == 0 else 1 / sr


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


class GLIDJaccardMLEEstimator(LIDMLEEstimator):
    """
    MLE estimator for node LIDs based on Jaccard node similarity
    """

    def __init__(self, graph, k):
        super().__init__("G-Jaccard-LID", graph, k)

    def compute_distance(self, src, dst):
        return jaccard_distance(self.nx_graph, src, dst)


class GLIDAdamicAdarMLEEstimator(LIDMLEEstimator):
    """
    MLE estimator for node LIDs based on Adamic-Adar node similarity
    """

    def __init__(self, graph, k):
        super().__init__("G-AdamicAdar-LID", graph, k)

    def compute_distance(self, src, dst):
        return adamic_adar_distance(self.nx_graph, src, dst)


class DistanceAnalyzer:
    """
    DistanceAnalyzer computes both graph-based and embedding-based
    distances and correlates them
    """

    def __init__(self, graph, embedding):
        self.graph = graph
        self.embedding = embedding
        self.nx_graph = graph.to_networkx()
        self.node_vectors = [embedding[n[0]] for n in graph.nodes()]
        self.emb_dist = []
        self.sp_dist = []

        # print("Computing simrank matrix")
        # self.simrank = nx.simrank_similarity_numpy(graph.to_networkx())
        # self.compute_distances()

    def compute_distances(self):
        print("Computing distances...")
        self.emb_dist = []
        self.sp_dist = []
        # self.sr_dist = []
        # self.jaccard_dist = []
        # self.aa_dist = []

        nodes = self.graph.nodes()
        num_nodes = len(nodes)
        for i in range(num_nodes):
            if i % 50 == 0:
                print(i, "/", num_nodes, " nodes finished")
            src = nodes[i][0]
            for j in range(num_nodes):
                dst = nodes[j][0]
                if src != dst:
                    self.emb_dist.append(
                        np.linalg.norm(self.node_vectors[src] - self.node_vectors[dst])
                    )
                    self.sp_dist.append(shortest_path_distance(self.nx_graph, src, dst))
                    # self.sr_dist.append(1 / self.simrank[src][dst])
                    # self.jaccard_dist.append(jaccard_distance(self.nx_graph, src, dst))
                    # self.aa_dist.append(adamic_adar_distance(self.nx_graph, src, dst))

    def compute_distances_rnd_sample(self, sample_size, seed=None):
        print("Computing distances...")
        self.emb_dist = []
        self.sp_dist = []

        if seed != None:
            random.seed(seed)

        nodes = self.graph.nodes()
        num_nodes = len(nodes)
        if sample_size > num_nodes:
            sample_size = num_nodes

        sample1 = random.sample(nodes, sample_size)
        sample2 = random.sample(nodes, sample_size)

        for i in range(sample_size):
            src = sample1[i][0]
            dst = sample2[i][0]
            if src != dst:
                self.emb_dist.append(
                    np.linalg.norm(self.node_vectors[src] - self.node_vectors[dst])
                )
                self.sp_dist.append(shortest_path_distance(self.nx_graph, src, dst))

    def distance_correlations(self, to_print=True):
        corr = kendalltau(self.emb_dist, self.sp_dist)
        if to_print:
            print("KendallTau(EMB, ShortestPath) = ", corr)
        return corr

        """
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

        return corrs
        """


#  LID estimated using natural communities


class Community(object):
    """
    Taken from https://github.com/GiulioRossetti/cdlib/blob/master/cdlib/algorithms/internal/lfm.py
    G. Rossetti, L. Milli, R. Cazabet. CDlib: a Python Library to Extract, Compare and Evaluate Communities
        from Complex Networks. Applied Network Science Journal. 2019. DOI:10.1007/s41109-019-0165-9
    """

    def __init__(self, g, alpha=1.0):
        self.g = g
        self.alpha = alpha
        self.nodes = set()
        self.k_in = 0
        self.k_out = 0

    def add_node(self, node):
        neighbors = set(self.g.neighbors(node))
        node_k_in = len(neighbors & self.nodes)
        node_k_out = len(neighbors) - node_k_in
        self.nodes.add(node)
        self.k_in += 2 * node_k_in
        self.k_out = self.k_out + node_k_out - node_k_in

    def remove_vertex(self, node):
        neighbors = set(self.g.neighbors(node))
        community_nodes = self.nodes
        node_k_in = len(neighbors & community_nodes)
        node_k_out = len(neighbors) - node_k_in
        self.nodes.remove(node)
        self.k_in -= 2 * node_k_in
        self.k_out = self.k_out - node_k_out + node_k_in

    def cal_add_fitness(self, node):
        neighbors = set(self.g.neighbors(node))
        old_k_in = self.k_in
        old_k_out = self.k_out
        vertex_k_in = len(neighbors & self.nodes)
        vertex_k_out = len(neighbors) - vertex_k_in
        new_k_in = old_k_in + 2 * vertex_k_in
        new_k_out = old_k_out + vertex_k_out - vertex_k_in
        new_fitness = new_k_in / (new_k_in + new_k_out) ** self.alpha
        old_fitness = old_k_in / (old_k_in + old_k_out) ** self.alpha
        return new_fitness - old_fitness

    def cal_remove_fitness(self, node):
        neighbors = set(self.g.neighbors(node))
        new_k_in = self.k_in
        new_k_out = self.k_out
        node_k_in = len(neighbors & self.nodes)
        node_k_out = len(neighbors) - node_k_in
        old_k_in = new_k_in - 2 * node_k_in
        old_k_out = new_k_out - node_k_out + node_k_in
        old_fitness = old_k_in / (old_k_in + old_k_out) ** self.alpha
        new_fitness = new_k_in / (new_k_in + new_k_out) ** self.alpha
        return new_fitness - old_fitness

    def recalculate(self):
        for vid in self.nodes:
            fitness = self.cal_remove_fitness(vid)
            if fitness < 0.0:
                return vid
        return None

    def get_neighbors(self):
        neighbors = set()
        for node in self.nodes:
            neighbors.update(set(self.g.neighbors(node)) - self.nodes)
        return neighbors

    def get_fitness(self):
        return float(self.k_in) / ((self.k_in + self.k_out) ** self.alpha)


class LFMnx:
    """
    Modified from https://github.com/GiulioRossetti/cdlib/blob/master/cdlib/algorithms/internal/lfm.py
    G. Rossetti, L. Milli, R. Cazabet. CDlib: a Python Library to Extract, Compare and Evaluate Communities
        from Complex Networks. Applied Network Science Journal. 2019. DOI:10.1007/s41109-019-0165-9
    """

    def __init__(self, g, alpha=1.0):
        if g.is_directed():
            self.g = g.to_undirected()
        else:
            self.g = g
        self.alpha = alpha

    def identify_natural_community(self, seed):
        c = Community(self.g, self.alpha)
        c.add_node(seed)

        to_be_examined = c.get_neighbors()
        while to_be_examined:
            # largest fitness to be added
            m = {}
            for node in to_be_examined:
                fitness = c.cal_add_fitness(node)
                m[node] = fitness
            to_be_add = sorted(m.items(), key=lambda x: x[1], reverse=True)[0]

            # stop condition
            if to_be_add[1] < 0.0:
                break
            c.add_node(to_be_add[0])

            to_be_remove = c.recalculate()
            while to_be_remove is not None:
                c.remove_vertex(to_be_remove)
                to_be_remove = c.recalculate()

            to_be_examined = c.get_neighbors()

        return list(c.nodes)


class NLIDEstimator(LIDEstimator):
    """
    Base class for LID estimators based on natural communities
    """

    def __init__(self, estimator_name, graph):
        super().__init__(estimator_name, graph)
        self.community_detector = LFMnx(graph.to_networkx())

    @abstractmethod
    def compute_distance(self, src, dst):
        pass

    def max_community_distance(self, src, src_community):
        maxd = 0
        for n in src_community:
            d = self.compute_distance(src, n)
            if d > maxd:
                maxd = d

        return maxd

    def estimate_lids(self):
        numn = len(self.nodes)
        for i in range(numn):
            src = self.nodes[i][0]
            src_community = self.community_detector.identify_natural_community(src)
            if len(src_community) <= 1:
                self.lid_values[src] = 0
            else:
                maxd = self.max_community_distance(src, src_community)

                # count how many nodes are from src at maxd distance
                counter = 0
                for j in range(numn):
                    dst = self.nodes[j][0]
                    d = self.compute_distance(src, dst)
                    if d <= maxd:
                        counter += 1

                # print(len(src_community), counter, "max_D = ", maxd)
                self.lid_values[src] = -1.0 * np.log(len(src_community) / counter)


class NLIDShortestPathEstimator(NLIDEstimator):
    """
    NLID estimated using shortest path distance
    """

    def __init__(self, graph):
        super().__init__("NLID-SP", graph)

    def compute_distance(self, src, dst):
        return shortest_path_distance(self.nx_graph, src, dst)


class NLIDSimRankEstimator(NLIDEstimator):
    """
    NLID estimated using SimRank
    """

    def __init__(self, graph):
        self.simrank = nx.simrank_similarity_numpy(graph.to_networkx())
        super().__init__("NLID-SR", graph)

    def compute_distance(self, src, dst):
        sr = self.simrank[src][dst]
        return 1 if sr == 0 else 1 / sr


class NLIDJaccardEstimator(NLIDEstimator):
    """
    NLID estimated using Jaccard similarity
    """

    def __init__(self, graph):
        super().__init__("NLID-Jaccard", graph)

    def compute_distance(self, src, dst):
        return jaccard_distance(self.nx_graph, src, dst)


class NLIDAdamicAdarEstimator(NLIDEstimator):
    """
    NLID estimated using AdamicAdar similarity
    """

    def __init__(self, graph):
        super().__init__("NLID-Jaccard", graph)

    def compute_distance(self, src, dst):
        return adamic_adar_distance(self.nx_graph, src, dst)
