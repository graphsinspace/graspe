"""
GRASPE -- Graphs in Space: Graph Embeddings for Machine Learning on Complex Data

LID-based evaluation

author: svc@dmi.uns.ac.rs
"""

import heapq
from abc import ABC, abstractmethod
from statistics import mean, stdev

import networkx as nx
import numpy as np


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

    def get_lid_values(self):
        return self.lid_values

    def get_avg_lid(self):
        return mean(self.lid_values.values())

    def get_stdev_lid(self):
        return stdev(list(self.lid_values.values()))

    def get_max_lid(self):
        return max(list(self.lid_values.values()))

    def get_min_lid(self):
        return min(list(self.lid_values.values()))

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
        self.node_vectors = {}  #[embedding[n[0]] for n in graph.nodes()]
        for n in graph.nodes():
            self.node_vectors[n[0]] = embedding[n[0]]
            
        super().__init__("EMB-LID", graph, k)

    def compute_distance(self, src, dst):
        return np.linalg.norm(self.node_vectors[src] - self.node_vectors[dst])


def shortest_path_distance(nx_graph, src, dst):
    try:
        d = nx.shortest_path_length(nx_graph, source=src, target=dst)
        return d
    except nx.NetworkXNoPath:
        return nx_graph.number_of_nodes()



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


class NCLIDEstimator(LIDEstimator):
    """
    NCLID estimator based on natural communities
    """

    def __init__(self, graph, alpha=1.0):
        super().__init__("NCLID", graph)
        self.community_detector = LFMnx(graph.to_networkx(), alpha=alpha)
        self.nc_size = dict()
        self.max_nc_size = 0
        self.natural_community = dict()

    def compute_distance(self, src, dst):
        return shortest_path_distance(self.nx_graph, src, dst)

    def max_community_distance(self, src, src_community):
        maxd = 0
        for n in src_community:
            d = self.compute_distance(src, n)
            if d > maxd:
                maxd = d

        return maxd


    def count_nodes_at_distance(self, src, maxd):
        # count how many nodes are from src at maxd distance
        counter = 1   # src counted
                
        queue = [(src, 0)]
        visited = set([src])
        while len(queue) > 0:
            curr = queue.pop(0)
            dst, depth = curr[0], curr[1]
                    
            if depth >= maxd:
                break
                    
            cneis = nx.all_neighbors(self.nx_graph, dst)

            for c in cneis:
                if not c in visited:
                    visited.add(c)
                    queue.append((c, depth + 1))
                    counter += 1

        return counter

    def estimate_lids(self):
        numn = len(self.nodes)
        for i in range(numn):
            src = self.nodes[i][0]
            src_community = self.community_detector.identify_natural_community(src)
            self.natural_community[src] = src_community
            len_src_community = len(src_community)
            if len(src_community) <= 1:
                self.lid_values[src] = 0
                self.nc_size[src] = 0
            else:
                maxd = self.max_community_distance(src, src_community)
                counter = self.count_nodes_at_distance(src, maxd)

                self.lid_values[src] = -1.0 * np.log(len_src_community / counter)
                self.nc_size[src] = len_src_community
                if len_src_community > self.max_nc_size:
                    self.max_nc_size = len_src_community


    def is_in_natural_community(self, seed_node, other_node):
        return other_node in self.natural_community[seed_node]


    def nc_size_distr(self, out_file):
        dstr = self.nc_size_distr_str()
        outf = open(out_file, "w")
        outf.write(dstr)
        outf.close()


    def nc_size_distr_str(self):
        distr = [0] * (self.max_nc_size + 1)
        numn = len(self.nodes)
        for i in range(numn):
            src = self.nodes[i][0]
            distr[self.nc_size[src]] += 1

        dstr = "NC_SIZE,#NODES\n"
        for i in range(len(distr)):
            if distr[i] == 0:
                continue

            dstr += str(i) + "," + str(distr[i]) + "\n"
            
        return dstr


    def nc_len(self, node):
        return self.nc_size[node]
