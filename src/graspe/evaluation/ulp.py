"""
GRASPE -- Graphs in Space: Graph Embeddings for Machine Learning on Complex Data

Unsupervised link prediction

author: {lucy, svc}@dmi.uns.ac.rs
"""

from common.dataset_pool import DatasetPool
from random import sample
import numpy as np
import heapq as hq


class DistanceNodesPair:
    def __init__(self, dist, node_pair):
        self.dist = dist
        self.node_pair = node_pair

    def __lt__(self, other):
        return self.dist < other.dist

    def __gt__(self, other):
        return self.dist > other.dist

    def __le__(self, other):
        return self.dist <= other.dist

    def __ge__(self, other):
        return self.dist >= other.dist

    def __eq__(self, other):
        return self.dist == other.dist

    def __ne__(self, other):
        return self.dist != other.dist

    def get_node_pair(self):
        return self.node_pair

    def get_dist(self):
        return self.dist


class UnsupervisedLinkPrediction:
    def __init__(self, dataset, emb_factory, hidden_fraction=0.05):
        self.dataset = dataset
        self.emb_factory = emb_factory
        self.hidden_fraction=hidden_fraction

        # load graph
        g = DatasetPool.load(self.dataset)
        g.remove_selfloop_edges()
        self.graph = g.to_undirected()

        # sample hidden edges
        self.hide_edges()

        # remove hidden edges
        self.remove_hidden_edges()


    def hide_edges(self):
        edgeset = self.graph.edges()
        normedges = []
        for e in edgeset:
            if e[0] < e[1]:
                normedges.append((e[0], e[1]))
        
        num_edges = len(edgeset) // 2
        self.sample_size = round(num_edges * self.hidden_fraction)
        self.hesample = sample(normedges, self.sample_size)
        

    def remove_hidden_edges(self):
        for e in self.hesample:
            self.graph.remove_edge(e[0], e[1])
            self.graph.remove_edge(e[1], e[0])


    def get_graph(self):
        return self.graph


    def eval(self):
        gmethods = self.emb_factory.get_gmethods()
        for method in gmethods:
            method[0].embed()
            self.compute_emb_dist(method[0])
            pred = hq.nsmallest(self.sample_size, self.dists)
            prec = self.get_precision(pred)
            out = self.dataset + "," + method[1] + "," + str(method[2]) + "," + str(prec)
            print(out)

    
    def compute_emb_dist(self, embedding):
        self.dists = []

        nodeset = self.graph.nodes()
        for i in range(len(nodeset)):
            node1 = nodeset[i]
            for j in range(i + 1, len(nodeset)):
                node2 = nodeset[j]
                e1 = (node1[0], node2[0])
                e2 = (node2[0], node1[0])
                if e1 not in self.graph.edges() and e2 not in self.graph.edges():
                    dist = np.linalg.norm(embedding[node1[0]] - embedding[node2[0]])
                    hq.heappush(self.dists, DistanceNodesPair(dist, (node1[0], node2[0])))


    def get_precision(self, pred):
        cnt = 0
        for p in pred:
            np = p.get_node_pair()
            pair1 = (np[0], np[1])
            pair2 = (np[1], np[0])
            if pair1 in self.hesample or pair2 in self.hesample:
                cnt += 1

        return float(cnt) / float(len(pred))