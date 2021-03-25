from common.graph import Graph
import random as random
from random import sample
from embeddings.embedding_node2vec import Node2VecEmbedding
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
    def __init__(self, graph, h, embedding_method):
        self._graph = graph
        self._h = h
        self._embedding_method = embedding_method
        self.__predict()

    def __predict(self):
        edgeset = self._graph.edges()
        hes_seed = 20
        random.seed(hes_seed)
        hesample = random.sample(edgeset, self._h)
        self._hidden_edges = []
        for he in hesample:
            src = he[0]
            dst = he[1]
            if src > dst:
                src = he[1]
                dst = he[0]
            self._hidden_edges.append((src, dst))

        print("Sampled edges are")
        for he in self._hidden_edges:
            print(he)
        print("\n")

        nodeset = self._graph.nodes()
        self._newgraph = Graph()
        for node in nodeset:
            self._newgraph.add_node(node[0], node[1])

        for edge in edgeset:
            if edge not in self._hidden_edges:
                self._newgraph.add_edge(edge[0], edge[1])

        self._newembedding = Node2VecEmbedding(self._newgraph, 10, 0.1, 0.5, seed=42)
        self._newembedding.embed()

        self._dists = []
        for i in range(len(nodeset)):
            node1 = nodeset[i]
            for j in range(i + 1, len(nodeset)):
                node2 = nodeset[j]
                e = (node1[0], node2[0])
                if e not in self._newgraph.edges():
                    d = np.linalg.norm(
                        self._newembedding._embedding[node1[0]]
                        - self._newembedding._embedding[node2[0]]
                    )
                    hq.heappush(self._dists, DistanceNodesPair(d, (node1[0], node2[0])))

        pred = hq.nsmallest(self._h, self._dists)
        self._prediction = [p.get_node_pair() for p in pred]
        print("\nPredictions are")
        for p in pred:
            print(
                p.get_node_pair(),
                " dist = ",
                p.get_dist(),
                " among hidden links",
                p.get_node_pair() in self._hidden_edges,
            )

    def get_precision_at_k(self, k):
        cnt = 0
        for i in range(0, k):
            if self._prediction[i] in self._hidden_edges:
                cnt += 1

        return float(cnt) / float(k)

    def get_map(self, k):
        nodeset = self._graph.nodes()
        map_sum = 0.0
        relevant_nodes = 0

        for node in nodeset:
            prednode = 0
            enode = 0

            for i in range(0, k):
                e = self._prediction[i]
                if e in self._hidden_edges and (node[0] == e[0] or node[0] == e[1]):
                    prednode += 1

            for e in self._hidden_edges:
                if node[0] == e[0] or node[0] == e[1]:
                    enode += 1

            if enode != 0:
                relevant_nodes += 1
                score = float(prednode) / float(enode)
                # print(node, " -- MAP = ", score)
                map_sum += score

        return map_sum / float(relevant_nodes)
