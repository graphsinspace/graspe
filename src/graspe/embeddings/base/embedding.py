import bisect
import heapq
from abc import ABC, abstractmethod

import numpy as np

from common.graph import Graph


class Embedding(ABC):
    """
    A model-class for graph embedding.

    Attributes
    ----------
    _embedding : dict
        Graph embedding. Keys of the dict are ids of the nodes,
        while the values are numpy arrays that holds the nodes' embeddings.
    _labels : dict
        Nodes' labels.
    _g : common.graph.Graph
        The original graph.
    _d : int
        Dimensionality of the embedding.
    """

    def __init__(self, g, d):
        """
        Parameters
        ----------
        g : common.graph.Graph
            The original graph.
        d : int
            Dimensionality of the embedding.
        """
        self._g = g
        self._d = d
        self._embedding = None
        self._labels = {x[0]: g.get_label(x[0]) for x in g.nodes()}
        self._dists = {}

    def __getitem__(self, node):
        """
        Returns the embedding of the given node.

        Parameters
        ----------
        node : int
            Identifier of the node.

        Returns
        ----------
        - numpy.array - Embedding of the node.
        """
        return self._embedding[node]

    def embed(self):
        """
        Embedding algorithm that must be implemented in subclasses.
        """
        if self.requires_labels() and not self._g.is_labeled():
            raise Exception(
                "{} works only with labeled graphs.".format(type(self).__name__)
            )

    @abstractmethod
    def requires_labels(self):
        """
        Determines if the algorithm requires labels or not.
        """
        pass

    def get_dist(self, node1, node2, save=False):
        if node1 > node2:
            node1, node2 = node2, node1
        if node1 in self._dists and node2 in self._dists[node1]:
            return self._dists[node1][node2]
        dist = np.linalg.norm(self._embedding[node1] - self._embedding[node2])
        if save:
            if not node1 in self._dists:
                self._dists[node1] = {}
            self._dists[node1][node2] = dist
        return dist

    def reconstruct(self, k, cache_dists=False):
        """
        Reconstructs the graph from its embedding.

        Parameters
        ----------
        k : int
            The number of the links in the resulting graph reconstruction.
            Must be an even number (the reconstruction algorithm cannot create a graph with odd number of links).
        cache_dists : boolean
            If true, the pairwise distances will be cached for later use.

        Returns
        ----------
        - common.graph.Graph - Graph reconstruction.
        """
        if k % 2 == 1:
            raise ValueError(f"k must be an even number: {k}")
        nodes = list(self._embedding.keys())
        dists = []
        for i in range(len(nodes)):
            node1 = nodes[i]
            for j in range(i + 1, len(nodes)):
                node2 = nodes[j]
                dist = self.get_dist(node1, node2, cache_dists)
                dists.append(
                    (
                        dist,
                        node1,
                        node2,
                    )
                )
        heapq.heapify(dists)

        g = Graph()
        for node in nodes:
            g.add_node(node, self._labels[node])
        for edge in heapq.nsmallest(k // 2, dists):
            g.add_edge(edge[1], edge[2])
            g.add_edge(edge[2], edge[1])
        return g

    def get_knng(self, k, cache_dists=False):
        """
        Returns k-NN graph based on the embedding.

        k : int
        cache_dists : boolean
            If true, the pairwise distances will be cached for later use.
        """
        nodes = list(self._embedding.keys())
        if len(nodes) < k + 1:
            print("[ERROR] Number of nodes must be greater than k.")
            return None
        dists = {x: [] for x in nodes}
        for i in range(len(nodes)):
            node1 = nodes[i]
            for j in range(i + 1, len(nodes)):
                node2 = nodes[j]
                dist = self.get_dist(node1, node2, cache_dists)
                bisect.insort(
                    dists[node1],
                    (
                        dist,
                        node2,
                    ),
                )
                bisect.insort(
                    dists[node2],
                    (
                        dist,
                        node1,
                    ),
                )
                dists[node1] = dists[node1][: int(k)]
                dists[node2] = dists[node2][: int(k)]

        g = Graph()
        for node in nodes:
            g.add_node(node, self._labels[node])
        for node in nodes:
            for neighbor in dists[node]:
                g.add_edge(node, neighbor[1])
        return g

    def get_label(self, node):
        """
        Returns label for the given node

        Parameters
        ----------
        node : int
            Id of a node.

        If a node with the given id exists, and if that node has a label, the method returns the node's label.
        Otherwise the method returns None.
        """
        if not node in self._labels:
            return None
        return self._labels[node]

    def get_dataset(self):
        """
        Returns the embedding in a dataset format (data, labels).
        """
        data = []
        labels = []
        for node in self._embedding:
            data.append(self._embedding[node])
            labels.append(self._labels[node])
        return data, labels

    def to_file(self, path):
        """
        Outputs the embedding into a file

        Parameters
        ----------
        path : string
            The output path.
        """
        if not self._embedding:
            raise Exception("method embed has not been called yet")

        out = ""
        for node_id in self._embedding:
            l = self._labels[node_id]
            out += (
                str(node_id)
                + ":"
                + str(l if l != None else "")
                + ":"
                + ",".join([str(x) for x in self._embedding[node_id]])
                + "\n"
            )
        f = open(path, "w")
        f.write(out)
        f.close()

    @classmethod
    def from_file(cls, path):
        e = DummyEmbedding()
        e._embedding = {}
        e._labels = {}
        e._dists = {}
        try:
            f = open(path, "r")
        except OSError:
            print("Unexisting file " + path)
            return None
        try:
            for line in f:
                line_s = line.split(":")
                node_id = line_s[0]
                try:
                    node_id = int(node_id)
                except:
                    print(
                        f"WARN: cannot convert node_id '{node_id}' to a numerical value."
                    )
                node_label = line_s[1] if line_s[1] != "" else None
                node_embedding = [float(x) for x in line_s[2].split(",")]
                e._embedding[node_id] = np.array(node_embedding)
                e._labels[node_id] = node_label
        except Exception as e:
            print("Invalid file format.")
            print(e)
            return None
        f.close()
        return e


class DummyEmbedding(Embedding):
    def __init__(self):
        pass

    def embed(self):
        pass

    def requires_labels(self):
        pass
