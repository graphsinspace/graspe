import numpy as np
from abc import ABC, abstractmethod
from common.graph import Graph
import heapq


class Embedding(ABC):
    """
    A model-class for graph embedding.

    Attributes
    ----------
    _embedding : dict
        Graph embedding. Keys of the dict are ids of the nodes,
        while the values are numpy arrays that holds the nodes' embeddings.
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

    @abstractmethod
    def embed(self):
        """
        Embedding algorithm that must be implemented in subclasses.
        """
        pass

    def reconstruct(self, k):
        """
        Reconstructs the graph from its embedding.

        Parameters
        ----------
        k : int
            The number of the links in the resulting graph reconstruction.

        Returns
        ----------
        - common.graph.Graph - Graph reconstruction.
        """
        nodes = list(self._embedding.keys())
        dists = []
        for i in range(len(nodes)):
            node1 = nodes[i]
            for j in range(i + 1, len(nodes)):
                node2 = nodes[j]
                dists.append(
                    (
                        np.linalg.norm(self._embedding[node1] - self._embedding[node2]),
                        node1,
                        node2,
                    )
                )
        heapq.heapify(dists)

        g = Graph()
        for node in nodes:
            g.add_node(node, self._g.get_label(node))
        for edge in heapq.nsmallest(k, dists):
            g.add_edge(edge[1], edge[2])
        return g

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
            out += (
                str(node_id)
                + ":"
                + ",".join([str(x) for x in self._embedding[node_id]])
                + "\n"
            )
        f = open(path, "w")
        f.write(out)
        f.close()
