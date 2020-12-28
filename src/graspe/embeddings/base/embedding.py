from abc import ABC, abstractmethod

from common.graph import Graph


class Embedding(ABC):
    """
    A model-class for graph embedding.

    Attributes
    ----------
    _embedding: ?
        Graph embedding.
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

    def __getitem__(self, node):
        """
        Returns the embedding of the given node.

        Parameters
        ----------
        node : ?
            Identifier of the node.

        Returns
        ----------
        - ? - Embedding of the node.
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
        return Graph()

    def to_file(self, path):
        if not self._embedding:
            raise Exception('method embed has not been called yet')
        
        out = ''
        for node_id in self._embedding:
            out += str(node_id) + ':' + ','.join([str(x) for x in self._embedding[node_id]]) + '\n'
        f = open(path, "w")
        f.write(out)
        f.close()