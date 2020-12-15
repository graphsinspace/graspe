from abc import ABC, abstractmethod

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
        self._embed(g, d)

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
    
    @abstractmethod
    def _embed(self, g, d, args={}):
        """
        Embedding algorithm that must be implemented in subclasses.

        Parameters
        ----------
        g : common.graph.Graph
            The original graph.
        d : int
            Dimensionality of the embedding.
        args : dict
            Additional arguments for the specific algorithms.
        """
        self._embedding = {}