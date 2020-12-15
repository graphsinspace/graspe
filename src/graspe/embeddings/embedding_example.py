from embeddings.embedding import Embedding

class EmbeddingExample(Embedding):
    """
    Implementation of a concrete embedding algorithm.

    Attributes
    ----------
    _embedding: ?
        Graph embedding.
    """

    def _embed(self, g, d, args={}):
        """
        Embedding algorithm X.

        Parameters
        ----------
        g : common.graph.Graph
            The original graph.
        d : int
            Dimensionality of the embedding.
        args : dict
            Additional arguments for the specific algorithms.
        """
        super()._embed(g, d, args)
        
        # Perform embedding.