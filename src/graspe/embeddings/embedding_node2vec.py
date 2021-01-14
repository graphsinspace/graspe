from common.graph import Graph
from embeddings.base.embedding import Embedding

from node2vec import Node2Vec

class Node2VecEmbedding(Embedding):
    def __init__(
        self,
        g,
        d,
        p,
        q,
        walk_length=10,
        num_walks=200,
        workers=1
    ):
        super().__init__(g,d)
        self._p = p
        self._q = q
        self._walk_length = walk_length
        self._num_walks = num_walks
        self._workers = workers


    def embed(self):
        print("usao u embed")
        nodes = self._g.nodes()
        print(nodes)
        print(self._g.edges())

        node2vec = Node2Vec(graph=self._g, p=self._p, q=self._q, dimensions=self._d, walk_length=self._walk_length, num_walks=self._num_walks, workers=self._workers)
        embedding = node2vec.fit()
        self._embedding = {}
        i = 0
        for node in embedding:
            self._embedding[nodes[i][0]] = node
            i += 1
