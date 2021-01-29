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
        workers=1,
        seed=42
    ):
        super().__init__(g,d)
        self._p = p
        self._q = q
        self._walk_length = walk_length
        self._num_walks = num_walks
        self._workers = workers
        self._seed = seed


    def embed(self):
        nodes = self._g.nodes()

        node2vec = Node2Vec(graph=self._g.to_networkx(), p=self._p, q=self._q, dimensions=self._d, walk_length=self._walk_length, num_walks=self._num_walks, workers=self._workers, seed=self._seed)
        embedding = node2vec.fit()
        vectors = embedding.wv

        self._embedding = {}
        for node in nodes:
            nid = node[0]
            strnid = str(nid)
            emb = vectors[strnid]
            self._embedding[nid] = emb
