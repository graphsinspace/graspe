from deepwalk import graph as dwgraph
from gensim.models import Word2Vec
from common.graph import Graph
from embeddings.base.embedding import Embedding

import numpy as np


class DeepWalkEmbedding(Embedding):
    def __init__(
        self,
        g,
        d,
        path_number=10,
        path_length=80,
        workers=4,
        window=5,
    ):

        super().__init__(g, d)
        self._path_number = path_number
        self._path_length = path_length
        self._workers = workers
        self._window = window

    def from_networkx(self, G_input, undirected=True):
        G = dwgraph.Graph()

        for x in G_input.nodes():
            G[x].append(x)
            for y in G_input[x]:
                G[x].append(y)

        if undirected:
            G.make_undirected()

        return G

    def embed(self):
        super().embed()

        nodes = self._g.nodes()
        nxg = self._g.to_networkx()
        dwg = self.from_networkx(nxg)

        walks = dwgraph.build_deepwalk_corpus(
            dwg, num_paths=self._path_number, path_length=self._path_length
        )
        model = Word2Vec(walks, size=self._d, window=self._window)
        vectors = model.wv

        self._embedding = {}

        for node in nodes:
            nid = node[0]
            strnid = str(nid)
            emb = vectors[strnid] if strnid in vectors else np.zeros(self._d)
            self._embedding[nid] = emb

    def requires_labels(self):
        return False
