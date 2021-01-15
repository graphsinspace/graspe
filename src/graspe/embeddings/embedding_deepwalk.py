from common.graph import Graph
from embeddings.base.embedding import Embedding

import numpy as np

from karateclub import DeepWalk


class DeepWalkEmbedding(Embedding):
    def __init__(
        self,
        g,
        d,
        walk_number=10,
        walk_length=80,
        workers=4,
        window_size=5,
        epochs=1,
        learning_rate=0.05,
        min_count=1,
        seed=42,
    ):

        super().__init__(g, d)
        self._walk_number = walk_number
        self._walk_length = walk_length
        self._workers = workers
        self._window_size = window_size
        self._epochs = epochs
        self._learning_rate = learning_rate
        self._min_count = min_count
        self._seed = seed

    def embed(self):

        nodes = self._g.nodes()

        deep_walk = DeepWalk(
            dimensions=self._d,
            walk_number=self._walk_number,
            walk_length=self._walk_length,
            workers=self._workers,
            epochs=self._epochs,
            learning_rate=self._learning_rate,
            min_count=self._min_count,
            seed=self._seed,
        )

        deep_walk.fit(self._g.to_networkx().to_undirected())

        embedding = deep_walk.get_embedding()

        self._embedding = {}
        i = 0
        for node in embedding:
            self._embedding[nodes[i][0]] = node
            i += 1
