from embeddings.embedding_gcn import GCNEmbedding
from embeddings.embedding_node2vec import Node2VecEmbedding
from embeddings.embedding_gae import GAEEmbedding
from embeddings.embedding_deepwalk import DeepWalkEmbedding
from embeddings.embedding_sdne import SDNEEmbedding
from embeddings.base.embedding import Embedding

import os
import sys
from timeit import default_timer as timer
from abc import ABC, abstractmethod


class EmbFactory(ABC):
    def __init__(self, dim, quiet, preset):
        presets = {
            "_": ["GCN", "GAE", "SDNE", "DW", "N2V"],
            "N2V": ["N2V", "N2V_p1_q0.5", "N2V_p1_q2", "N2V_p0.5_q1", "N2V_p2_q1"],
        }

        self._dim = dim
        self._quiet = quiet

        if preset in presets:
            self._ids = presets[preset]
        else:
            self._ids = presets["_"]
            print("Preset {} does not exist. Switching to default.".format(preset))

        self._init_embeddings()

        self._ids = list(filter(lambda e: e in self._ems, self._ids))

    @abstractmethod
    def _init_embeddings(self):
        pass

    def get_embedding(self, index):
        if index < 0 or index >= len(self._ids):
            print(
                "[ERROR] Embedding index {} is out of range. Returning None.".format(
                    index
                )
            )
            return None
        return self.get_embedding_by_name(self._ids[index])

    def get_embedding_by_name(self, name):
        if not name in self._ems:
            print(
                "[ERROR] Embedding id {} does not exist. Returning None.".format(name)
            )
            return None
        return self._ems[name]

    def num_methods(self):
        return len(self._ids)

    def get_name(self, index):
        return self._ids[index]

    def get_full_name(self, graph_name, index):
        return self.get_full_name_for_name(graph_name, self.get_name(index))

    def get_full_name_for_name(self, graph_name, name):
        return graph_name + "_d" + str(self._dim) + "_" + name


class LazyEmbFactory(EmbFactory):
    def __init__(self, graph, dim, quiet=False, epochs=200, preset="_"):
        self._graph = graph
        self._epochs = epochs
        super().__init__(dim, quiet, preset)
        if not graph.is_labeled():
            self._ids = list(
                filter(lambda e: self._ems[e].requires_labels(), self._ids)
            )

    def _init_embeddings(self):
        self._ems = {
            "GCN": GCNEmbedding(self._graph, self._dim, self._epochs),
            "GAE": GAEEmbedding(self._graph, self._dim, epochs=self._epochs),
            "SDNE": SDNEEmbedding(
                self._graph, self._dim, epochs=self._epochs, verbose=0
            ),
            "DW": DeepWalkEmbedding(self._graph, self._dim),
            "N2V": Node2VecEmbedding(self._graph, self._dim),
            "N2V_p1_q0.5": Node2VecEmbedding(self._graph, self._dim, p=1, q=0.5),
            "N2V_p1_q2": Node2VecEmbedding(self._graph, self._dim, p=1, q=2),
            "N2V_p0.5_q1": Node2VecEmbedding(self._graph, self._dim, p=0.5, q=1),
            "N2V_p2_q1": Node2VecEmbedding(self._graph, self._dim, p=2, q=1),
        }

    def get_embedding_by_name(self, name):
        e = super().get_embedding_by_name(name)

        if not e:
            if not self._quiet:
                print("* ", name, "embedding not found")
            return None

        try:
            start = timer()
            e.embed()
            end = timer()
            t = end - start
            if not self._quiet:
                print("* ", name, "embedding created, time = ", t, "[s]")
        except:
            if not self._quiet:
                print("[WARNING]", id, "not working for given graph", sys.exc_info()[0])
            return None

        return e


class EagerEmbFactory(LazyEmbFactory):
    def __init__(self, graph, dim, quiet=False, epochs=200, exclude=[], preset="_"):
        super().__init__(graph, dim, quiet, epochs, preset)
        self._ids = list(filter(lambda e: not e in exclude, self._ids))

    def _init_embeddings(self):
        super()._init_embeddings()
        for i in range(len(self._ids)):
            LazyEmbFactory.get_embedding_by_name(self, self.get_name(i))

    def get_embedding_by_name(self, name):
        EmbFactory.get_embedding_by_name(self, name)


class FileEmbFactory(EmbFactory):
    def __init__(self, graph_name, directory, dim, quiet=False, preset="_"):
        self._graph_name = graph_name
        self._directory = directory
        super().__init__(dim, quiet, preset)

    def _init_embeddings(self):
        self._ems = {}
        for name in self._ids:
            e = Embedding.from_file(
                os.path.join(
                    self._directory,
                    self.get_full_name_for_name(self._graph_name, name) + ".embedding",
                )
            )
            if e:
                self._ems[name] = e
                if not self._quiet:
                    print("* ", name, "embedding loaded")
