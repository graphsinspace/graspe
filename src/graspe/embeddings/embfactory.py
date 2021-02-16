from embeddings.embedding_gcn import GCNEmbedding
from embeddings.embedding_node2vec import Node2VecEmbedding
from embeddings.embedding_gae import GAEEmbedding
from embeddings.embedding_deepwalk import DeepWalkEmbedding
from embeddings.embedding_sdne import SDNEEmbedding

import sys
from timeit import default_timer as timer


class LazyEmbFactory:
    def __init__(self, graph, dim, quiet=False, epochs=200, preset="_"):
        presets = {
            "_": ["GCN", "GAE", "SDNE", "DW", "N2V"],
            "N2V": ["N2V", "N2V_p1_q0.5", "N2V_p1_q2", "N2V_p0.5_q1", "N2V_p2_q1"],
        }
        if preset in presets:
            self.ids = presets[preset]
        else:
            self.ids = presets["_"]
            print("Preset {} does not exist. Switching to default.".format(preset))

        self.ems = {
            "GCN": GCNEmbedding(graph, dim, epochs),
            "GAE": GAEEmbedding(graph, dim, epochs=epochs),
            "SDNE": SDNEEmbedding(graph, dim, epochs=epochs, verbose=0),
            "DW": DeepWalkEmbedding(graph, dim),
            "N2V": Node2VecEmbedding(graph, dim),
            "N2V_p1_q0.5": Node2VecEmbedding(graph, dim, p=1, q=0.5),
            "N2V_p1_q2": Node2VecEmbedding(graph, dim, p=1, q=2),
            "N2V_p0.5_q1": Node2VecEmbedding(graph, dim, p=0.5, q=1),
            "N2V_p2_q1": Node2VecEmbedding(graph, dim, p=2, q=1),
        }

        if not graph.is_labeled():
            self.ids = filter(lambda e: self.ems[e].requires_labels(), self.ids)

        self.quiet = quiet

    def get_embedding(self, index):
        if index >= len(self.ids):
            print(
                "[ERROR] Embedding index {} is out of range. Returning None.".format(
                    index
                )
            )
            return None
        return self.get_embedding_by_id(self.ids[index])

    def get_embedding_by_id(self, id):
        if not id in self.ems:
            print("[ERROR] Embedding id {} does not exist. Returning None.".format(id))
            return None
        try:
            start = timer()
            self.ems[id].embed()
            end = timer()
            t = end - start
            if not self.quiet:
                print("* ", id, "embedding created, time = ", t, "[s]")

            return self.ems[id]
        except:
            if not self.quiet:
                print("[WARNING]", id, "not working for given graph", sys.exc_info()[0])

            return None

    def num_methods(self):
        return len(self.ids)

    def get_name(self, index):
        return self.ids[index]


class EagerEmbFactory(LazyEmbFactory):
    def __init__(self, graph, dim, quiet=False, epochs=200, exclude=[], preset="_"):
        super().__init__(graph, dim, quiet, epochs, preset)

        self.embs = []

        if not quiet:
            print("Creating embedding methods started...")

        for i in range(len(self.ids)):
            if self.ids[i] in exclude:
                continue

            emb = super().get_embedding(i)
            if emb != None:
                self.embs.append((emb, self.ids[i]))

    def get_embedding(self, index):
        return self.embs[index][0]

    def num_methods(self):
        return len(self.embs)

    def get_name(self, index):
        return self.embs[index][1]
