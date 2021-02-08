from embeddings.embedding_gcn import GCNEmbedding
from embeddings.embedding_node2vec import Node2VecEmbedding
from embeddings.embedding_gae import GAEEmbedding
from embeddings.embedding_deepwalk import DeepWalkEmbedding
from embeddings.embedding_sdne import SDNEEmbedding

import sys
from timeit import default_timer as timer


class LazyEmbFactory:
    def __init__(self, graph, dim, quiet=False, epochs=200):
        self.ids = ["GCN", "GAE", "SDNE", "DW", "N2V"]

        self.ems = [
            GCNEmbedding(graph, dim, epochs),
            GAEEmbedding(graph, dim, epochs=epochs),
            SDNEEmbedding(graph, dim, epochs=epochs, verbose=0),
            DeepWalkEmbedding(graph, dim, epochs=epochs),
            Node2VecEmbedding(graph, dim),
        ]

        self.quiet = quiet

    def get_embedding(self, index):
        try:
            start = timer()
            self.ems[index].embed()
            end = timer()
            t = end - start
            if not self.quiet:
                print("* ", self.ids[index], "embedding created, time = ", t, "[s]")

            return self.ems[index]
        except:
            if not self.quiet:
                print(
                    "[WARNING]",
                    self.ids[index],
                    "not working for given graph",
                    sys.exc_info()[0],
                )

            return None

    def num_methods(self):
        return len(self.ems)

    def get_name(self, index):
        return self.ids[index]


class EagerEmbFactory(LazyEmbFactory):
    def __init__(self, graph, dim, quiet=False, epochs=200, exclude=[]):
        super().__init__(graph, dim, quiet, epochs)

        self.embs = []

        if not quiet:
            print("Creating embedding methods started...")

        for i in range(len(self.ems)):
            if self.ids[i] in exclude:
                continue

            emb = super().get_embedding(i)
            if emb != None:
                self.embs.append((emb, self.ids[i]))

    def num_methods(self):
        return len(self.embs)

    def get_embedding(self, index):
        return self.embs[index][0]

    def get_name(self, index):
        return self.embs[index][1]
