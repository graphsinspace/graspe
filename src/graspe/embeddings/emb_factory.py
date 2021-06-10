import os
import sys
import itertools

from abc import ABC, abstractmethod
from timeit import default_timer as timer

from embeddings.base.embedding import Embedding
from embeddings.embedding_deepwalk import DeepWalkEmbedding
from embeddings.embedding_gae import GAEEmbedding
from embeddings.embedding_gcn import GCNEmbedding
from embeddings.embedding_graphsage import GraphSageEmbedding
from embeddings.embedding_node2vec import Node2VecEmbedding
from embeddings.embedding_ha_node2vec import (
    HANode2VecNumWalksHubsLessEmbedding,
    HANode2VecNumWalksHubsMoreEmbedding,
    HANode2VecNumWalksHubsLessLogEmbedding,
    HANode2VecNumWalksHubsMoreLogEmbedding,
    HANode2VecLoPLoQEmbedding,
    HANode2VecLoPHiQEmbedding,
    HANode2VecHiPLoQEmbedding,
    HANode2VecHiPHiQEmbedding,
    HANode2VecLoPLoQLogEmbedding,
    HANode2VecLoPHiQLogEmbedding,
    HANode2VecHiPLoQLogEmbedding,
    HANode2VecHiPHiQLogEmbedding,
)
from embeddings.embedding_sdne import SDNEEmbedding


class EmbFactory(ABC):
    def __init__(self, dim, quiet, preset, algs=None):
        presets = {
            "_": ["SDNE", "DW", "N2V"],
            "N2V": ["N2V", "N2V_p1_q0.5", "N2V_p1_q2", "N2V_p0.5_q1", "N2V_p2_q1"],
            "HA_N2V": [
                "N2V",
                "HA_N2V_NumWalks_HubsLess",
                "HA_N2V_NumWalks_HubsMore",
                "HA_N2V_NumWalks_HubsLess_Log",
                "HA_N2V_NumWalks_HubsMore_Log",
                "HA_N2V_LoP_LoQ",
                "HA_N2V_LoP_HiQ",
                "HA_N2V_HiP_LoQ",
                "HA_N2V_HiP_HiQ",
                "HA_N2V_LoP_LoQ_Log",
                "HA_N2V_LoP_HiQ_Log",
                "HA_N2V_HiP_LoQ_Log",
                "HA_N2V_HiP_HiQ_Log",
            ],
        }

        # begin GCN builder
        self.gcn_algs = list(
            itertools.product(
                ["GCN"],  # name
                ["tanh", "relu"],  # act_fn
                [0.01, 0.1],  # learning rate
                [100, 200],  # epochs
                [(128,), (128, 128), (256, 256), (256, 512, 256)],  # layer configs
            )
        )
        self.gcn_names = [
            f"{a[0]}_{a[1]}_{a[2]}_{a[3]}_{'_'.join(str(d) for d in a[4])}"
            for a in self.gcn_algs
        ]
        presets["GCN"] = self.gcn_names
        # end GCN builder

        self.graphsage_algs = list(
            itertools.product(
                ["GraphSAGE"],  # name
                ["tanh", "relu"],  # act_fn
                [0.01, 0.1],  # learning rate
                [100, 200],  # epochs
                [(128,), (128, 128), (256, 256), (256, 512, 256)],  # layer configs
            )
        )
        self.graphsage_names = [
            f"{a[0]}_{a[1]}_{a[2]}_{a[3]}_{'_'.join(str(d) for d in a[4])}"
            for a in self.graphsage_algs
        ]
        presets["GraphSAGE"] = self.graphsage_names

        self.gae_algs = list(
            itertools.product(
                ["GAE"],  # name
                ["tanh", "relu"],  # act_fn
                ["V", "NV"],  # Variational, non-variational
                ["L", "NL"],  # Linear, non-linear
                [0.01, 0.1],  # learning rate
                [100, 200],  # epochs
                [(128,), (128, 128), (256, 256), (256, 512, 256)],  # layer configs
            )
        )
        self.gae_names = [
            f"{a[0]}_{a[1]}_{a[2]}_{a[3]}_{a[4]}_{a[5]}_{'_'.join(str(d) for d in a[6])}"
            for a in self.gae_algs
        ]
        presets["GAE"] = self.gae_names

        self.sdne_algs = list(
            itertools.product(
                ["SDNE"],  # name
                [1e-5, 1e-4],  # nu1 (regularizer l1)
                [1e-4, 1e-5],  # nu2 (regularizer l2)
                [5.0],  # beta loss
                [1, 0.9],  # alpha loss
                [128, 256],  # batch size
                [100, 200],  # epochs
                [[32, 16], [32, 32], [64, 32], [128, 64, 128]],  # hidden size
            )
        )
        self.sdne_names = [
            f"{a[0]}_{a[1]}_{a[2]}_{a[3]}_{a[4]}_{a[5]}_{a[6]}_{'_'.join(str(d) for d in a[7])}"
            for a in self.sdne_algs
        ]
        presets["SDNE"] = self.sdne_names

        self._dim = dim
        self._quiet = quiet

        if algs:
            self._ids = algs
        elif preset in presets:
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
        if name not in self._ems:
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
    def __init__(self, graph, dim, quiet=False, epochs=200, preset="_", algs=None):
        self._graph = graph
        self._epochs = epochs
        super().__init__(dim, quiet, preset, algs)
        if not graph.is_labeled():
            self._ids = list(
                filter(lambda e: not self._ems[e].requires_labels(), self._ids)
            )

    def _init_embeddings(self):
        self._ems = {
            "SDNE": SDNEEmbedding(
                self._graph, self._dim, epochs=self._epochs, verbose=0
            ),
            "DW": DeepWalkEmbedding(self._graph, self._dim),
            "N2V": Node2VecEmbedding(self._graph, self._dim),
            "N2V_p1_q0.5": Node2VecEmbedding(self._graph, self._dim, p=1, q=0.5),
            "N2V_p1_q2": Node2VecEmbedding(self._graph, self._dim, p=1, q=2),
            "N2V_p0.5_q1": Node2VecEmbedding(self._graph, self._dim, p=0.5, q=1),
            "N2V_p2_q1": Node2VecEmbedding(self._graph, self._dim, p=2, q=1),
            "HA_N2V_NumWalks_HubsLess": HANode2VecNumWalksHubsLessEmbedding(
                self._graph, self._dim
            ),
            "HA_N2V_NumWalks_HubsMore": HANode2VecNumWalksHubsMoreEmbedding(
                self._graph, self._dim
            ),
            "HA_N2V_NumWalks_HubsLess_Log": HANode2VecNumWalksHubsLessLogEmbedding(
                self._graph, self._dim
            ),
            "HA_N2V_NumWalks_HubsMore_Log": HANode2VecNumWalksHubsMoreLogEmbedding(
                self._graph, self._dim
            ),
            "HA_N2V_LoP_LoQ": HANode2VecLoPLoQEmbedding(self._graph, self._dim),
            "HA_N2V_LoP_HiQ": HANode2VecLoPHiQEmbedding(self._graph, self._dim),
            "HA_N2V_HiP_LoQ": HANode2VecHiPLoQEmbedding(self._graph, self._dim),
            "HA_N2V_HiP_HiQ": HANode2VecHiPHiQEmbedding(self._graph, self._dim),
            "HA_N2V_LoP_LoQ_Log": HANode2VecLoPLoQLogEmbedding(self._graph, self._dim),
            "HA_N2V_LoP_HiQ_Log": HANode2VecLoPHiQLogEmbedding(self._graph, self._dim),
            "HA_N2V_HiP_LoQ_Log": HANode2VecHiPLoQLogEmbedding(self._graph, self._dim),
            "HA_N2V_HiP_HiQ_Log": HANode2VecHiPHiQLogEmbedding(self._graph, self._dim),
        }

        for name, config in zip(self.gcn_names, self.gcn_algs):
            self._ems[name] = GCNEmbedding(
                self._graph,
                self._dim,
                epochs=config[3],
                lr=config[2],
                act_fn=config[1],
                layer_configuration=config[4],
            )

        for name, config in zip(self.graphsage_names, self.graphsage_algs):
            self._ems[name] = GraphSageEmbedding(
                self._graph,
                self._dim,
                epochs=config[3],
                lr=config[2],
                act_fn=config[1],
                layer_configuration=config[4],
            )

        for name, config in zip(self.gae_names, self.gae_algs):
            self._ems[name] = GAEEmbedding(
                self._graph,
                self._dim,
                act_fn=config[1],
                variational=True if config[2] == "V" else False,
                linear=True if config[3] == "L" else False,
                lr=config[4],
                epochs=config[5],
                layer_configuration=config[6],
            )

        for name, config in zip(self.sdne_names, self.sdne_algs):
            self._ems[name] = SDNEEmbedding(
                self._graph,
                self._dim,
                nu1=config[1],
                nu2=config[2],
                beta=config[3],
                alpha=config[4],
                batch_size=config[5],
                epochs=config[6],
                hidden_size=config[7],
            )

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
            import traceback

            traceback.print_exc()
            print("[ERROR]", name, "not working for given graph", sys.exc_info()[0])
            return None

        return e


class EagerEmbFactory(LazyEmbFactory):
    def __init__(
        self, graph, dim, quiet=False, epochs=200, exclude=[], preset="_", algs=None
    ):
        super().__init__(graph, dim, quiet, epochs, preset, algs)
        self._ids = list(filter(lambda e: not e in exclude, self._ids))

    def _init_embeddings(self):
        super()._init_embeddings()
        for i in range(len(self._ids)):
            LazyEmbFactory.get_embedding_by_name(self, self.get_name(i))

    def get_embedding_by_name(self, name):
        EmbFactory.get_embedding_by_name(self, name)


class FileEmbFactory(EmbFactory):
    def __init__(self, graph_name, directory, dim, quiet=False, preset="_", algs=None):
        self._graph_name = graph_name
        self._directory = directory
        super().__init__(dim, quiet, preset, algs)

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
