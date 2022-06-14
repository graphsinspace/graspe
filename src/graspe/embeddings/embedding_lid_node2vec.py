a"""
GRASPE -- Graphs in Space: Graph Embeddings for Machine Learning on Complex Data

LID-aware node2vec

author: svc@dmi.uns.ac.rs
"""

import sys

sys.path.append("../")

from embeddings.embedding_node2vec import (
    Node2VecEmbeddingFastBase,
    Node2VecEmbeddingSlowBase,
)
from evaluation.lid_eval import NCLIDEstimator


class LIDNode2VecBase:
    """
    LID-aware Node2Vec base with elastic walk_length and num_walks parameters
    """

    def __init__(self, g, walk_length=80, num_walks=10, alpha=1):
        self.nclid = NCLIDEstimator(g, alpha=alpha)
        self.nclid.estimate_lids()

        self.nw_dict = dict()
        self.wl_dict = dict()
        for n in g.nodes():
            node = n[0]
            lid_val = self.nclid.get_lid(node)
            self.nw_dict[node] = int((1 + lid_val) * num_walks)
            coef = 1 / lid_val if lid_val > 0 else 1
            self.wl_dict[node] = int(coef * walk_length)


class LIDNode2VecElasticW(Node2VecEmbeddingFastBase, LIDNode2VecBase):
    """
    LID-aware Node2Vec with elastic num_walks and walk_length parameters
    """

    def __init__(
        self, g, d, p=1, q=1, walk_length=80, num_walks=10, workers=10, seed=42, alpha=1
    ):
        Node2VecEmbeddingFastBase.__init__(self, g, d, workers, seed)
        LIDNode2VecBase.__init__(self, g, walk_length, num_walks, alpha)
        self._p = p
        self._q = q

    def num_walks(self, node):
        return self.nw_dict[node]

    def walk_length(self, node):
        return self.wl_dict[node]

    def p_value(self, node1, node2):
        return self._p

    def q_value(self, node1, node2):
        return self._q


class LIDNode2VecElasticWPQ(Node2VecEmbeddingFastBase, LIDNode2VecBase):
    """
    LID-aware Node2Vec with elastic num_walks, walk_length and q parameters
    """

    def __init__(
        self, g, d, p=1, q=1, walk_length=80, num_walks=10, workers=10, seed=42, alpha=1
    ):
        Node2VecEmbeddingFastBase.__init__(self, g, d, workers, seed)
        LIDNode2VecBase.__init__(self, g, walk_length, num_walks, alpha)
        self._p = p
        self._q = q

    def num_walks(self, node):
        return self.nw_dict[node]

    def walk_length(self, node):
        return self.wl_dict[node]

    def in_natural_community(self, node1, node2):
        in_nc = self.nclid.is_in_natural_community(node1, node2)
        return in_nc

    def p_value(self, node1, node2):
        if self.in_natural_community(node2, node1):
            return self._p
        else:
            lid_val = self.nclid.get_lid(node2)
            return self._p + lid_val

    def q_value(self, node1, node2):
        if self.in_natural_community(node1, node2):
            return self._q
        else:
            lid_val = self.nclid.get_lid(node1)
            return self._q + lid_val
