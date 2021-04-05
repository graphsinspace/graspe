import math
from abc import abstractmethod

from embeddings.embedding_node2vec import (
    Node2VecEmbeddingFastBase,
    Node2VecEmbeddingSlowBase,
)


class HANode2VecFastEmbedding(Node2VecEmbeddingFastBase):
    def __init__(
        self, g, d, p=1, q=1, walk_length=80, num_walks=10, workers=10, seed=42
    ):
        super().__init__(g, d, workers, seed)
        self._p = p
        self._q = q
        self._walk_length = walk_length
        self._num_walks = num_walks
        self._hubness = g.get_hubness()


class HANode2VecSlowEmbedding(Node2VecEmbeddingSlowBase):
    def __init__(
        self, g, d, p=1, q=1, walk_length=80, num_walks=10, workers=10, seed=42
    ):
        super().__init__(g, d, workers, seed)
        self._p = p
        self._q = q
        self._walk_length = walk_length
        self._num_walks = num_walks
        self._hubness = g.get_hubness()


class HANode2VecNumWalksEmbedding(HANode2VecFastEmbedding):
    def __init__(
        self, g, d, p=1, q=1, walk_length=80, num_walks=10, workers=10, seed=42
    ):
        super().__init__(g, d, p, q, walk_length, num_walks, workers, seed)
        self._node_num_walks = self.get_node_num_walks(num_walks * g.nodes_cnt())
        # print("Total num walks {}, real num walks {}".format(num_walks * g.nodes_cnt(), sum(self._node_num_walks.values())))

    def num_walks(self, node):
        # print("Num walks for node {}, with hubness {}, is {}".format(node, self._hubness[node], self._node_num_walks[node]))
        return self._node_num_walks[node]

    def walk_length(self, node):
        return self._walk_length

    def p_value(self, node1, node2):
        return self._p

    def q_value(self, node1, node2):
        return self._q

    @abstractmethod
    def get_node_num_walks(self, num_walks_total):
        pass

    @staticmethod
    def distribute_walks(values, num_walks_total):
        values_sum = sum(values.values())
        values_norm = {node: values[node] / values_sum for node in values}
        return {
            node: int(round(num_walks_total * values_norm[node]))
            for node in values_norm
        }


class HANode2VecNumWalksHubsLessEmbedding(HANode2VecNumWalksEmbedding):
    def get_node_num_walks(self, num_walks_total):
        reciprocal_hubness = {
            node: 1 / self._hubness[node] if self._hubness[node] > 0 else 1
            for node in self._hubness
        }
        return HANode2VecNumWalksEmbedding.distribute_walks(
            reciprocal_hubness, num_walks_total
        )


class HANode2VecNumWalksHubsMoreEmbedding(HANode2VecNumWalksEmbedding):
    def get_node_num_walks(self, num_walks_total):
        return HANode2VecNumWalksEmbedding.distribute_walks(
            self._hubness, num_walks_total
        )


class HANode2VecNumWalksHubsLessLogEmbedding(HANode2VecNumWalksEmbedding):
    def get_node_num_walks(self, num_walks_total):
        log_hubness = {
            node: (
                1 + (math.log(self._hubness[node]) if self._hubness[node] > 0 else 0)
            )
            for node in self._hubness
        }
        reciprocal_log_hubness = {node: 1 / log_hubness[node] for node in log_hubness}
        return HANode2VecNumWalksEmbedding.distribute_walks(
            reciprocal_log_hubness, num_walks_total
        )


class HANode2VecNumWalksHubsMoreLogEmbedding(HANode2VecNumWalksEmbedding):
    def get_node_num_walks(self, num_walks_total):
        log_hubness = {
            node: (
                1 + (math.log(self._hubness[node]) if self._hubness[node] > 0 else 0)
            )
            for node in self._hubness
        }
        return HANode2VecNumWalksEmbedding.distribute_walks(
            log_hubness, num_walks_total
        )
