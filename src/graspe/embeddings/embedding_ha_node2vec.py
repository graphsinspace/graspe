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
        self._log_hubness = {
            node: (
                1 + (math.log(self._hubness[node]) if self._hubness[node] > 0 else 0)
            )
            for node in self._hubness
        }
        self._max_hubness = max(self._hubness.values())
        self._max_log_hubness = max(self._log_hubness.values())
        self._inverse_hubness = {
            node: self._max_hubness - self._hubness[node] for node in self._hubness
        }
        self._inverse_log_hubness = {
            node: self._max_log_hubness - self._log_hubness[node]
            for node in self._log_hubness
        }


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
        self._log_hubness = {
            node: (
                1 + (math.log(self._hubness[node]) if self._hubness[node] > 0 else 0)
            )
            for node in self._hubness
        }
        self._max_hubness = max(self._hubness.values())
        self._max_log_hubness = max(self._log_hubness.values())
        self._inverse_hubness = {
            node: self._max_hubness - self._hubness[node] for node in self._hubness
        }
        self._inverse_log_hubness = {
            node: self._max_log_hubness - self._log_hubness[node]
            for node in self._log_hubness
        }


class HANode2VecNumWalksEmbedding(HANode2VecFastEmbedding):
    def __init__(
        self, g, d, p=1, q=1, walk_length=80, num_walks=10, workers=10, seed=42
    ):
        super().__init__(g, d, p, q, walk_length, num_walks, workers, seed)
        self._node_num_walks = self.get_node_num_walks(num_walks * g.nodes_cnt())
        # print("Total num walks {}, real num walks {}".format(num_walks * g.nodes_cnt(), sum(self._node_num_walks.values())))

    def num_walks(self, node):
        if self._node_num_walks[node] == 0:
            print(
                "Num walks for node {}, with hubness {}, is {}".format(
                    node, self._hubness[node], self._node_num_walks[node]
                )
            )
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
            if int(round(num_walks_total * values_norm[node])) > 0
            else 1
            for node in values_norm
        }


class HANode2VecNumWalksHubsLessEmbedding(HANode2VecNumWalksEmbedding):
    def get_node_num_walks(self, num_walks_total):
        return HANode2VecNumWalksEmbedding.distribute_walks(
            self._inverse_hubness, num_walks_total
        )


class HANode2VecNumWalksHubsMoreEmbedding(HANode2VecNumWalksEmbedding):
    def get_node_num_walks(self, num_walks_total):
        return HANode2VecNumWalksEmbedding.distribute_walks(
            self._hubness, num_walks_total
        )


class HANode2VecNumWalksHubsLessLogEmbedding(HANode2VecNumWalksEmbedding):
    def get_node_num_walks(self, num_walks_total):
        return HANode2VecNumWalksEmbedding.distribute_walks(
            self._inverse_log_hubness, num_walks_total
        )


class HANode2VecNumWalksHubsMoreLogEmbedding(HANode2VecNumWalksEmbedding):
    def get_node_num_walks(self, num_walks_total):
        return HANode2VecNumWalksEmbedding.distribute_walks(
            self._log_hubness, num_walks_total
        )


class HANode2VecPQEmbedding(HANode2VecSlowEmbedding):
    def __init__(
        self, g, d, max_p=4, max_q=4, walk_length=80, num_walks=10, workers=10, seed=42
    ):
        super().__init__(g, d, 1, 1, walk_length, num_walks, workers, seed)
        self._max_p = max_p
        self._max_q = max_q
        self._p_values = self.get_p_vals()
        self._q_values = self.get_q_vals()
        # for node in self._hubness:
        #     print("hubness {}, p {}, q {}".format(self._hubness[node], self._p_values[node], self._q_values[node]))
        # print("----------------------------")

    def num_walks(self, node):
        return self._num_walks

    def walk_length(self, node):
        return self._walk_length

    def p_value(self, start_node, node1, node2):
        return self._p_values[start_node]

    def q_value(self, start_node, node1, node2):
        return self._q_values[start_node]

    @abstractmethod
    def get_p_vals(self):
        pass

    @abstractmethod
    def get_q_vals(self):
        pass

    def interpolate(self, values, max_value, interpolation_limit):
        return {
            node: ((values[node] if values[node] > 0 else 1) / max_value)
            * interpolation_limit
            for node in values
        }


class HANode2VecLoPLoQEmbedding(HANode2VecPQEmbedding):
    def get_p_vals(self):
        return self.interpolate(self._inverse_hubness, self._max_hubness, self._max_p)

    def get_q_vals(self):
        return self.interpolate(self._inverse_hubness, self._max_hubness, self._max_q)


class HANode2VecLoPHiQEmbedding(HANode2VecPQEmbedding):
    def get_p_vals(self):
        return self.interpolate(self._inverse_hubness, self._max_hubness, self._max_p)

    def get_q_vals(self):
        return self.interpolate(self._hubness, self._max_hubness, self._max_q)


class HANode2VecHiPLoQEmbedding(HANode2VecPQEmbedding):
    def get_p_vals(self):
        return self.interpolate(self._hubness, self._max_hubness, self._max_p)

    def get_q_vals(self):
        return self.interpolate(self._inverse_hubness, self._max_hubness, self._max_q)


class HANode2VecHiPHiQEmbedding(HANode2VecPQEmbedding):
    def get_p_vals(self):
        return self.interpolate(self._hubness, self._max_hubness, self._max_p)

    def get_q_vals(self):
        return self.interpolate(self._hubness, self._max_hubness, self._max_q)


class HANode2VecLoPLoQLogEmbedding(HANode2VecPQEmbedding):
    def get_p_vals(self):
        return self.interpolate(
            self._inverse_log_hubness, self._max_log_hubness, self._max_p
        )

    def get_q_vals(self):
        return self.interpolate(
            self._inverse_log_hubness, self._max_log_hubness, self._max_q
        )


class HANode2VecLoPHiQLogEmbedding(HANode2VecPQEmbedding):
    def get_p_vals(self):
        return self.interpolate(
            self._inverse_log_hubness, self._max_log_hubness, self._max_p
        )

    def get_q_vals(self):
        return self.interpolate(self._log_hubness, self._max_log_hubness, self._max_q)


class HANode2VecHiPLoQLogEmbedding(HANode2VecPQEmbedding):
    def get_p_vals(self):
        return self.interpolate(self._log_hubness, self._max_log_hubness, self._max_p)

    def get_q_vals(self):
        return self.interpolate(
            self._inverse_log_hubness, self._max_log_hubness, self._max_q
        )


class HANode2VecHiPHiQLogEmbedding(HANode2VecPQEmbedding):
    def get_p_vals(self):
        return self.interpolate(self._log_hubness, self._max_log_hubness, self._max_p)

    def get_q_vals(self):
        return self.interpolate(self._log_hubness, self._max_log_hubness, self._max_q)
