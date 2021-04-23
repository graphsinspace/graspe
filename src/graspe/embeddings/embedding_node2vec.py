import random
from abc import abstractmethod

import numpy as np
from gensim.models import Word2Vec
from node2vec import Node2Vec

from embeddings.base.embedding import Embedding


class Node2VecEmbeddingBase(Embedding):
    def __init__(self, g, d, workers=4, seed=42):
        super().__init__(g, d)
        self._workers = workers
        self._seed = seed
        self.alias_nodes = {}
        self.alias_edges = {}

    def get_alias_edge(self, edge, start_node=None):
        """
        Get the alias edge setup lists for a given edge.
        """
        G = self._g
        node1 = edge[0]
        node2 = edge[1]

        p = self._p_value(start_node, node1, node2)
        q = self._q_value(start_node, node1, node2)

        unnormalized_probs = []
        for e in sorted(G.edges(node2, data=True)):
            dst_nbr = e[1]
            weight = e[2]["w"] if "w" in e[2] else 1
            if dst_nbr == node1:
                unnormalized_probs.append(weight / p)
            elif G.has_edge(dst_nbr, node1):
                unnormalized_probs.append(weight)
            else:
                unnormalized_probs.append(weight / q)
        norm_const = sum(unnormalized_probs)
        normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]

        return Node2VecEmbeddingBase.alias_setup(normalized_probs)

    def preprocess_transition_probs(self):
        """
        Preprocessing of transition probabilities for guiding the random walks.
        """
        self.alias_nodes = {}
        self.alias_edges = {}
        for node in self._g.nodes():
            node_id = node[0]
            node_edges = sorted(self._g.edges(node[0], data=True))
            unnormalized_probs = [
                (edge[2]["w"] if "w" in edge[2] else 1) for edge in node_edges
            ]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [
                float(u_prob) / norm_const for u_prob in unnormalized_probs
            ]
            self.alias_nodes[node_id] = Node2VecEmbeddingBase.alias_setup(
                normalized_probs
            )

    def node2vec_walk(self, walk_length, start_node):
        """
        Simulate a random walk starting from start node.
        """
        G = self._g
        alias_nodes = self.alias_nodes

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = sorted([edge[1] for edge in G.edges(cur)])
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(
                        cur_nbrs[
                            Node2VecEmbeddingBase.alias_draw(
                                alias_nodes[cur][0], alias_nodes[cur][1]
                            )
                        ]
                    )
                else:
                    prev = walk[-2]
                    current_edge = (prev, cur)
                    if current_edge in self.alias_edges:
                        alias_edges = self.alias_edges[current_edge]
                    else:
                        alias_edges = self.get_alias_edge(current_edge, start_node)
                    next_walk = cur_nbrs[
                        Node2VecEmbeddingBase.alias_draw(alias_edges[0], alias_edges[1])
                    ]
                    walk.append(next_walk)
            else:
                break

        return walk

    def simulate_walks(self):
        """
        Repeatedly simulate random walks from each node.
        """
        G = self._g
        walks = []
        for node in list(G.nodes()):
            n = node[0]
            num_walks = self.num_walks(n)
            walk_length = self.walk_length(n)
            for _ in range(num_walks):
                walks.append(self.node2vec_walk(walk_length=walk_length, start_node=n))
        random.shuffle(walks)
        return walks

    @staticmethod
    def alias_setup(probs):
        """
        Compute utility lists for non-uniform sampling from discrete distributions.
        """
        K = len(probs)
        q = np.zeros(K)
        J = np.zeros(K, dtype=np.int)

        smaller = []
        larger = []
        for kk, prob in enumerate(probs):
            q[kk] = K * prob
            if q[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)

        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            J[small] = large
            q[large] = q[large] + q[small] - 1.0
            if q[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        return J, q

    @staticmethod
    def alias_draw(J, q):
        """
        Draw sample from a non-uniform discrete distribution using alias sampling.
        """
        K = len(J)

        kk = int(np.floor(np.random.rand() * K))
        if np.random.rand() < q[kk]:
            return kk
        else:
            return J[kk]

    def embed(self):
        super().embed()
        self.preprocess_transition_probs()
        walks = self.simulate_walks()
        walks = [list(map(str, walk)) for walk in walks]
        model = Word2Vec(
            walks,
            size=self._d,
            min_count=0,
            sg=1,
            workers=self._workers,
            seed=self._seed,
        )
        self._embedding = {}
        for node in self._g.nodes():
            self._embedding[node[0]] = model[str(node[0])]

    def requires_labels(self):
        return False

    @abstractmethod
    def _p_value(self, start_node, node1, node2):
        pass

    @abstractmethod
    def _q_value(self, start_node, node1, node2):
        pass

    @abstractmethod
    def num_walks(self, node):
        pass

    @abstractmethod
    def walk_length(self, node):
        pass


class Node2VecEmbeddingFastBase(Node2VecEmbeddingBase):
    """
    Node2Vec version with preprocessing step that makes the algorithm faster.
    Inherit this class if p and q values do not depend on walks' start nodes.
    """

    def preprocess_transition_probs(self):
        """
        Preprocessing of transition probabilities for guiding the random walks.
        """
        super().preprocess_transition_probs()
        for edge in self._g.edges():
            self.alias_edges[edge] = self.get_alias_edge(edge)

    def _p_value(self, start_node, node1, node2):
        return self.p_value(node1, node2)

    def _q_value(self, start_node, node1, node2):
        return self.q_value(node1, node2)

    @abstractmethod
    def p_value(self, node1, node2):
        pass

    @abstractmethod
    def q_value(self, node1, node2):
        pass


class Node2VecEmbeddingSlowBase(Node2VecEmbeddingBase):
    """
    Node2Vec version without preprocessing step, i.e. slower Node2Vec version.
    Inherit this class ONLY if p and q values depend on walks' start nodes.
    """

    def _p_value(self, start_node, node1, node2):
        return self.p_value(start_node, node1, node2)

    def _q_value(self, start_node, node1, node2):
        return self.q_value(start_node, node1, node2)

    @abstractmethod
    def p_value(self, start_node, node1, node2):
        pass

    @abstractmethod
    def q_value(self, start_node, node1, node2):
        pass


class Node2VecEmbedding(Node2VecEmbeddingFastBase):
    def __init__(
        self, g, d, p=1, q=1, walk_length=80, num_walks=10, workers=10, seed=42
    ):
        super().__init__(g, d, workers, seed)
        self._p = p
        self._q = q
        self._walk_length = walk_length
        self._num_walks = num_walks

    def p_value(self, node1, node2):
        return self._p

    def q_value(self, node1, node2):
        return self._q

    def num_walks(self, node):
        return self._num_walks

    def walk_length(self, node):
        return self._walk_length


class Node2VecEmbeddingNative(Embedding):
    def __init__(
        self, g, d, p=1, q=1, walk_length=80, num_walks=10, workers=10, seed=42
    ):
        super().__init__(g, d)
        self._p = p
        self._q = q
        self._walk_length = walk_length
        self._num_walks = num_walks
        self._workers = workers
        self._seed = seed

    def embed(self):
        nodes = self._g.nodes()

        node2vec = Node2Vec(
            graph=self._g.to_networkx(),
            p=self._p,
            q=self._q,
            dimensions=self._d,
            walk_length=self._walk_length,
            num_walks=self._num_walks,
            workers=self._workers,
            seed=self._seed,
            quiet=True,
        )
        embedding = node2vec.fit()
        vectors = embedding.wv

        self._embedding = {}
        for node in nodes:
            nid = node[0]
            strnid = str(nid)
            emb = vectors[strnid]
            self._embedding[nid] = emb

    def requires_labels(self):
        return False
