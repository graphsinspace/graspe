from common.graph import Graph
from embeddings.base.embedding import Embedding

import numpy as np
from gensim.models import Word2Vec
import random
from abc import abstractmethod

class Node2VecEmbeddingBase(Embedding):
    def __init__(
        self, g, d, walk_length=80, num_walks=10, workers=4, seed=42
    ):
        super().__init__(g, d)
        self._walk_length = walk_length
        self._num_walks = num_walks
        self._workers = workers
        self._seed = seed
    
    def get_alias_edge(self, node1, node2):
        """
        Get the alias edge setup lists for a given edge.
        """
        G = self._g
        p = self.p_value(node1, node2)
        q = self.q_value(node1, node2)

        unnormalized_probs = []
        for edge in sorted(G.edges(node2, data=True)):
            dst_nbr = edge[1]
            weight = edge[2]['w'] if 'w' in edge[2] else 1
            if dst_nbr == node1:
                unnormalized_probs.append(weight/p)
            elif G.has_edge(dst_nbr, node1):
                unnormalized_probs.append(weight)
            else:
                unnormalized_probs.append(weight/q)
        norm_const = sum(unnormalized_probs)
        normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]

        return Node2VecEmbeddingBase.alias_setup(normalized_probs)

    def preprocess_transition_probs(self):
        """
        Preprocessing of transition probabilities for guiding the random walks.
        """
        G = self._g		
        alias_nodes = {}
        for node in G.nodes():
            node_id = node[0]
            node_edges = sorted(G.edges(node[0], data=True))
            unnormalized_probs = [(edge[2]['w'] if 'w' in edge[2] else 1) for edge in node_edges]
            norm_const = sum(unnormalized_probs)
            normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]
            alias_nodes[node_id] = Node2VecEmbeddingBase.alias_setup(normalized_probs)

        alias_edges = {}
        triads = {}
        
        for edge in G.edges():
            alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges
    
    def node2vec_walk(self, walk_length, start_node):
        '''
        Simulate a random walk starting from start node.
        '''
        G = self._g
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = sorted([edge[1] for edge in G.edges(cur)])
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(cur_nbrs[Node2VecEmbeddingBase.alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    prev = walk[-2]
                    next = cur_nbrs[Node2VecEmbeddingBase.alias_draw(alias_edges[(prev, cur)][0], 
                        alias_edges[(prev, cur)][1])]
                    walk.append(next)
            else:
                break

        return walk

    def simulate_walks(self, num_walks, walk_length):
        '''
        Repeatedly simulate random walks from each node.
        '''
        G = self._g
        walks = []
        nodes = list(G.nodes())
        for walk_iter in range(num_walks):
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node[0]))
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
            q[kk] = K*prob
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

        kk = int(np.floor(np.random.rand()*K))
        if np.random.rand() < q[kk]:
            return kk
        else:
            return J[kk]
    
    def embed(self):
        super().embed()
        self.preprocess_transition_probs()
        walks = self.simulate_walks(self._num_walks, self._walk_length)
        walks = [list(map(str, walk)) for walk in walks]
        model = Word2Vec(walks, size=self._d, min_count=0, sg=1, workers=self._workers, seed=self._seed)
        self._embedding = {}
        for node in self._g.nodes():
            self._embedding[node[0]] = model[str(node[0])]
        
    def requires_labels(self):
        return False

    @abstractmethod
    def p_value(self, node1, node2):
        pass

    @abstractmethod
    def q_value(self, node1, node2):
        pass

class Node2VecEmbedding(Node2VecEmbeddingBase):
    def __init__(
        self, g, d, p=1, q=1, walk_length=80, num_walks=10, workers=4, seed=42
    ):
        super().__init__(g, d, walk_length, num_walks, workers, seed)
        self._p = p
        self._q = q

    def p_value(self, node1, node2):
        return self._p

    def q_value(self, node1, node2):
        return self._q


from node2vec import Node2Vec

class Node2VecEmbedding_Native(Embedding):
    def __init__(
        self,
        g,
        d,
        p=1,
        q=1,
        walk_length=80,
        num_walks=10,
        workers=10,
        seed=42
    ):
        super().__init__(g,d)
        self._p = p
        self._q = q
        self._walk_length = walk_length
        self._num_walks = num_walks
        self._workers = workers
        self._seed = seed


    def embed(self):
        nodes = self._g.nodes()

        node2vec = Node2Vec(graph=self._g.to_networkx(), p=self._p, q=self._q, dimensions=self._d, walk_length=self._walk_length, num_walks=self._num_walks, workers=self._workers, seed=self._seed, quiet=True)
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
