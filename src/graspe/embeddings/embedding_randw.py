"""
GRASP: Graphs in Space

Embedding methods based on custom random walk schemes

author: Milos Savic (svc@dmi.uns.ac.rs)
"""

from gensim.models import Word2Vec
import random
from abc import abstractmethod

from embeddings.base.embedding import Embedding

from networkx.algorithms.core import core_number

import sys
sys.path.append("../")

from evaluation.lid_eval import LFMnx

import numpy as np


class RWEmbBase(Embedding):
    def __init__(self, g, d, workers=4, seed=42):
        super().__init__(g, d)
        self._workers = workers
        self._seed = seed

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
                walks.append(
                    self.rndwalks_from_node(walk_length=walk_length, start_node=n)
                )
        random.shuffle(walks)
        return walks

    def rndwalks_from_node(self, walk_length, start_node):
        """
        Simulate a random walk starting from start node.
        """
        G = self._g

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = sorted([edge[1] for edge in G.edges(cur)])
            if len(cur_nbrs) > 0:
                next_node = self.select_next_node(start_node, cur, cur_nbrs)
                if next_node == None:
                    break
                walk.append(next_node)
            else:
                break

        return walk

    @abstractmethod
    def select_next_node(self, start_node, current_node, neighbours):
        pass

    @abstractmethod
    def num_walks(self, node):
        pass

    @abstractmethod
    def walk_length(self, node):
        pass

    def embed(self):
        super().embed()
        walks = self.simulate_walks()
        walks = [list(map(str, walk)) for walk in walks]
        model = Word2Vec(
            walks,
            vector_size=self._d,
            min_count=0,
            sg=1,
            workers=self._workers,
            seed=self._seed,
        )
        self._embedding = {}
        for node in self._g.nodes():
            self._embedding[node[0]] = model.wv[str(node[0])]

    def requires_labels(self):
        return False


class UnbiasedWalk(RWEmbBase):
    def __init__(self, g, d, num_walks=10, walk_length=80, workers=4, seed=42):
        super().__init__(g, d, workers, seed)
        self.nw = num_walks
        self.wl = walk_length

    def select_next_node(self, start_node, current_node, neighbours):
        return random.sample(neighbours, 1)[0]

    def num_walks(self, node):
        return self.nw

    def walk_length(self, node):
        return self.wl


class NaturalCommunities:
    def __init__(self, g, alpha=1):
        self.g = g
        self.community_detector = LFMnx(g.to_networkx(), alpha=alpha)
        self.natural_community = dict()

    def detect(self):
        nodes = self.g.nodes()
        numn = len(nodes)
        for i in range(numn):
            src = nodes[i][0]
            src_community = self.community_detector.identify_natural_community(src)
            self.natural_community[src] = src_community

    def is_in_natural_community(self, seed_node, other_node):
        return other_node in self.natural_community[seed_node]


class NCWalk(RWEmbBase):
    def __init__(
        self, g, d, num_walks=10, walk_length=80, p=0.85, workers=4, seed=42, alpha=1
    ):
        super().__init__(g, d, workers, seed)
        self.nw = num_walks
        self.wl = walk_length
        self.nc = NaturalCommunities(g, alpha=alpha)
        self.nc.detect()
        self.p = p

    def select_next_node(self, start_node, current_node, neighbours):
        r = random.random()

        if r <= self.p:
            ncnodes = [
                n for n in neighbours if self.nc.is_in_natural_community(start_node, n)
            ]
            if len(ncnodes) > 0:
                return random.sample(ncnodes, 1)[0]
            else:
                return random.sample(neighbours, 1)[0]

        return random.sample(neighbours, 1)[0]

    def num_walks(self, node):
        return self.nw

    def walk_length(self, node):
        return self.wl


class RNCWalk(RWEmbBase):
    def __init__(
        self, g, d, num_walks=10, walk_length=80, p=0.85, workers=4, seed=42, alpha=1
    ):
        super().__init__(g, d, workers, seed)
        self.nw = num_walks
        self.wl = walk_length
        self.nc = NaturalCommunities(g, alpha=alpha)
        self.nc.detect()
        self.p = p

    def select_next_node(self, start_node, current_node, neighbours):
        r = random.random()
        
        if r <= self.p:
            ncnodes = [
                n
                for n in neighbours
                if self.nc.is_in_natural_community(current_node, n)
            ]
            if len(ncnodes) > 0:
                return random.sample(ncnodes, 1)[0]
            else:
                return random.sample(neighbours, 1)[0]

        return random.sample(neighbours, 1)[0]

    def num_walks(self, node):
        return self.nw

    def walk_length(self, node):
        return self.wl


class ShellWalk(RWEmbBase):
    def __init__(
        self,
        g,
        d,
        num_walks=10,
        walk_length=80,
        p=0.85,
        workers=4,
        seed=42,
        inverted=False,
    ):
        super().__init__(g, d, workers, seed)
        self.nw = num_walks
        self.wl = walk_length
        self.cores = core_number(g.to_networkx())
        self.inverted = inverted
        self.p = p

    def select_next_node(self, start_node, current_node, neighbours):
        r = random.random()

        if r <= self.p:
            if self.inverted:
                corenodes = [
                    n for n in neighbours if self.cores[n] <= self.cores[current_node]
                ]
            else:
                corenodes = [
                    n for n in neighbours if self.cores[n] >= self.cores[current_node]
                ]

            if len(corenodes) > 0:
                return random.sample(corenodes, 1)[0]
            else:
                return random.sample(neighbours, 1)[0]

        return random.sample(neighbours, 1)[0]

    def num_walks(self, node):
        return self.nw

    def walk_length(self, node):
        return self.wl

"""

author: Aleksandar Tomcic (aleksandart@dmi.uns.ac.rs)
"""

# SC - Same Class
class SCWalk(RWEmbBase):
    def __init__(
        self, g, d, num_walks=10, walk_length=80, p=0.85, workers=4, seed=42
    ):
        super().__init__(g, d, workers, seed)
        self.nw = num_walks
        self.wl = walk_length
        self.p = p

    def select_next_node(self, start_node, current_node, neighbours):

        G = self._g
        
        r = random.random()

        if r <= self.p:
            same_label = [n for n in neighbours if G.get_label(n) == G.get_label(start_node)]
            different_label = [n for n in neighbours if G.get_label(n) != G.get_label(start_node)]

            if len(same_label) > 0:
                return random.sample(same_label, 1)[0]
            else if len(different_label) > 0:
                return random.sample(different_label, 1)[0] # Maybe to get node with smallest weight between current_node and node in different_labels?
                  
        return random.sample(neighbours, 1)[0]

    def num_walks(self, node):
        return self.nw

    def walk_length(self, node):
        return self.wl


class HubWalk(RWEmbBase):
    def __init__(
        self, g, d, num_walks=10, walk_length=80, p=0.85, workers=4, seed=42
    ):
        super().__init__(g, d, workers, seed)
        self.nw = num_walks
        self.wl = walk_length
        self.p = p

    def select_next_node(self, start_node, current_node, neighbours):
    
        G = self._g

        r = random.random()

        if r <= self.p:
            
            neighbours_of_neighbours = {}

            for n in neighbours:
                n_o_n = [edge[1] for edge in G.edges(n) if G.get_label(edge[1]) != G.get_label(start_node)] # BAD

                neighbours_of_neighbours[n] = n_o_n

            neighbours_of_neighbours = ' '.join(sorted(neighbours_of_neighbours, key = lambda key: len(neighbours_of_neighbours[key]), reverse=True))

            if len(neighbours_of_neighbours) > 0:
                return list(neighbours_of_neighbours.keys())[0]
            else:
                return random.sample(neighbours, 1)[0]

        return random.sample(neighbours, 1)[0]

    def num_walks(self, node):
        return self.nw

    def walk_length(self, node):
        return self.wl


class HubWalkUniform(RWEmbBase):
    def __init__(
        self, g, d, num_walks=10, walk_length=80, p=0.85, workers=4, seed=42
    ):
        super().__init__(g, d, workers, seed)
        self.nw = num_walks
        self.wl = walk_length
        self.p = p

    def select_next_node(self, start_node, current_node, neighbours):

        G = self._g

        r = random.random()

        if r <= self.p:

            neighbours_of_neighbours = {}

            for n in neighbours:
                n_o_n = [edge[1] for edge in G.edges(n) if G.get_label(edge[1]) != G.get_label(start_node)]

                neighbours_of_neighbours[n] = n_o_n

                # This time without sorting i.e. we go uniform
            
            if len(neighbours_of_neighbours) > 0:
                return random.sample(list(neighbours_of_neighbours.keys()), 1)[0]
            else:
                return random.sample(neighbours, 1)[0]

    def num_walks(self, node):
        return self.nw
    
    def walk_length(self, node):
        return self.wl


class HubWalkDistribution(RWEmbBase):
    def __init__(
        self, g, d, num_walks=10, walk_length=80, p=0.85, workers=4, seed=42
    ):
        super().__init__(g, d, workers, seed)
        self.nw = num_walks
        self.wl = walk_length
        self.p = p

    def select_next_node(self, start_node, current_node, neighbours):

        G = self._g

        r = random.random()

        if r <= self.p:

            neighbours_of_neighbours = {}

            for n in neighbours:
                n_o_n = [edge[1] for edge in G.edges(n) if G.get_label(edge[1]) != G.get_label(start_node)]

                neighbours_of_neighbours[n] = n_o_n

            if len(neighbours_of_neighbours) > 0:
                full_length = 0
                for k in neighbours_of_neighbours.keys():
                    full_length += len(neighbours_of_neighbours[k])
                

                probabilities = []
                for k in neighbours_of_neighbours.keys():
                    probabilities.append(len(neighbours_of_neighbours[k]) / full_length)

                return np.random.choice(list(neighbours_of_neighbours.keys()
                                        1, 
                                        p = probabilities)[0]
        else:
            return random.sample(neighbours, 1)[0]

    def num_walks(self, node):
        return self.nw
    
    def walk_length(self, node):
        return self.wl
