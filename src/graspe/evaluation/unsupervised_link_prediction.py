from common.graph import Graph
import random as random
from random import sample
from embeddings.embedding_node2vec import Node2VecEmbedding
import numpy as np

class UnsupervisedLinkPrediction():
    def __init__(
        self,
        graph,
        h,
        k,
        embedding_method
    ):
        self._graph = graph
        self._h = h
        self._k = k
        self._embedding_method = embedding_method
        self.__predict()


    def __predict(self):
        edgeset = self._graph.edges()
        hidden_edges = random.sample(edgeset, self._h)
        nodeset = self._graph.nodes()
        newgraph = Graph()
        for node in nodeset:
            newgraph.add_node(node[0], node[1])

        for edge in edgeset:
            if edge not in hidden_edges:
                newgraph.add_edge(edge[0], edge[1])

        #TODO: how to make a "random" embedding????
        #embedding method field as idea
        newembedding = Node2VecEmbedding(newgraph, 10, 0.1, 0.5)
        newembedding.embed()


        dists = []
        for i in range(len(nodeset)):
            node1 = nodeset[i]
            for j in range(i + 1, len(nodeset)):
                node2 = nodeset[j]
                e = (node1[0], node2[0])
                if e not in newgraph.edges():
                    dists.append(((node1[0], node2[0]), np.linalg.norm(newembedding._embedding[node1[0]] - newembedding._embedding[node2[0]])))

        distssorted = sorted(dists, key=lambda p: p[1])
        prediction = distssorted[-self._k:]
        
        cnt = 0
        for e in prediction:
            if e[0] in hidden_edges:
                cnt=cnt+1

        self._precisionATk = float(cnt)/float(self._k)

        sum = 0.0

        for node in nodeset:
            prednode = 0
            enode = 0
            
            for e in prediction:
                if node[0]==e[0][0] or node[0]==e[0][1]:
                    prednode = prednode + 1

            
            for e in hidden_edges:
                if node[0]==e[0] or node[0]==e[1]:
                    enode = enode + 1

            if enode!=0:
                sum = sum + float(prednode)/float(enode)


        self._map = sum/float(self._graph.nodes_cnt())


    def get_precisionATk(self):
        return self._precisionATk

    def get_map(self):
        return self._map