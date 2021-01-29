from common.graph import Graph
import random as random
from random import sample
from embeddings.embedding_node2vec import Node2VecEmbedding
import numpy as np
import heapq as hq

class UnsupervisedLinkPrediction():
    def __init__(
        self,
        graph,
        h,
        embedding_method
    ):
        self._graph = graph
        self._h = h
        self._embedding_method = embedding_method
        self.__predict()


    def __predict(self):
        edgeset = self._graph.edges()
        self._hidden_edges = random.sample(edgeset, self._h)
        nodeset = self._graph.nodes()
        self._newgraph = Graph()
        for node in nodeset:
            self._newgraph.add_node(node[0], node[1])

        for edge in edgeset:
            if edge not in self._hidden_edges:
                self._newgraph.add_edge(edge[0], edge[1])

        #TODO: how to make a "random" embedding????
        #embedding method field as idea
        self._newembedding = Node2VecEmbedding(self._newgraph, 10, 0.1, 0.5, seed=42)
        self._newembedding.embed()


        self._dists = []
        for i in range(len(nodeset)):
            node1 = nodeset[i]
            for j in range(i + 1, len(nodeset)):
                node2 = nodeset[j]
                e = (node1[0], node2[0])
                if e not in self._newgraph.edges():
                    #dists.append(((node1[0], node2[0]), np.linalg.norm(self._newembedding._embedding[node1[0]] - self._newembedding._embedding[node2[0]])))
                    hq.heappush(self._dists,((np.linalg.norm(self._newembedding._embedding[node1[0]] - self._newembedding._embedding[node2[0]]), (node1[0], node2[0]))))

        #TODO: prepraviti u pravi heap
        #self._distssorted = sorted(dists, key=lambda p: p[1])
        self._precisionATh = self.get_precisionATk(self._h)
        self._mapATh = self.get_map(self._h)
        


    def get_precisionATk(self, k):
        prediction = hq.nsmallest(k, self._dists, key=lambda p: p[0])
        print("dists=",self._dists)
        print("preds=", prediction)
        
        cnt = 0
        for e in prediction:
            if e[1] in self._hidden_edges:
                cnt=cnt+1

        return float(cnt)/float(k)

    def get_map(self, k):
        prediction = hq.nsmallest(k, self._dists, key=lambda p: p[0]) #key=lambda p: p[0]
        nodeset = self._graph.nodes()
        sum = 0.0

        for node in nodeset:
            prednode = 0
            enode = 0
            
            for e in prediction:
                if node[0]==e[1][0] or node[0]==e[1][1]:
                    prednode = prednode + 1

            
            for e in self._hidden_edges:
                if node[0]==e[0] or node[0]==e[1]:
                    enode = enode + 1

            if enode!=0:
                sum = sum + float(prednode)/float(enode)


        return sum/float(self._graph.nodes_cnt())
        

    def get_mapATh(self):
        return self._mapATh

    def get_precisionATh(self):
        return self._precisionATh