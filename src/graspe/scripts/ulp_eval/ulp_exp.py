
"""
GRASPE -- Graphs in Space: Graph Embeddings for Machine Learning on Complex Data

Unsupervised link prediction experiments

author: {lucy, svc}@dmi.uns.ac.rs
"""

from evaluation.ulp import UnsupervisedLinkPrediction
from embeddings.embedding_lid_node2vec import LIDNode2VecElasticW, LIDNode2VecElasticWPQ
from embeddings.embedding_node2vec import Node2VecEmbedding
import sys

class EmbFactory:
    def __init__(self, dataset):
        self.alpha = 1.0
        if dataset == "amazon_electronics_photo" or dataset == "amazon_electronics_computers":
            self.alpha = 1.15
        
        conf = []
        with open('n2v_bestemb_conf.csv') as f:
            lines = [line.rstrip() for line in f]
            for l in lines:
                if l.startswith(dataset + ","):
                    tok = l.split(",")
                    dim = int(tok[1])
                    p = float(tok[2])
                    q = float(tok[3])

                    conf.append((dim, p, q))

        self.conf = conf


    def init_methods_for(self, graph):
        self.graph = graph
        self.gmethods = []

        for c in self.conf:
            d, p, q = c[0], c[1], c[2]
            emb = Node2VecEmbedding(graph, d, p=p, q=q)
            self.gmethods.append((emb, "n2v", d))

        for c in self.conf:
            d, p, q = c[0], c[1], c[2]
            emb = LIDNode2VecElasticW(graph, d, p=p, q=q, alpha=self.alpha)
            self.gmethods.append((emb, "lid-n2v-1", d))
        
        for c in self.conf:
            d, p, q = c[0], c[1], c[2]
            emb = LIDNode2VecElasticWPQ(graph, d, p=p, q=q, alpha=self.alpha)
            self.gmethods.append((emb, "lid-n2v-2", d))    

    def get_gmethods(self):
        return self.gmethods

    

if __name__ == "__main__":
    dataset = sys.argv[1]
    frac = float(sys.argv[2])
    ef = EmbFactory(dataset)
    ulp = UnsupervisedLinkPrediction(dataset, ef, hidden_fraction=frac)
    ef.init_methods_for(ulp.get_graph())
    ulp.eval()
    

