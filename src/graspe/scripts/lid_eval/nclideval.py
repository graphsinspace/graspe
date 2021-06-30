"""
GRASPE -- Graphs in Space: Graph Embeddings for Machine Learning on Complex Data

NCLID evaluation

author: svc@dmi.uns.ac.rs
"""


from common.dataset_pool import DatasetPool
from evaluation.lid_eval import NCLIDEstimator

from datetime import datetime

from statistics import mean, stdev
from scipy.stats import skew

datasets = [
    "karate_club_graph",
    "les_miserables_graph",
    "florentine_families_graph",
    "cora_ml",
    "citeseer",
    "amazon_electronics_photo",
    "amazon_electronics_computers",
    "pubmed",
    "cora",
    "dblp"
]

if __name__ == "__main__":
    from os import sys
    print(datetime.now())
    print("Evaluating natural communities")
    
    print("DATASET,AVG-NCLID,STD-NCLID,CV-NCLID,SKW-NCLID,MIN-NCLID,MAX-NCLID")
    tl = []
    for d in datasets:
        graph = DatasetPool.load(d)
        graph.remove_selfloop_edges()
        graph = graph.to_undirected()

        nclid = NCLIDEstimator(graph)
        nclid.estimate_lids()
        nclids = list(nclid.get_lid_values().values())
        nclens = [nclid.nc_len(node[0]) for node in graph.nodes()]
        
        avg_nclid = mean(nclids)
        std_nclid = stdev(nclids)
        cv_nclid  = std_nclid / avg_nclid
        min_nclid = min(nclids)
        max_nclid = max(nclids)
        skw_nclid = skew(nclids)

        s = d + "," + str(avg_nclid) + "," + str(std_nclid) + "," + str(cv_nclid) + "," + str(skw_nclid) + ","
        s += str(min_nclid) + "," + str(max_nclid)

        print(s)

        avg_nclens = mean(nclens)
        std_nclens = stdev(nclens)
        cv_nclens = std_nclens / avg_nclens
        min_nclens = min(nclens)
        max_nclens = max(nclens)
        skw_nclens = skew(nclens)

        t = d + "," + str(avg_nclens) + "," + str(std_nclens) + "," + str(cv_nclens) + "," + str(skw_nclens) + ","
        t += str(min_nclens) + "," + str(max_nclens)
        tl.append(t)

    print("\n")
    print("DATASET,AVG-NCLEN,STD-NCLEN,CV-NCLEN,SKW-NCLEN,MIN-NCLEN,MAX-NCLEN")
    for t in tl:
        print(t)

    print(datetime.now())
