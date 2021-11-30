from common.dataset_pool import DatasetPool
from embeddings.embedding_randw import SCWalk, HubWalkDistribution
import sys, os

def eval(method, dim, graph, emb, numl):
    rg = emb.reconstruct(numl)
    prec = graph.link_precision(rg)
    rec, _ = graph.recall(rg)
    mapv, _ = graph.map_value(rg)
    f1 = 2 * rec * mapv / (rec + mapv)

    s = method + "," + str(dim) + "," + str(prec) + "," + str(rec) + "," + str(mapv) + "," + str(f1)
    print(s)
    return s

if __name__ == "__main__":
    dataset = sys.argv[1]
    print("Evaluating CUSTOM RW embeddings for ", dataset)
    graph = DatasetPool.load(dataset)
    graph.remove_selfloop_edges()
    graph = graph.to_undirected()
    numl = graph.edges_cnt()

    
    DIMS = [10, 25, 50, 100, 200]

    out_log = open("CRWEVAL_" + dataset + ".csv", "w")
    out_log.write("METHOD, DIM, PREC, REC, MAP, F1\n")

    for d in DIMS:
        emb = SCWalk(graph, d)
        emb.embed()
        s = eval("SCWALK", d, graph, emb, numl)
        out_log.write(s + "\n")
        emb.to_file("/home/tomcica/graspe/src/graspe/scripts/customrws/customrw_embeddings_aleksandar/" + dataset + "_" + str(d) + "_SCWALK" + "_p085.embedding")

        emb = HubWalkDistribution(graph, d)
        emb.embed()
        s = eval("HUBDISTRIBUTION", d, graph, emb, numl)
        out_log.write(s + "\n")
        emb.to_file("/home/tomcica/graspe/src/graspe/scripts/customrws/customrw_embeddings_aleksandar/" + dataset + "_" + str(d) + "_HUBDISTRIBUTION" + "_p085.embedding")

    out_log.close()