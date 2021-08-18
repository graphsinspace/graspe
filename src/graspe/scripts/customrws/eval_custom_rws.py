from common.dataset_pool import DatasetPool
from embeddings.embedding_randw import UnbiasedWalk, NCWalk, RNCWalk, ShellWalk
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
    print("Evaluating CUSTOM-RW embeddings for ", dataset)
    graph = DatasetPool.load(dataset)
    graph.remove_selfloop_edges()
    graph = graph.to_undirected()
    numl = graph.edges_cnt()

    """
    top_dir = "customrw_embeddings"
    if not os.path.exists(top_dir):
        os.mkdir(top_dir)
    """

    alpha = 1.0
    if dataset == "amazon_electronics_photo" or dataset == "amazon_electronics_computers":
        alpha = 1.15

    DIMS = [10, 25, 50, 100, 200]
    K = 5

    out_log = open("CRWEVAL_" + dataset + ".csv", "w")
    out_log.write("METHOD,DIM,PREC,REC,MAP,F1\n")

    for d in DIMS:
        for i in range(K):
            emb = UnbiasedWalk(graph, d)
            emb.embed()
            s = eval("UNBIASED", d, graph, emb, numl)
            out_log.write(s + "\n")

            emb = NCWalk(graph, d)
            emb.embed()
            s = eval("NCWALK", d, graph, emb, numl)
            out_log.write(s + "\n")

            emb = RNCWalk(graph, d)
            emb.embed()
            s = eval("RNCWALK", d, graph, emb, numl)
            out_log.write(s + "\n")

            emb = ShellWalk(graph, d)
            emb.embed()
            s = eval("SHELLWALK", d, graph, emb, numl)
            out_log.write(s + "\n")

            emb = ShellWalk(graph, d, inverted=True)
            emb.embed()
            s = eval("INVSHELLWALK", d, graph, emb, numl)
            out_log.write(s + "\n")

    out_log.close()

