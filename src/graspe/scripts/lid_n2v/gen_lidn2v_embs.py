
from embeddings.embedding_lid_node2vec import LIDNode2VecElasticW, LIDNode2VecElasticWPQ
from os import sys
import os
from datetime import datetime
from common.dataset_pool import DatasetPool

if __name__ == "__main__":
    dataset = sys.argv[1]
    print(datetime.now())
    print("Generating LID-aware n2v embeddings for ", dataset)

    alpha = 1.0
    if dataset == "amazon_electronics_photo" or dataset == "amazon_electronics_computers":
        alpha = 1.15

    print(dataset, "alpha=", alpha)

    graph = DatasetPool.load(dataset)
    graph.remove_selfloop_edges()
    graph = graph.to_undirected()

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

    out_dir = "genembeddings/lid_n2v_embeddings/"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print("Directory " + out_dir + " created")
    else:    
        print("Directory " + out_dir + " already exists")

    print("Generating lid-n2v elastic w embeddings")
    for c in conf:
        d, p, q = c[0], c[1], c[2]
        emb = LIDNode2VecElasticW(graph, d, p=c[1], q=c[2], alpha=alpha)
        emb.embed()

        output_file = out_dir + "lidn2vew-" + dataset + "-" + str(d) + "-" + str(p).replace(".", "_") + "-" + str(q).replace(".", "_") + ".embedding"
        print(output_file)
        emb.to_file(output_file)


    print("\nGenerating lid-n2v elastic wpq embeddings")
    for c in conf:
        d, p, q = c[0], c[1], c[2]
        emb = LIDNode2VecElasticWPQ(graph, d, p=c[1], q=c[2], alpha=alpha)
        emb.embed()

        output_file = out_dir + "lidn2vewpq-" + dataset + "-" + str(d) + "-" + str(p).replace(".", "_") + "-" + str(q).replace(".", "_") + ".embedding"
        print(output_file)
        emb.to_file(output_file)

    print(datetime.now())