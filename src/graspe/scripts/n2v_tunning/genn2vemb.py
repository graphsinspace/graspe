from common.dataset_pool import DatasetPool
from embeddings.embedding_node2vec import Node2VecEmbedding
import sys
import os
from datetime import datetime

class EmbGen:
    def __init__(self, g, d, p, q):
        self.emb = Node2VecEmbedding(g, d, p=p, q=q, walk_length=80, num_walks=10)
        self.emb.embed()

    def save(self, output_file):
        self.emb.to_file(output_file)


if __name__ == "__main__":
    p_q_vals = [0.25, 0.5, 1, 2, 4]
    dims = [10, 25, 50, 100, 200]

    dataset = sys.argv[1]
    print("Generating node2vec embeddings for ", dataset)
    graph = DatasetPool.load(dataset).to_undirected()

    top_dir = "genembeddings"
    if not os.path.exists(top_dir):
        os.mkdir(top_dir)

    out_dir = "genembeddings/" + dataset + "_embeddings"

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        print("Directory " + out_dir + " created")
    else:    
        print("Directory " + out_dir + " already exists")

    print(datetime.now())

    for d in dims:
        for p in p_q_vals:
            for q in p_q_vals:
                gen = EmbGen(graph, d, p, q)
                out_emb_name = out_dir + "/" + "n2v-" + dataset + "-" + str(d) + "-" + str(p).replace(".", "_") + "-" + str(q).replace(".", "_") + ".embedding"
                gen.save(out_emb_name)
                print("Embedding for ", d, p, q, "generated")

    print("Generating embedding finished...")
    print(datetime.now())


