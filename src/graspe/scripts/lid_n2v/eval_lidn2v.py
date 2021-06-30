from common.dataset_pool import DatasetPool
from embeddings.base.embedding import Embedding
import sys
from os import listdir
from os.path import isfile, join


class EmbEval:
    def __init__(self, dataset, graph, emb_dir="genembeddings/lid_n2v_embeddings"):
        print(dataset)
        self.graph = graph

        self.emb_dir = emb_dir

        self.lidn2vew = []
        self.lidn2vewpq = []
        files = [f for f in listdir(emb_dir) if isfile(join(emb_dir, f))]
        for file_name in files:
            if file_name.startswith("lidn2vew-" + dataset + "-"):
                self.lidn2vew.append(self.parse_file_name(file_name))
            elif file_name.startswith("lidn2vewpq-" + dataset + "-"):
                self.lidn2vewpq.append(self.parse_file_name(file_name))

        self.lidn2vew.sort(key=lambda tup: tup[2])
        self.lidn2vewpq.sort(key=lambda tup: tup[2])

    def parse_file_name(self, file_name):
        toks = file_name.split(".")[0].split("-")
        dataset = toks[1]
        dim = int(toks[2])
        p = float(toks[3].replace("_", "."))
        q = float(toks[4].replace("_", "."))
        return (file_name, dataset, dim, p, q)

    def eval(self):
        print("EVALUATING lid-n2v-ew")
        self.evaluate(self.lidn2vew)

        print("\nEVALUATING lid-n2v-ewqp")
        self.evaluate(self.lidn2vewpq)

    
    def evaluate(self, base):
        print("DATASET,DIM,p,q,PREC,REC,MAP,F1")
        for b in base:
            emb_file, dataset, dim, p, q = b
            emb_file_path = join(self.emb_dir, emb_file)

            emb = Embedding.from_file(emb_file_path)
            numl = self.graph.edges_cnt()
            rg = emb.reconstruct(numl)

            prec = self.graph.link_precision(rg)
            rec, _ = self.graph.recall(rg)
            mapv, _ = self.graph.map_value(rg)

            f1 = 2 * rec * mapv / (rec + mapv)

            s = dataset + "," + str(dim) + "," + str(p) + "," + str(q) + ","
            s += str(prec) + "," + str(rec) + "," + str(mapv) + "," + str(f1)

            print(s)

if __name__ == "__main__":
    dataset = sys.argv[1]
    print("Evaluating embeddings for ", dataset)
    graph = DatasetPool.load(dataset)
    graph.remove_selfloop_edges()
    graph = graph.to_undirected()

    e = EmbEval(dataset, graph)
    e.eval()