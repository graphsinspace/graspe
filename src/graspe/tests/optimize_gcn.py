from common.dataset_pool import DatasetPool
from embeddings.embedding_gcn import GCNEmbedding
import numpy as np
from tqdm import tqdm

NUM_ITER = 20


def gcn_tuning():
    datasets = ['karate_club_graph']
    precs = np.zeros(NUM_ITER)
    maps = np.zeros(NUM_ITER)
    recs = np.zeros(NUM_ITER)
    f1s = np.zeros(NUM_ITER)
    for it in tqdm(range(NUM_ITER)):
        for dataset in datasets:
            if dataset in [
                "davis_southern_women_graph",
                "florentine_families_graph",
                "les_miserables_graph",
            ]:
                continue
            graph = DatasetPool.load(dataset)
            # print("labels =", graph.to_dgl().ndata["label"])
            # graph.set_community_labels()
            # print("community labels =", graph.to_dgl().ndata["label"])
            graph.to_undirected()
            emb_m = GCNEmbedding(
                g=graph,
                d=50,
                epochs=200,
                deterministic=False,
                lr=0.1,
                layer_configuration=(128, 128,),
                act_fn="tanh",
                lid_aware=False,
                lid_k=20
            )
            emb_m.embed()
            num_links = graph.edges_cnt()
            reconstructed_graph = emb_m.reconstruct(num_links)

            # graph reconstruction evaluation
            precision_val = graph.link_precision(reconstructed_graph)
            map_val, _ = graph.map_value(reconstructed_graph)
            recall_val, _ = graph.recall(reconstructed_graph)
            f1_val = (2 * precision_val * recall_val) / (
                    precision_val + recall_val
            )

            precs[it] = precision_val
            maps[it] = map_val
            recs[it] = recall_val
            f1s[it] = f1_val
    print(
        "PRECISION@K = ", precs.mean(),
        "MAP = ", maps.mean(),
        ", RECALL = ", recs.mean(),
        ", F1 = ", f1s.mean(),
    )


if __name__ == "__main__":
    gcn_tuning()
