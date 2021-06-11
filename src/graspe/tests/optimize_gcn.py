from common.dataset_pool import DatasetPool
from embeddings.embedding_gcn import GCNEmbedding


def gcn_tuning():
    datasets = ['cora']
    for dataset in datasets:
        if dataset in [
            "davis_southern_women_graph",
            "florentine_families_graph",
            "les_miserables_graph",
        ]:
            continue
        graph = DatasetPool.load(dataset)
        graph.set_community_labels()
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
        print(
            "PRECISION@K = ", precision_val,
            "MAP = ", map_val,
            ", RECALL = ", recall_val,
            ", F1 = ", f1_val,
        )


if __name__ == "__main__":
    gcn_tuning()
