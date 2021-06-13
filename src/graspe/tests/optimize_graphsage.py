from common.dataset_pool import DatasetPool
from embeddings.embedding_graphsage import GraphSageEmbedding


def graphsage_tuning():
    # datasets = DatasetPool.get_datasets()
    datasets = ['karate_club_graph']
    for dataset in datasets:
        if dataset in [
            "davis_southern_women_graph",
            "florentine_families_graph",
            "les_miserables_graph",
        ]:
            continue
        graph = DatasetPool.load(dataset)
        print("labels =", graph.to_dgl().ndata["label"])
        graph.set_community_labels()
        print("community labels =", graph.to_dgl().ndata["label"])
        graph.to_undirected()
        emb_m = GraphSageEmbedding(
            g=graph,
            d=100,
            epochs=100,
            deterministic=False,
            lr=0.1,
            layer_configuration=(256, 512, 256,),
            act_fn="tanh",
            lid_aware=False,
            lid_k=20,
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
    graphsage_tuning()
