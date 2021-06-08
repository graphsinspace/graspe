from common.dataset_pool import DatasetPool
from embeddings.embedding_gcn import GCNEmbedding


def gcn_tuning():
    for dataset in [
        "cora_ml",
        "amazon_electronics_computers",
        "amazon_electronics_photo",
        "citeseer",
        "cora",
        "dblp",
        "pubmed",
        "karate_club_graph",
        "davis_southern_women_graph",
        "florentine_families_graph",
        "les_miserables_graph",
    ]:
        iter = 1
        with open("/home/dusan/graspe_gcn_res/{}.csv".format(dataset), "w") as file:
            file.write("PARAMS,PRECISION@K,MAP,RECALL,F1\n")
        graph = DatasetPool.load(dataset).to_undirected()
        print("... WORKING ON {} ...".format(dataset.upper()))
        for d in [10, 25, 50, 100, 200]:
            for epochs in [50, 100, 200, 400]:
                for n_layers in [1, 2, 3, 4]:
                    for dropout in [0.0, 0.1, 0.2, 0.3]:
                        print(
                            "... starting to work on iter {} out of {} ... ".format(
                                iter, 320
                            )
                        )
                        emb_m = GCNEmbedding(
                            graph,
                            d=d,
                            epochs=epochs,
                            n_layers=n_layers,
                            dropout=dropout,
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
                            "PRECISION@K = ",
                            precision_val,
                            "MAP = ",
                            map_val,
                            ", RECALL = ",
                            recall_val,
                            ", F1 = ",
                            f1_val,
                        )
                        with open(
                            "/home/dusan/graspe_gcn_res/{}.csv".format(dataset), "a"
                        ) as file:
                            file.write(
                                "d={}_epochs={}_n_layers={}_dropout={},{},{},{},{}\n".format(
                                    d,
                                    epochs,
                                    n_layers,
                                    dropout,
                                    precision_val,
                                    map_val,
                                    recall_val,
                                    f1_val,
                                )
                            )

                        iter += 1


if __name__ == "__main__":
    gcn_tuning()
