from common.dataset_pool import DatasetPool
from embeddings.embedding_graphsage import GraphSageEmbedding


def graphsage_tuning():
    datasets = DatasetPool.get_datasets()
    for dataset in datasets:
        if dataset in [
            "davis_southern_women_graph",
            "florentine_families_graph",
            "les_miserables_graph",
        ]:
            continue
        iter = 1
        with open(
            "/home/dusan/graspe_graphsage_res/{}.csv".format(dataset), "w"
        ) as file:
            file.write("PARAMS,PRECISION@K,MAP,RECALL,F1\n")
        graph = DatasetPool.load(dataset).to_undirected()
        print("... WORKING ON {} ...".format(dataset.upper()))
        for d in [10, 25, 50, 100, 200]:
            for epochs in [100, 200]:
                for n_layers in [1, 2, 3, 4]:
                    for dropout in [0.0, 0.2]:
                        for hidden in [32, 64]:
                            print(
                                "... starting to work on iter {} out of {} ... ".format(
                                    iter, 160
                                )
                            )
                            emb_m = GraphSageEmbedding(
                                graph,
                                d=d,
                                hidden=hidden,
                                epochs=epochs,
                                layers=n_layers,
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
                                    "d={}_epochs={}_n_layers={}_dropout={}_hidden={},{},{},{},{}\n".format(
                                        d,
                                        epochs,
                                        n_layers,
                                        dropout,
                                        hidden,
                                        precision_val,
                                        map_val,
                                        recall_val,
                                        f1_val,
                                    )
                                )

                            iter += 1


if __name__ == "__main__":
    graphsage_tuning()
