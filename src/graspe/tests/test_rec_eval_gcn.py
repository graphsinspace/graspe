import sys
sys.path.append('../')

from common.dataset_pool import DatasetPool
from embeddings.embedding_gcn import GCNEmbedding


def test_reconstuction_evaluation():
    dataset = 'cora_ml'
    with open('./{}_undir.csv'.format(dataset), 'w') as file:
        file.write('PRECISION@K,MAP,RECALL,F1\n')
    with open('./{}_dir.csv'.format(dataset), 'w') as file:
        file.write('PRECISION@K,MAP,RECALL,F1\n')
    # first case
    for d in [10, 25, 50, 100, 200]:
        for epochs in [10, 50, 100]:
            for n_layers in [1, 2, 3, 4]:
                for dropout in [0.0, 0.1, 0.2, 0.3]:
                    print('...starting on {} out of {}... ')
                    graph = DatasetPool.load(dataset)
                    emb_m = GCNEmbedding(graph, d=d, epochs=epochs, n_layers=n_layers, dropout=dropout)
                    emb_m.embed()

                    undirected_projection = graph.to_undirected()
                    # print("#nodes = ", undirected_projection.nodes_cnt())
                    # print("#edges = ", undirected_projection.edges_cnt())

                    num_links = undirected_projection.edges_cnt()

                    reconstructed_graph = emb_m.reconstruct(num_links)
                    # print("#nodes = ", reconstructed_graph.nodes_cnt())
                    # print("#edges = ", reconstructed_graph.edges_cnt())

                    # graph reconstruction evaluation
                    precision_val = undirected_projection.link_precision(reconstructed_graph)
                    map_val, _ = undirected_projection.map_value(reconstructed_graph)
                    recall_val, _ = undirected_projection.recall(reconstructed_graph)
                    f1_val = (2 * precision_val * recall_val) / (precision_val + recall_val)
                    print("PRECISION@K = ", precision_val, "MAP = ", map_val,
                          ", RECALL = ", recall_val, ', F1 = ', f1_val)
                    with open('./cora_ml_undir.csv', 'a') as file:
                        file.write('{},{},{}.{}\n'.format(precision_val, map_val, recall_val, f1_val))

                    # second case
                    graph = DatasetPool.load(dataset).to_undirected()
                    emb_m = GCNEmbedding(graph, d=d, epochs=epochs, n_layers=n_layers, dropout=dropout)
                    emb_m.embed()

                    # print("#nodes = ", graph.nodes_cnt())
                    # print("#edges = ", graph.edges_cnt())

                    num_links = graph.edges_cnt()

                    reconstructed_graph = emb_m.reconstruct(num_links)
                    # print("#nodes = ", reconstructed_graph.nodes_cnt())
                    # print("#edges = ", reconstructed_graph.edges_cnt())

                    # graph reconstruction evaluation
                    precision_val = graph.link_precision(reconstructed_graph)
                    map_val, _ = graph.map_value(reconstructed_graph)
                    recall_val, _ = graph.recall(reconstructed_graph)
                    f1_val = (2 * precision_val * recall_val) / (precision_val + recall_val)
                    print("PRECISION@K = ", precision_val, "MAP = ", map_val,
                          ", RECALL = ", recall_val, ', F1 = ', f1_val)
                    with open('./cora_ml_dir.csv', 'a') as file:
                        file.write('{},{},{}.{}\n'.format(precision_val, map_val, recall_val, f1_val))


if __name__ == "__main__":
    test_reconstuction_evaluation()
