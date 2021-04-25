from common.dataset_pool import DatasetPool
from embeddings.embedding_gcn import GCNEmbedding


def test_reconstuction_evaluation():
    dataset = 'cora_ml'
    with open('./{}_results.csv'.format(dataset), 'w') as file:
        file.write('PARAMS,PRECISION@K,MAP,RECALL,F1\n')
    # first case
    iter = 1
    for d in [10, 25, 50, 100, 200]:
        for epochs in [10, 50, 100, 200]:
            for n_layers in [1, 2, 3, 4]:
                for dropout in [0.0, 0.1, 0.2, 0.3]:
                    print('...starting to work on iter {} out of {}... '.format(iter, 4*4*4*3))
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
                    print('PRECISION@K = ', precision_val, 'MAP = ', map_val,
                          ', RECALL = ', recall_val, ', F1 = ', f1_val)
                    with open('./{}_results.csv'.format(dataset), 'a') as file:
                        file.write('d={}_epochs={}_n_layers={}_dropout={},{},{},{}.{}\n'.format(
                            d,
                            epochs,
                            n_layers,
                            dropout,
                            precision_val,
                            map_val,
                            recall_val,
                            f1_val))

                    iter += 1

if __name__ == '__main__':
    test_reconstuction_evaluation()
