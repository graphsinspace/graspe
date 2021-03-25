from common.dataset_pool import DatasetPool
from embeddings.embedding_node2vec import Node2VecEmbedding


def test_reconstuction_evaluation():
    # first case

    graph = DatasetPool.load("cora_ml")
    emb_m = Node2VecEmbedding(graph, 10, 1, 1)
    emb_m.embed()

    undirected_projection = graph.to_undirected()
    print("#nodes = ", undirected_projection.nodes_cnt())
    print("#edges = ", undirected_projection.edges_cnt())

    num_links = undirected_projection.edges_cnt()

    reconstructed_graph = emb_m.reconstruct(num_links)
    print("#nodes = ", reconstructed_graph.nodes_cnt())
    print("#edges = ", reconstructed_graph.edges_cnt())

    # graph reconstruction evaluation
    precision_val = undirected_projection.link_precision(reconstructed_graph)
    map_val, _ = undirected_projection.map_value(reconstructed_graph)
    recall_val, _ = undirected_projection.recall(reconstructed_graph)

    print("PRECISION@K = ", precision_val, "MAP = ", map_val, ", RECALL = ", recall_val)

    # second case

    graph = DatasetPool.load("cora_ml").to_undirected()
    emb_m = Node2VecEmbedding(graph, 10, 1, 1)
    emb_m.embed()

    print("#nodes = ", graph.nodes_cnt())
    print("#edges = ", graph.edges_cnt())

    num_links = graph.edges_cnt()

    reconstructed_graph = emb_m.reconstruct(num_links)
    print("#nodes = ", reconstructed_graph.nodes_cnt())
    print("#edges = ", reconstructed_graph.edges_cnt())

    # graph reconstruction evaluation
    precision_val = graph.link_precision(reconstructed_graph)
    map_val, _ = graph.map_value(reconstructed_graph)
    recall_val, _ = graph.recall(reconstructed_graph)

    print("PRECISION@K = ", precision_val, "MAP = ", map_val, ", RECALL = ", recall_val)


if __name__ == "__main__":
    test_reconstuction_evaluation()
