from common.dataset_pool import DatasetPool
from embeddings.embedding_node2vec import Node2VecEmbedding
from evaluation.clustering import ClusteringEval


def test_clustering():
    graph = DatasetPool.load("karate_club_graph")
    embedding = Node2VecEmbedding(graph, 10, 0.1, 0.5)
    embedding.embed()

    clustering_eval = ClusteringEval(graph, embedding, "aGglomerative")
    print("adjMI = ", clustering_eval.get_adjusted_mutual_info_score())
    print("adjRand = ", clustering_eval.get_adjusted_rand_score())
    print("fm = ", clustering_eval.get_fowlkes_mallows_score())
    print("homo = ", clustering_eval.get_homogeneity_score())
    print("rand = ", clustering_eval.get_rand_score())
    print("vm = ", clustering_eval.get_v_measure_score())
    print("compl = ", clustering_eval.get_completness_score())
    print("mi = ", clustering_eval.get_mutual_info_score())
    print("nmi = ", clustering_eval.get_normalized_mutual_info_score())

    assert clustering_eval
