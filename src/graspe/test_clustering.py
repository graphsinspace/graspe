from common.dataset_pool import DatasetPool
from embeddings.embedding_node2vec import Node2VecEmbedding
from evaluation.clustering import ClusteringEval

graph = DatasetPool.load("karate_club_graph")
embedding = Node2VecEmbedding(graph, 10, 0.1, 0.5)
embedding.embed()

eval = ClusteringEval(graph, embedding, "aGglomerative")
print("adjMI = ", eval.get_adjusted_mutual_info_score())
print("adjRand = ", eval.get_adjusted_rand_score())
print("fm = ", eval.get_fowlkes_mallows_score())
print("homo = ", eval.get_homogeneity_score())
print("rand = ", eval.get_rand_score())
print("vm = ", eval.get_v_measure_score())
print("compl = ", eval.get_completness_score())
print("mi = ", eval.get_mutual_info_score())
print("nmi = ", eval.get_normalized_mutual_info_score())
