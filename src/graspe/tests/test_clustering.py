import sys
import os
sys.path.append('/s01/dmi/lucy/graspe/src/graspe')

from common.dataset_pool import DatasetPool
from embeddings.embedding_node2vec import Node2VecEmbedding
from embeddings.embedding_gae import GAEEmbedding
from embeddings.embedding_graphsage import GraphSageEmbedding
from evaluation.clustering import ClusteringEval


def test_clustering():
    f = open("test_cl.txt", "w")
    ds_names = DatasetPool.get_datasets()
    for name in ds_names:
        f.write("dateset = " + name + "\n")
        graph = DatasetPool.load(name)
        #node2vec
        f.write("node2vec")
        embedding = Node2VecEmbedding(graph, 10, 0.5, 1)
        embedding.embed()

        clustering_eval = ClusteringEval(graph, embedding, "kmeans")
        f.write("adjMI = " + clustering_eval.get_adjusted_mutual_info_score() + "\n")
        f.write("adjRand = " + clustering_eval.get_adjusted_rand_score() + "\n")
        f.write("fm = " + clustering_eval.get_fowlkes_mallows_score() + "\n")
        f.write("homo = " + clustering_eval.get_homogeneity_score() + "\n")
        f.write("rand = " + clustering_eval.get_rand_score() + "\n")
        f.write("vm = " + clustering_eval.get_v_measure_score() + "\n")
        f.write("compl = " + clustering_eval.get_completness_score() + "\n")
        f.write("mi = " + clustering_eval.get_mutual_info_score() + "\n")
        f.write("nmi = " + clustering_eval.get_normalized_mutual_info_score() + "\n")
        f.write("silhouette = " + clustering_eval.get_silhouette_score() + "\n")
        f.write("calinski_harabasz_score = " + clustering_eval.get_calinski_harabasz_score() + "\n")
        f.write("davis_bolding = " + clustering_eval.get_davis_bolding_score() + "\n")
        f.write("\n\n\n")
        f.flush()

        #gae
        f.write("gae\n")
        embedding = GAEEmbedding(graph, 10)
        embedding.embed()

        clustering_eval = ClusteringEval(graph, embedding, "kmeans")
        f.write("adjMI = " + clustering_eval.get_adjusted_mutual_info_score() + "\n")
        f.write("adjRand = " + clustering_eval.get_adjusted_rand_score() + "\n")
        f.write("fm = " + clustering_eval.get_fowlkes_mallows_score() + "\n")
        f.write("homo = " + clustering_eval.get_homogeneity_score() + "\n")
        f.write("rand = " + clustering_eval.get_rand_score() + "\n")
        f.write("vm = " + clustering_eval.get_v_measure_score() + "\n")
        f.write("compl = " + clustering_eval.get_completness_score() + "\n")
        f.write("mi = " + clustering_eval.get_mutual_info_score() + "\n")
        f.write("nmi = " + clustering_eval.get_normalized_mutual_info_score() + "\n")
        f.write("silhouette = " + clustering_eval.get_silhouette_score() + "\n")
        f.write("calinski_harabasz_score = " + clustering_eval.get_calinski_harabasz_score() + "\n")
        f.write("davis_bolding = " + clustering_eval.get_davis_bolding_score() + "\n")
        f.write("\n\n\n")
        f.flush()

        #graphsage
        f.write("graph sage\n")
        embedding = GraphSageEmbedding(graph, 10, 20)
        embedding.embed()

        clustering_eval = ClusteringEval(graph, embedding, "kmeans")
        f.write("adjMI = " + clustering_eval.get_adjusted_mutual_info_score() + "\n")
        f.write("adjRand = " + clustering_eval.get_adjusted_rand_score() + "\n")
        f.write("fm = " + clustering_eval.get_fowlkes_mallows_score() + "\n")
        f.write("homo = " + clustering_eval.get_homogeneity_score() + "\n")
        f.write("rand = " + clustering_eval.get_rand_score() + "\n")
        f.write("vm = " + clustering_eval.get_v_measure_score())
        f.write("compl = " + clustering_eval.get_completness_score() + "\n")
        f.write("mi = " + clustering_eval.get_mutual_info_score() + "\n")
        f.write("nmi = " + clustering_eval.get_normalized_mutual_info_score() + "\n")
        f.write("silhouette = " + clustering_eval.get_silhouette_score() + "\n")
        f.write("calinski_harabasz_score = " + clustering_eval.get_calinski_harabasz_score() + "\n")
        f.write("davis_bolding = " + clustering_eval.get_davis_bolding_score() + "\n")
        f.write("\n\n\n")

    f.flush()
    f.close()
    assert clustering_eval

print("pokusavam da poteram test")
test_clustering()
print("poterao test")