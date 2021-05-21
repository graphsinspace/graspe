import sys
import os
sys.path.append('/s01/dmi/lucy/graspe/src/graspe')

from common.dataset_pool import DatasetPool
from embeddings.embedding_node2vec import Node2VecEmbedding
from embeddings.embedding_gae import GAEEmbedding
from embeddings.embedding_graphsage import GraphSage
from evaluation.clustering import ClusteringEval


def test_clustering():
    f = open("test_cl.txt", "w")
    ds_names = DatasetPool.get_datasets
    for name in ds_names:
        f.write("dateset = %s\n".format(name))
        graph = DatasetPool.load(name)
        
        #node2vec
        f.write("node2vec\n")
        embedding = Node2VecEmbedding(graph, 10, 0.5, 1)
        embedding.embed()

        clustering_eval = ClusteringEval(graph, embedding, "kmeans")
        f.write("adjMI = %f\n".format(clustering_eval.get_adjusted_mutual_info_score()))
        f.write("adjRand = %f\n".format(clustering_eval.get_adjusted_rand_score()))
        f.write("fm = %f\n".format(clustering_eval.get_fowlkes_mallows_score()))
        f.write("homo = %f\n".format(clustering_eval.get_homogeneity_score()))
        f.write("rand = %f\n".format(clustering_eval.get_rand_score()))
        f.write("vm = %f\n".format(clustering_eval.get_v_measure_score()))
        f.write("compl = %f\n".format(clustering_eval.get_completness_score()))
        f.write("mi = %f\n".format(clustering_eval.get_mutual_info_score()))
        f.write("nmi = %f\n".format(clustering_eval.get_normalized_mutual_info_score()))
        f.write("silhouette = %f\n".format(clustering_eval.get_silhouette_score()))
        f.write("calinski_harabasz_score = %f\n".format(clustering_eval.get_calinski_harabasz_score()))
        f.write("davis_bolding = %f\n".format(clustering_eval.get_davis_bolding_score()))
        f.write("\n\n\n")
        f.flush()

        #gae
        f.write("gae\n")
        embedding = GAEEmbedding(graph, 10)
        embedding.embed()

        clustering_eval = ClusteringEval(graph, embedding, "kmeans")
        f.write("adjMI = %f\n".format(clustering_eval.get_adjusted_mutual_info_score()))
        f.write("adjRand = %f\n".format(clustering_eval.get_adjusted_rand_score()))
        f.write("fm = %f\n".format(clustering_eval.get_fowlkes_mallows_score()))
        f.write("homo = %f\n".format(clustering_eval.get_homogeneity_score()))
        f.write("rand = %f\n".format(clustering_eval.get_rand_score()))
        f.write("vm = %f\n".format(clustering_eval.get_v_measure_score()))
        f.write("compl = %f\n".format(clustering_eval.get_completness_score()))
        f.write("mi = %f\n".format(clustering_eval.get_mutual_info_score()))
        f.write("nmi = %f\n".format(clustering_eval.get_normalized_mutual_info_score()))
        f.write("silhouette = %f\n".format(clustering_eval.get_silhouette_score()))
        f.write("calinski_harabasz_score = %f\n".format(clustering_eval.get_calinski_harabasz_score()))
        f.write("davis_bolding = %f\n".format(clustering_eval.get_davis_bolding_score()))
        f.write("\n\n\n")
        f.flush()

        #graphsage
        f.write("graph sage\n")
        embedding = GraphSage(graph, 10, 20)
        embedding.embed()

        clustering_eval = ClusteringEval(graph, embedding, "kmeans")
        f.write("adjMI = %f\n".format(clustering_eval.get_adjusted_mutual_info_score()))
        f.write("adjRand = %f\n".format(clustering_eval.get_adjusted_rand_score()))
        f.write("fm = %f\n".format(clustering_eval.get_fowlkes_mallows_score()))
        f.write("homo = %f\n".format(clustering_eval.get_homogeneity_score()))
        f.write("rand = %f\n".format(clustering_eval.get_rand_score()))
        f.write("vm = %f\n".format(clustering_eval.get_v_measure_score()))
        f.write("compl = %f\n".format(clustering_eval.get_completness_score()))
        f.write("mi = %f\n".format(clustering_eval.get_mutual_info_score()))
        f.write("nmi = %f\n".format(clustering_eval.get_normalized_mutual_info_score()))
        f.write("silhouette = %f\n".format(clustering_eval.get_silhouette_score()))
        f.write("calinski_harabasz_score = %f\n".format(clustering_eval.get_calinski_harabasz_score()))
        f.write("davis_bolding = %f\n".format(clustering_eval.get_davis_bolding_score()))
        f.write("\n\n\n")

    f.flush()
    f.close()
    assert clustering_eval
