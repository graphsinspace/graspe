#import sys
import os

#sys.path.append('/home/knezevicd/grasp/')

from embeddings.base.embedding import Embedding
from common.dataset_pool import DatasetPool
from evaluation.clustering_eval import ClusteringEval

def test_clustering():
    f = open("test_cl.csv", "w")
    listOfFiles = []
    ds_names = DatasetPool.get_datasets()

    for (dirpath, dirnames, filenames) in os.walk('/home/shared/grasp/n2v_best_embeddings'):
        listOfFiles += [os.path.join(dirpath, file) for file in filenames]

    for embFile in listOfFiles:
        embedding = Embedding.from_file(embFile)
        graph = None
        for name in ds_names:
            if name in embFile:
                graph = DatasetPool.load(name)

            f.write("name,adjMI,adjRand,fm,homo,rand,vm,compl,mi,nmi,silhouette,cal_har_sc,davis_bol\n")
            clustering_eval = ClusteringEval(graph, embedding, "kmeans")
            f.write(embFile + "," + clustering_eval.get_adjusted_mutual_info_score())
            f.write("," + clustering_eval.get_adjusted_rand_score())
            f.write("," + clustering_eval.get_fowlkes_mallows_score())
            f.write("," + clustering_eval.get_homogeneity_score())
            f.write("," + clustering_eval.get_rand_score())
            f.write("," + clustering_eval.get_v_measure_score())
            f.write("," + clustering_eval.get_completness_score())
            f.write("," + clustering_eval.get_mutual_info_score())
            f.write("," + clustering_eval.get_normalized_mutual_info_score())
            f.write("," + clustering_eval.get_silhouette_score())
            f.write("," + clustering_eval.get_calinski_harabasz_score())
            f.write("," + clustering_eval.get_davis_bolding_score())
            f.write("\n")
            f.flush()

    f.flush()
    f.close()
    

print("pokusavam da poteram test")
test_clustering()
print("poterao test")