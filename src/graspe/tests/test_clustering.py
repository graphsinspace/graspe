import sys
import os

from graspe.embeddings.base.embedding import Embedding
from graspe.common.dataset_pool import DatasetPool
from graspe.evaluation.clustering_eval import ClusteringEval
sys.path.append('/s01/dmi/lucy/graspe/src/graspe')

def test_clustering():
    f = open("test_cl.txt", "w")
    listOfFiles = []
    ds_names = DatasetPool.get_datasets()

    for (dirpath, dirnames, filenames) in os.walk('/home/shared/grasp/embeddings'):
        listOfFiles += [os.path.join(dirpath, file) for file in filenames]

    for embFile in listOfFiles:
        embedding = Embedding.from_file(embFile)
        for name in ds_names:
            if name in embFile:
                graph = DatasetPool.load(name)

            f.write(name + "\n")
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

    f.flush()
    f.close()
    assert clustering_eval

print("pokusavam da poteram test")
test_clustering()
print("poterao test")