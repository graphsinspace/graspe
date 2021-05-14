from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics.cluster import completeness_score
from sklearn.metrics.cluster import fowlkes_mallows_score
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics.cluster import mutual_info_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import rand_score
from sklearn.metrics.cluster import v_measure_score

from sklearn.metrics.cluster import silhouette_score
from sklearn.metrics.cluster import calinski_harabasz_score
from sklearn.metrics.cluster import davies_bouldin_score


class ClusteringEval:
    def __init__(self, graph, embedding, clustering_type):
        self._graph = graph
        self._embedding = embedding
        self._labels = [n[1]["label"] for n in graph.nodes()]
        self._node_vectors = [embedding[n[0]] for n in graph.nodes()]
        self._ajusted_mutual_info_score = 0.0
        self._adjusted_rand_score = 0.0
        self._completeness_score = 0.0
        self._fowlkes_mallows_score = 0.0
        self._homogeneity_score = 0.0
        self._mutual_info_score = 0.0
        self._normalized_mutual_info_score = 0.0
        self._rand_score = 0.0
        self._v_measure_score = 0.0
        self._silhouette_score = 0.0
        self._calinski_harabasz_score = 0.0
        self._davies_boldin_score = 0.0
        self.__evaluate(clustering_type)

    def __evaluate(self, clustering_type):
        no_clusters = len(set(self._labels))
        function = self.__get_clustering_function(clustering_type)
        clustering = function(no_clusters)
        clusters = clustering.fit_predict(self._node_vectors)
        self._ajusted_mutual_info_score = adjusted_mutual_info_score(
            self._labels, clusters
        )
        self._adjusted_rand_score = adjusted_rand_score(self._labels, clusters)
        self._completeness_score = completeness_score(self._labels, clusters)
        self._fowlkes_mallows_score = fowlkes_mallows_score(self._labels, clusters)
        self._homogeneity_score = homogeneity_score(self._labels, clusters)
        self._mutual_info_score = mutual_info_score(self._labels, clusters)
        self._normalized_mutual_info_score = normalized_mutual_info_score(
            self._labels, clusters
        )
        self._rand_score = rand_score(self._labels, clusters)
        self._v_measure_score = v_measure_score(self._labels, clusters)
        self._silhouette_score = silhouette_score(self._embedding, clusters)
        self._calinski_harabasz_score = calinski_harabasz_score(self._embedding, clusters)
        self._davies_boldin_score = davies_bouldin_score(self._embedding, clusters)


    def __get_clustering_function(self, clustering_type):
        cltype = clustering_type.lower()
        if cltype == "agglomerative":
            return AgglomerativeClustering
        elif cltype == "kmeans":
            return KMeans
        else:
            return SpectralClustering

    def get_adjusted_mutual_info_score(self):
        return self._ajusted_mutual_info_score

    def get_adjusted_rand_score(self):
        return self._adjusted_rand_score

    def get_completness_score(self):
        return self._completeness_score

    def get_fowlkes_mallows_score(self):
        return self._fowlkes_mallows_score

    def get_homogeneity_score(self):
        return self._homogeneity_score

    def get_mutual_info_score(self):
        return self._mutual_info_score

    def get_normalized_mutual_info_score(self):
        return self._normalized_mutual_info_score

    def get_rand_score(self):
        return self._rand_score

    def get_v_measure_score(self):
        return self._v_measure_score

    def get_silhouette_score(self):
        return self._silhouette_score

    def get_calinski_harabasz_score(self):
        return self._calinski_harabasz_score
    
    def get_davis_bolding_score(self):
        return self._davies_boldin_score
