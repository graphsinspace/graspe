import skmultilearn.cluster.base.LabelGraphClustererBase as clusterer
import evaluation.clustering.ClusteringEval

def get_method(name):
    name_sm = name.lower()

    if(name_sm == "louvain"):
        return clusterer.Louvain()
    elif(name_sm == "walktrap"):
        return None #TODO: promeniti
    else: 
        return clusterer.PropagationClustering()


def pecentage_equal_labels(name, graph, embedding):
    method = get_method(name)
    graph_mat = graph.to_adj_matrix()
    labels = method.fit_transform(graph_mat) #niz labela za svaki cvor
    cl_eval = ClusteringEval(graph, embedding, 'kmeans')
    clusters_eval = cl_eval.evaluate_extern_labels(labels) #niz labela za svaki cvor
    num_equal = 0
    for i in range(len(labels)):
        for j in range(len(labels)):
            if(labels[i] == labels[j] and clusters_eval[i] == clusters_eval[j]):
                num_equal = num_equal + 1

    percentage = num_equal/(len(labels) * len(labels))     
    return percentage