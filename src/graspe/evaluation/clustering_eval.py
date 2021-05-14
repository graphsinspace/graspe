import skmultilearn.cluster.base.LabelGraphClustererBase as clusterer

def get_method(name):
    name_sm = name.lower()

    if(name_sm == "louvain"):
        return clusterer.Louvain()
    elif(name_sm == "walktrap"):
        return None #TODO: promeniti
    else: 
        return clusterer.PropagationClustering()


def evaluate(name, graph, embedding):
    method = get_method(name)
    graph_mat = graph.to_adj_matrix()
    labels = method.fit_transform(graph_mat)
    return labels