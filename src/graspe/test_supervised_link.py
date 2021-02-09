from common.dataset_pool import DatasetPool
from embeddings.embedding_node2vec import Node2VecEmbedding

import networkx as nx
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score
)

if __name__ == '__main__':
    graph = DatasetPool.load('florentine_families_graph')
    embedding = Node2VecEmbedding(graph, 10, 0.1, 0.5)
    start_time = time.time()
    embedding.embed()
    print('{}s needed for embedding of the graph'.format(time.time() - start_time))

    x = []
    y = []

    nodes = list(embedding._embedding.keys())
    print('num nodes: ', len(nodes))

    for i in range(len(nodes)):
        node1 = nodes[i]
        for j in range(i + 1, len(nodes)):
            node2 = nodes[j]
            x.append(np.concatenate([embedding._embedding[node1], embedding._embedding[node2]], axis=-1))
            y.append(1 if node2 in graph.to_networkx().neighbors(node1) else 0)

    # if the graph is directed, the adjacency matrix is not symmetric
    # i.e. we need to iterate over the lower triangular part
    if nx.is_directed(graph.to_networkx()):
        for i in range(len(nodes)):
            node1 = nodes[i]
            for j in range(i):
                node2 = nodes[j]
                x.append(np.concatenate([embedding._embedding[node1], embedding._embedding[node2]], axis=-1))
                y.append(1 if node2 in graph.to_networkx().neighbors(node1) else 0)

    x = np.stack(x, axis=0)
    y = np.array(y)

    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.33, random_state=42)

    rf = RandomForestClassifier(n_estimators=10)
    rf.fit(xtrain, ytrain)
    predictions = rf.predict(xtest)

    acc = accuracy_score(ytest, predictions)
    prec = precision_score(ytest, predictions)
    recall = recall_score(ytest, predictions)

    print('RandomForestClassifier accuracy score : ', acc)
    print('RandomForestClassifier precision score : ', prec)
    print('RandomForestClassifier recall score : ', recall)
