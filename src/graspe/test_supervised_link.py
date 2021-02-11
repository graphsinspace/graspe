from common.dataset_pool import DatasetPool
from embeddings.embedding_node2vec import Node2VecEmbedding

import networkx as nx
import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

if __name__ == "__main__":
    percentages = np.linspace(0.1, 1, 10)
    graph = DatasetPool.load("karate_club_graph")
    embedding = Node2VecEmbedding(graph, 10, 0.1, 0.5)
    start_time = time.time()
    embedding.embed()
    print("{}s needed for embedding of the graph".format(time.time() - start_time))

    x = []
    y = []

    nodes = graph.nodes()
    print("num nodes: ", len(nodes))
    for i in range(len(nodes)):
        node1 = nodes[i][0]
        for j in range(i + 1, len(nodes)):
            node2 = nodes[j][0]
            x.append(np.concatenate([embedding[node1], embedding[node2]], axis=-1))
            y.append(1 if graph.has_edge(node1, node2) else 0)

    # if the graph is directed, the adjacency matrix is not symmetric
    # i.e. we need to iterate over the lower triangular part
    if nx.is_directed(graph.to_networkx()):
        for i in range(len(nodes)):
            node1 = nodes[i][0]
            for j in range(i):
                node2 = nodes[j][0]
                x.append(np.concatenate([embedding[node1], embedding[node2]], axis=-1))
                y.append(1 if graph.has_edge(node1, node2) else 0)

    x = np.stack(x, axis=0)
    y = np.array(y)
    print('number of positive samples in dataset : ', len(y[y == 1]))
    num_negative = len(np.where(y == 0)[0])
    for percentage in percentages:
        # subsampling
        neg_indices = np.where(y == 0)[0]  # indices of negative samples
        np.random.shuffle(neg_indices)  # shuffle (inplace)
        neg_indices_sub = neg_indices[:int(percentage * num_negative)]  # subsample
        pos_indices = np.where(y == 1)[0]  # indices of positive samples
        indices = np.concatenate([neg_indices_sub, pos_indices], axis=-1)
        np.random.shuffle(indices)  # shuffle (inplace)
        x_sub = x[indices]
        y_sub = y[indices]
        neg = len(y_sub[y_sub == 0]) / len(y_sub)
        print('number of samples in dataset @ {}% of negative class : '.format(percentage*100), len(x_sub))
        print('percentage of negative class in formed dataset @ {}% of negative class : '.format(percentage*100), neg)
        xtrain, xtest, ytrain, ytest = train_test_split(
            x_sub, y_sub, test_size=0.33, random_state=42
        )

        rf = RandomForestClassifier(n_estimators=10)
        rf.fit(xtrain, ytrain)
        predictions = rf.predict(xtest)

        acc = accuracy_score(ytest, predictions)
        prec = precision_score(ytest, predictions)
        recall = recall_score(ytest, predictions)

        print('RandomForestClassifier accuracy score @ {}% of negative class : '.format(percentage*100), acc)
        print('RandomForestClassifier precision score @ {}% of negative class : '.format(percentage*100), prec)
        print('RandomForestClassifier recall score @ {}% of negative class : '.format(percentage*100), recall)
        print('\n')

