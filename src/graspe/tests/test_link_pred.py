import networkx as nx
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from common.dataset_pool import DatasetPool
from embeddings.embedding_node2vec import Node2VecEmbedding


def test_link_pred():

    graph = DatasetPool.load("karate_club_graph")
    embedding = Node2VecEmbedding(graph, 10, 0.1, 0.5)
    embedding.embed()

    x = []
    y = []
    nodes = graph.nodes()
    for i in range(len(nodes)):
        node1 = nodes[i]
        for j in range(i + 1, len(nodes)):
            node2 = nodes[j]
            x.append(np.linalg.norm(embedding[node1] - embedding[node2]))
            y.append(1 if node2 in graph.to_networkx().neighbors(node1) else 0)

    # if the graph is directed, the adjacency matrix is not symmetric
    # i.e. we need to iterate over the lower triangular part
    if nx.is_directed(graph.to_networkx()):
        for i in range(len(nodes)):
            node1 = nodes[i]
            for j in range(i):
                node2 = nodes[j]
                x.append(np.linalg.norm(embedding[node1] - embedding[node2]))
                y.append(1 if node2 in graph.to_networkx().neighbors(node1) else 0)

    # print(x)
    # print(y)

    xtrain, xtest, ytrain, ytest = train_test_split(
        np.array(x).reshape(-1, 1), y, test_size=0.33, random_state=42
    )
    # print(xtrain)

    # print(xtest)

    lr = LogisticRegression()
    lr.fit(xtrain, ytrain)
    predictions = lr.predict_proba(xtest)
    ras = roc_auc_score(ytest, predictions[:, 1])
    print(ras)

    assert ras is not None
