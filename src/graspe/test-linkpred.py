from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from embeddings.embedding_node2vec import Node2VecEmbedding
from common.dataset_pool import DatasetPool
import numpy as np
import networkx as nx
from sklearn.metrics import roc_auc_score


graph = DatasetPool.load("karate_club_graph")
embedding = Node2VecEmbedding(graph, 10, 0.1, 0.5)
embedding.embed()

x = []
y = []
nodes = list(embedding._embedding.keys())
# print(nodes)
for i in range(len(nodes)):
    node1 = nodes[i]
    for j in range(i + 1, len(nodes)):
        node2 = nodes[j]
        x.append(
            np.linalg.norm(embedding._embedding[node1] - embedding._embedding[node2])
        )
        y.append(1 if node2 in graph.to_networkx().neighbors(node1) else 0)

# if the graph is directed, the adjacency matrix is not symmetric
# i.e. we need to iterate over the lower triangular part

# if nx.is_directed(graph.to_networkx()):
#     for i in range(len(nodes)):
#         node1 = nodes[i]
#         for j in range(i):
#             node2 = nodes[j]
#             x.append(np.concatenate([embedding._embedding[node1], embedding._embedding[node2]], axis=-1))
#             y.append(1 if node2 in graph.to_networkx().neighbors(node1) else 0)

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
