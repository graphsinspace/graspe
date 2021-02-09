from common.dataset_pool import DatasetPool
from embeddings.embedding_node2vec import Node2VecEmbedding
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score
)

if __name__ == '__main__':
    graph = DatasetPool.load("amazon_electronics_photo")
    embedding = Node2VecEmbedding(graph, 10, 0.1, 0.5)
    embedding.embed()

    x = []
    y = []

    nodes = list(embedding._embedding.keys())

    for i in range(len(nodes)):
        node1 = nodes[i]
        for j in range(i + 1, len(nodes)):
            node2 = nodes[j]
            x.append(np.concatenate([embedding._embedding[node1], embedding._embedding[node2]], axis=-1))
            y.append(1 if j in graph.to_networkx().neighbors(i) else 0)
    x = np.stack(x, axis=0)
    y = np.array(y)

    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.33, random_state=42)

    lr = LogisticRegression()
    lr.fit(xtrain, ytrain)
    predictions = lr.predict(xtest)

    acc = accuracy_score(ytest, predictions)
    prec = precision_score(ytest, predictions)
    recall = recall_score(ytest, predictions)

    print('LogisticRegression accuracy score : ', acc)
    print('LogisticRegression precision score : ', prec)
    print('LogisticRegression recall score : ', recall)

    rf = RandomForestClassifier(n_estimators=10)
    rf.fit(xtrain, ytrain)
    predictions = rf.predict(xtest)

    acc = accuracy_score(ytest, predictions)
    prec = precision_score(ytest, predictions)
    recall = recall_score(ytest, predictions)

    print('RandomForestClassifier accuracy score : ', acc)
    print('RandomForestClassifier precision score : ', prec)
    print('RandomForestClassifier recall score : ', recall)
