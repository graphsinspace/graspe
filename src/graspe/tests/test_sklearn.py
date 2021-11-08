# Examples from @svc as tests

from sklearn import ensemble
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    completeness_score,
)
from sklearn.model_selection import train_test_split

from common.dataset_pool import DatasetPool
from embeddings.embedding_gcn import GCNEmbedding


def test_classification():
    # ucitavanje grafa
    graph = DatasetPool.load("karate_club_graph")

    # formiranje embedinga za ucitani graf
    dim = 10
    epochs = 100
    embedding = GCNEmbedding(graph, dim, epochs)
    embedding.embed()

    # priprema za klasterisanje / klasifikaciju
    nodes = graph.nodes()
    labels = [n[1]["label"] for n in nodes]
    node_vectors = [embedding[n[0]] for n in nodes]

    # klasifikacija
    train_data, test_data, train_labels, test_labels = train_test_split(
        node_vectors, labels, test_size=0.33
    )
    rf = ensemble.RandomForestClassifier(n_estimators=10).fit(train_data, train_labels)
    predicted_labels = rf.predict(test_data)

    # evaluacija klasifikacionog modela
    acc = accuracy_score(test_labels, predicted_labels)
    print("Accuracy", acc)
    print("Precisions: ", precision_score(test_labels, predicted_labels, average=None))
    print("Recalls: ", recall_score(test_labels, predicted_labels, average=None))

    assert acc > 0


def test_clustering():
    # ucitavanje grafa
    graph = DatasetPool.load("karate_club_graph")

    # formiranje embedinga za ucitani graf
    dim = 10
    epochs = 100
    embedding = GCNEmbedding(graph, dim, epochs)
    embedding.embed()

    # priprema za klasterisanje / klasifikaciju
    nodes = graph.nodes()
    labels = [n[1]["label"] for n in nodes]
    node_vectors = [embedding[n[0]] for n in nodes]

    """
    for n in nodes:
        nid = n[0]                # id cvora
        nattr = n[1]              # atributi cvora
        label = nattr['label']    # labela cvora
        labels.append(label)
        emb = embedding[nid]      # embedding cvora
        node_vectors.append(emb)
    """

    # klasterisanje
    km = KMeans(n_clusters=2)
    clusters = km.fit_predict(node_vectors)
    print(labels)
    print(clusters)

    # evaluacija klasterisanja
    acc = completeness_score(labels, clusters)
    print("ACC = ", acc)

    assert acc > 0


if __name__ == "__main__":
    test_clustering()
    test_classification()
