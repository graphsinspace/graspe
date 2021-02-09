from common.dataset_pool import DatasetPool
from embeddings.embedding_gcn import GCNEmbedding


def test_gcn_citeseer():
    g = DatasetPool.load("citeseer")
    e = GCNEmbedding(g, d=10, epochs=1)
    e.embed()
    assert e._embedding is not None
    for i in range(34):
        print(i, e[i])


if __name__ == "__main__":
    test_gcn_citeseer()
