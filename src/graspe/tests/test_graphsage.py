from common.dataset_pool import DatasetPool
from embeddings.embedding_gcn import GCNEmbedding
from embeddings.embedding_graphsage import GraphSageEmbedding


def test_graphsage():
    g = DatasetPool.load("cora_ml")
    e = GraphSageEmbedding(g, d=500, epochs=200)
    e.embed()
    assert e._embedding is not None
    # print(e._embedding)


if __name__ == "__main__":
    test_graphsage()
