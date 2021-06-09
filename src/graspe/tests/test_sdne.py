from common.dataset_pool import DatasetPool
from embeddings.embedding_sdne import SDNEEmbedding


def test_sdne():
    g = DatasetPool.load("citeseer")

    e = SDNEEmbedding(
        g=g,
        d=10,
        hidden_size=[32, 16],
        alpha=1,
        beta=5.0,
        nu1=1e-5,
        nu2=1e-4,
        batch_size=1024,
        epochs=100,
        verbose=1,
    )
    e.embed()
    assert e._embedding is not None
    for i in range(len(e._embedding)):
        print(i, e[i])


if __name__ == "__main__":
    test_sdne()
