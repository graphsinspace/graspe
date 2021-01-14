from common.dataset_pool import DatasetPool
from embeddings.embedding_gae import GAEEmbedding


def test_gae_normal():
    g = DatasetPool.load("karate_club_graph")
    gae_embedding = GAEEmbedding(g, d=10, epochs=5, variational=False, linear=False)
    gae_embedding.embed()
    assert gae_embedding._embedding is not None
    for i in range(34):
        print(i, gae_embedding[i])


def test_gae_variational():
    g = DatasetPool.load("karate_club_graph")
    gae_embedding = GAEEmbedding(g, d=10, epochs=5, variational=True, linear=False)
    gae_embedding.embed()
    assert gae_embedding._embedding is not None
    for i in range(34):
        print(i, gae_embedding[i])


def test_gae_normal_linear():
    g = DatasetPool.load("karate_club_graph")
    gae_embedding = GAEEmbedding(g, d=10, epochs=5, variational=False, linear=True)
    gae_embedding.embed()
    assert gae_embedding._embedding is not None
    for i in range(34):
        print(i, gae_embedding[i])


def test_gae_variational_linear():
    g = DatasetPool.load("karate_club_graph")
    gae_embedding = GAEEmbedding(g, d=10, epochs=5, variational=True, linear=True)
    gae_embedding.embed()
    assert gae_embedding._embedding is not None
    for i in range(34):
        print(i, gae_embedding[i])


if __name__ == "__main__":
    test_gae_variational()
    test_gae_normal()
    test_gae_normal_linear()
    test_gae_variational_linear()
