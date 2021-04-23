from common.dataset_pool import DatasetPool
from embeddings.embedding_gae import GAEEmbedding

g = DatasetPool.load("karate_club_graph")


def test_gae_normal():
    gae_embedding = GAEEmbedding(g, d=10, epochs=5, variational=False, linear=False)
    gae_embedding.embed()
    assert gae_embedding._embedding is not None
    for i in range(34):
        print(i, gae_embedding[i])


def test_gae_variational():
    gae_embedding = GAEEmbedding(g, d=10, epochs=5, variational=True, linear=False)
    gae_embedding.embed()
    assert gae_embedding._embedding is not None
    for i in range(34):
        print(i, gae_embedding[i])


def test_gae_normal_linear():
    gae_embedding = GAEEmbedding(g, d=10, epochs=5, variational=False, linear=True)
    gae_embedding.embed()
    assert gae_embedding._embedding is not None
    for i in range(34):
        print(i, gae_embedding[i])


def test_gae_variational_linear():
    gae_embedding = GAEEmbedding(g, d=10, epochs=5, variational=True, linear=True)
    gae_embedding.embed()
    assert gae_embedding._embedding is not None
    for i in range(34):
        print(i, gae_embedding[i])


def test_gae_cora():
    gae_embedding = GAEEmbedding(g, d=10, epochs=5, variational=False, linear=False)
    gae_embedding.embed()
    assert gae_embedding._embedding is not None
    for i in range(34):
        print(i, gae_embedding[i])


def test_gae_all():
    datasets = DatasetPool.get_datasets()
    print(datasets)
    for dataset_name in datasets:
        print(dataset_name)
        g = DatasetPool.load(dataset_name)
        gae_embedding = GAEEmbedding(g, d=1, epochs=1, variational=False, linear=False)
        gae_embedding.embed()
        assert gae_embedding._embedding is not None
        for i in range(34):
            print(i, gae_embedding[i])


if __name__ == "__main__":
    test_gae_variational()
    test_gae_normal()
    test_gae_normal_linear()
    test_gae_variational_linear()
    test_gae_cora()
    # test_gae_all() # takes long time
