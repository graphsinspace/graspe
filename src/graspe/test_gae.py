from common.dataset_pool import DatasetPool
from embeddings.embedding_gae import GAEEmbedding, GAEEmbeddingNCLIDAware
from evaluation.lid_eval import EmbLIDMLEEstimatorTorch

g = DatasetPool.load("karate_club_graph")
g = g.to_undirected()
g.remove_selfloop_edges()


def test_gae_normal():
    gae_embedding = GAEEmbedding(
        g,
        d=10,
        epochs=5,
        variational=False,
        linear=False,
        lr=0.01,
        layer_configuration=(8,),
        act_fn="relu",
    )
    gae_embedding.embed()
    assert gae_embedding._embedding is not None
    for i in range(len(gae_embedding._embedding)):
        print(i, gae_embedding[i])


def test_gae_variational():
    gae_embedding = GAEEmbedding(
        g,
        d=10,
        epochs=5,
        variational=True,
        linear=False,
        lr=0.01,
        layer_configuration=(8,),
        act_fn="relu",
    )
    gae_embedding.embed()
    assert gae_embedding._embedding is not None
    for i in range(len(gae_embedding._embedding)):
        print(i, gae_embedding[i])


def test_gae_normal_linear():
    gae_embedding = GAEEmbeddingNCLIDAware(
        g,
        d=25,
        epochs=1,
        variational=False,
        linear=True,
        lr=0.01,
        layer_configuration=(8,),
        act_fn="relu",
    )
    gae_embedding.embed()
    assert gae_embedding._embedding is not None
    for i in range(len(gae_embedding._embedding)):
        print(i, gae_embedding[i])


def test_gae_variational_linear():
    gae_embedding = GAEEmbedding(
        g,
        d=10,
        epochs=5,
        variational=True,
        linear=True,
        lr=0.01,
        layer_configuration=(8,),
        act_fn="relu",
    )
    gae_embedding.embed()
    assert gae_embedding._embedding is not None
    for i in range(len(gae_embedding._embedding)):
        print(i, gae_embedding[i])


def test_gae_cora():
    gae_embedding = GAEEmbedding(
        g,
        d=10,
        epochs=5,
        variational=False,
        linear=False,
        lr=0.01,
        layer_configuration=(8,),
        act_fn="relu",
    )
    gae_embedding.embed()
    assert gae_embedding._embedding is not None
    for i in range(len(gae_embedding._embedding)):
        print(i, gae_embedding[i])


def test_gae_all():
    datasets = DatasetPool.get_datasets()
    print(datasets)
    for dataset_name in datasets:
        print(dataset_name)
        g = DatasetPool.load(dataset_name)
        gae_embedding = GAEEmbedding(
            g,
            d=10,
            epochs=5,
            variational=False,
            linear=False,
            lr=0.01,
            layer_configuration=(8,),
            act_fn="relu",
        )
        gae_embedding.embed()
        assert gae_embedding._embedding is not None
        for i in range(len(gae_embedding._embedding)):
            print(i, gae_embedding[i])


def test_lid_aware_gae():
    g = DatasetPool.load("karate_club_graph")

    # LID-aware GAE:
    e = GAEEmbedding(
        g, d=10, epochs=100, variational=False, linear=False, lid_aware=True, lid_k=20
    )
    e.embed()

    # normal GAE:
    e2 = GAEEmbedding(g, d=10, epochs=100, variational=False, linear=False)
    e2.embed()

    tlid = EmbLIDMLEEstimatorTorch(g, e, 20)
    tlid.estimate_lids()
    print("LID sum (LID-Aware)", tlid.get_total_lid())
    print("recon loss (LID-Aware)", g.link_precision(e.reconstruct(k=20)))

    tlid = EmbLIDMLEEstimatorTorch(g, e2, 20)
    tlid.estimate_lids()
    print("LID sum (normal)", tlid.get_total_lid())
    print("recon loss (normal)", g.link_precision(e2.reconstruct(k=20)))


if __name__ == "__main__":
    # test_lid_aware_gae()
    # test_gae_variational()
    # test_gae_normal()
    test_gae_normal_linear()
    # test_gae_variational_linear()
    # test_gae_cora()
    # test_gae_all() # takes long time
