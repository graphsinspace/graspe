from common.dataset_pool import DatasetPool
from embeddings.embedding_graphsage import GraphSageEmbedding
from evaluation.lid_eval import EmbLIDMLEEstimatorTorch


def test_graphsage():
    g = DatasetPool.load("karate_club_graph")
    e = GraphSageEmbedding(
        g,
        d=100,
        epochs=200,
        lr=0.01,
        layer_configuration=(128, 128),
        act_fn="tanh",
        hub_aware=True,
        hub_fn='identity'
    )
    e.embed()
    assert e._embedding is not None
    # print(e._embedding)


def test_lid_aware_graphsage():
    g = DatasetPool.load("karate_club_graph")
    # LID-aware graphsage:
    e = GraphSageEmbedding(g, d=10, epochs=100, lid_aware=True, lid_k=20)
    e.embed()

    # normal graphsage:
    e2 = GraphSageEmbedding(g, d=10, epochs=100)
    e2.embed()

    tlid = EmbLIDMLEEstimatorTorch(g, e, 20)
    tlid.estimate_lids()
    print("sum (LID-Aware)", tlid.get_total_lid())
    print("recon loss (LID-Aware)", g.link_precision(e.reconstruct(k=20)))

    tlid = EmbLIDMLEEstimatorTorch(g, e2, 20)
    tlid.estimate_lids()
    print("sum (normal)", tlid.get_total_lid())
    print("recon loss (normal)", g.link_precision(e2.reconstruct(k=20)))


def test_graphsage_all():
    datasets = DatasetPool.get_datasets()
    print(datasets)
    for dataset_name in datasets:
        if dataset_name in [
            "davis_southern_women_graph",
            "florentine_families_graph",
            "les_miserables_graph",
        ]:
            continue
        print(dataset_name)
        g = DatasetPool.load(dataset_name)
        e = GraphSageEmbedding(
            g,
            d=100,
            epochs=100,
            lr=0.05,
            layer_configuration=(128, 256, 128),
            act_fn="tanh",
        )
        e.embed()
        assert e._embedding is not None


if __name__ == "__main__":
    # test_lid_aware_graphsage()
    # test_graphsage_all()
    test_graphsage()
