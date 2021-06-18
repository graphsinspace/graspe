from common.dataset_pool import DatasetPool
from embeddings.embedding_gcn import GCNEmbedding
from evaluation.lid_eval import (
    LIDMLEEstimator,
    EmbLIDMLEEstimator,
    EmbLIDMLEEstimatorTorch,
)


def test_gcn_citeseer():
    g = DatasetPool.load("amazon_electronics_computers")
    e = GCNEmbedding(
        g, d=10, epochs=1, lr=0.05, layer_configuration=(128, 256, 128), act_fn="tanh"
    )
    e.embed()
    assert e._embedding is not None
    for i in range(34):
        print(i, e[i])


def test_gcn_lid():
    g = DatasetPool.load("karate_club_graph")
    hubness = g.get_hubness()
    print(hubness)
    print(len(g.nodes()))
    e = GCNEmbedding(
        g, d=100, epochs=1, lr=0.05, layer_configuration=(128, 256, 128), act_fn="tanh"
    )
    e.embed()
    assert e._embedding is not None

    lid = EmbLIDMLEEstimator(g, e, 20)
    lid.estimate_lids()
    print("lid values", lid.lid_values)
    print("avg", lid.get_avg_lid())
    print("max", lid.get_max_lid())
    print("min", lid.get_min_lid())
    print("stdev", lid.get_stdev_lid())
    print("sum", sum(lid.get_lid_values().values()))

    tlid = EmbLIDMLEEstimatorTorch(g, e, 20)
    tlid.estimate_lids()
    print("lid values", tlid.lid_values)
    print("avg", tlid.get_avg_lid())
    print("max", tlid.get_max_lid())
    print("min", tlid.get_min_lid())
    print("stdev", tlid.get_stdev_lid())
    print("sum", tlid.get_total_lid())

    assert tlid.get_avg_lid() - lid.get_avg_lid() <= 1e-5, (
        tlid.get_avg_lid() - lid.get_avg_lid()
    )
    assert tlid.get_max_lid() - lid.get_max_lid() <= 1e-5, (
        tlid.get_max_lid() - lid.get_max_lid()
    )
    assert tlid.get_min_lid() - lid.get_min_lid() <= 1e-5, (
        tlid.get_min_lid() - lid.get_min_lid()
    )
    assert tlid.get_stdev_lid() - lid.get_stdev_lid() <= 1e-5, (
        tlid.get_stdev_lid() - lid.get_stdev_lid()
    )
    assert tlid.get_total_lid() - sum(lid.get_lid_values().values()) <= 1e-5, (
        tlid.get_stdev_lid() - lid.get_stdev_lid()
    )


def test_lid_aware_gcn():
    g = DatasetPool.load("karate_club_graph")
    # LID-aware GCN:
    e = GCNEmbedding(
        g,
        d=100,
        epochs=1,
        lr=0.05,
        layer_configuration=(128, 256, 128),
        act_fn="tanh",
        lid_aware=True,
        lid_k=20,
    )
    e.embed()

    # normal GCN:
    e2 = GCNEmbedding(
        g, d=100, epochs=1, lr=0.05, layer_configuration=(128, 256, 128), act_fn="tanh"
    )
    e2.embed()

    tlid = EmbLIDMLEEstimatorTorch(g, e, 20)
    tlid.estimate_lids()
    print("sum (LID-Aware)", tlid.get_total_lid())
    print("recon loss (LID-Aware)", g.link_precision(e.reconstruct(k=20)))

    tlid = EmbLIDMLEEstimatorTorch(g, e2, 20)
    tlid.estimate_lids()
    print("sum (normal)", tlid.get_total_lid())
    print("recon loss (normal)", g.link_precision(e2.reconstruct(k=20)))


def test_gcn_all():
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
        e = GCNEmbedding(g, d=1, epochs=1)
        e.embed()
        assert e._embedding is not None


if __name__ == "__main__":
    # test_lid_aware_gcn()
    test_gcn_lid()
    test_gcn_citeseer()
    # test_gcn_all()
