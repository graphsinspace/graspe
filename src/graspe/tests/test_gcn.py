from common.dataset_pool import DatasetPool
from embeddings.embedding_gcn import GCNEmbedding
from evaluation.lid_eval import LIDMLEEstimator, EmbLIDMLEEstimator


def test_gcn_citeseer():
    g = DatasetPool.load("citeseer")
    e = GCNEmbedding(
        g, d=100, epochs=1, lr=0.05, layer_configuration=(128, 256, 128), act_fn="tanh"
    )
    e.embed()
    assert e._embedding is not None
    for i in range(34):
        print(i, e[i])


def test_gcn_lid():
    g = DatasetPool.load("karate_club_graph")
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
    test_gcn_lid()
    # test_gcn_citeseer()
    # test_gcn_all()
