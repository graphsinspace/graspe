from common.dataset_pool import DatasetPool
from embeddings.embedding_gcn import GCNEmbedding


def test_gcn_citeseer():
    g = DatasetPool.load("citeseer")
    e = GCNEmbedding(g, d=10, epochs=1)
    e.embed()
    assert e._embedding is not None
    for i in range(34):
        print(i, e[i])


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
    test_gcn_citeseer()
    test_gcn_all()
