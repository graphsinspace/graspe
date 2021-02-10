from common.dataset_pool import DatasetPool
from embeddings.embedding_gcn import GCNEmbedding
from embeddings.embedding_graphsage import GraphSageEmbedding


def test_graphsage():
    g = DatasetPool.load("cora_ml")
    e = GraphSageEmbedding(g, d=10, epochs=200)
    e.embed()
    assert e._embedding is not None
    # print(e._embedding)


def test_graphsage_all():
    datasets = DatasetPool.get_datasets()
    needs_self_loop = ["amazon_electronics_computers", "amazon_electronics_photo"]
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
        e = GraphSageEmbedding(g, d=10, epochs=1)
        e.embed()
        assert e._embedding is not None


if __name__ == "__main__":
    test_graphsage_all()
    # test_graphsage()
