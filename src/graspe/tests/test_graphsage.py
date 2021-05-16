from common.dataset_pool import DatasetPool
from embeddings.embedding_graphsage import GraphSageEmbedding


def test_graphsage():
    g = DatasetPool.load("cora_ml")
    e = GraphSageEmbedding(g, d=100, epochs=100, lr=0.05, layer_configuration=(128, 256, 128), act_fn="tanh")
    e.embed()
    assert e._embedding is not None
    # print(e._embedding)


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
        e = GraphSageEmbedding(g, d=100, epochs=100, lr=0.05, layer_configuration=(128, 256, 128), act_fn="tanh")
        e.embed()
        assert e._embedding is not None


if __name__ == "__main__":
    # test_graphsage_all()
    test_graphsage()
