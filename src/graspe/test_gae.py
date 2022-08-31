import statistics
from common.dataset_pool import DatasetPool
from embeddings.embedding_gae import GAEEmbedding, GAEEmbeddingNCLIDAware
from evaluation.lid_eval import EmbLIDMLEEstimatorTorch

g = DatasetPool.load("karate_club_graph")
g = g.to_undirected()
g.remove_selfloop_edges()


def test_gae():
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

    g_undirected = g.to_undirected()
    g_undirected.remove_selfloop_edges()
    recg = gae_embedding.reconstruct(g_undirected.edges_cnt())
    avg_map, maps = g_undirected.map_value(recg)
    avg_recall, recalls = g_undirected.recall(recg)
    f1 = [
        (
            (2 * maps[node] * recalls[node]) / (maps[node] + recalls[node])
            if maps[node] + recalls[node] != 0
            else 0
        )
        for node in maps
    ]
    avg_f1 = statistics.mean(f1)
    results = {
        "map": avg_map,
        "recall": avg_recall,
        "f1": avg_f1,
    }

    print(results)
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


if __name__ == "__main__":
    # test_lid_aware_gae()
    # test_gae_variational()
    # test_gae_normal()
    test_gae()
    # test_gae_variational_linear()
    # test_gae_cora()
    # test_gae_all() # takes long time
