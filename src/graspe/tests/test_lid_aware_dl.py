import inspect
import traceback

from common.dataset_pool import DatasetPool
from embeddings.embedding_gae import GAEEmbedding
from embeddings.embedding_gcn import GCNEmbedding
from embeddings.embedding_graphsage import GraphSageEmbedding
from evaluation.lid_eval import (
    LIDMLEEstimator,
    EmbLIDMLEEstimator,
    EmbLIDMLEEstimatorTorch,
)


def test_lidaware():
    datasets = DatasetPool.get_datasets()
    datasets = ["karate_club_graph"]
    print("Testing with", datasets)
    for dataset_name in datasets:
        try:
            print("Dataset name:", dataset_name)
            g = DatasetPool.load(dataset_name)
            print("Number of nodes:", g.nodes_cnt())

            gcns = (
                GCNEmbedding(g, d=20, epochs=100, lid_aware=True, lid_k=20),
                GCNEmbedding(g, d=20, epochs=100, lid_aware=False),
            )

            gaes = (
                GAEEmbedding(
                    g,
                    d=20,
                    layer_configuration=(64, 128),
                    epochs=100,
                    lid_aware=True,
                    lid_k=20,
                ),
                GAEEmbedding(
                    g, d=20, layer_configuration=(64, 128), epochs=100, lid_aware=False
                ),
            )

            graphsages = (
                GraphSageEmbedding(
                    g, d=20, epochs=100, lid_aware=True, lid_k=20, verbose=False
                ),
                GraphSageEmbedding(g, d=20, epochs=100, lid_aware=False, verbose=False),
            )

            for e1, e2 in [gcns, gaes, graphsages]:
                _test_method(g, e1, e2)
        except Exception as exc:
            print(f"Test failed for {dataset_name}!")
            print(str(exc))
            traceback.print_exc()


def _test_method(g, e1, e2):
    e1.embed()
    e2.embed()

    _eval_method(e1, g)
    _eval_method(e2, g)


def _eval_method(e, g):
    print("Method:", e.__class__.__name__)
    print("Method params:", [(i, j) for i, j in e.__dict__.items() if i[:1] != "_"])

    if e.lid_aware:
        print("LID Aware:")
    else:
        print("Not LID Aware:")

    tlid = EmbLIDMLEEstimatorTorch(g, e, 20)
    tlid.estimate_lids()
    print("LID =", tlid.get_total_lid())
    print("Recon loss=", g.link_precision(e.reconstruct(k=20)))

    num_links = g.edges_cnt()
    reconstructed_graph = e.reconstruct(num_links)

    precision_val = g.link_precision(reconstructed_graph)
    map_val, _ = g.map_value(reconstructed_graph)
    recall_val, _ = g.recall(reconstructed_graph)
    f1_val = (2 * precision_val * recall_val) / (precision_val + recall_val)

    print("PRECISION@K =", precision_val)
    print("MAP =", map_val)
    print("RECALL =", recall_val)
    print("F1 =", f1_val)


if __name__ == "__main__":
    test_lidaware()
