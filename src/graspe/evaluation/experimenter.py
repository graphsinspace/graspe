import statistics
from common.dataset_pool import DatasetPool
from embeddings.emb_factory import LazyEmbFactory, FileEmbFactory


class Experimenter:
    def __init__(self, graphs, dim, preset, algs=None, embedding_cache=""):
        self._factories = {}
        for graph in graphs:
            if embedding_cache:
                factory = FileEmbFactory(
                    graph, embedding_cache, dim, True, preset, algs
                )
            else:
                factory = LazyEmbFactory(
                    DatasetPool.load(graph), dim, True, preset=preset, algs=algs
                )
            self._factories[graph] = factory

    def run(self, out=None):
        results = {}
        for graph in self._factories:
            g = DatasetPool.load(graph)
            g_undirected = g.to_undirected()
            factory = self._factories[graph]
            results[graph] = {}
            for i in range(factory.num_methods()):
                e = factory.get_embedding(i)
                e_name = factory.get_name(i)
                recg = e.reconstruct(g_undirected.edges_cnt())
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
                results[graph][e_name] = {
                    "map": avg_map,
                    "recall": avg_recall,
                    "f1": avg_f1,
                }
        if out:
            with open(out, "w") as fp:
                import json

                json.dump(results, fp)
        return results
