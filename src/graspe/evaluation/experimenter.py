import json
import os
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
        if out:
            if os.path.exists(out):
                with open(out) as json_file:
                    old_results = json.load(json_file)
                    for graph in old_results:
                        if not graph in results:
                            results[graph] = {}
                        for emb in old_results[graph]:
                            results[graph][emb] = old_results[graph][emb]
        for graph in self._factories:
            g = DatasetPool.load(graph)
            g_undirected = g.to_undirected()
            g_undirected.remove_selfloop_edges()
            factory = self._factories[graph]
            if not graph in results:
                results[graph] = {}
            for i in range(factory.num_methods()):
                e_name = factory.get_name(i)
                if e_name in results[graph]:
                    results[graph][e_name] = old_results[graph][e_name]
                    print(
                        "Results for graph {} and embedding {} already exist.".format(
                            graph, e_name
                        )
                    )
                    continue
                e = factory.get_embedding(i)
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
                        json.dump(results, fp, indent=4)
        return results
