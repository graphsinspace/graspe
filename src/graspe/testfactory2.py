from embeddings.embfactory import EagerEmbFactory, LazyEmbFactory
from common.dataset_pool import DatasetPool

from timeit import default_timer as timer

datasets = [
    "karate_club_graph"
    #"cora_ml"
]

for d in datasets:
    graph = DatasetPool.load(d)
    gnx = graph.to_networkx()
    nodes = gnx.number_of_nodes()
    links = gnx.number_of_edges()
    print(d, "#nodes = ", nodes, "#links = ", links)
    f = LazyEmbFactory(graph, 20)
    
    print("Reconstruction errors")
    for i in range(f.num_methods()):
        start = timer()
        e = f.get_embedding(i)
        name = f.get_name(i)
        rg = e.reconstruct(links)
        lp = graph.link_precision(rg)
        mp = graph.map(rg)
        end = timer()
        t = end - start
        print("----- ", name, ", lp = ", lp, ", map = ", mp, ", time = ", t, "[s]")

    print("\n")