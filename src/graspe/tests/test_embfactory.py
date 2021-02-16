from embeddings.embfactory import LazyEmbFactory, EagerEmbFactory, FileEmbFactory
from common.dataset_pool import DatasetPool
import os

def test_lazy(g, dim):
    print("---- Test Lazy Factory ----")
    test_factory(LazyEmbFactory(g, dim, preset="N2V"))

def test_eager(g, dim):
    print("---- Test Eager Factory ----")
    test_factory(EagerEmbFactory(g, dim, preset="N2V"))

def test_file(g, dim, directory):
    print("---- Test File Factory ----")
    test_factory(FileEmbFactory(g, directory, dim, preset="N2V"))

def test_factory(fact):
    print(".. Iterate over embeddings")
    for i in range(fact.num_methods()):
        e = fact.get_embedding(i)

if __name__ == "__main__":
    g_name = "karate_club_graph"
    g = DatasetPool.load(g_name)
    directory = os.path.join("..", "..", "data", "embeddings")
    dim = 2
    test_lazy(g, dim)
    test_eager(g, dim)
    test_file(g_name, dim, directory)