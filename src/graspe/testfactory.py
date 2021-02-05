
from embeddings.embfactory import EagerEmbFactory, LazyEmbFactory
from common.dataset_pool import DatasetPool

dim = 20
graph = DatasetPool.load("cora_ml")

print("Eager factory")
f = EagerEmbFactory(graph, dim)
for i in range(f.num_methods()):
    name = f.get_name(i)
    emb = f.get_embedding(i)
    print("Radim nesto sa", name)
    

print("\nLazy factory")
f = LazyEmbFactory(graph, dim)
for i in range(f.num_methods()):
    name = f.get_name(i)
    emb = f.get_embedding(i)
    if emb == None:
        print(name, " ne radi za ovaj graf")
    else:
        print("Radim nesto sa", name)