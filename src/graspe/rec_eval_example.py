from common.dataset_pool import DatasetPool
from embeddings.embedding_node2vec import Node2VecEmbedding_Native

# graph = DatasetPool.load("cora_ml")                    # first case
graph = DatasetPool.load("cora_ml").to_undirected()      # second case
emb_m = Node2VecEmbedding_Native(graph, 10, 1, 1)
emb_m.embed()

undirected_projection = graph.to_undirected()
print("#nodes = ", undirected_projection.nodes_cnt())
print("#edges = ", undirected_projection.edges_cnt())

# ovde uzimamo pola od ukupnog broja grana: 
# undirected_projection je neusmeren, ali je interno reprezentovan usmerenim
# grafom sa reciprocnim linkovima (A -> B i B -> A za svaki link A -- B)
num_links = undirected_projection.edges_cnt() // 2

reconstructed_graph = emb_m.reconstruct(num_links)
print("#nodes = ", reconstructed_graph.nodes_cnt())
print("#edges = ", reconstructed_graph.edges_cnt())
map_val = undirected_projection.map(reconstructed_graph)
precision_val = undirected_projection.link_precision(reconstructed_graph)
recall_val, _ = undirected_projection.recall(reconstructed_graph)

print("PRECISION@K = ", precision_val, "MAP = ", map_val, ", RECALL = ", recall_val)





