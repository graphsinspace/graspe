from common.dataset_pool import DatasetPool
from common.graph import Graph
from evaluation.unsupervised_link_prediction import UnsupervisedLinkPrediction

graph = DatasetPool.load("karate_club_graph")
#print(graph.edges())
#edgeset = graph.edges()
h = 20 # broj hidden grana
k = 20 #parametar k, broj predikcija za selektovanje
#hidden = random.sample(edgeset, h)
#print(proba)
#print(graph.nodes())
#nodeset = graph.nodes()
#newgraph = Graph()
#kopiranje cvorova
#for node in nodeset:
#    newgraph.add_node(node[0], node[1])

#kopiranje grana koje nisu hidden
#for edge in edgeset:
#    if edge not in hidden:
#        newgraph.add_edge(edge[0], edge[1])

# embNovi = Node2VecEmbedding(newgraph, 10, 0.1, 0.5)
# embNovi.embed()

#ideja: uzeti distance svih mogucih parova, staviti u mapu grana:vrednost
#sortirati mapu ako moze i uzeti prvih k grana kao predikcije

# dists = []
# for i in range(len(nodeset)):
#     node1 = nodeset[i]
#     for j in range(i + 1, len(nodeset)):
#         node2 = nodeset[j]
#         e = (node1[0], node2[0])
#         if e not in newgraph.edges():
#             dists.append(((node1[0], node2[0]), np.linalg.norm(embNovi._embedding[node1[0]] - embNovi._embedding[node2[0]])))

# #print(dists)
# distssorted = sorted(dists, key=lambda p: p[1])
# #print(distssorted)
# #print()
# #print()
# #print(distssorted[-k:])

# print(hidden)
# print()
# print()

# pred = distssorted[-k:]
# print(pred)
# print()
# print()

# cnt = 0
# for e in pred:
#     if e[0] in hidden:
#         cnt=cnt+1

# #print("cnt=", cnt)
# precisionATk = float(cnt)/float(k)
# #print(precisionATk)

# sum = 0.0

# for node in nodeset:
#     prednode = 0
#     enode = 0
#     #odredjivanje prednode
#     for e in pred:
#         if node[0]==e[0][0] or node[0]==e[0][1]:
#             prednode = prednode + 1

#     #odredjivanje enode
#     for e in hidden:
#         if node[0]==e[0] or node[0]==e[1]:
#             enode = enode + 1

#     if enode!=0:
#         sum = sum + float(prednode)/float(enode)


# map = sum/float(graph.nodes_cnt())
# print(map)

ulp = UnsupervisedLinkPrediction(graph, h, k, "node2vec")
patk = ulp.get_precisionATk()
map = ulp.get_map()

print("precision@k=", patk)
print("map=", map)