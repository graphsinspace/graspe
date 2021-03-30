import os
import pandas as pd
import statistics
import math
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme()

def read_file(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
    return float(lines[3].strip().split(": ")[1])

directory = "native_hubness_map_stats"
random_graphs = ["barabasi-albert", "erdos-renyi", "newman-watts-strogatz"]
k_vals = [5, 10, 20]
n_vals = [1000, 10000]

data = []

for n in n_vals:
    for k in k_vals:
        p = k/(n-1)
        ba_graph = "barabasi-albert_n{}_m{}".format(n, k)
        er_graph = "erdos-renyi_n{}_p{}".format(n, p)        
        ba_map = read_file(os.path.join(directory, ba_graph+"_d10_N2V.txt"))
        er_map = read_file(os.path.join(directory, er_graph+"_d10_N2V.txt"))
        data.append(["barabasi-albert", str(n), k, ba_map])
        data.append(["erdos-renyi", str(n), k, er_map])
        
f_data = pd.DataFrame(data=data, columns=["random graph", "size", "k", "map"])
g = sns.FacetGrid(f_data, row="random graph", height=2, aspect=2, sharex=False)
g.map_dataframe(sns.barplot, hue="k", y="map", x="size", palette=sns.color_palette("crest", 3))
plt.tight_layout()
g.add_legend(title="k")
plt.savefig("random_graph_properties_influence.png")
plt.close()