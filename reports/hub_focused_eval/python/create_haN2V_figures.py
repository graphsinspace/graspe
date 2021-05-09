import os
import pandas as pd
import statistics
import math
import seaborn as sns
import matplotlib.pyplot as plt
import json

sns.set_theme()

data = []
algorithms = set()
with open("../ha_n2v_experiments_pq0-4.json") as json_file:
    json_data = json.load(json_file)
    for dataset in json_data:
        for algorithm in json_data[dataset]:
            algorithms.add(algorithm)
            for measure in json_data[dataset][algorithm]:
                data.append(
                    [
                        dataset,
                        algorithm,
                        measure,
                        json_data[dataset][algorithm][measure],
                    ]
                )

f_data = pd.DataFrame(data=data, columns=["dataset", "algorithm", "measure", "value"])
g = sns.FacetGrid(f_data, row="dataset", height=1.5, aspect=4, sharey=False)
g.map_dataframe(
    sns.barplot,
    hue="algorithm",
    y="value",
    x="measure",
    palette=sns.color_palette("hls", len(algorithms)),
)
plt.xticks(
    fontweight="light",
    horizontalalignment="right",
)
plt.tight_layout()
g.add_legend()
plt.savefig("../figures/ha_n2v.png")
plt.close()
