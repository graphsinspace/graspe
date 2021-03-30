import os
import pandas as pd
import statistics
import math
import seaborn as sns
import matplotlib.pyplot as plt
import common as cmn

sns.set_theme()

algorithm = "N2V"
d = 10
results = filter(lambda r: r.algorithm == algorithm and r.d == d, cmn.get_result_files(".."))

datasets = set()
overall_chart_data = []
for result in results:
    datasets.add(result.dataset)
    for correl in result.correls:
        for stat in result.correls[correl]:
            overall_chart_data.append([correl.replace("Correlation between", ""), result.dataset, stat, result.correls[correl][stat]])

f_data = pd.DataFrame(data=overall_chart_data, columns=["correl", "dataset", "measures", "correlation"])
g = sns.FacetGrid(f_data, row="correl", height=1.5, aspect=4)
g.map_dataframe(sns.barplot, x="dataset", y="correlation", hue="measures", palette=sns.color_palette("tab10"))
plt.xticks( 
    fontweight='light',
    rotation=90,
    horizontalalignment='right',
)
plt.tight_layout()
g.add_legend()
plt.savefig("../figures/correlations.png")
plt.close()