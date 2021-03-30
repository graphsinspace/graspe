import os
import pandas as pd
import statistics
import math
import seaborn as sns
import matplotlib.pyplot as plt
import common as cmn

results = cmn.get_result_files("..")
d = 10

organized_results = {}
for result in results:
    if result.d != d:
        continue 
    for correl in result.correls:
        if not correl in organized_results:
            organized_results[correl] = {}
        correl_dict = organized_results[correl]
        for stat in cmn.Result.stats:
            if not stat in correl_dict:
                correl_dict[stat] = {}
            stat_dict = correl_dict[stat]
            if not result.dataset in stat_dict:
                stat_dict[result.dataset] = []
            stat_dict[result.dataset].append(result.correls[correl][stat])

overall_chart_data = []
for correl in organized_results:
    for stat in organized_results[correl]:
        stds = []
        for dataset in organized_results[correl][stat]:
            stds.append(statistics.stdev(organized_results[correl][stat][dataset]))
        overall_chart_data.append([correl.replace("Correlation between", ""), stat, statistics.mean(stds)])

fig, ax = plt.subplots(figsize=[12, 6])
f_data = pd.DataFrame(data=overall_chart_data, columns=["correlation variables", "measures", "std"])
f = sns.barplot(ax=ax, x="correlation variables", y="std", hue="measures", data=f_data)
f.set_xlabel("")
plt.xticks(
    fontsize="small", 
    fontweight='light',
    rotation=30,
    horizontalalignment='right',
)
plt.tight_layout()
plt.savefig("../figures/n2v_variants_std.png")
plt.close()