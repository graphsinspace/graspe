import os
import pandas as pd
import statistics
import math
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats
import json
import common as cmn

sns.set_theme()

results = cmn.get_result_files("..")
d = 10
algorithm = "N2V"

with open('dataset_properties.json',) as f:
    datasets = json.load(f)

ds_props = ["nodes_cnt", "edges_cnt", "edges_density", "avg_hubness", "max_hubness_norm"]
measures = ["map", "recall", "f1"]
values = {x:[] for x in ds_props+measures}
data = []
for result in results:
    if not result.dataset in datasets or result.d != d or result.algorithm != algorithm:
        continue
    for measure in measures:
        values[measure].append(getattr(result, "avg_{}".format(measure)))
        for ds_prop in ds_props:
            data.append([measure, ds_prop, datasets[result.dataset][ds_prop], getattr(result, "avg_{}".format(measure))])
    for ds_prop in ds_props:
        values[ds_prop].append(datasets[result.dataset][ds_prop])

for ds_prop in ds_props:
    for measure in measures:
        pearson = scipy.stats.pearsonr(values[measure], values[ds_prop])[0]
        spearman = scipy.stats.spearmanr(values[measure], values[ds_prop])[0]
        kendall = scipy.stats.kendalltau(values[measure], values[ds_prop])[0]
        print(
            "{} vs. {}: {}, {}, {}".format(
                ds_prop, measure, pearson, spearman, kendall
            )
        )

f_data = pd.DataFrame(data=data, columns=["measure", "dataset property", "value of dataset property", "value of measure"])
g = sns.FacetGrid(f_data, row="dataset property", col="measure", height=2, aspect=4, sharex=False, sharey=False)
g.map_dataframe(sns.regplot, x="value of dataset property", y="value of measure", truncate=True)
plt.tight_layout()
plt.savefig("../figures/dataset_properties_influence.png")
plt.close()
