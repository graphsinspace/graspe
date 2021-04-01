import os
import pandas as pd
import statistics
import math
import seaborn as sns
import matplotlib.pyplot as plt
import common as cmn

sns.set_theme()

results = cmn.get_result_files("..")
d = 10
algorithm = "N2V"

data_prob = []
data_mwu = []
for measure in ["map", "recall", "f1"]:
    for result in results:
        if result.d != d or result.algorithm != algorithm:
            continue
        data_prob.append(
            [
                result.dataset,
                measure,
                "low-hubness points",
                getattr(result, "avg_{}_low_hubness".format(measure)),
            ]
        )
        data_prob.append(
            [
                result.dataset,
                measure,
                "high-hubness points",
                getattr(result, "avg_{}_high_hubness".format(measure)),
            ]
        )
        data_prob.append(
            [
                result.dataset,
                "probability of having higher {}".format(measure),
                "low-hubness points",
                getattr(result, "s_prob_{}_low_hubness".format(measure)),
            ]
        )
        data_prob.append(
            [
                result.dataset,
                "probability of having higher {}".format(measure),
                "high-hubness points",
                getattr(result, "s_prob_{}_high_hubness".format(measure)),
            ]
        )
        data_mwu.append(
            [
                result.dataset,
                "mwu for {}".format(measure),
                getattr(result, "mwu_{}_low_high_hubness".format(measure)),
            ]
        )

f_data = pd.DataFrame(
    data=data_prob, columns=["dataset", "type", "sub-type", "value"]
)
g = sns.FacetGrid(f_data, row="type", height=2, aspect=4, sharey=False)
g.map_dataframe(
    sns.barplot,
    x="dataset",
    y="value",
    hue="sub-type",
    palette=sns.color_palette("tab20"),
)
plt.xticks(
    fontweight="light",
    rotation=30,
    horizontalalignment="right",
)
plt.tight_layout()
g.add_legend()
plt.savefig("../figures/low-high_hubness-prob.png")
plt.close()

fig, ax = plt.subplots(figsize=[6, 3])
f_data = pd.DataFrame(data=data_mwu, columns=["dataset", "mwu", "p-value"])
g = sns.FacetGrid(f_data, row="mwu", height=2, aspect=4, sharey=False)
g.map_dataframe(
    sns.barplot,
    x="dataset",
    y="p-value",
)
plt.xticks(
    fontweight="light",
    rotation=30,
    horizontalalignment="right",
)
plt.tight_layout()
plt.savefig("../figures/low-high_hubness-mwu.png")
plt.close()
