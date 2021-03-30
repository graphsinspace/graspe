import os
import re


class Result:

    stats = [
        "pearson",
        "spearman",
        "kendall",
    ]

    properties = [
        "avg_hubness",
        "low_hubness_count",
        "high_hubness_count",
        "avg_map",
        "avg_map_low_hubness",
        "avg_map_high_hubness",
        "mwu_map_low_high_hubness",
        "s_prob_map_low_hubness",
        "s_prob_map_high_hubness",
        "avg_recall",
        "avg_recall_low_hubness",
        "avg_recall_high_hubness",
        "mwu_recall_low_high_hubness",
        "s_prob_recall_low_hubness",
        "s_prob_recall_high_hubness",
        "avg_f1_low_hubness",
        "avg_f1_high_hubness",
        "mwu_f1_low_high_hubness",
        "s_prob_f1_low_hubness",
        "s_prob_f1_high_hubness",
        "avg_knng_map",
        "avg_knng_recall",
    ]

    def __init__(self, path):
        self.path = path
        filename = os.path.basename(path).split(".")[0]
        splt = re.split("_d\d+_", filename)
        self.d = int(re.search("_d\d+_", filename).group(0)[2:-1])
        self.dataset = splt[0]
        self.algorithm = splt[1]
        self.parse()

    def parse(self):
        self.titles = {}
        self.correls = {}
        with open(self.path) as fp:
            lines = [[y.strip() for y in x.split(":")] for x in fp.readlines()]
            for i in range(len(Result.properties)):
                property_name = Result.properties[i]
                property_title = Result.properties[i]
                property_value = float(lines[i][1])
                self.titles[property_name] = property_title
                setattr(self, property_name, property_value)
            for i in range(len(Result.properties), len(lines)):
                correl_name = lines[i][0].replace(" (pearson, spearman, kendall)", "")
                correl_vals = [float(x) for x in lines[i][1].split(", ")]
                self.correls[correl_name] = {
                    "pearson": correl_vals[0],
                    "spearman": correl_vals[1],
                    "kendall": correl_vals[2],
                }


def get_result_files(path):
    files = []
    for filename in os.listdir(path):
        if filename.endswith(".txt"):
            files.append(Result(os.path.join(path, filename)))
    return files


# Correlation between native hubness and reconstructed hubness (pearson, spearman, kendall): -0.13791170390388646, -0.6172889163579542, -0.4477484640559191
# Correlation between native hubness and knng hubness (k=5) (pearson, spearman, kendall): 0.04445359703441836, -0.08685370827756446, -0.062321189033560466
# Correlation between native hubness and knng hubness (k=auto) (pearson, spearman, kendall): 0.028345035804948925, -0.16179984011621365, -0.11179528257807621
# Correlation between reconstructed hubness and knng hubness (k=5) (pearson, spearman, kendall): 0.21259754760263808, 0.36349851592792637, 0.2691602292018133
# Correlation between reconstructed hubness and knng hubness (k=auto) (pearson, spearman, kendall): 0.32720614475263066, 0.4996049674675015, 0.36283246494386495
# Correlation between knng hubness (k=5) and knng hubness (k=auto) (pearson, spearman, kendall): 0.7262576286599234, 0.7194978779143578, 0.5568243406661563
# Correlation between native hubness and map (pearson, spearman, kendall): 0.239384941216979, 0.19570175231823805, 0.16562226844528147
# Correlation between reconstructed hubness and map (pearson, spearman, kendall): -0.24386941409552734, 0.2013735151010858, 0.07502440782918739
# Correlation between knng hubness (k=5) and map (pearson, spearman, kendall): 0.11350045933795759, 0.259940590533355, 0.19651189332994995
# Correlation between knng hubness (k=auto) and map (pearson, spearman, kendall): 0.01721518037186282, 0.23050083513710962, 0.1652068342460226
# Correlation between native hubness and recall (pearson, spearman, kendall): -0.09540118669839097, -0.05623467824453358, -0.04755652309244151
# Correlation between reconstructed hubness and recall (pearson, spearman, kendall): 0.0638622482591899, 0.4187887007776697, 0.3335870189782735
# Correlation between knng hubness (k=5) and recall (pearson, spearman, kendall): 0.13675181621979884, 0.28414975399381254, 0.21668242749721076
# Correlation between knng hubness (k=auto) and recall (pearson, spearman, kendall): 0.135052282795157, 0.30475048438072816, 0.22610107375611654
# Correlation between native hubness and f1 (pearson, spearman, kendall): -0.05749681441609761, 0.05936696420520983, 0.04792412256179353
# Correlation between reconstructed hubness and f1 (pearson, spearman, kendall): -0.12580133298708682, 0.2993751433441922, 0.20331142738044122
# Correlation between knng hubness (k=5) and f1 (pearson, spearman, kendall): 0.10102728857918133, 0.2678698464488544, 0.20205493466385926
# Correlation between knng hubness (k=auto) and f1 (pearson, spearman, kendall): 0.028220595107114676, 0.25358325324591235, 0.18335050366312225
