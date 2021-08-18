
import sys
import pandas as pd

methods = ["UNBIASED", "NCWALK", "RNCWALK", "SHELLWALK", "INVSHELLWALK"]

def summary(df, m):
    dfm = df[df.METHOD == m]
    means = dfm.groupby(['DIM']).mean()
    print("Method", m)
    print(means)
    maxF1 = means["F1"].max()
    print("Maximal F1 = ", maxF1)
    print()
    return maxF1


if __name__ == "__main__":
    file = sys.argv[1]

    df = pd.read_csv(file)
    maxF1 = 0
    maxF1Method = "None"
    for m in methods:
        mf1 = summary(df, m)
        if mf1 > maxF1:
            maxF1 = mf1
            maxF1Method = m

    print("Maximal F1 = ", maxF1, " by", maxF1Method)

            