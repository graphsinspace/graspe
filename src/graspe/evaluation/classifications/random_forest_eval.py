from os import listdir
from os.path import isfile, join

mypath = 'n2v_BEST_EMBEDDINGS'

embeddings = [f for f in listdir(mypath) if isfile(join(mypath, f))]

import sys

# print(sys.path)

# sys.path.append('/home/aleksandar/GRASP/graspe/src/graspe/evaluation/classifications/base/classifier')
sys.path.append('/home/aleksandar/GRASP/graspe/src/graspe/')

# print(onlyfiles)
# print("\n")

# fo = open(onlyfiles[0],"wt")
# print(fo.name)

from skl_classifiers import RandomForest

# Dodati u fajl i pokrenuti i za drugi dataset

for embedding in listdir(mypath):
    if isfile(join(mypath, embedding)):
        print("Dosao")
        print(embedding)
        try:
            rf = RandomForest(join(mypath, embedding), n_estimators=100)
            print(rf.classify())
        except:
            print("ERROR")
        print("Nisam")
