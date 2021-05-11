from os import listdir
from os.path import isfile, join

mypath = 'n2v_BEST_EMBEDDINGS'

embeddings = [f for f in listdir(mypath) if isfile(join(mypath, f))]

import sys

import os

sys.path.append('/home/aleksandar/GRASP/graspe/src/graspe/')


from skl_classifiers import RandomForest

# Dodati u fajl i pokrenuti i za drugi dataset

with open('RF_N2V.txt', 'a') as file_lid:

    for embedding in listdir(mypath)[0:2]:
        if isfile(join(mypath, embedding)):
            file_lid.write("===\n")
            file_lid.write(embedding + '\n')
            try:
                rf = RandomForest(join(mypath, embedding), n_estimators=100)
                rf.fit_predict()
                acc = rf.accuracy()
                prec = rf.precision()
                rec = rf.recall()
                
                file_lid.write("Accuracy: " + str(acc) + os.linesep)
                file_lid.write("Precision: " + str(prec) + os.linesep)
                file_lid.write("Recall: " + str(rec) + os.linesep)
                
            except Exception as e:
                print(str(e))
                file_lid.write("ERROR\n")
            file_lid.write("===\n")

print("THE END")
