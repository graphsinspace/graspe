import os
import glob
from os import listdir
from os.path import isfile, join
from skl_classifiers import RandomForest

mypath = 'n2v_BEST_EMBEDDINGS'
mypath_test = 'n2v_BEST_EMBEDDINGS_broken/*.embedding'


def main():
    with open('RF_N2V.txt', 'a') as file_lid:
        for embedding_path in glob.glob(mypath_test, recursive=True):
            print("Working with", embedding_path)
            if isfile(embedding_path):
                file_lid.write("===\n")
                file_lid.write(embedding_path + '\n')
                try:
                    rf = RandomForest(embedding_path, n_estimators=5, skip_split=True)  # pazi na skip_split

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


if __name__ == "__main__":
    main()
