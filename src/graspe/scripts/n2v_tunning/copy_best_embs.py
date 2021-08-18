import os
from shutil import copyfile


with open("n2v_bestemb_conf.csv") as f:
    lines = [line.rstrip() for line in f]
    for l in lines:
        toks = l.split(",")
        folder = "genembeddings/" + toks[0] + "_embeddings"
        file = "n2v-" + toks[0] + "-" + toks[1] + "-" + toks[2].replace(".", "_") + "-" + toks[3].replace(".", "_") + ".embedding"
        path = folder + "/" + file
        print(path)
        dst = "genembeddings/n2v_BEST_EMBEDDINGS/" + file
        print(dst)  
        if os.path.exists(path):
            print("Postoji")
            copyfile(path, dst)
            print("Iskopiran...")
        else:
            print("Ne postoji")
