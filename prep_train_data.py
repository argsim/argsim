import os
import numpy as np


datadir = "/home/jan/Documents/uni/Argumentation_Mining/argsim/data/train_data/IBM_Debater_(R)_CE-EMNLP-2015.v3"


sentences = []
for subdir, dirs, files in os.walk(datadir):

    for file in files:
        if file == "evidence.txt":
            text = open(os.path.join(subdir, file)).read()
            text = text.split("\n")
            for sentence in text:
                try:
                    sentences.append(sentence.split("\t")[2])
                except IndexError:
                    continue
        if file == "claims.txt":
            text = open(os.path.join(subdir, file)).read()
            text = text.split("\n")
            for sentence in text[1:]:
                break
                try:
                    sentences.append(sentence.split("\t")[1])
                except IndexError:
                    continue

np.save("/home/jan/Documents/uni/Argumentation_Mining/argsim/data/train_data/IBM_Debater_(R)_CE-EMNLP-2015.v3/train_data",np.asarray(sentences))
