import os
import numpy as np

############################################
### Processes and saves the Test Dataset ###
############################################

datadir = '../data/reason/reason'
labels, arguments,idx2lbl,lbl2idx = [], [], {}, {}

for dirs, subdirs, files in os.walk(datadir):
    # don't look into unwanted folders
    if 'labels' and 'folds' in subdirs:
        subdirs.remove('labels')
        subdirs.remove('folds')
    # skip readme file
    for file in files:
        if file == "readme.txt":
            continue
        else:
            text = open(os.path.join(dirs, file),encoding = "Windows-1252").read()
            text = text.split("\n")
            for idx, sentence in enumerate(text):
                if sentence[:7] == "Label##":
                    count = 1
                    try:
                        while text[idx+count][:7] != "Label##":
                            labels.append(text[idx][7:])
                            arguments.append(text[idx+count][6:])
                            count += 1
                    except IndexError:
                        continue

# Creates lookup dictionaries
for idx,lbl in enumerate(set(labels)):
    idx2lbl[idx] = lbl
    lbl2idx[lbl] = idx
num_lbls = [lbl2idx[lbl] for lbl in labels]

data = np.asarray([arguments, num_lbls, idx2lbl, lbl2idx])
np.save('../data/test_data.npy', data)
