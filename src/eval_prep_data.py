import os
import numpy as np
from util_io import load_txt, save_txt

############################################
### Processes and saves the Test Dataset ###
############################################

datadir = '../data/reason/reason'

file2fold = {}
for filename in os.listdir("{}/folds".format(datadir)):
    topic, fold = filename.split("-")
    for i in range(1, 6):
        for line in load_txt("{}/folds/{}".format(datadir, filename)):
            file2fold[line.split()[0]] = fold

labels, arguments = [], []
for dirs, subdirs, files in os.walk(datadir):
    topic = dirs.split("/")[-1]
    # don't look into unwanted folders
    if 'labels' and 'folds' in subdirs:
        subdirs.remove('labels')
        subdirs.remove('folds')
    for file in files:
        # skip readme file
        if file == "readme.txt":
            continue
        else:
            fold = file2fold[file.split(".")[0]]
            text = list(load_txt(os.path.join(dirs, file), encoding="Windows-1252"))
            for idx, sentence in enumerate(text):
                if sentence[:7] == "Label##":
                    # the other class is usually Other but in abortion it's other
                    stance, reason = sentence[7:].lower().split("-")
                    if "other" == reason: continue #exclude OTHER class
                    label = "{}-{}-{}-{}".format(topic, fold, stance, reason)
                    count = 1
                    try:
                        while text[idx+count][:6] == "Line##":
                            labels.append(label)
                            arguments.append(text[idx+count][6:])
                            count += 1
                    except IndexError:
                        continue

# save the data
save_txt("../data/test_data.txt", arguments)
np.save("../data/test_labels.npy", np.asarray(labels))
