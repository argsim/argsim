import os
import numpy as np

############################################
### Processes and saves the Test Dataset ###
############################################

datadir = '../data/reason/reason'

labels, arguments = [], []

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
                        while text[idx+count][:6] == "Line##":
                            labels.append(text[idx][7:])
                            arguments.append(text[idx+count][6:])
                            count += 1
                    except IndexError:
                        continue

# save the data
data = "\n".join(arguments)
file = open('..argsim/data/test_data.txt', 'w')
file.write(data)
file.close()
np.save("..argsim/data/test_labels.npy", np.asarray(labels))
