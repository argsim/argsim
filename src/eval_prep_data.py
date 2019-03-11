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

folds, labels, arguments = [], [], []
for topic in 'abortion', 'gayRights', 'marijuana', 'obama':
    dirname = "{}/{}".format(datadir, topic)
    for filename in sorted(os.listdir(dirname)):
        fold = int(file2fold[filename.split(".")[0]]) - 1
        text = list(load_txt(os.path.join(dirname, filename), encoding="Windows-1252"))
        for idx, sentence in enumerate(text):
            if sentence[:7] == "Label##":
                stance, reason = sentence[7:].lower().split("-")
                if "other" == reason: continue #exclude OTHER class
                label = "{}-{}-{}".format(topic, stance, reason)
                count = 1
                try:
                    while text[idx+count][:6] == "Line##":
                        folds.append(fold)
                        labels.append(label)
                        arguments.append(text[idx+count][6:])
                        count += 1
                except IndexError:
                    continue

# save the data
save_txt("../data/test_data.txt", arguments)
np.save("../data/test_labels.npy", np.asarray(labels))
np.save("../data/test_folds.npy", np.asarray(folds))






#######
# NEW #


import os
import numpy as np
from util_io import load_txt, save_txt, clean
import re

datadir = '../data/reason/reason'

output = []
for topic in 'abortion', 'gayRights', 'marijuana', 'obama':
    dirname = "{}/{}".format(datadir, topic)
    for filename in sorted(os.listdir(dirname)):
        text = open(dirname+"/"+filename, encoding="Windows-1252").read().lower()

        #text = open("../data/reason/reason/abortion/A6.data.rsn", encoding="Windows-1252").read().lower()
        post = clean(re.search(r"(.|\n)*(?=(label##|\n\n*))", text).group())
        try:
            label = " ".join(re.findall(r"(?<=label##).*(?=\nline##)", text))
        except AttributeError:
            label = []

        output.append([post, label, topic])

np.savez_compressed("../data/test_data", output=output)
x=np.load("../data/test_data.npz")['output']
