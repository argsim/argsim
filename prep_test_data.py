import os
import numpy as np
from collections import Counter
from operator import itemgetter
import heapq

def least_common(array, to_find=None):
    counter = Counter(array)
    if to_find is None:
        return sorted(counter.items(), key=itemgetter(1), reverse=False)
    return heapq.nsmallest(to_find, counter.items(), key=itemgetter(1))


datadir = '/home/jan/Documents/uni/Argumentation_Mining/dataset/data'


labels, arguments = [], []

for subdir, dirs, files in os.walk(datadir):
    for file in files:
        #print(os.path.join(subdir, file))
        text = open(os.path.join(subdir, file),encoding = "Windows-1252").read()
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

counts = Counter(labels)
least_common(counts)
