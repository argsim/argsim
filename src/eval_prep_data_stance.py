import os
import numpy as np
from util_io import load_txt, save_txt, clean

topics = tuple("abortion gayRights marijuana obama".split())

top2folds = {top: tuple(
    set(load_txt("../data/reason/stance/folds/{}_folds/Fold-{}".format(top, fold)))
    for fold in range(1, 6))
             for top in topics}

dataset = []
for top, folds in top2folds.items():
    path = "../data/reason/stance/{}".format(top)
    for fold, names in enumerate(folds):
        for name in names:
            data = list(load_txt("{}/{}.data".format(path, name)))
            assert len(data) == 1
            data = data[0]
            meta = dict(line.split("=") for line in load_txt("{}/{}.meta".format(path, name)))
            stn = meta['Stance']
            try:
                stn = int(meta['Stance'])
            except ValueError:
                print(top, name)
                continue
            assert stn == -1 or stn == 1
            dataset.append((data, fold, top, stn))

text, fold, top, stn = zip(*dataset)
text = np.array(list(map(clean, text)))
fold = np.array(fold)
top = np.array(top)
stn = np.array([int(s > 0) for s in stn])

np.savez_compressed("../data/stance.npz", text= text, fold= fold, top= top, stn= stn)
