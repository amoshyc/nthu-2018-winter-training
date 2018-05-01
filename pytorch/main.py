import pathlib
from datetime import datetime

import torch
from dataset import MNISTtrain, MNISTvalid, MNISTvis
from model import MNISTClassifier

ckpt_dir = pathlib.Path(f'./ckpt/{datetime.now():%m-%d %H:%M:%S}/')
ckpt_dir.mkdir(parents=True)

device = torch.device('cpu')
clf = MNISTClassifier(ckpt_dir, device)
clf.fit(MNISTtrain, MNISTvalid, MNISTvis)
