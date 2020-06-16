import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter





root = "experiment_save/batchsize512_lr05_100epoch"

np_file_list = []

for path, subdirs, files in os.walk(root):
    for name in files:
        if name.endswith(".npy"):
            np_file_list.append(os.path.join(path, name))



writer = SummaryWriter(log_dir=root)


for np_file in np_file_list:
    tag = np_file.split("/")[-1].split(".")[0].split("-")[1]
    print("ON TAG  " + str(tag))
    arr = np.load(np_file)
    for i, elem in enumerate(arr):
        writer.add_scalar(tag, elem, i)

