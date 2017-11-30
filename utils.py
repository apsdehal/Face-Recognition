from __future__ import print_function

import os
import sys
import shutil
import random


def move_data(folder, data_folder):

    if os.path.isdir(os.path.join(folder, "train")):
        print("Train folder already exists, Exiting")
        return
    shutil.move(data_folder, os.path.join(folder, "train"))

    val_path = os.path.join(folder, "val")
    train_path = os.path.join(folder, "train")
    if not os.path.isdir(val_path):
        os.makedirs(val_path)

    for dir in os.listdir(train_path):
        if os.path.isdir(os.path.join(train_path, dir)):
            os.makedirs(os.path.join(val_path, dir))
            k = random.choice(os.listdir(os.path.join(train_path, dir)))
            os.rename(os.path.join(train_path, dir, k),
                      os.path.join(val_path, dir, k))


if __name__ == '__main__':
    move_data(sys.argv[1], sys.argv[2])
