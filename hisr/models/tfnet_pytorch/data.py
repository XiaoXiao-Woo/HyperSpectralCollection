from os.path import exists, join, basename

from torchvision.transforms import Compose, ToTensor
from transforms import Stretch
from dataset import DatasetFromHdf5

def input_transform():
    return Compose([ToTensor(), Stretch()])

def target_transform():
    return Compose([ToTensor(), Stretch()])

def get_training_set(root_dir):
    # train_dir = join(root_dir, "train_cave(with_up)x4.h5")
    train_dir = join(root_dir, "train_cave(with_up)x4.h5")
    return DatasetFromHdf5(train_dir)

def get_test_set(root_dir):
    # train_dir = join(root_dir, "validation_cave(with_up)x4.h5")
    train_dir = join(root_dir, "validation_cave(with_up)x4.h5")
    return DatasetFromHdf5(train_dir)

def get_test(root_dir):
    train_dir = join(root_dir, "test_cave(with_up)x8_rgb.h5")
    # train_dir = join(root_dir, "test_harvard(with_up)x4_rgb.h5")
    # train_dir = join(root_dir, "test_harvard(with_up)x4_rgb500.h5")
    return DatasetFromHdf5(train_dir)
