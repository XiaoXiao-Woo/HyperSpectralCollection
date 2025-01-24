# import os
# import warnings

# import cv2
# import imageio
# import numpy as np
import torch

# from numpy.random import RandomState
# import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

# import math
# import torch.nn as nn
from hisr.common.dataUPHSI import DatasetFromHdf5, DummyDataset

# from hisr.common.chsp_dataset import SpectralPansharpeningDataset, LIIFDataset, SSIPDataset, GaussianBlur, \
# RandomBicubicSampling
from udl_vis.Basis.distributed import DistributedSampler, RandomSampler


# import h5py
# cv2.setNumThreads(1)


class HISRSession:
    def __init__(self, args):
        self.dataloaders = {}
        self.samples_per_gpu = args.samples_per_gpu
        self.workers_per_gpu = args.workers_per_gpu
        # self.patch_size = args.patch_size
        self.writers = {}
        self.args = args

    def get_train_dataloader(self, dataset_name, distributed, state_dataloader):
        dataset_type = self.args.dataset_type
        generator = torch.Generator()
        generator.manual_seed(self.args.seed)
        if state_dataloader is not None:
            generator.set_state(state_dataloader.cpu())
        # if hasattr(self.args, 'inr_transforms'):
        #     if dataset_name in ["PaviaCentre"]:
        #         inr_transforms = [
        #             GaussianBlur(dataset_name=dataset_name, scale=self.args.scale, patch_size=self.args.patch_size),
        #             RandomBicubicSampling(scale_min=2.0, scale_max=4.0)]
        #         self.args.inr_transforms = inr_transforms
        #         dataset = SSIPDataset(self.args, mode="train",
        #                               file_path='/'.join([self.args.data_dir, self.args.train_path]))

        #     else:
        #         raise NotImplementedError(f"{dataset_name} is not supported")
        if dataset_type == "Hdf5":
            if dataset_name in [
                "cave_x4",
                "harvard_x4",
                "cave_x8",
                "harvard_x8",
                "GF5_GF1",
            ]:
                dataset = DatasetFromHdf5(
                    getattr(self.args.dataset, f"{dataset_name}_train_path")
                )
        elif dataset_type == "Dummy":
            dataset = DummyDataset()
        else:
            print(f"{dataset_name} is not supported.")
            raise NotImplementedError(f"{dataset_name} is not supported.")
        # dataset = TestDataset(dataset_name)

        sampler = None
        if distributed:
            sampler = DistributedSampler(dataset, generator=generator)
        else:
            sampler = RandomSampler(dataset, generator=generator)

        dataloader = DataLoader(
            dataset,
            batch_size=self.samples_per_gpu,
            persistent_workers=(True if self.workers_per_gpu > 0 else False),
            shuffle=(sampler is None),
            num_workers=self.workers_per_gpu,
            drop_last=False,
            sampler=sampler,
        )
        return dataloader, sampler, generator

    def get_valid_dataloader(self, dataset_name, distributed):
        # if hasattr(self.args, 'SSIPDataset'):
        #     if dataset_name in ["PaviaCentre"]:
        #         inr_transforms = [
        #             GaussianBlur(dataset_name=dataset_name, scale=self.args.scale, patch_size=self.args.patch_size),
        #             RandomBicubicSampling(scale_min=2.0, scale_max=4.0)]
        #         self.args.inr_transforms = inr_transforms
        #         dataset = SSIPDataset(self.args, mode="train",
        #                               file_path='/'.join([self.args.data_dir, self.args.train_path]))

        #     else:
        #         raise NotImplementedError(f"{dataset_name} is not supported")
        if dataset_name in ["cave_x4", "harvard_x4", "GF5-GF1"]:
            dataset = DatasetFromHdf5(
                getattr(self.args.dataset, f"{dataset_name}_val_path")
            )
        else:
            print(f"{dataset_name} is not supported.")
            raise NotImplementedError
        # dataset = TestDataset(dataset_name)
        sampler = None
        if distributed:
            sampler = DistributedSampler(dataset, shuffle=False)
        else:
            sampler = RandomSampler(dataset)

        dataloader = DataLoader(
            dataset,
            batch_size=self.samples_per_gpu,
            persistent_workers=(True if self.workers_per_gpu > 0 else False),
            shuffle=(sampler is None),
            num_workers=self.workers_per_gpu,
            drop_last=False,
            sampler=sampler,
        )

        return dataloader, sampler

    def get_test_dataloader(self, dataset_name, distributed):
        # if hasattr(self.args, 'SSIPDataset'):
        #     if dataset_name in ["PaviaCentre"]:
        #         inr_transforms = [
        #             GaussianBlur(dataset_name=dataset_name, scale=self.args.scale, patch_size=self.args.patch_size),
        #             RandomBicubicSampling(scale_min=2.0, scale_max=4.0)]
        #         self.args.inr_transforms = inr_transforms
        #         dataset = SSIPDataset(self.args, mode="train",
        #                               file_path='/'.join([self.args.data_dir, self.args.train_path]))

        #     else:
        #         raise NotImplementedError(f"{dataset_name} is not supported")
        if dataset_name in [
            "cave_x4",
            "harvard_x4",
            "cave_x8",
            "harvard_x8",
            "GF5-GF1",
        ]:
            dataset = DatasetFromHdf5(
                getattr(self.args.dataset, f"{dataset_name}_test_path")
            )
        else:
            print(f"{dataset_name} is not supported.")
            raise NotImplementedError

        sampler = None
        if distributed:
            sampler = DistributedSampler(dataset, shuffle=False)
        else:
            sampler = RandomSampler(dataset)
        # dataset = TrainValDataset(dataset_name)
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            drop_last=False,
            sampler=sampler,
        )
        return dataloader, sampler


if __name__ == "__main__":
    # from option import args
    import argparse

    parser = argparse.ArgumentParser()

    args = parser.parse_args()
    args.samples_per_gpu = 8
    args.workers_per_gpu = 0
    args.data_dir = "C:/Datasets/hisr"
    args.dataset = "cave_x4"
    args.patch_size = 64

    # survey
    # wv3 9714 16-64
    # wv2 15084 16-64
    # gf2 19809 16-64
    # qb  17139 16-64
    sess = HISRSession(args)
    train_loader, _ = sess.get_dataloader(args.dataset, False)
    print(len(train_loader))

    # import scipy.io as sio
    #
    # x = sio.loadmat("D:/Datasets/pansharpening/training_data/train1.mat")
    # print(x.keys())

if __name__ == "__main__":
    import matplotlib.pyplot as plt
