import torch.utils.data as data
import torch
import h5py
import numpy as np

class DummyDataset(data.Dataset):
    def __init__(self, num_samples=100, factor=4):
        self.num_samples = num_samples
        self.GT = torch.randn(num_samples, 31, factor * 16, factor * 16)
        self.UP = torch.randn(num_samples, 31, factor * 16, factor * 16)
        self.LRHSI = torch.randn(num_samples, 31, 16, 16)
        self.RGB = torch.randn(num_samples, 3, factor * 16, factor * 16)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "gt": self.GT[idx],
            "up": self.UP[idx],
            "lrhsi": self.LRHSI[idx],
            "rgb": self.RGB[idx],
        }


class DatasetFromHdf5(data.Dataset):
    def __init__(self, file_path):
        super(DatasetFromHdf5, self).__init__()
        dataset = h5py.File(file_path)
        print(file_path, dataset.keys())
        self.GT = torch.from_numpy(np.array(dataset.get("GT"))).float()
        self.UP = torch.from_numpy(np.array(dataset.get("HSI_up"))).float()
        self.LRHSI = torch.from_numpy(np.array(dataset.get("LRHSI"))).float()
        self.RGB = torch.from_numpy(np.array(dataset.get("RGB"))).float()
        print(self.GT.shape, self.LRHSI.shape, self.RGB.shape)

    #####必要函数
    def __getitem__(self, index):
        # 'GT', 'HSI_up', 'LRHSI', 'RGB'
        return {'gt': self.GT[index, :, :, :],
                'up': self.UP[index, :, :, :],
                'lrhsi': self.LRHSI[index, :, :, :],
                'rgb': self.RGB[index, :, :, :]}

        #####必要函数

    def __len__(self):
        return self.GT.shape[0]


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # h5data = h5py.File("C:/Python/ProjectSets/TDMlab/hisr/data/Harvard/DL/harvard/validation_harvard(with_up)x4_rgb.h5")
    h5data = h5py.File("C:/Python/ProjectSets/TDMlab/hisr/data/CAVE/cave/train_cave(with_up)x4_rgb.h5")
    msi = np.array(h5data['RGB']).transpose(2, 3, 1, 0)  # 'GT', 'HSI_up', 'LRHSI', 'RGB'
    # print(h5data.keys())
    print(msi.shape)
    plt.figure()
    plt.imshow(msi[:, :, :, 2])
    plt.show()
