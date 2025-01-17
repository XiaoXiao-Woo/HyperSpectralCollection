import torch.utils.data as data
import torch
import h5py
import math
from torch.nn import functional as F
from imresize_pytorch import imresize
from psf2otf import psf2otf_torch
from functools import partial

class RandomBicubicSampling:
    """Generate LQ image from GT (and crop), which will randomly pick a scale.

    Args:
        scale_min (float): The minimum of upsampling scale, inclusive.
            Default: 1.0.
        scale_max (float): The maximum of upsampling scale, exclusive.
            Default: 4.0.
        patch_size (int): The cropped lr patch size.
            Default: None, means no crop.

        Scale will be picked in the range of [scale_min, scale_max).
    """

    def __init__(self,
                 scale_min=1.0,
                 scale_max=4.0,
                 patch_size=None
                 ):
        assert scale_max >= scale_min
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.patch_size = patch_size

    def make_random(self):
        self.scale = np.random.uniform(self.scale_min, self.scale_max)

    def __call__(self, img, key_name):
        """Call function.

        Args:
            results (dict): A dict containing the necessary information and
                data for augmentation. 'gt' is required.

        Returns:
            dict: A dict containing the processed data and information.
                modified 'gt', supplement 'lq' and 'scale' to keys.
        """

        results = {}
        scale = self.scale
        # print(img.shape)
        if self.patch_size is None:
            h_lr = math.floor(img.shape[-2] / scale + 1e-9)
            w_lr = math.floor(img.shape[-1] / scale + 1e-9)
            img = img[..., :round(h_lr * scale), :round(w_lr * scale)]
            # img_down = resize_fn(img, (w_lr, h_lr), scale)
            crop_lr = imresize(img, 1/scale)

        else:
            w_lr = self.patch_size
            w_hr = round(w_lr * scale)
            x0 = np.random.randint(0, img.shape[-3] - w_hr)
            y0 = np.random.randint(0, img.shape[-2] - w_hr)
            crop_hr = img[x0:x0 + w_hr, y0:y0 + w_hr, :]
            crop_lr = imresize(crop_hr, 1 / scale)


        # results['gt'] = crop_hr
        results[key_name] = crop_lr
        results['scale'] = scale
        # results['scale'] = scale

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f' scale_min={self.scale_min}, '
                     f'scale_max={self.scale_max}, '
                     f'patch_size={self.patch_size}, '
                     )

        return repr_str


class GaussianBlur:

    def __init__(self, dataset_name, scale, patch_size: int):

        if dataset_name in ["PaviaCentre"]:
            sigma = (1 / (2 * (2.7725887) / scale ** 2)) ** 0.5
        else:
            raise NotImplementedError("Please contact authors to know about sigma for other datasets.")

        psf = torch.from_numpy(gen_psf(sigma))
        self.otf = psf2otf_torch(psf, (patch_size, patch_size))


    def __call__(self, x, keyname) -> dict:

        return {keyname: torch.fft.ifft2(self.otf * torch.fft.fft2(x))}




class SpectralPansharpeningDataset(data.Dataset):
    def __init__(self, file_path, num_spectra, patch_size=None):
        super(SpectralPansharpeningDataset, self).__init__()
        self.patch_size = patch_size
        dataset = h5py.File(file_path)
        print(file_path, dataset.keys())
        self.GT = np.asarray(dataset.get("GT"))
        self.GT = self.GT / self.GT.max()
        self.UP = np.asarray(dataset.get("HSI_up"))
        self.LRHSI = np.asarray(dataset.get("LRHSI"))
        self.PAN = np.asarray(dataset.get("PAN"))

        self.indices = (np.asarray(dataset.get("indices" + str(num_spectra)), dtype=np.int32) - 1).flatten().tolist()

        self.PAN = self.PAN / self.PAN.max()
        self.patch_size = self.GT.shape[-1]

    def __getitem__(self, index):

        return {'gt': torch.from_numpy(self.GT[index, self.indices, ...]).float(),
                'up': torch.from_numpy(self.UP[index, self.indices, ...]).float(),
                'lrhsi': torch.from_numpy(self.LRHSI[index, self.indices, ...]).float(),
                'pan': torch.from_numpy(self.PAN[index, ...]).float()}

    def __len__(self):
        return self.GT.shape[0]


class LIIFDataset(data.Dataset):
    def __init__(self, cfg, file_path, patch_size=None):
        super(LIIFDataset, self).__init__()
        self.patch_size = patch_size
        self.inr_transforms = cfg.inr_transforms

        dataset = h5py.File(file_path)
        print(file_path, dataset.keys())
        self.GT = np.asarray(dataset.get("GT"))
        self.GT = self.GT / self.GT.max()
        self.UP = np.asarray(dataset.get("HSI_up"))
        self.LRHSI = np.asarray(dataset.get("LRHSI"))
        self.PAN = np.asarray(dataset.get("PAN"))
        self.PAN = self.PAN / self.PAN.max()
        self.patch_size = self.GT.shape[-1]

    def __getitem__(self, index):
        data = {'gt': torch.from_numpy(self.GT[index, ...]).float(),
                'gt_pan': torch.from_numpy(self.PAN[index, ...]).float()}

        keys = list(data.keys())

        for idx, key in enumerate(keys):
            if key == 'gt':
                key_name = 'ms'
                self.inr_transforms.make_random()
                print(self.inr_transforms.scale)
            else:
                key_name = 'pan'

            data.update(self.inr_transforms(data[key], key_name))

        return data

    def __len__(self):
        return self.GT.shape[0]






class SSIPDataset(data.Dataset):
    # Spatial-Spectral Implicit HyperspetralPansharpening
    def __init__(self, cfg, mode, file_path, patch_size=None):
        super(SSIPDataset, self).__init__()

        self.scale = cfg.scale
        self.patch_size = patch_size
        self.inr_transforms = cfg.inr_transforms

        dataset = h5py.File(file_path)

        self.GT = np.asarray(dataset.get("GT"))
        self.GT = self.GT / self.GT.max()
        self.UP = np.asarray(dataset.get("HSI_up"))
        self.LRHSI = np.asarray(dataset.get("LRHSI"))
        self.PAN = np.asarray(dataset.get("PAN"))
        self.PAN = self.PAN / self.PAN.max()
        self.patch_size = self.GT.shape[-1]

        self.valid_waveLength = [10, 34, 50, 78, 90]

        self.dataset = dataset
        self.indices_list = sorted([k.replace('indices', '') for k in list(dataset.keys())[4:]], key=int)[:-1]
        self.train_waveLength = [idx for idx in self.indices_list if idx not in self.valid_waveLength]

        print(file_path, dataset.keys(), len(self.indices_list))

        if mode == "train":
            self.indices_list = self.train_waveLength
        else:
            self.indices_list = self.valid_waveLength

    def __getitem__(self, index):


        num_spectra = np.random.randint(0, len(self.indices_list))
        indices = (np.asarray(self.dataset.get("indices" + str(self.indices_list[num_spectra])), dtype=np.int32) - 1).flatten().tolist()


        data = {'gt_ms': torch.from_numpy(self.GT[index, indices, ...]).float(),
                'gt_pan': torch.from_numpy(self.PAN[index, ...]).float()}

        keys = list(data.keys())

        for idx, key in enumerate(keys):
            if key == 'gt_ms':
                key_name = 'ms'
            else:
                key_name = 'pan'
            for transform in self.inr_transforms:
                if isinstance(transform, RandomBicubicSampling):
                    transform.make_random()
                data.update(transform(data[key], key_name))
            if key_name == "ms":
                data['lms'] = imresize(data['ms'], self.scale)

        data.pop('gt_pan')
        data.update({'gt': torch.from_numpy(self.GT[index, ...]).float()})

        return data

    def __len__(self):
        return self.GT.shape[0]

class DatasetFromHdf5(data.Dataset):
    def __init__(self, file_path):
        super(DatasetFromHdf5, self).__init__()
        dataset = h5py.File(file_path)
        print(file_path, dataset.keys())
        self.GT = dataset.get("GT")
        self.UP = dataset.get("HSI_up")
        self.LRHSI = dataset.get("LRHSI")
        self.RGB = dataset.get("RGB")

    #####必要函数
    def __getitem__(self, index):
        return {'gt': torch.from_numpy(self.GT[index, :, :, :]).float(),
                'up': torch.from_numpy(self.UP[index, :, :, :]).float(),
                'lrhsi': torch.from_numpy(self.LRHSI[index, :, :, :]).float(),
                'rgb': torch.from_numpy(self.RGB[index, :, :, :]).float()}

        #####必要函数

    def __len__(self):
        return self.GT.shape[0]



def show_CAVE():
    # h5data = h5py.File("C:/Python/ProjectSets/TDMlab/hisr/data/Harvard/DL/harvard/validation_harvard(with_up)x4_rgb.h5")
    h5data = h5py.File("C:/Python/ProjectSets/TDMlab/hisr/data/CAVE/cave/train_cave(with_up)x4_rgb.h5")
    msi = np.array(h5data['RGB']).transpose(2, 3, 1, 0)  # 'GT', 'HSI_up', 'LRHSI', 'RGB'

    return msi


def show_Pavia_centre():
    # h5data = h5py.File("C:/Python/ProjectSets/TDMlab/hisr/data/Harvard/DL/harvard/validation_harvard(with_up)x4_rgb.h5")
    h5data = h5py.File(r"D:\Datasets\hyper\Pavia_centre\train_pavia_new_4.h5")
    print(h5data.keys())

    gt = np.array(h5data['GT']).transpose(2, 3, 1, 0)  # 'GT', 'HSI_up', 'LRHSI', 'RGB'
    pan = np.array(h5data['PAN']).transpose(2, 3, 1, 0)  # 'GT', 'HSI_up', 'LRHSI', 'RGB'

    return gt, pan

def show_Spectral():
    # gt, pan = show_Pavia_centre()
    dataset = SpectralPansharpeningDataset(r"D:\Datasets\hyper\Botswana\train_Botswana96_86.h5", 145)
    loader = data.DataLoader(dataset, batch_size=1, num_workers=0)
    for idx, batch in enumerate(loader):
        print(batch.keys())
        gt = batch['gt']
        print(gt.shape)
        plt.figure()
        plt.imshow(gt[0, [60, 30, 10], :, :].permute(1, 2, 0) / gt.max())
        plt.show()

def show_LIIF():
    dataset = LIIFDataset(r"G:\woo\Datasets\hisr\PaviaCentre\valid_PaviaCentre96_15.h5", 2.0, 3.9, 102)
    loader = data.DataLoader(dataset, batch_size=1, num_workers=0)
    for idx, batch in enumerate(loader):
        print(batch.keys())
        ms, pan = batch['ms'], batch['pan']
        print(ms.shape, pan.shape)
        # plt.figure()
        # plt.imshow(ms[0, [60, 30, 10], :, :].permute(1, 2, 0))
        # plt.show()


def gen_psf(sig):
    # from scipy.signal import convolve2d as conv2
    import cv2
    def gaussian_kernel(dimension_x, dimension_y, sigma):
        x = cv2.getGaussianKernel(dimension_x, sigma)
        y = cv2.getGaussianKernel(dimension_y, sigma)
        kernel = x.dot(y.T)
        return kernel

    # Gaussian fspecial kernel
    psf = gaussian_kernel(9, 9, sig)

    return psf

def show_SSIPDataset():
    from udl_vis.Basis.config import Config
    # gt, pan = show_Pavia_centre()
    dataset = SSIPDataset(Config(dict(scale=4,
                                      inr_transforms=[
                                          GaussianBlur(dataset_name="PaviaCentre", scale=4, patch_size=96),
                                          RandomBicubicSampling(scale_min=2.0, scale_max=4.0)]
                                      )), mode="train", file_path=r"G:\woo\Datasets\hisr\PaviaCentre\valid_PaviaCentre96_15.h5")
    loader = data.DataLoader(dataset, batch_size=1, num_workers=0)
    for idx, batch in enumerate(loader):
        print(batch.keys())
        gt, pan, lms, ms = batch['gt'], batch['pan'], batch['lms'], batch['ms']
        print(gt.shape, pan.shape, lms.shape, ms.shape, batch['scale'].item())
        plt.figure()
        plt.imshow(gt[0, np.random.choice(range(gt.shape[1]), 3), :, :].permute(1, 2, 0) / gt.max())
        plt.show()


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import numpy as np

    # show_Spectral()
    show_SSIPDataset()

