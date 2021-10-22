import torch
from torch.utils.data import Dataset
from torchvision import transforms

import numpy as np

from skimage import data, feature, filters
from skimage.color import rgb2gray
from skimage.feature import corner_harris, corner_peaks
from skimage.transform import warp, AffineTransform
from skimage.draw import ellipse, rectangle

from PIL import Image


class EdgeDetectionDataset(Dataset):
    def __init__(self, files, type='canny', RGB=True):
        super().__init__()
        self.files = sorted(files)
        self.len_ = len(self.files)
        self.type = type
        self.RGB = RGB

    def __len__(self):
        return self.len_

    def load_sample(self, file):
        image = Image.open(file)
        image.load()
        return image

    def to_grayscale(self, img_tensor):
        img_arr = img_tensor.permute(1, 2, 0).numpy()
        img_gray = rgb2gray(img_arr)
        return img_gray

    def __getitem__(self, index):
        transform = transforms.Compose([
            transforms.Resize(size=(64, 64)),
            transforms.ToTensor()
        ])

        x_true = self.load_sample(self.files[index])
        x_true = transform(x_true)

        x_gray = self.to_grayscale(x_true)

        if self.type == 'canny':
            x_filtered = feature.canny(x_gray, sigma=1)

        elif self.type == 'sobel':
            x_filtered = filters.sobel(x_gray)

        x_filtered = torch.Tensor(x_filtered).unsqueeze(0)

        if self.RGB:
            return x_true, x_filtered

        else:
            x_gray = transforms.ToTensor()(x_gray)
            return x_gray, x_filtered


class BinarizationDataset(Dataset):
    def __init__(self, files, RESIZE=True, RGB=True):
        super().__init__()
        self.files = sorted(files)
        self.len_ = len(self.files)
        self.RGB = RGB
        self.RESIZE = RESIZE

    def __len__(self):
        return self.len_

    def load_sample(self, file):
        image = Image.open(file)
        image.load()
        return image

    def transform(self, x, size=256):
        if self.RESIZE:
            return transforms.Compose([
                transforms.RandomResizedCrop(size=size),
                transforms.ToTensor()])(x)  # tensor
        else:
            return transforms.ToTensor()(x)  # tensor

    def to_grayscale(self, img_tensor):
        img_arr = img_tensor.permute(1, 2, 0).numpy()
        img_gray = rgb2gray(img_arr)
        return torch.tensor(img_gray).unsqueeze(0)

    def niblack_binarization(self, x, window_size=15, k=0.2):
        thresh_niblack = filters.threshold_niblack(x, window_size=window_size, k=k)
        binary_niblack = x > thresh_niblack
        binary_niblack = torch.FloatTensor(binary_niblack)
        return binary_niblack

    def __getitem__(self, index):
        x_true = self.load_sample(self.files[index])  # true image
        x_true = self.transform(x_true, size=256)  # RGB or RGB + noise

        if x_true.shape[0] != 1:
            x_true = self.to_grayscale(x_true)

        x_filtered = self.niblack_binarization(x_true.numpy(), 17, 0.8)
        return x_true, x_filtered


class CornerDetectionDataset(Dataset):
    def __init__(self, N):
        super().__init__()
        self.len_ = N

    def __len__(self):
        return self.len_

    def generate_image(self, img_shape=(256, 256)):
        # Sheared checkerboard
        tform = AffineTransform(scale=tuple(np.random.uniform(low=1, high=2, size=2)),
                                rotation=np.random.uniform(low=0, high=1, size=1),
                                shear=np.random.uniform(low=0, high=1, size=1),
                                translation=tuple(np.random.uniform(low=0, high=180, size=2)))

        image1 = warp(data.checkerboard()[:90, :90], tform.inverse,
                      output_shape=img_shape)

        # Ellipse
        rr, cc = ellipse(np.random.uniform(low=50, high=180, size=1),
                         np.random.uniform(low=50, high=180, size=1),
                         np.random.uniform(low=10, high=60, size=1),
                         np.random.uniform(low=10, high=70, size=1))

        image2 = np.zeros_like(image1)
        image2[rr, cc] = 1

        # Rectangle
        rr, cc = rectangle(tuple(np.random.randint(low=10, high=100, size=2)),
                           extent=tuple(np.random.randint(low=10, high=100, size=2)),
                           shape=img_shape)

        image3 = np.zeros_like(image1)
        image3[rr, cc] = 1

        image = image1 + image2 + image3

        coords = corner_peaks(corner_harris(image), min_distance=5, threshold_rel=0.02)

        # ground truth of corners
        heatmap = np.zeros_like(image)
        for x, y in coords:
            heatmap[x, y] = 1

        image = transforms.ToTensor()(image)
        heatmap = transforms.ToTensor()(heatmap)
        return image, heatmap

    def __getitem__(self, index):
        image, heatmap = self.generate_image()
        return image, heatmap.float()