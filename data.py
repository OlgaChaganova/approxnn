from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
import os
from pathlib import Path
from PIL import Image


def load_sample(file):
    image = Image.open(file)
    image.load()
    return transforms.ToTensor()(image)


def RGB_images(files):
    rgb = []
    for file in files:
        img = load_sample(file)
        if img.shape[0] == 3:
            rgb.append(file)
    return rgb


def get_data_for_edge_detection(data_dir):
    TRAIN_DIR = Path(os.path.join(data_dir, 'train'))
    TEST_DIR = Path(os.path.join(data_dir, 'test'))
    VALID_DIR = Path(os.path.join(data_dir, 'val'))

    train_files = list(TRAIN_DIR.rglob('*.JPEG'))
    test_files = list(TEST_DIR.rglob('*.JPEG'))
    valid_files = list(VALID_DIR.rglob('*.JPEG'))

    train_files = RGB_images(train_files)
    test_files = RGB_images(test_files)
    valid_files = RGB_images(valid_files)
    return train_files, valid_files, test_files


def get_data_for_binarization(data_dir1, data_dir2):
    files1 = list(data_dir1.rglob('*.png'))
    files2 = list(data_dir2.rglob('*.png'))

    files = files1 + files2

    ids = np.random.permutation(len(files))
    train_ids, test_ids = train_test_split(ids, test_size=0.3)
    valid_ids, test_ids = train_test_split(test_ids, test_size=0.5)

    train_files = [files[id] for id in train_ids]
    valid_files = [files[id] for id in valid_ids]
    test_files = [files[id] for id in test_ids]

    return train_files, valid_files, test_files


def get_dataloader(dataset, files, batch_size, drop_last=False, shuffle=True):
    data_set = dataset(files)
    data_loader = DataLoader(data_set, batch_size=batch_size, drop_last=drop_last, shuffle=shuffle)
    return data_loader
