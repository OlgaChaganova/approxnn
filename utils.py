from torchvision import transforms
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def save_plt(history, epoch, type):
    if type == 'epoch':
        plt.plot(history['train loss by epoch'], label="train" if epoch == 0 else "", color='dodgerblue')
        plt.plot(history['valid loss by epoch'], label="val" if epoch == 0 else "", color='orange')
        plt.xlabel("Epoch")
        plt.xticks(np.arange(0, len(history['train loss by epoch']), 1))

    elif type == 'batch':
        plt.plot(history['train loss by batch'], label="train" if epoch == 0 else "", color='dodgerblue')
        plt.plot(history['valid loss by batch'], label="val" if epoch == 0 else "", color='orange')
        plt.xlabel("Epoch")
        plt.xticks(np.arange(0, len(history['train loss by batch']), 1))

    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(Path('training plot.png'))


def save_PIL(img_batch, fp, random_samples=False, num_samples=None, nrow=8):
    to_PIL = transforms.ToPILImage()

    if random_samples:
        ids = np.random.permutation(len(img_batch))[:num_samples]
        img_batch = img_batch[ids]

    pil_img_batch = to_PIL(make_grid(img_batch, nrow=nrow))
    pil_img_batch.save(fp=fp)


