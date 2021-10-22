import torch
import torch.nn as nn
from torchsummary import summary
import sys

import config
from datasets import EdgeDetectionDataset, BinarizationDataset, CornerDetectionDataset
from data import get_data_for_edge_detection, get_data_for_binarization, get_dataloader
from models import ConvNet
from train import train

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


if __name__ == '__main__':
    if sys.argv[1] == 'canny':
        train_files, valid_files, test_files = get_data_for_edge_detection(config.DATA_DIR_EDGE)
        dataset = EdgeDetectionDataset

    elif sys.argv[1] == 'niblack':
        train_files, valid_files, test_files = get_data_for_binarization(config.DATA_DIR_BIN1, config.DATA_DIR_BIN2)
        dataset = BinarizationDataset

    elif sys.argv[1] == 'harris':
        pass

    train_loader = get_dataloader(dataset, train_files, config.BATCH_SIZE)
    valid_loader = get_dataloader(dataset, valid_files, config.BATCH_SIZE)
    test_loader = get_dataloader(dataset, test_files, config.BATCH_SIZE)

    model = ConvNet(in_channels=config.IN_CHANNELS,
                    out_channels=config.OUT_CHANNELS,
                    kernel_size=config.KERNEL_SIZE,
                    stride=config.STRIDE,
                    padding=config.PADDING).to(config.DEVICE)

    optimizer = torch.optim.Adam(model.parameters())
    loss = nn.BCELoss()

    print('Model:')
    print(model)

    print('Parameters of the model:')
    print(summary(model, (1, 64, 64)))

    print('Starting training:')
    history = train(model, optimizer, loss, train_loader, valid_loader, config.EPOCHS, config.DEVICE)
