import torch
from pathlib import Path

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paths to data
DATA_DIR_BIN1 = Path('NoisyOffice/RealNoisyOffice/real_noisy_images_grayscale' )
DATA_DIR_BIN2 = Path('NoisyOffice/SimulatedNoisyOffice/simulated_noisy_images_grayscale')
DATA_DIR_EDGE = 'tiny-imagenet-200'

# Hyperparameters
BATCH_SIZE = 8
EPOCHS = 3
IN_CHANNELS = 1
OUT_CHANNELS = [4, 8, 16, 1]
KERNEL_SIZE = [3, 3, 3, 3]
STRIDE = [1, 1, 1, 1]
PADDING = [1, 1, 1, 1]
