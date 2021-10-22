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
