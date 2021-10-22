import torch.nn as nn


class ConvNet(nn.Module):
    """
    Simple convolutional model.
    """

    def __init__(self, in_channels: int, out_channels: list, kernel_size: list, stride: list, padding: list,
                 padding_mode: str = 'zeros', threshold: float = 0.5):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.padding_mode = padding_mode
        self.threshold = threshold
        self.layers = []

        self.layers.extend([nn.Conv2d(in_channels=self.in_channels,
                                      out_channels=self.out_channels[0],
                                      kernel_size=self.kernel_size[0],
                                      stride=self.stride[0],
                                      padding=self.padding[0],
                                      padding_mode=self.padding_mode),
                            nn.ReLU()])

        for i in range(1, len(self.out_channels) - 1):
            self.layers.extend([nn.Conv2d(in_channels=self.out_channels[i - 1],
                                          out_channels=self.out_channels[i],
                                          kernel_size=self.kernel_size[i],
                                          stride=self.stride[i],
                                          padding=self.padding[i],
                                          padding_mode=self.padding_mode),
                                nn.ReLU()])

        self.layers.extend([nn.Conv2d(in_channels=self.out_channels[-2],
                                      out_channels=self.out_channels[-1],
                                      kernel_size=self.kernel_size[-1],
                                      stride=self.stride[-1],
                                      padding=self.padding[-1],
                                      padding_mode=self.padding_mode),
                            nn.Sigmoid()])

        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        probas = self.model(x.float())
        y_hat = (probas >= self.threshold).float()
        return probas, y_hat
