import torch
import torch.nn as nn

class UNetDecoder(nn.Module):
    def __init__(self, input_dim=512, output_channels=1, img_size=128):
        super(UNetDecoder, self).__init__()
        # Fully connected layer to map embeddings to an image-like tensor
        self.fc = nn.Linear(input_dim, img_size * img_size)

        # Encoder-like layers
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)

        # Decoder-like layers
        self.upconv1 = nn.ConvTranspose2d(16, output_channels, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 1, 128, 128)  # Reshape for image-like data
        x = self.pool(torch.relu(self.conv1(x)))
        x = torch.sigmoid(self.upconv1(x))
        return x
