
from model_utils import Encoder ,Decoder
import torch
import torch.nn as nn

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self ,x):
        z = self.encoder(x)
        x_hat = self.decoder(z)

        return x_hat







