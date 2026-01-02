import torch 
import numpy as np 
import torch.nn as nn 
import torch.nn.functional as F 

# [----------Encoder Block ---------]
class Encoder(nn.Module):
    def __init__(self,):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=1 , out_channels=32 ,stride=1 ,kernel_size=3 ,padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, stride=2, kernel_size=3 ,padding=1)  
        self.conv3 = nn.Conv2d(in_channels=64 ,out_channels=64, stride=2, kernel_size=3 ,padding=1)
        self.conv4 = nn.Conv2d(in_channels=64 ,out_channels=64, stride=1 ,kernel_size=3 ,padding=1)
        self.fc1 = nn.Linear(3136,2)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x) ,0.01)
        x = F.leaky_relu(self.conv2(x) ,0.01)
        x = F.leaky_relu(self.conv3(x) ,0.01)
        x = F.leaky_relu(self.conv4(x) ,0.01)
        x = torch.flatten(x ,start_dim=1) # the dimension 0 will  stay as its that is Batch dimension and the rest will be multiplied ise channels ,hieght , width 
        z = self.fc1(x)

        return z 
    

# [--------------Decoder Block ------------------]

class Decoder(nn.Module):
    def __init__(self,):
        super().__init__()
        self.fc = nn.Linear(2 ,3136)
        self.deconv1 = nn.ConvTranspose2d(in_channels=64 ,out_channels=64 ,stride=1 ,kernel_size=3 ,padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=64 ,out_channels=64 ,stride=2 ,kernel_size=3 ,padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(in_channels=64 ,out_channels=32 ,stride=2 ,kernel_size=3, padding=1 ,output_padding=1)
        self.deconv4 = nn.ConvTranspose2d(in_channels=32 ,out_channels=1 ,stride=1, kernel_size=3, padding=1)

    def forward(self ,z):
        x = self.fc(z)
        x = x.view(-1,64,7,7) # Exactly opposite of torch.flatten in Encoder Block
        x = F.leaky_relu(self.deconv1(x), 0.01)
        x = F.leaky_relu(self.deconv2(x), 0.01)
        x = F.leaky_relu(self.deconv3(x), 0.01)
        x = self.deconv4(x)
        x = torch.sigmoid(x)
        return x 
    
    