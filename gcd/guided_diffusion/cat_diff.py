import torch
import torch.nn as nn

class cator(nn.Module):
    def __init__(self, channels, num_img):
        super(cator, self).__init__()
        # self.conv = nn.Conv2d(in_channels=self.AB_channel,out_channels=self.AB_channel, kernel_size=1)

    def forward(self, A_img, B_img):
        # Assuming A_img and B_img have the same spatial dimensions
        # concat_img = torch.cat((A_img, B_img), dim=1)  # Concatenate along the channel dimension
        diff = A_img - B_img
        return diff
