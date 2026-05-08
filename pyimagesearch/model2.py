import torch
import torch.nn as nn

def double_convolution(in_channels, out_channels):
    """
    In the original paper implementation, the convolution operations were
    not padded but we are padding them here. This is because, we need the
    output result size to be same as input size.
    """
    conv_op = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.ReLU(inplace=True)
    )
    return conv_op


class UNet2(nn.Module):
    def __init__(self, num_classes):
        super(UNet2, self).__init__()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        # Contracting path.
        # Each convolution is applied twice.
        self.down_convolution_1 = double_convolution(1, 4)
        self.down_convolution_2 = double_convolution(4, 8)
        self.down_convolution_3 = double_convolution(8, 16)
        self.down_convolution_4 = double_convolution(16, 32)
        self.down_convolution_5 = double_convolution(32, 64)
        # Expanding path.
        self.up_transpose_1 = nn.ConvTranspose2d(
            in_channels=64, out_channels=32,
            kernel_size=2,
            stride=2)
        # Below, `in_channels` again becomes 1024 as we are concatinating.
        self.up_convolution_1 = double_convolution(64, 32)

        self.up_transpose_2 = nn.ConvTranspose2d(
            in_channels=32, out_channels=16,
            kernel_size=2,
            stride=2)
        self.up_convolution_2 = double_convolution(32, 16)

        self.up_transpose_3 = nn.ConvTranspose2d(
            in_channels=16, out_channels=8,
            kernel_size=2,
            stride=2)
        self.up_convolution_3 = double_convolution(16, 8)

        self.up_transpose_4 = nn.ConvTranspose2d(
            in_channels=8, out_channels=4,
            kernel_size=2,
            stride=2)
        self.up_convolution_4 = double_convolution(8, 4)

        # output => `out_channels` as per the number of classes.
        self.out = nn.Conv2d(
            in_channels=4, out_channels=num_classes,
            kernel_size=1
        )
    def forward(self, x):
        down_1 = self.down_convolution_1(x)
        down_2 = self.max_pool2d(down_1)
        down_3 = self.down_convolution_2(down_2)
        down_4 = self.max_pool2d(down_3)
        down_5 = self.down_convolution_3(down_4)
        down_6 = self.max_pool2d(down_5)
        down_7 = self.down_convolution_4(down_6)
        down_8 = self.max_pool2d(down_7)
        down_9 = self.down_convolution_5(down_8)
        # *** DO NOT APPLY MAX POOL TO down_9 ***

        up_1 = self.up_transpose_1(down_9)
        # contcat_1 = torch.cat([down_7, up_1], 1)
        # print(down_7.shape)
        # print(up_1.shape)
        # print(contcat_1.shape)
        # x = self.up_convolution_1(contcat_1)
        x = self.up_convolution_1(torch.cat([down_7, up_1], 1))
        up_2 = self.up_transpose_2(x)
        x = self.up_convolution_2(torch.cat([down_5, up_2], 1))
        up_3 = self.up_transpose_3(x)
        x = self.up_convolution_3(torch.cat([down_3, up_3], 1))
        up_4 = self.up_transpose_4(x)
        x = self.up_convolution_4(torch.cat([down_1, up_4], 1))
        out = self.out(x)
        return out

