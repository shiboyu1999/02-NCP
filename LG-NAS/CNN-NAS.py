import torch
import torch.nn as nn
import torch.nn.functional as f

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(BasicBlock, self).__init__()
        padding = kernel_size // 2  # out_size = (in_size + padding * 2 - kernel_size) / stride + 1

        # 第一个卷积层
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 第二个卷积层
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 残差连接
        self.shortcut = nn.Sequential()
        if stride !=1  or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        y = f.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))

        out = y + self.shortcut(identity)
        out = f.relu(out)
        return out




class CandBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, num_conv=1):
        super(CandBasicBlock, self).__init__()
        self.num_conv = num_conv
        padding = kernel_size // 2  # out_size = (in_size + padding * 2 - kernel_size) / stride + 1 

        # 单层卷积
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 双层卷积
        if num_conv == 2:
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False)
            self.bn2 = nn.BatchNorm2d(out_channels)
        # 三层卷积
        elif num_cong == 3:
            self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False)
            self.bn3 = nn.BatchNorm2d(out_channels)
        else:
            raise ValueError("num_conv must be 1, 2 or 3")
        

