import torch
import torch.nn as nn
import torch.nn.functional as f

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
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

        self.candidate_block = nn.ModuleList()

        # 单层卷积
        if num_conv == 1:
            self.block1 = BasicBlock(in_channels, out_channels, kernel_size, stride=stride)
            self.candidate_block.append(self.block1)
        
        elif num_conv == 2:
            self.block2_1 = BasicBlock(in_channels, out_channels, kernel_size, stride=stride)
            self.block2_2 = BasicBlock(out_channels, out_channels, kernel_size, stride=stride)
            self.candidate_block.append(self.block2_1)
            self.candidate_block.append(self.block2_2)
        
        elif num_conv == 3:
            self.block3_1 = BasicBlock(in_channels, out_channels, kernel_size, stride=stride)
            self.block3_2 = BasicBlock(out_channels, out_channels, kernel_size, stride=stride)
            self.block3_3 = BasicBlock(out_channels, out_channels, kernel_size, stride=stride)
            self.candidate_block.append(self.block3_1)
            self.candidate_block.append(self.block3_2)
            self.candidate_block.append(self.block3_3)
        
        else:
            raise ValueError("num_conv must be 1, 2 or 3!")
        
    def forward(self, x):
        for layer in self.candidate_block:
            x = layer(x)
        return x

class SuperNet(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, num_layers=4):
        super(SuperNet, self).__init__()
        # 初始卷积层
        self.conv1 = nn.Conv2d(in_channels, out_channels=64, kernel_size=7, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.Maxpool(kernel_size=3, stride=2, padding=1)

        # 候选层
        for i in range(num_layers):
            setattr(self, f'candlayer_{i}', nn.ModuleList())
            current_candlayer = getattr(self, f'candlayer_{i}')
            current_candlayer.append(CandBasicBlock(in_channels, out_channels, kernel_size=j, num_conv=1) for j in [1, 3, 5, 7])
            current_candlayer.append(CandBasicBlock(in_channels, out_channels, kernel_size=j, num_conv=2) for j in [1, 3, 5, 7])
            current_candlayer.append(CandBasicBlock(in_channels, out_channels, kernel_size=j, num_conv=3) for j in [1, 3, 5, 7])
    
    def forward(self, x_0, x_1, x_2, x_3):
        out_1 = [self.candlayer_0[i](x_0) for i in range(12)]
        out_2 = [self.candlayer_1[i](x_1) for i in range(12)]
        out_3 = [self.candlayer_2[i](x_2) for i in range(12)]
        out_4 = [self.candlayer_3[i](x_3) for i in range(12)]
        return out_1, out_2, out_3, out_4

def train_supernet(supernet, ansnet, train_loader):
    optimizer = torch.optim.Adam(supernet.parameters(), lr=0.001)
    criterion = nn.mseloss()




