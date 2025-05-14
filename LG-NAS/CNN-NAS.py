import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import os
from datetime import datetime
import argparse
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()
        mid_channels = out_channels // self.expansion

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)

        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(identity)
        return F.relu(out)


class CandBottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, num_bottleneck=1):
        super(CandBottleneckBlock, self).__init__()
        layers = []
        for i in range(num_bottleneck):
            layers.append(
                Bottleneck(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    stride if i == 0 else 1
                )
            )
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


# ----------- BasicBlock ----------
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(BasicBlock, self).__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        out = y + self.shortcut(identity)
        return F.relu(out)

# ---------- CandBasicBlock ----------
class CandBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, num_conv=1):
        super(CandBasicBlock, self).__init__()
        self.num_conv = num_conv
        self.candidate_block = nn.Sequential()

        layers = []
        for i in range(num_conv):
            layers.append(
                BasicBlock(in_channels if i == 0 else out_channels, out_channels, kernel_size, stride if i == 0 else 1)
            )
        self.candidate_block = nn.Sequential(*layers)

    def forward(self, x):
        return self.candidate_block(x)

# ---------- SuperNet ----------
class SuperNet(nn.Module):
    def __init__(self, block_type="basic"):
        super(SuperNet, self).__init__()

        BlockClass = BasicBlock if block_type == "basic" else Bottleneck
        CandBlockClass = CandBasicBlock if block_type == "basic" else CandBottleneckBlock

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.candlayer_0 = nn.ModuleList([CandBlockClass(64, 64, num_conv=n) for n in [1, 2, 3]])
        self.candlayer_1 = nn.ModuleList([CandBlockClass(64, 128, num_conv=n) for n in [1, 2, 3]])
        self.candlayer_2 = nn.ModuleList([CandBlockClass(128, 256, num_conv=n) for n in [1, 2, 3]])
        self.candlayer_3 = nn.ModuleList([CandBlockClass(256, 512, num_conv=n) for n in [1, 2]])

    def forward(self, img, *t_feats):
        assert len(t_feats) >= 4

        x0 = F.relu(self.bn1(self.conv1(img)))
        x0 = self.maxpool(x0)
        out0 = [self.candlayer_0[i](x0) for i in range(len(self.candlayer_0))]
        out1 = [self.candlayer_1[i](t_feats[0]) for i in range(len(self.candlayer_1))]
        out2 = [self.candlayer_2[i](t_feats[1]) for i in range(len(self.candlayer_2))]
        out3 = [self.candlayer_3[i](t_feats[2]) for i in range(len(self.candlayer_3))]
        return out0, out1, out2, out3


def train_supernet(supernet, teacher_model, train_loader, device, log_path="supernet_log.txt", epochs=10, alpha=0.5):
    supernet.to(device)
    teacher_model.to(device)
    teacher_model.eval()

    optimizer = torch.optim.Adam(supernet.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as f:
        f.write(f"SuperNet Training Log - {datetime.now()}\n\n")

    for epoch in range(epochs):
        total_loss = 0.0
        for step, (images, _) in enumerate(train_loader):
            images = images.to(device)

            with torch.no_grad():
                t_out = teacher_model(images)
                t_feats = [t_out[i] for i in [0, 1, 2, 3]]  # ResNet features
                gene_outputs = t_out[4]

            s_out = supernet(images, *t_feats)  # out0, out1, out2, out3


            loss = 0.0
            for i in range(4):  # layer 0, 1, 2, 3
                cand_blocks = s_out[i]
                t_feat = t_feats[i]
                for s_block in cand_blocks:
                    # loss1: 与 teacher 对齐
                    loss_teacher = criterion(s_block, t_feat)
                    # loss2: 与 gene 输出对齐
                    loss_gene = criterion(s_block, gene_out)
                    # 总损失 = teacher 对齐 + gene 对齐
                    loss += alpha * loss_teacher + (1 - alpha) * loss_gene

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if step % 10 == 0:
                log = f"[Epoch {epoch+1}/{epochs}] Step {step}, Loss: {loss.item():.4f}"
                print(log)
                with open(log_path, "a") as f:
                    f.write(log + "\n")

        avg_loss = total_loss / len(train_loader)
        epoch_log = f"==> Epoch [{epoch+1}/{epochs}] Average Loss: {avg_loss:.4f}\n"
        print(epoch_log)
        with open(log_path, "a") as f:
            f.write(epoch_log)

#  取出中间值
class ResNetWithFeatures(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool)
        self.layer1 = backbone.layer1  # usually has 3 BasicBlocks
        self.layer2 = backbone.layer2  # 4 blocks
        self.layer3 = backbone.layer3  # 6 blocks
        self.layer4 = backbone.layer4  # 3 blocks for resnet34

    def forward(self, x):
        x = self.layer0(x)

        f1 = self._extract_layer_features(self.layer1, x)  # output of last block
        f2 = self._extract_layer_features(self.layer2, f1)
        f3 = self._extract_layer_features(self.layer3, f2)

        # 特殊处理 layer4
        layer4_feats = []
        input_l4 = f3
        for idx, block in enumerate(self.layer4):
            input_l4 = block(input_l4)
            layer4_feats.append(input_l4)

        f4_penultimate = layer4_feats[-2]  # 倒数第二个 block 的输出
        f4_last = layer4_feats[-1]         # 最后一个 block 的输出

        return [f1, f2, f3, f4_penultimate, f4_last]

    def _extract_layer_features(self, layer, x):
        for block in layer:
            x = block(x)
        return x

def get_args():
    parser = argparse.ArgumentParser(description="Train SuperNet with teacher model")

    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--log_path', type=str, default='logs/supernet_log.txt', help='Log file path')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for training')
    parser.add_argument('--dataset_root', type=str, default='./data', help='Path to download/load dataset')
    parser.add_argument('--arch', type=str, default='resnet34', choices=['resnet34', 'resnet50'], help='Teacher model architecture')


    return parser.parse_args()

def main():
    args = get_args()

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 加载 ImageFolder 数据集
    train_dataset = ImageFolder(root=args.dataset_root, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    # 根据 arch 决定教师模型和 supernet 结构
    if args.arch == 'resnet34':
        teacher = models.resnet34(pretrained=True)
        block_type = "basic"
    elif args.arch == 'resnet50':
        teacher = models.resnet50(pretrained=True)
        block_type = "bottleneck"
    else:
        raise ValueError(f"Unsupported architecture: {args.arch}")

    teacher_model = ResNetWithFeatures(teacher)
    supernet = SuperNet(block_type=block_type)

    # 开始训练
    train_supernet(
        supernet, 
        teacher_model, 
        train_loader, 
        device=args.device, 
        log_path=args.log_path, 
        epochs=args.epochs
    )


if __name__ == '__main__':
    main()
