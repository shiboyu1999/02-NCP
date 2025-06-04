import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import os
from datetime import datetime
import argparse
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm
import time

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(Bottleneck, self).__init__()
        mid_channels = out_channels // self.expansion

        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)

        padding = kernel_size // 2
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
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
        # print("x.shape: ", x.shape)
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(identity)
        return F.relu(out)


class CandBottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, num_bottleneck=1):
        super(CandBottleneckBlock, self).__init__()
        layers = []
        for i in range(num_bottleneck):
            layers.append(
                Bottleneck(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride if i == 0 else 1
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

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        if block_type == "basic":
            self.align_f1 = FeatureAlign(in_channels=64, out_channels=512)
            self.align_f2 = FeatureAlign(in_channels=128, out_channels=512)
            self.align_f3 = FeatureAlign(in_channels=256, out_channels=512)
        else:
            self.align_f1 = FeatureAlign(in_channels=256, out_channels=2048)
            self.align_f2 = FeatureAlign(in_channels=512, out_channels=2048)
            self.align_f3 = FeatureAlign(in_channels=1024, out_channels=2048)

        if block_type == "basic":
            self.candlayer_0 = nn.ModuleList([CandBasicBlock(64, 64, kernel_size=k, num_conv=n, stride=1) for n in [1, 2, 3] for k in [1, 3, 5, 7]])
            self.candlayer_1 = nn.ModuleList([CandBasicBlock(64, 128, kernel_size=k,  num_conv=n, stride=2) for n in [1, 2, 3] for k in [1, 3, 5, 7]])
            self.candlayer_2 = nn.ModuleList([CandBasicBlock(128, 256, kernel_size=k, num_conv=n, stride=2) for n in [1, 2, 3] for k in [1, 3, 5, 7]])
            self.candlayer_3 = nn.ModuleList([CandBasicBlock(256, 512, kernel_size=k, num_conv=n, stride=2) for n in [1, 2] for k in [1, 3, 5, 7]])
        else:
            self.candlayer_0 = nn.ModuleList([CandBottleneckBlock(64, 256, kernel_size=k, num_bottleneck=n, stride=1) for n in [1, 2, 3] for k in [1, 3, 5, 7]])
            self.candlayer_1 = nn.ModuleList([CandBottleneckBlock(256, 512, kernel_size=k, num_bottleneck=n, stride=2) for n in [1, 2, 3] for k in [1, 3, 5, 7]])
            self.candlayer_2 = nn.ModuleList([CandBottleneckBlock(512, 1024, kernel_size=k, num_bottleneck=n, stride=2) for n in [1, 2, 3] for k in [1, 3, 5, 7]])
            self.candlayer_3 = nn.ModuleList([CandBottleneckBlock(1024, 2048, kernel_size=k, num_bottleneck=n, stride=2) for n in [1, 2] for k in [1, 3, 5, 7]])

    def forward(self, img, *t_feats):
        assert len(t_feats) >= 4

        x0 = F.relu(self.bn1(self.conv1(img)))
        x0 = self.maxpool(x0)
        # print("x0.shape: ", x0.shape)
        out0 = [self.candlayer_0[i](x0) for i in range(len(self.candlayer_0))]
        # print("out0.shape: ", out0[0].shape)
        out1 = [self.candlayer_1[i](t_feats[0]) for i in range(len(self.candlayer_1))]
        # print("out1.shape: ", out1[0].shape)
        out2 = [self.candlayer_2[i](t_feats[1]) for i in range(len(self.candlayer_2))]
        # print("out2.shape: ", out2[0].shape)
        out3 = [self.candlayer_3[i](t_feats[2]) for i in range(len(self.candlayer_3))]
        # print("out3.shape: ", out3[0].shape)
        align_gene_out0 = [self.align_f1(out0[i]) for i in range(len(out0))]
        align_gene_out1 = [self.align_f2(out1[i]) for i in range(len(out1))]
        align_gene_out2 = [self.align_f3(out2[i]) for i in range(len(out2))]
        align_gene_out3 = out3
        return out0, out1, out2, out3, align_gene_out0, align_gene_out1, align_gene_out2, align_gene_out3

def evaluate_supernet(supernet, teacher_model, val_loader, device, save_path="cand_selection.txt"):
    supernet.eval()
    teacher_model.eval()
    supernet.to(device)
    teacher_model.to(device)

    criterion = nn.MSELoss(reduction='none')  # 不平均，逐样本保留

    # 初始化每个 candidate block 的累计损失
    num_blocks_per_layer = [len(supernet.candlayer_0), len(supernet.candlayer_1),
                            len(supernet.candlayer_2), len(supernet.candlayer_3)]
    block_losses = [torch.zeros(n).to(device) for n in num_blocks_per_layer]
    block_counts = [0] * 4  # 样本计数器

    with torch.no_grad():
        for images, _ in tqdm(val_loader, desc="Evaluating SuperNet"):
            images = images.to(device)
            teacher_outputs = teacher_model(images)
            t_feats = [teacher_outputs[i] for i in [0, 1, 2, 3]]

            x0 = F.relu(supernet.bn1(supernet.conv1(images)))
            x0 = supernet.maxpool(x0)

            cand_inputs = [x0] + t_feats[:3]  # 输入 x0, t1, t2 for layer 0,1,2
            cand_layers = [supernet.candlayer_0, supernet.candlayer_1, supernet.candlayer_2, supernet.candlayer_3]

            for layer_idx in range(4):
                t_feat = t_feats[layer_idx]
                input_feat = cand_inputs[layer_idx]
                candidates = cand_layers[layer_idx]

                for i, cand_block in enumerate(candidates):
                    output = cand_block(input_feat)
                    loss = criterion(output, t_feat).mean(dim=[1,2,3]).mean()  # 每样本 MSE，后取平均
                    block_losses[layer_idx][i] += loss

                block_counts[layer_idx] += 1

    selected_blocks = []
    for layer_idx in range(4):
        avg_losses = block_losses[layer_idx] / block_counts[layer_idx]
        topk = torch.topk(-avg_losses, k=4)  # 最小的4个，负号变最大
        top_indices = topk.indices.cpu().tolist()
        selected_blocks.append(top_indices)

    # 写入文件
    with open(save_path, "w") as f:
        for i, indices in enumerate(selected_blocks):
            f.write(f"Layer {i}: {', '.join(map(str, indices))}\n")

    print(f"Top-4 candidate block indices per layer saved to {save_path}")
    return selected_blocks

def train_supernet(supernet, teacher_model, train_loader, device, log_path="supernet_log.txt", epochs=10, alpha=0.5):
    supernet.to(device)
    teacher_model.to(device)
    teacher_model.eval()

    optimizer = torch.optim.Adam(supernet.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    log_path = os.path.join(os.path.dirname(log_path), "supernt_log.txt")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w") as f:
        f.write(f"SuperNet Training Log - {datetime.now()}\n\n")
    
    total_step = len(train_loader)

    for epoch in range(epochs):
        total_loss = 0.0
        for step, (images, _) in enumerate(train_loader):
            # 计算每个step的时间
            start_time = time.time()
            images = images.to(device)
            with torch.no_grad():
                t_out = teacher_model(images)
                t_feats = [t_out[i] for i in [0, 1, 2, 3]]  # ResNet features
                gene_outputs = t_out[4]

            s_out = supernet(images, *t_feats)  # out0, out1, out2, out3, align_gene_out0, align_gene_out1, align_gene_out2, align_gene_out3

            loss = 0.0
            for i in range(4):  # layer 0, 1, 2, 3
                cand_blocks = s_out[i]
                t_feat = t_feats[i]
                aligned_cand_blocks = s_out[i + 4]  # align_gene_out0, align_gene_out1, align_gene_out2, align_gene_out3
                for s_block, s_aligned in zip(cand_blocks, aligned_cand_blocks):
                    # print(f"Block {i} - Teacher Feature Shape: {t_feat.shape}, Aligned Block Shape: {s_aligned.shape}, Candidate Block Shape: {s_block.shape}")
                    # loss1: 与 teacher 对齐
                    loss_teacher = criterion(s_block, t_feat)
                    # loss2: 与 gene 输出对齐
                    loss_gene = criterion(s_aligned, gene_outputs)
                    # 总损失 = teacher 对齐 + gene 对齐
                    loss += alpha * loss_teacher + (1 - alpha) * loss_gene

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            step_time = time.time() - start_time
            if step % 10 == 0:
                log = f"[Epoch {epoch+1}/{epochs}] [Step {step}/{total_step}], Loss: {loss.item():.4f}, Step Time: {step_time:.4f}s"
                print(log)
                log_path = os.path.join(os.path.dirname(log_path), "supernt_log.txt")
                with open(log_path, "a") as f:
                    f.write(log + "\n")

        avg_loss = total_loss / len(train_loader)
        epoch_log = f"==> Epoch [{epoch+1}/{epochs}] Average Loss: {avg_loss:.4f}\n"
        print(epoch_log)
        with open(log_path, "a") as f:
            f.write(epoch_log)
        
        checkpoint_model_path = os.path.join(os.path.dirname(log_path), "supernet_checkpoint.pt")
        torch.save(
            {
                "model": supernet.state_dict(),
                "epoch": epoch
            },
            checkpoint_model_path
        )

    final_model_path = os.path.join(os.path.dirname(log_path), "supernet_final.pt")
    torch.save(supernet.state_dict(), final_model_path)


class FeatureAlign(nn.Module):
    def __init__(self, in_channels, out_channels=512, target_size=(7, 7)):
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(target_size)
        )

    def forward(self, x):
        return self.project(x)

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
    parser.add_argument('--eval', action='store_true', help='Evaluate the model')


    return parser.parse_args()

def main():
    args = get_args()

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    if args.eval:
        val_dataset = ImageFolder(root=args.dataset_root.replace("train", "val"), transform=transform)
        print(f"Validation dataset size: {len(val_dataset)}")
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    else:
        # 加载 ImageFolder 数据集
        train_dataset = ImageFolder(root=args.dataset_root, transform=transform)
        print(f"Training dataset size: {len(train_dataset)}")
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
    if args.eval:
        print("Evaluating SuperNet...")
        # 加载supernet的权重
        checkpoint_path = os.path.join(os.path.dirname(args.log_path), "supernet_final.pt")
        supernet.load_state_dict(torch.load(checkpoint_path))
        evaluate_supernet(
        supernet=supernet,
        teacher_model=teacher_model,
        val_loader=val_loader,
        device=args.device,
        save_path=os.path.join(os.path.dirname(args.log_path), "cand_selection.txt")
        )
        return

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
