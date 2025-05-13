"""
基于梯度方差信息和注意力重要性的ViT架构的学习基因提取（多进程优化版）
"""
import timm 
import os
import sys
import pickle
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms, datasets
from collections import defaultdict
from multiprocessing import Pool, cpu_count
from functools import partial
from tqdm import tqdm

# 华为NPU环境配置
sys.path.append('/usr/local/Ascend/ascend-toolkit/latest/python/site-packages')
import torch_npu
from torch_npu.npu import amp
from torch_npu.contrib import transfer_to_npu

class LoadSubImageNetTask(Dataset):
    def __init__(self, task_dir, transform=None):
        """
        加载一个 ImageNet 子任务的数据。

        参数:
            task_dir (str): 当前任务的文件夹路径，如 'ILSVRC2012_split/task_00'
            transform (callable): 图像预处理 transform
        """
        self.dataset = datasets.ImageFolder(task_dir, transform=transform)

    def __getitem__(self, index):
        return self.dataset[index]

    def __len__(self):
        return len(self.dataset)

# ------------------------
# ViT模型与训练（优化混合精度）
# ------------------------
class ViTBlockGradTracker:
    def __init__(self, model):
        self.model = model
        self.grad_dict = defaultdict(list)
        self.attn_importance_dict = defaultdict(list)
        self._layers = []
        self.active = False  # ✅ 是否启用记录的开关

        # 获取所有ViT Block层
        for name, module in model.named_modules():
            if isinstance(module, timm.models.vision_transformer.Block):
                self._layers.append((name, module))

        self.handles = []
        for layer_name, layer in self._layers:
            # 注册梯度追踪钩子
            grad_handle = layer.register_full_backward_hook(self._make_grad_hook(layer_name))
            self.handles.append(grad_handle)
            # 注册注意力重要性追踪钩子
            attn_handle = layer.attn.register_forward_hook(self._make_attn_hook(layer_name))
            self.handles.append(attn_handle)
        
    def _make_grad_hook(self, layer_name):
        def grad_hook(module, input, output):
            if not self.active:
                return
            total_norm = 0.0
            for param in module.parameters():
                if param.grad is not None:
                    total_norm += param.grad.norm(2).item()
            self.grad_dict[layer_name].append(total_norm)
        return grad_hook
    
    def _make_attn_hook(self, layer_name):
        def attn_hook(module, input, output):
            if not self.active:
                return
            if hasattr(module, 'qkv'):
                qkv = module.qkv(input[0])  # input: (B, N, C)
                B, N, C_total = qkv.shape
                C = C_total // 3
                qkv = qkv.reshape(B, N, 3, module.num_heads, C // module.num_heads).permute(2, 0, 3, 1, 4)
                q, k, _ = qkv[0], qkv[1], qkv[2]

                attn_scores = (q @ k.transpose(-2, -1)) * module.scale
                attn = attn_scores.softmax(dim=-1)  # (B, num_heads, N, N)
                cls_attn = attn[:, :, 0, 1:]
                importance = cls_attn.mean(dim=(0, 2)).mean()
                self.attn_importance_dict[layer_name].append(importance.detach().cpu())
        return attn_hook

    
    def get_grad_attn_matrix(self):
        # 转化为[任务数目 ✖️ block数量] 矩阵
        grad_matrix = []
        attn_matrix = []
        for block_name, _ in self._layers:
            # 将列表转换为张量，然后移动到CPU
            grad_tensor = torch.tensor(self.grad_dict[block_name])
            grad_matrix.append(grad_tensor.cpu().numpy())
            
            attn_tensor = torch.tensor(self.attn_importance_dict[block_name])
            attn_matrix.append(attn_tensor.cpu().numpy())
        return np.array(grad_matrix), np.array(attn_matrix)

# ------------------------
# 训练流程
# ------------------------
def save_checkpoint(model, optimizer, epoch, task_id, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    checkpoint_file = os.path.join(save_path, 'vit-base-grad-checkpoint.pth')
    torch.save({'task_id': task_id,
                     'epoch': epoch,
                     'state': model.state_dict(),
                     'optim': optimizer.state_dict(),
                     }, checkpoint_file)

    
def create_vit_model(num_classes=20, arch='vit_base_patch16_224'):
    model = timm.create_model(arch, pretrained=False, num_classes=num_classes)
    return model


def train_ansnet(model, task_loader, save_dir, task_id, epoch_per_task=30, lr=0.0001):
    model.train()
    """优化后的训练函数（关键修改点2：添加混合精度）"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # scaler = amp.GradScaler()  # 华为NPU混合精度
    
    total_steps = epoch_per_task * len(task_loader)
    step = 0
    for epoch in tqdm(range(epoch_per_task)):
        for i, (inputs, labels) in enumerate(task_loader):
            step += 1
            inputs = inputs.npu()
            labels = labels.npu()
            
            # is_last_batch = (epoch == epoch_per_task - 1) and (i == len(task_loader) - 1)
            # if grad_tracker is not None:
            #     grad_tracker.active = is_last_batch

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    save_checkpoint(model, optimizer, epoch, task_id, save_dir)
    return model

def save_grad_matrix(grad_dict, attn_importance_dict, save_path):
    """每个任务都保存一次grad和attn重要性信息"""
    grad_save_path = os.path.join(save_path, "grad_matrix_vit.txt")
    attn_importance_dict_save_path = os.path.join(save_path, "attn_importance_matrix_vit.txt")

    with open(grad_save_path, 'a') as f:
        np.savetxt(f, grad_dict.reshape(1, -1), fmt="%.6f", delimiter=",")
        f.write("\n")  # 每个任务的梯度换一行
    
    with open(attn_importance_dict_save_path, 'a') as f:
        np.savetxt(f, attn_importance_dict.reshape(1, -1), fmt="%.6f", delimiter=",")
        f.write("\n")  # 每个任务的梯度换一行

def parse_args():
    parser = argparse.ArgumentParser()

    # 任务相关参数
    parser.add_argument('--num-tasks', type=int, default=50, help='Number of tasks')
    parser.add_argument('--class-per-task', type=int, default=20, help='Class per task')
    # 训练相关参数
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-epochs-per-task', type=int, default=30)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--save-path', type=str, default='./results/Grad-vit-checkpoints_last_iter/')
    # 数据集相关参数
    parser.add_argument('--dataset-root', type=str, default='./imagenet')
    # 模型相关参数
    parser.add_argument('--arch', type=str, choices=['vit-small', 'vit-base'], default='vit-base')
    parser.add_argument('--resume', type=str, default=None)

    args = parser.parse_args()
    return args

# ------------------------
# 主流程优化
# ------------------------
def main():
    args = parse_args()
    args.save_path = os.path.join(args.save_path, args.arch)
    os.makedirs(args.save_path, exist_ok=True)
    start_task = 0
    if args.resume and os.path.isfile(args.resume):
        print(f"Resuming model from {args.resume}")
        checkpoint = torch.load(args.resume, map_location='npu')
        model.load_state_dict(checkpoint['model'])
        start_task = checkpoint['task_id']+1
        
    for task_id in range(start_task, args.num_tasks):
        print(f"\nTask {task_id}/{args.num_tasks}")

        # 当前任务的数据路径
        task_folder = os.path.join(args.dataset_root, f"task_{task_id:02d}")
        assert os.path.exists(task_folder), f"Task folder not found: {task_folder}"

        # 加载当前任务数据
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        task_dataset = ImageFolder(task_folder, transform=transform)
        # print(task_dataset.class_to_idx)

        # 构建 DataLoader
        loader = DataLoader(
            task_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=min(8, cpu_count()),
            pin_memory=True,
            persistent_workers=True
        )

        # 模型初始化或继续训练
        if task_id == 0:
            model = create_vit_model(num_classes=args.class_per_task, arch='vit_base_patch16_224').npu()
        print(model)
        
        # grad_tracker = ViTBlockGradTracker(model)
        # model = train_ansnet(model, loader, args.save_path, task_id, grad_tracker=grad_tracker)
        tracker = ViTBlockGradTracker(model)
        trained_model = train_ansnet(model, loader, save_dir=args.save_path, task_id=task_id)
        
        # 执行虚拟前向传播以捕获最终梯度
        sample, _ = next(iter(loader))
        sample = sample.cuda()
        output = trained_model(sample)
        loss = output.mean() # 虚拟损失
        loss.backward()
        
        grad, attn = tracker.get_grad_attn_matrix()
        save_grad_matrix(grad[:, -1], attn[:, -1], args.save_path)


if __name__ == '__main__':
    main()