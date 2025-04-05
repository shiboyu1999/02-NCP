"""
基于梯度方差信息和注意力重要性的ViT架构的学习基因提取
"""

import timm 
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageNet, ImageFolder
from torchvision import transforms
from torchvision.models import resnet34, resnet50
from torchvision.models.resnet import BasicBlock, Bottleneck
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import argparse
import time

# ------------------------
# ViT Block梯度追踪模块
# ------------------------
class ViTBlockGradTracker:
    def __init__(self, model):
        self.model = model
        self.grad_dict = defaultdict(list)
        self.attn_importance_dict = defaultdict(list)
        self._layers = []

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
            total_norm = 0.0
            for param in module.parameters():
                if param.grad is not None:
                    total_norm += param.grad.norm(2).item()
            self.grad_dict[layer_name].append(total_norm)
        return grad_hook
    
    def _make_attn_hook(self, layer_name):
        def attn_hook(module, input, output):
            # 计算每个注意力头的贡献程度
            attn = output[1]
            # 计算每个注意力头对分类的贡献程度
            cls_attn = attn[:, :, 0, 1:]
            importance = cls_attn.mean(dim=(0,2)).mean()  # 指定层所有head的平均重要性
            self.attn_importance_dict[layer_name].append(importance.detach())
        return attn_hook
    
    def get_grad_attn_matrix(self):
        # 转化为[任务数目 ✖️ block数量] 矩阵
        grad_matrix = []
        attn_matrix = []
        for block_name, _ in self._layers:
            grad_matrix.append(self.grad_dict[block_name])
            attn_matrix.append(self.attn_importance_dict[block_name])
        return np.array(grad_matrix), np.array(attn_matrix)

# ------------------------
# ViT 模型定义
# ------------------------
def create_vit_model(num_classes=20, arch='vit_base_patch16_224'):
    model = timm.create_model(arch, pretrained=False, num_classes=num_classes)
    return model

# ------------------------
# 训练流程
# ------------------------
def save_checkpoint(model, optimizer, epoch, task_id, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save({'task_id': task_id,
                     'epoch': epoch,
                     'state': model.state_dict(),
                     'optim': optimizer.state_dict(),
                     }, save_path)
    
def train_ansnet(model, task_loader, save_dir, task_id, epoch_per_task=30, lr=0.0001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in tqdm(range(epoch_per_task)):
        for input, label in (task_loaders):
            input = input.cuda()
            label = label.cuda()

            output = model(input)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    save_checkpoint(model, optimizer, epoch, task_id, save_dir)

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
    parser.add_argument('--save-path', type=str, default='./results/Grad-vit-checkpoints/')
    # 数据集相关参数
    parser.add_argument('--dataset-root', type=str, default='./imagenet')
    # 模型相关参数
    parser.add_argument('--arch', type=str, choices=['vit-small', 'vit-base'], default='vit-base')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    args.save_path = os.path.join(args.save_path, args.arch)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # 初始化模型
    if args.arch == 'vit-base':
        model = create_vit_model(num_classes=args.class_per_task, arch='vit_base_patch16_224')
    elif args.arch == 'vit-small':
        model = create_vit_model(num_classes=args.class_per_task, arch='vit_base_patch16_224')
    else:
        raise ValueError(f"Unknown architecture {args.arch}")
    model.cuda()

    # 初始化数据集
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 加载数据集
    dataset = ImageFolder(root=args.dataset_root, transform=train_transform)
    class_to_idx = dataset.class_to_idx  # {类标签: 类索引}
    classes = dataset.classes  # 类别名称列表

    # 预先计算每个类标签的样本索引
    class_indices = {class_label: [] for class_label in class_to_idx.keys()}
    for idx, (data, label) in enumerate(dataset):
        class_label = classes[label]
        class_indices[class_label].append(idx)

    # 创建任务拆分器
    num_classes = len(classes)
    num_tasks = args.num_tasks
    class_per_task = args.class_per_task

    # 记录最终的梯度矩阵
    final_grad_matrix = []

    # 开始训练
    for task_id in range(0, args.num_tasks):
        print(f"#####################Task {task_id}#########################")
        # 获取当前任务的类标签
        task_classes = classes[task_id * class_per_task:(task_id + 1) * class_per_task]
        print(f"Task {task_id} classes: {task_classes}")

        # 获取当前任务对应的索引
        task_indices = []
        for class_label in task_classes:
            task_indices.extend(class_indices[class_label])  # 获取所有类标签对应的样本索引

        # 根据任务的样本索引创建子集
        # 创建 Subset 数据集
        subset_dataset = Subset(dataset, task_indices)

        # **重新映射标签** (从全局标签映射到 0 到 class_per_task-1)
        def map_labels(dataset_subset):
            # 获取类标签的映射表
            label_map = {class_to_idx[cls_name]: i for i, cls_name in enumerate(task_classes)}
            # 对数据的标签进行重新映射
            for i in range(len(dataset_subset)):
                img, label = dataset_subset[i]
                new_label = label_map[label]
                dataset_subset.dataset.samples[dataset_subset.indices[i]] = (dataset_subset.dataset.samples[dataset_subset.indices[i]][0], new_label)
        
        # 映射标签
        map_labels(subset_dataset)

        task_loaders = DataLoader(subset_dataset, 
                                 batch_size=args.batch_size,
                                 shuffle=True,
                                 drop_last=True,
                                 num_workers=args.num_workers,
                                 pin_memory=torch.cuda.is_available())
        print(f"Task {task_id} data loader length: {len(task_loaders)}")
        
        tracker = ViTBlockGradTracker(model)
        trained_model = train_ansnet(model, task_loaders, save_dir=args.save_path, task_id=task_id, epoch_per_task=args.num_epochs_per_task, lr=args.lr)
        
        # 执行虚拟前向传播以捕获最终梯度
        sample, _ = next(iter(task_loaders))
        sample = sample.cuda()
        output = trained_model(sample)
        loss = output.mean() # 虚拟损失
        loss.backward()

        # 记录梯度信息
        task_gradients, task_attn_importance = tracker.get_grad_attn_matrix()
        save_grad_matrix(task_gradients[:, -1], task_attn_importance[:, -1], save_path=args.save_path)

        model = trained_model











