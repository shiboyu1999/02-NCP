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
            grad_handle = layer.register_backward_hook(self._make_grad_hook(layer_name))
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
    
    def _get_metrics(self):
        """返回梯度稳定性与注意力重要性矩阵"""
        return self.grad_dict, self.attn_importance_dict

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
    
def train_ansnet(model, task_loader, save_path, task_id, epoch_per_task=30, lr=0.0001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epoch_per_task):
        for idx, input, label in enumerate(task_loader):
            input = input.cuda()
            label = label.cuda()

            output = model(input)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    save_checkpoint(model, optimizer, epoch, task_id, save_path)












