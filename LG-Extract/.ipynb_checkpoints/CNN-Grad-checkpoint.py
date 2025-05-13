"""
基于梯度信息的CNN架构(ResNet)的祖先模型的学习基因的抽取
"""
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
# 数据集划分
# ------------------------
class TaskSpliter:
    """将ImageNet划分为50个子任务，每个子任务20个类别，不重复"""
    def __init__(self, num_classes=1000, num_tasks=50, class_per_task=20):
        # 加载数据集类别
        self.classes = np.arange(num_classes)

        # 随机划分任务
        np.random.seed(1999)
        shuffled_classes = np.random.permutation(self.classes)
        self.tasks_classes = shuffled_classes.reshape(num_tasks, class_per_task)

    def get_task_indices(self, task_id):
        """获取指定任务中的类别"""
        tasks_classes = self.tasks_classes[task_id]
        mask = np.isin(self.classes, tasks_classes)
        return np.where(mask)[0]
    

# ------------------------
# ResNet Block梯度追踪模块
# ------------------------
class BlockGradTracker:
    def __init__(self, model):
        self.model = model
        self.gradient_dict = defaultdict(list)
        self._blocks = []

        self._find_blocks(self.model)

        # 为每个block注册钩子
        self.handles = []
        for block_name, block in self._blocks:
            handle = block.register_backward_hook(self._make_hook(block_name))
            self.handles.append(handle)

    def _find_blocks(self, module, name=""):
        for child_name, child in module.named_children():
            if isinstance(child, (nn.ModuleList, nn.Sequential)):
                self._find_blocks(child, f"{name}.{child_name}" if name else child_name)
            elif isinstance(child, (BasicBlock, Bottleneck)):
                self._blocks.append((f"{name}.{child_name}" if name else child_name, child))
    

    def _make_hook(self, block_name):
        def hook(module, grad_input, grad_output):
            # 计算梯度L2范数
            total_norm = 0.0
            for p in module.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item()
            self.gradient_dict[block_name].append(total_norm / len(list(module.parameters())))
        return hook
    
    def get_gradient_matrix(self):
        # 转化为[任务数目 ✖️ block数量] 矩阵
        matrix = []
        for block_name, _ in self._blocks:
            matrix.append(self.gradient_dict[block_name])
        return np.array(matrix)
    
    def __del__(self):
        # 移除钩子
        for handle in self.handles:
            handle.remove()

# ------------------------
# 训练流程
# ------------------------
def save_checkpint(model, optimizer, task_id, epoch, save_dir):
    checkpoint = {
        'task_id': task_id,
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(checkpoint, os.path.join(save_dir, "checkpoint.pth"))

def train_ansnet(model, task_loaders, save_dir, task_id, num_epochs=30, lr=0.0001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    
    model.train()
    for epoch  in tqdm(range(num_epochs)):
        for inputs, labels in (task_loaders):
            inputs, labels = inputs.cuda(), labels.cuda()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    save_checkpint(model, optimizer, task_id, epoch, save_dir)
    return model

# ------------------------
# 保存gradient_matrix函数
# ------------------------
def save_gradient_matrix(gradient_matrix, save_path):
    # 确保保存路径存在
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    file_path = os.path.join(save_path, "gradient_matrix.txt")

    # 以追加模式写入每个任务的梯度
    with open(file_path, "a") as f:
        np.savetxt(f, gradient_matrix.reshape(1, -1), fmt="%.6f", delimiter=",")
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
    parser.add_argument('--save-path', type=str, default='./results/Grad-cnn-checkpoints/')
    # 数据集相关参数
    parser.add_argument('--dataset-root', type=str, default='./imagenet')
    # 模型相关参数
    parser.add_argument('--arch', type=str, choices=['resnet34', 'resnet50'], default='resnet34')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    args.save_path = os.path.join(args.save_path, args.arch)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # 初始化模型
    if args.arch == 'resnet34':
        model = resnet34(pretrained=False)
    elif args.arch == 'resnet50':
        model = resnet50(pretrained=False)
    else:
        raise ValueError(f"Unknown architecture {args.arch}")
    model.fc = torch.nn.Linear(model.fc.in_features, args.class_per_task)  # 修改分类器
    model = model.cuda()
    
    # 数据集预处理
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 记录任务开始时间
    start_time = time.time()
    
    # 加载数据集
    dataset = ImageFolder(root=args.dataset_root, transform=transform)
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

    end_time = time.time()
    # 计算并打印总时长
    elapsed_time = end_time - start_time
    print(f"数据处理时间开销: ({elapsed_time/60:.2f} minutes)")

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
        
        tracker = BlockGradTracker(model)
        trained_model = train_ansnet(model, task_loaders, save_dir=args.save_path, task_id=task_id, num_epochs=args.num_epochs_per_task, lr=args.lr)
        
        # 执行虚拟前向传播以捕获最终梯度
        sample, _ = next(iter(task_loaders))
        sample = sample.cuda()
        output = trained_model(sample)
        loss = output.mean() # 虚拟损失
        loss.backward()

        # 记录梯度信息
        task_gradients = tracker.get_gradient_matrix()[:, -1] # 取最后一次反向传播的梯度
        gradient_matrix = np.array(task_gradients)
        save_gradient_matrix(gradient_matrix, args.save_path)

        model = trained_model