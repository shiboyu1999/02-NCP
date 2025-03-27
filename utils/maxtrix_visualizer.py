import numpy as np
import matplotlib.pyplot as plt

# 读取并转换矩阵
def read_matrix(file_path):
    matrix = []
    with open(file_path, 'r') as file:
        for line in file:
            row = list(map(float, line.strip().split(',')))
            matrix.append(row)
    return np.array(matrix)

# 线性归一化到 [0, 1]
def normalize(matrix):
    min_val = np.min(matrix)
    max_val = np.max(matrix)
    if max_val - min_val == 0:
        return np.zeros_like(matrix)
    return (matrix - min_val) / (max_val - min_val)

# 幂次变换来增强对比度
def power_transform(matrix, gamma=0.5):
    matrix = normalize(matrix)
    return np.power(matrix, gamma)  # γ < 1 提高小值，γ > 1 提高大值

# 绘制热力图
def plot_heatmap(matrix):
    matrix = matrix.T  # 转置，横坐标为tasks，纵坐标为features
    plt.figure(figsize=(12, 6))  # 设置画布大小
    plt.imshow(matrix, cmap='Blues', aspect='auto')  # 使用冷色系Blues渐变
    plt.colorbar(label='Intensity (Gamma Adjusted)')  # 颜色条
    
    # 设置标题和标签
    plt.title('Gradient Matrix with Enhanced Contrast', fontsize=16, fontweight='bold')
    plt.xlabel('Tasks', fontsize=14)
    plt.ylabel('Features', fontsize=14)
    
    # 显示完整坐标
    plt.xticks(ticks=np.arange(0, matrix.shape[1]), labels=np.arange(1, matrix.shape[1] + 1), rotation=90)
    plt.yticks(ticks=np.arange(0, matrix.shape[0]), labels=np.arange(1, matrix.shape[0] + 1))
    
    # 美观调整
    plt.grid(False)
    plt.tight_layout()
    plt.savefig('./matrix_visualizer_gamma.png', dpi=300)
    plt.show()

# 主程序
file_path = '/home/shiboyu/code/02-NCP/results/Grad-cnn-checkpoints/resnet34/gradient_matrix.txt'  # 替换为你的实际路径
matrix = read_matrix(file_path)

# 使用 gamma=0.5 提升小值对比
gamma_value = 0.5  # 你可以尝试 0.4, 0.3 进一步提升小值
transformed_matrix = power_transform(matrix, gamma=gamma_value)
plot_heatmap(transformed_matrix)
