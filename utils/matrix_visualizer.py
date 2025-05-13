import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter  # 新增：用于趋势平滑

# 读取并转换矩阵
def read_matrix(file_path):
    matrix = []
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
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
    plt.imshow(matrix, cmap='YlGnBu', aspect='auto')  # 使用冷色系Blues渐变
    # plt.colorbar(label='Intensity (Gamma Adjusted)')  # 颜色条
    
    # 设置标题和标签
    # plt.title('Gradient Matrix with Enhanced Contrast', fontsize=16, fontweight='bold')
    plt.xlabel('Tasks', fontsize=14)
    plt.ylabel('Layers', fontsize=14)
    
    # 显示完整坐标
    plt.xticks(ticks=np.arange(0, matrix.shape[1]), labels=np.arange(1, matrix.shape[1] + 1), rotation=90)
    plt.yticks(ticks=np.arange(0, matrix.shape[0]), labels=np.arange(1, matrix.shape[0] + 1))
    
    # 美观调整
    plt.grid(False)
    plt.tight_layout()
    plt.gca().set_aspect('equal')
    plt.savefig('/home/shiboyu/code/02-NCP/results/Grad-cnn-checkpoints/resnet34/matrix_visualizer_gamma.png', dpi=300)

def plot_chart(matrix):
    matrix = matrix.T*100 # 转置，横坐标为tasks，纵坐标为features
    # 绘制每个 block 的梯度变化曲线
    plt.figure(figsize=(10, 5))
    for i in range(16):
        plt.plot(range(50), matrix[i, :], label=f'Block {i + 1}')

    # 设置标题和标签
    # plt.title('Gradient Matrix with Enhanced Contrast', fontsize=16, fontweight='bold')
    plt.xlabel('Task', fontsize=14)
    plt.ylabel('Grad Norm', fontsize=14)
    
    # 显示完整坐标
    plt.xticks(ticks=np.arange(0, matrix.shape[1]), labels=np.arange(1, matrix.shape[1] + 1), rotation=90)
    
    # 美观调整
    plt.grid(False)
    plt.tight_layout()
    plt.savefig('/home/shiboyu/code/02-NCP/results/Grad-cnn-checkpoints/resnet34/matrix_visualizer_trend.png', dpi=300)

def plot_trend(matrix):
    matrix = matrix.T * 100  # 转置并缩放
    
    plt.figure(figsize=(20, 8))  # 更宽的画布
    
    # 为每个block计算趋势线（使用Savitzky-Golay滤波器）
    window_size = 9  # 滑动窗口大小（奇数）
    poly_order = 3   # 多项式阶数
    
    for i in range(16):
        raw_data = matrix[i, :]
        # 计算趋势线（关键修改）
        trend = savgol_filter(raw_data, window_size, poly_order)
        
        # 绘制原始数据（半透明细线）
        plt.plot(range(50), raw_data, 
                 color=plt.cm.tab20(i), alpha=0.15, linewidth=0.8)
        # 绘制趋势线（突出显示）
        plt.plot(range(50), trend, 
                 label=f'Block {i+1}', 
                 color=plt.cm.tab20(i), linewidth=2.5)

    # 坐标轴优化
    plt.xlabel('Task', fontsize=14)
    plt.ylabel('Grad Norm (Normalized)', fontsize=14)
    plt.xticks(ticks=np.arange(0, 50, 5),  # 降低横坐标密度
               labels=np.arange(1, 51, 5), 
               rotation=45)
    
    # 添加趋势说明
    plt.text(0.02, 0.95, 
             'Trend Lines (Savitzky-Golay Filter)\nWindow=9, Polynomial=3',
             transform=plt.gca().transAxes,
             fontsize=10, alpha=0.8,
             bbox=dict(facecolor='white', alpha=0.7))
    
    # 图例优化
    plt.legend(ncol=8, loc='upper center', 
              bbox_to_anchor=(0.5, 1),
              frameon=True, 
              facecolor='white')
    
    plt.grid(axis='y', alpha=0.3)  # 添加横向网格线
    plt.tight_layout()
    plt.savefig('/home/shiboyu/code/02-NCP/results/Grad-cnn-checkpoints/resnet50/matrix_visualizer_trend.png', dpi=300, bbox_inches='tight')

# 主程序
file_path = '/home/shiboyu/code/02-NCP/results/Grad-cnn-checkpoints/resnet50/resnet50-grad-gradient_matrix.txt'  # 替换为你的实际路径
matrix = read_matrix(file_path)

# 使用 gamma=0.5 提升小值对比
gamma_value = 0.5  # 你可以尝试 0.4, 0.3 进一步提升小值
transformed_matrix = power_transform(matrix, gamma=gamma_value)

# plot_heatmap(transformed_matrix)
plot_trend(transformed_matrix)
