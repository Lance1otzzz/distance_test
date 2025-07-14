# analyze_distances.py

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

def load_nodes(dataset_dir: str) -> np.ndarray:
    """
    从指定的数据集目录加载 nodes.txt 文件。

    Args:
        dataset_dir (str): 包含 nodes.txt 的目录路径。

    Returns:
        np.ndarray: 一个 NumPy 数组，每行是一个高维向量。

    Raises:
        FileNotFoundError: 如果 nodes.txt 文件不存在。
    """
    nodes_path = os.path.join(dataset_dir, 'nodes.txt')
    print(f"Loading data from: {nodes_path}")

    if not os.path.exists(nodes_path):
        raise FileNotFoundError(f"Error: nodes.txt not found in '{dataset_dir}'")

    # 使用 numpy.loadtxt 可以高效地加载纯数字文件
    data = np.loadtxt(nodes_path, dtype=np.float32)
    print(f"Successfully loaded {data.shape[0]} points with {data.shape[1]} dimensions.")
    return data

def sample_distances(data: np.ndarray, num_samples: int) -> np.ndarray:
    """
    从数据集中随机抽取点对，并计算它们之间的欧氏距离。

    Args:
        data (np.ndarray): 包含所有点的 NumPy 数组。
        num_samples (int): 要抽样的点对数量。

    Returns:
        np.ndarray: 包含所有计算出的距离的 NumPy 数组。
    """
    num_points = data.shape[0]
    if num_points < 2:
        raise ValueError("Cannot sample pairs from a dataset with less than 2 points.")

    distances = []
    print(f"Calculating distances for {num_samples} random pairs...")

    # 随机生成所有需要的点对索引
    # replace=True 允许一个点被多次选中，这在抽样数量远大于点数时是可接受的
    indices1 = np.random.randint(0, num_points, size=num_samples)
    indices2 = np.random.randint(0, num_points, size=num_samples)

    # 过滤掉一个点和自身配对的情况
    valid_pairs = indices1 != indices2
    indices1 = indices1[valid_pairs]
    indices2 = indices2[valid_pairs]

    # 使用 tqdm 创建一个进度条
    for i in tqdm(range(len(indices1)), desc="Calculating distances"):
        p1 = data[indices1[i]]
        p2 = data[indices2[i]]
        # np.linalg.norm 是计算欧氏距离的高效方法
        dist = np.linalg.norm(p1 - p2)
        distances.append(dist)

    return np.array(distances)

def plot_distance_distribution(distances: np.ndarray, output_filename: str, dataset_name: str):
    """
    使用 seaborn 和 matplotlib 绘制距离的分布直方图和密度曲线。

    Args:
        distances (np.ndarray): 距离数组。
        output_filename (str): 保存图像的文件名。
        dataset_name (str): 数据集的名称，用于图表标题。
    """
    print("Generating plot...")

    # 设置绘图风格
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(12, 7))

    # 绘制直方图和核密度估计(KDE)曲线
    # kde=True 会自动添加密度曲线
    sns.histplot(distances, kde=True, bins=50, color='skyblue', stat='density')

    # 计算并标注一些统计数据
    mean_dist = np.mean(distances)
    median_dist = np.median(distances)
    std_dist = np.std(distances)

    plt.axvline(mean_dist, color='r', linestyle='--', label=f'Mean: {mean_dist:.2f}')
    plt.axvline(median_dist, color='g', linestyle='-', label=f'Median: {median_dist:.2f}')

    plt.title(f'Distribution of Pairwise Distances in "{dataset_name}" Dataset', fontsize=16)
    plt.xlabel('Euclidean Distance', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend()

    # 在图表上添加统计文本
    stats_text = (f'Samples: {len(distances)}\n'
                  f'Std Dev: {std_dist:.2f}\n'
                  f'Min: {np.min(distances):.2f}\n'
                  f'Max: {np.max(distances):.2f}')
    plt.text(0.95, 0.95, stats_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round,pad=0.5', fc='wheat', alpha=0.5))

    # 保存图像
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved successfully to '{output_filename}'")

    # (可选) 显示图像
    # plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze and plot the distribution of pairwise distances in a high-dimensional dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("dataset_dir",
                        type=str,
                        help="Path to the dataset directory (e.g., 'data/SinaNet').")

    parser.add_argument("-s", "--samples",
                        type=int,
                        default=100000,
                        help="Number of random pairs to sample for distance calculation.")

    parser.add_argument("-o", "--output",
                        type=str,
                        default="distance_distribution.png",
                        help="Filename for the output plot.")

    args = parser.parse_args()

    try:
        # 1. 加载数据
        vector_data = load_nodes(args.dataset_dir)

        # 2. 抽样并计算距离
        distances = sample_distances(vector_data, args.samples)

        # 3. 绘图
        dataset_name = os.path.basename(args.dataset_dir) # 从路径中获取数据集名称
        plot_distance_distribution(distances, args.output, dataset_name)

    except (FileNotFoundError, ValueError) as e:
        print(e)
        exit(1)