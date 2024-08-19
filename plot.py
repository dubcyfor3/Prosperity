import torch
import matplotlib.pyplot as plt
from networks import FC, Conv2D, MaxPool2D, LIFNeuron, Attention, create_network, conv2d_2_fc


def plot_matrix(matrix: torch.Tensor, filename):
    # 使用seaborn绘图
    cmap = plt.cm.colors.ListedColormap(["white", "#ADD8E6"])

    # 定义颜色的边界和归一化
    bounds = [0, 0.5, 1]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

    # 显示矩阵
    plt.imshow(matrix, cmap=cmap, norm=norm, aspect='equal')

    # 添加网格线
    plt.grid(which='major', color='black', linestyle='-', linewidth=0.5)

    # 关闭坐标轴标签
    plt.xticks([])
    plt.yticks([])

    # 保存为PDF格式且背景透明
    plt.savefig('{}'.format(filename), transparent=True)
    plt.close()

if __name__ == '__main__':
    
    model = 'spikformer'
    dataset_train = 'cifar10_train'
    dataset_test = 'cifar10_test'

    nn_train = create_network(model, 'data/{}_{}.pkl'.format(model, dataset_train))
    nn_test = create_network(model, 'data/{}_{}.pkl'.format(model, dataset_test))

    # 提取第一层和第19层的激活矩阵并绘制，第一个图片对应的矩阵

    # 第一层
    for network, label in [(nn_train, 'train'), (nn_test, 'test')]:
        first_layer = network[0]
        if isinstance(first_layer, Conv2D):
            first_layer = conv2d_2_fc(first_layer)
        tensor_first_layer = first_layer.activation_tensor.sparse_map
        # 选择第一个样本的激活矩阵
        tensor_first_layer = tensor_first_layer[0].reshape(-1, tensor_first_layer.shape[-1])
        plot_matrix(tensor_first_layer, f'first_layer_{label}.pdf')

    # 第19层
    for network, label in [(nn_train, 'train'), (nn_test, 'test')]:
        # 假设网络有至少19层
        nineteenth_layer = network[18]  # 第19层的索引是18
        if isinstance(nineteenth_layer, Conv2D):
            nineteenth_layer = conv2d_2_fc(nineteenth_layer)
        tensor_nineteenth_layer = nineteenth_layer.activation_tensor.sparse_map
        # 选择第一个样本的激活矩阵
        tensor_nineteenth_layer = tensor_nineteenth_layer[0].reshape(-1, tensor_nineteenth_layer.shape[-1])
        plot_matrix(tensor_nineteenth_layer, f'nineteenth_layer_{label}.pdf')