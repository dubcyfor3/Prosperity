from networks import FC, Conv2D, create_network, conv2d_2_fc
import torch
import matplotlib.pyplot as plt

from utils import get_density


def product_sparsify_whole_matrix(input_tensor: torch.Tensor):
    input_tensor = input_tensor.reshape(-1, input_tensor.shape[-1])
    tile_size_k = 16
    tile_size_m = 256
    sparsified_tensor = torch.zeros_like(input_tensor)


    for m_id in range(0, input_tensor.shape[0], tile_size_m):
        for k_id in range(0, input_tensor.shape[1], tile_size_k):
            
            cur_tile_size_m = min(tile_size_m, input_tensor.shape[0] - m_id)
            cur_tile_siz_k = min(tile_size_k, input_tensor.shape[1] - k_id)
            
            tile = input_tensor[m_id:m_id+cur_tile_size_m, k_id:k_id+cur_tile_siz_k]

            sparsified_tile = tile.clone()
            for i in range(tile.shape[0]):
                cur_row = tile[i]
                nnz = torch.sum(cur_row != 0).item()
                if nnz < 2:
                    continue

                and_result = torch.logical_and(cur_row, tile)
                equalities = torch.eq(and_result, tile)
                is_subset = torch.all(equalities, dim=-1)

                equalities = torch.eq(cur_row, tile)
                is_equal = torch.all(equalities, dim=-1)
                is_bigger_index = torch.arange(tile.shape[0]) >= i

                is_excluded = torch.logical_and(is_equal, is_bigger_index)
                is_real_subset = torch.logical_and(is_subset, ~is_excluded)

                if torch.sum(is_real_subset) == 0:
                    continue


                subset_row = tile[is_real_subset]
                subset_row_nnz = torch.sum(subset_row != 0, dim=-1)
                max_subset_size = torch.max(subset_row_nnz).item()
                max_subset = subset_row[torch.argmax(subset_row_nnz)]
                # if max_subset_size > 1: # can also reuse even when the size is 1
                sparsified_tile[i] = torch.logical_xor(sparsified_tile[i], max_subset)

            sparsified_tensor[m_id:m_id+cur_tile_size_m, k_id:k_id+cur_tile_siz_k] = sparsified_tile

        return sparsified_tensor

def plot_matrix(matrix: torch.Tensor, filename):

    # use seaborn plot
    cmap = plt.cm.colors.ListedColormap(["white", "#ADD8E6"])

    # Define the bounds and normalization for the colors
    bounds = [0, 0.5, 1]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

    # save the matrix
    plt.imshow(matrix, cmap=cmap, norm=norm, aspect='equal')


    # Add a grid with thin lines
    plt.grid(which='major', color='black', linestyle='-', linewidth=0.5)

    # Turn off the axis labels
    plt.xticks([])
    plt.yticks([])
    # store as pdf and transparent
    plt.savefig('{}'.format(filename), transparent=True)
    return


if __name__ == '__main__':

    
    model = 'spikformer'
    dataset = 'cifar100'

    nn = create_network(model, 'data/{}_{}.pkl'.format(model, dataset))
    layer_id = 46
    layer = nn[layer_id]
    if isinstance(layer, Conv2D):
        layer = conv2d_2_fc(layer)
    elif isinstance(layer, FC):
        pass
    else:
        raise ValueError("This layer is not supported")
    tensor = layer.activation_tensor.sparse_map

    tensor = tensor.reshape(-1, tensor.shape[-1])
    prosparsity_tensor = product_sparsify_whole_matrix(tensor)

    # select submatrix and visualize
    length = 64
    tensor = tensor[0:length, 0:length]
    prosparsity_tensor = prosparsity_tensor[0:length, 0:length]
    plot_matrix(tensor, '{}_bit_sparsity.png'.format(model))
    print('bit density: ', get_density(tensor))
    plot_matrix(prosparsity_tensor, '{}_product_sparsity.png'.format(model))
    print('product density: ', get_density(prosparsity_tensor))
