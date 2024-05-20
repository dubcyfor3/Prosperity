import math
import torch
def ceil_a_by_b(a, b):
    return int(math.ceil(float(a) / b))

def get_density(tensor: torch.Tensor):
    return float(torch.sum(tensor != 0).item()) / torch.numel(tensor)

import torch

def img2col(input_tensor, kernel_size, stride=1, padding=0):
    """
    Applies the img2col transformation to an input tensor.
    
    Args:
    input_tensor (torch.Tensor): Input tensor of shape (batch_size, C, H, W)
    kernel_size (int): Size of the kernel
    stride (int): Stride of the convolution
    padding (int): Padding added to all four sides of the input tensor
    
    Returns:
    torch.Tensor: Output column tensor
    """
    # Add padding to the input tensor
    if padding > 0:
        input_tensor = torch.nn.functional.pad(input_tensor, (padding, padding, padding, padding))
    
    # Unfold the tensor along height and width
    patches = input_tensor.unfold(2, kernel_size, stride).unfold(3, kernel_size, stride)
    
    # Reshape to form the column tensor
    output = patches.permute(0, 2, 3, 1, 4, 5).reshape(input_tensor.shape[0], -1, kernel_size*kernel_size*input_tensor.shape[1])
    return output


if __name__ == '__main__':
    # create a tensor of shape [4, 48, 32, 32]
    input_tensor = torch.rand(4, 48, 32, 32)
    kernel_size = 3
    stride = 1
    padding = 1
    column_tensor = img2col(input_tensor, kernel_size, stride, padding)