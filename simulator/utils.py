import math
import torch

class Stats:
    def __init__(self):
        self.total_cycles = 0
        self.mem_stall_cycles = 0
        self.compute_cycles = 0
        self.num_ops = 0
        self.LIF_latency = 0
        self.preprocess_stall_cycles = 0
        self.mem_namespace = ['dram', 'g_act', 'g_wgt', 'g_psum']
        self.reads = {space: 0 for space in self.mem_namespace}
        self.writes = {space: 0 for space in self.mem_namespace}
        self.bit_density = 0
        self.product_density = 0
        self.ops_sparsity = 0
        self.rank_two_sparsity = 0
        self.avg_rank_one_prefix = 0
        self.avg_rank_two_prefix = 0
        self.cycle_breakdown = None

    def __add__(self, other):
        if not isinstance(other, Stats):
            raise Exception("unsupported type")
        else:
            added_stats = Stats()
            added_stats.total_cycles = self.total_cycles # handle cycles manually
            added_stats.mem_stall_cycles = self.mem_stall_cycles + other.mem_stall_cycles
            added_stats.compute_cycles = self.compute_cycles + other.compute_cycles
            added_stats.preprocess_stall_cycles = self.preprocess_stall_cycles + other.preprocess_stall_cycles
            added_stats.num_ops = self.num_ops + other.num_ops
            added_stats.LIF_latency = 0
            added_stats.mem_namespace = self.mem_namespace
            added_stats.reads = {space: self.reads[space] + other.reads[space] for space in self.mem_namespace}
            added_stats.writes = {space: self.writes[space] + other.writes[space] for space in self.mem_namespace}
            
            return added_stats

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
    torch.Tensor: Output column tensor of shape (batch_size, num_patches, C*kernel_size*kernel_size)
    """
    # Add padding to the input tensor
    if padding > 0:
        input_tensor = torch.nn.functional.pad(input_tensor, (padding, padding, padding, padding))
    
    # Unfold the tensor along height and width
    patches = input_tensor.unfold(2, kernel_size, stride).unfold(3, kernel_size, stride)
    
    # Reshape to form the column tensor
    output = patches.permute(0, 2, 3, 1, 4, 5).reshape(input_tensor.shape[0], -1, kernel_size*kernel_size*input_tensor.shape[1]).contiguous()
    return output

def pad_to_power_of_2(tensor, dim):
    """
    Pad a specific dimension of a tensor to make it a multiple of the nearest power of 2.
    
    Args:
    tensor (torch.Tensor): Input tensor
    dim (int): Dimension to pad
    
    Returns:
    torch.Tensor: Padded tensor
    """
    size = tensor.size(dim)
    nearest_power_of_2 = 2 ** math.ceil(math.log2(size))
    pad_size = nearest_power_of_2 - size
    
    if pad_size == 0:
        return tensor
    
    pad_shape = [0] * (2 * tensor.dim())
    pad_shape[-2 * dim - 1] = pad_size
    
    return torch.nn.functional.pad(tensor, pad_shape)

import pandas as pd

def read_position(file_name, column_name, row_name):
    try:
        # Read the CSV file
        df = pd.read_csv(file_name)
        
        # Check if column exists
        if column_name not in df.columns:
            return None
            
        # Check if row exists (assuming row_name is in first column)
        row_idx = df.iloc[:, 0][df.iloc[:, 0] == row_name].index
        if len(row_idx) == 0:
            return None
            
        # Get the value at the intersection
        value = df.at[row_idx[0], column_name]
        
        # Return None if value is NaN or empty
        if pd.isna(value) or value == '':
            return None
            
        return value
        
    except FileNotFoundError:
        return None
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None

def write_position(file_name, column_name, row_name, data):
    # Read existing CSV file into a pandas DataFrame
    df = pd.read_csv(file_name)
    
    # Handle the case where the DataFrame is empty (no rows, only headers)
    if df.empty:
        # Create a new row initialized with the row_name and -1.0 for other columns
        df.loc[0] = [row_name] + [-1.0] * (len(df.columns) - 1)
    else:
        # If the row_name doesn't exist, append it
        if row_name not in df.iloc[:, 0].values:
            new_row = pd.DataFrame([[row_name] + [-1.0] * (len(df.columns) - 1)], 
                                   columns=df.columns)
            df = pd.concat([df, new_row], ignore_index=True)
    
    # If column_name doesn't exist, add it
    if column_name not in df.columns:
        df[column_name] = -1.0
    
    # Update the specific cell
    row_idx = df.iloc[:, 0][df.iloc[:, 0] == row_name].index[0]
    df.at[row_idx, column_name] = data
    
    # Write back to CSV
    df.to_csv(file_name, index=False)

def write_title(file_name, title):
    # create a new file and write the title in csv format
    with open(file_name, 'w') as f:
        for i in range(len(title) - 1):
            f.write(title[i] + ',')
        f.write(title[-1] + '\n')

    



if __name__ == '__main__':
    # create a tensor of shape [4, 48, 32, 32]
    # input_tensor = torch.rand(4, 48, 32, 32)
    # kernel_size = 3
    # stride = 1
    # padding = 1
    # column_tensor = img2col(input_tensor, kernel_size, stride, padding)

    model_list = ['vgg', 'resnet']
    write_title('artifact_eval/test.csv', ['model_name'] + model_list)
