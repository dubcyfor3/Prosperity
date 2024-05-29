import logging
from collections import OrderedDict
import pickle
import torch
from utils import img2col, get_density

def spikformer_config(dataset='cifar10'):
    if dataset == 'cifar10' or dataset == 'cifar100':
        dim = 384
        batch_size = 1
        time_steps = 4
        depth = 4
        num_head = 12
        mlp_ratio = 4
        image_size = [32, 32, 16, 8]
        sequence_length = image_size[-1] * image_size[-1]
    elif dataset == 'cifar10dvs':
        dim = 256
        batch_size = 1
        time_steps = 16
        depth = 2
        num_head = 16
        mlp_ratio = 4
        image_size = [64, 32, 16, 8]
        sequence_length = image_size[-1] * image_size[-1]
    else:
        raise ValueError('Unknown dataset')

    spikformer_SPS = OrderedDict([
        ('conv2d_1', [image_size[0], dim // 8, dim // 4, 3, 1, 1, batch_size, time_steps]),
        ('lif_1', [image_size[0] * image_size[0] * dim // 4, batch_size, time_steps]),
        ('conv2d_2', [image_size[1], dim // 4, dim // 2, 3, 1, 1, batch_size, time_steps]),
        ('lif_2', [image_size[1] * image_size[1] * dim // 2, batch_size, time_steps]),
        ('maxpool2d_2', [image_size[1], dim // 2, 3, 1, 2, batch_size, time_steps]),
        ('conv2d_3', [image_size[2], dim // 2, dim, 3, 1, 1, batch_size, time_steps]),
        ('lif_3', [image_size[2] * image_size[2] * dim, batch_size, time_steps]),
        ('maxpool2d_3', [image_size[2], dim, 3, 1, 2, batch_size, time_steps]),
        ('conv2d_rpe', [image_size[3], dim, dim, 3, 1, 1, batch_size, time_steps]),
        ('lif_rpe', [image_size[3] * image_size[3] * dim, batch_size, time_steps]),
    ])
    spikformer_encoder = OrderedDict([
        ('fc_q', [dim, dim * 3, sequence_length, batch_size, time_steps]),
        ('lif_q', [dim * sequence_length * 3, batch_size, time_steps]),
        # qkv is fused
        # ('fc_k', [dim, dim, 64, batch_size, time_steps]),
        # ('lif_k', [dim * 64, batch_size, time_steps]),
        # ('fc_v', [dim, dim, 64, batch_size, time_steps]),
        # ('lif_v', [dim * 64, batch_size, time_steps]),
        ('attention', [dim, sequence_length, num_head, batch_size, time_steps]),
        ('lif_attn', [dim * sequence_length, batch_size, time_steps]),
        ('fc_o', [dim, dim, sequence_length, batch_size, time_steps]),
        ('lif_o', [dim * sequence_length, batch_size, time_steps]),
        ('fc_1', [dim, dim * mlp_ratio, sequence_length, batch_size, time_steps]),
        ('lif_1', [dim * mlp_ratio * sequence_length, batch_size, time_steps]),
        ('fc_2', [dim * mlp_ratio, dim, sequence_length, batch_size, time_steps]),
        ('lif_2', [dim * sequence_length, batch_size, time_steps]),
    ])
    spikformer = OrderedDict([(key + '_sps', value) for key, value in spikformer_SPS.items()])
    for i in range(depth):
        encoder_with_idx = OrderedDict([(key + '_enc_' + str(i), value) for key, value in spikformer_encoder.items()])
        spikformer.update(encoder_with_idx)

    return spikformer

def SDT_config(dataset):
    if dataset == 'cifar10' or dataset == 'cifar100':
        dim = 512
        batch_size = 1
        time_steps = 4
        depth = 2
        num_head = 8
        mlp_ratio = 4
        image_size = [32, 32, 16, 8]
        sequence_length = image_size[-1] * image_size[-1]
    elif dataset == 'cifar10dvs':
        dim = 512
        batch_size = 1
        time_steps = 10
        depth = 2
        num_head = 8
        mlp_ratio = 4
        image_size = [64, 32, 16, 8]
        sequence_length = image_size[-1] * image_size[-1]

        
    SDT_sps = OrderedDict([('conv2d_1', [image_size[0], dim // 8, dim // 4, 3, 1, 1, batch_size, time_steps]),
                           ('lif_1', [image_size[0] * image_size[0] * dim // 4, batch_size, time_steps]),
                           ('maxpool2d_1', [image_size[0], dim // 4, 3, 1, 2, batch_size, time_steps]),
                           ('conv2d_2', [image_size[1], dim // 4, dim // 2, 3, 1, 1, batch_size, time_steps]),
                           ('lif_2', [image_size[1] * image_size[1] * dim // 2, batch_size, time_steps]),
                            ('maxpool2d_2', [image_size[1], dim // 2, 3, 1, 2, batch_size, time_steps]),
                            ('conv2d_3', [image_size[2], dim // 2, dim, 3, 1, 1, batch_size, time_steps]),
                            ('lif_3', [image_size[2] * image_size[2] * dim, batch_size, time_steps]),
                            ('maxpool2d_3', [image_size[2], dim, 3, 1, 2, batch_size, time_steps]),
                            ('conv2d_rpe', [image_size[3], dim, dim, 3, 1, 1, batch_size, time_steps]),
                            ('lif_sc', [image_size[3] * image_size[3] * dim, batch_size, time_steps])
    ])
    SDT_encoder = OrderedDict([('fc_q', [dim, dim * 3, sequence_length, batch_size, time_steps]),
                               ('lif_q', [dim * sequence_length * 3, batch_size, time_steps]),
                               ('attention', [dim, sequence_length, num_head, batch_size, time_steps]),
                               ('fc_o', [dim, dim, sequence_length, batch_size, time_steps]),
                               ('lif_o', [dim * sequence_length, batch_size, time_steps]),
                               ('fc_1', [dim, dim * mlp_ratio, sequence_length, batch_size, time_steps]),
                               ('lif_1', [dim * mlp_ratio * sequence_length, batch_size, time_steps]),
                               ('fc_2', [dim * mlp_ratio, dim, sequence_length, batch_size, time_steps]),
                               ('lif_2', [dim * sequence_length, batch_size, time_steps]),

    ])
    SDT = OrderedDict([(key + '_sps', value) for key, value in SDT_sps.items()])
    for i in range(depth):
        encoder_with_idx = OrderedDict([(key + '_enc_' + str(i), value) for key, value in SDT_encoder.items()])
        SDT.update(encoder_with_idx)
                        
    return SDT


def vgg16_config():
    batch_size = 1
    time_steps = 4
    vgg16 = OrderedDict([
        ('conv2d_1', [32, 64, 64, 3, 1, 1, batch_size, time_steps]),
        ('lif_1', [32 * 32 * 64, batch_size, time_steps]),
        ('maxpool2d_1', [32, 64, 2, 1, 2, batch_size, time_steps]),
        ('conv2d_2', [16, 64, 128, 3, 1, 1, batch_size, time_steps]),
        ('lif_2', [16 * 16 * 128, batch_size, time_steps]),
        ('conv2d_3', [16, 128, 128, 3, 1, 1, batch_size, time_steps]),
        ('lif_3', [16 * 16 * 128, batch_size, time_steps]),
        ('maxpool2d_2', [16, 128, 2, 1, 2, batch_size, time_steps]),
        ('conv2d_4', [8, 128, 256, 3, 1, 1, batch_size, time_steps]),
        ('lif_4', [8 * 8 * 256, batch_size, time_steps]),
        ('conv2d_5', [8, 256, 256, 3, 1, 1, batch_size, time_steps]),
        ('lif_5', [8 * 8 * 256, batch_size, time_steps]),
        ('conv2d_6', [8, 256, 256, 3, 1, 1, batch_size, time_steps]),
        ('lif_6', [8 * 8 * 256, batch_size, time_steps]),
        ('conv2d_7', [8, 256, 512, 3, 1, 1, batch_size, time_steps]),
        ('lif_7', [8 * 8 * 512, batch_size, time_steps]),
        ('conv2d_8', [8, 512, 512, 3, 1, 1, batch_size, time_steps]),
        ('lif_8', [8 * 8 * 512, batch_size, time_steps]),
        ('conv2d_9', [8, 512, 512, 3, 1, 1, batch_size, time_steps]),
        ('lif_9', [8 * 8 * 512, batch_size, time_steps]),
        ('conv2d_10', [8, 512, 512, 3, 1, 1, batch_size, time_steps]),
        ('lif_10', [8 * 8 * 512, batch_size, time_steps]),
        ('conv2d_11', [8, 512, 512, 3, 1, 1, batch_size, time_steps]),
        ('lif_11', [8 * 8 * 512, batch_size, time_steps]),
        ('conv2d_12', [8, 512, 512, 3, 1, 1, batch_size, time_steps]),
        ('lif_12', [8 * 8 * 512, batch_size, time_steps]),
        ('fc_0', [8 * 8 * 512, 4096, 1, batch_size, time_steps]),
        ('lif_13', [4096, batch_size, time_steps]),
        ('fc_1', [4096, 4096, 1, batch_size, time_steps]),
        ('lif_14', [4096, batch_size, time_steps]),
        # ('fc_2', [4096, 10, 1, batch_size, time_steps]),
        # ('lif_15', [10, batch_size, time_steps]),
    ])
    return vgg16

def compute_num_OPS(nn):
    total_ops = 0
    for op in nn:
        if isinstance(op, FC):
            M = op.batch_size * op.time_steps * op.sequence_length
            K = op.input_dim
            N = op.output_dim
            density = get_density(op.activation_tensor.sparse_map)
            total_ops += int(M * K * N * density)
        elif isinstance(op, Conv2D):
            eq_op = conv2d_2_fc(op)
            M = eq_op.batch_size * eq_op.time_steps * eq_op.sequence_length
            K = eq_op.input_dim
            N = eq_op.output_dim
            density = get_density(eq_op.activation_tensor.sparse_map)
            total_ops += int(M * K * N * density)
    
    return total_ops

class Tensor:
    def __init__(self, shape, dtype, sparse=False):
        self.is_activation = True
        self.sparse = True if dtype == 'spike' else sparse
        self.shape = shape
        self.dtype = dtype
        self.nbits = 8 if dtype == 'fp8' else 1
        self.sparse_map = None

    def get_size(self):
        return torch.tensor(self.shape).prod().item() * self.nbits

class FC:
    def __init__(self, name, input_dim, output_dim, sequence_length, batch_size, time_steps):
        self.name = name
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.activation_tensor = Tensor([batch_size, time_steps, sequence_length, input_dim], 'spike', sparse=True)
        self.weight_tensor = Tensor([input_dim, output_dim], 'fp8', sparse=False)
        self.output_tensor = Tensor([batch_size, time_steps, sequence_length, output_dim], 'fp8', sparse=False)

class Conv2D:
    def __init__(self, name, input_H, input_channel, output_channel, kernel_size, padding, stride, batch_size, time_steps, img2col=True):
        self.name = name
        self.input_H = input_H # assert image is square
        self.kernel_size = kernel_size
        self.output_H = (input_H - kernel_size + 2 * padding) // stride + 1
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.padding = padding
        self.stride = stride
        self.activation_tensor = Tensor([batch_size, time_steps, self.output_H * self.output_H, input_channel * kernel_size * kernel_size], 'spike', sparse=True)
        self.weight_tensor = Tensor([input_channel * kernel_size * kernel_size, output_channel], 'fp8', sparse=False)
        self.output_tensor = Tensor([batch_size, time_steps, self.output_H * self.output_H, output_channel], 'fp8', sparse=False)

class MaxPool2D:
    def __init__(self, name, input_H, channel, kernel_size, padding, stride, batch_size, time_steps, img2col=True):
        self.name = name
        self.input_H = input_H
        self.kernel_size = kernel_size
        self.output_H = (input_H - kernel_size + 2 * padding) // stride + 1
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.channel = channel
        self.activation_tensor = Tensor([batch_size, time_steps, self.output_H * self.output_H, channel, kernel_size * kernel_size], 'spike', sparse=True)
        self.output_tensor = Tensor([batch_size, time_steps, self.output_H * self.output_H, channel], 'spike', sparse=True)

class LIFNeuron:
    def __init__(self, name, num_neuron, batch_size, time_steps):
        self.name = name
        self.num_neuron = num_neuron
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.activation_tensor = Tensor([batch_size, time_steps, num_neuron], 'fp8', sparse=False)
        self.membrane_potential = Tensor([batch_size, time_steps, num_neuron], 'fp8', sparse=False)
        self.output_tensor = Tensor([batch_size, time_steps, num_neuron], 'spike', sparse=True)

class Attention:
    def __init__(self, name, dim, sequence_length, num_head, batch_size, time_steps, attention_type='spikformer'):
        self.name = name
        self.dim = dim
        self.sequence_length = sequence_length
        self.num_head = num_head
        self.dim_per_head = dim // num_head
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.attention_type = attention_type
        self.act_q_tensor = Tensor([batch_size, time_steps, num_head, sequence_length, dim // num_head], 'spike', sparse=True)
        self.act_k_tensor = Tensor([batch_size, time_steps, num_head, sequence_length, dim // num_head], 'spike', sparse=True)
        self.act_v_tensor = Tensor([batch_size, time_steps, num_head, sequence_length, dim // num_head], 'spike', sparse=True)
        if attention_type == 'spikformer':
            self.attn_tensor = Tensor([batch_size, time_steps, num_head, sequence_length, sequence_length], 'fp8', sparse=True)
            self.output_tensor = Tensor([batch_size, time_steps, num_head, sequence_length, dim // num_head], 'fp8', sparse=False)
        elif attention_type == 'SDT':
            self.kv_out_tensor = Tensor([batch_size, time_steps, num_head, sequence_length, dim // num_head], 'spike', sparse=True)
            self.output_tensor = Tensor([batch_size, time_steps, num_head, sequence_length, dim // num_head], 'spike', sparse=True)
        else:
            raise ValueError('Unknown attention type')


def conv2d_2_fc(operator: Conv2D) -> FC:
    eq_input_dim  = operator.kernel_size * operator.kernel_size * operator.input_channel
    eq_sequence_length = operator.output_H * operator.output_H
    eq_output_dim = operator.output_channel
    eq_sparse_map = img2col(operator.activation_tensor.sparse_map, operator.kernel_size, operator.stride, operator.padding)
    eq_fc = FC(operator.name + '_2fc', eq_input_dim, eq_output_dim, eq_sequence_length, operator.batch_size, operator.time_steps)
    eq_fc.activation_tensor.sparse_map = eq_sparse_map
    return eq_fc

def create_network(name, spike_info):
    dataset = spike_info.split('_')[1].split('.')[0]
    if name == 'spikformer':
        config = spikformer_config(dataset=dataset)
    elif name == 'SDT':
        config = SDT_config(dataset=dataset)
    elif name == 'vgg16':
        config = vgg16_config()
    else:
        raise ValueError('Unknown network name')
    ops = []
    for key, value in config.items():
        if key.startswith('conv2d'):
            ops.append(Conv2D(key, *value))
        elif key.startswith('lif'):
            ops.append(LIFNeuron(key, *value))
        elif key.startswith('maxpool2d'):
            ops.append(MaxPool2D(key, *value))
        elif key.startswith('fc'):
            ops.append(FC(key, *value))
        elif key.startswith('attention'):
            ops.append(Attention(key, *value, attention_type=name))
    with open(spike_info, 'rb') as f:
        sparse_act = pickle.load(f)
        for op in ops:
            if isinstance(op, Attention):
                op.act_q_tensor.sparse_map = sparse_act[op.name + '_q'].contiguous()
                op.act_k_tensor.sparse_map = sparse_act[op.name + '_k'].contiguous()
                op.act_v_tensor.sparse_map = sparse_act[op.name + '_v'].contiguous()
            elif op.activation_tensor.sparse:
                if op.name in sparse_act:
                    op.activation_tensor.sparse_map = sparse_act[op.name].contiguous()
    return ops
    
def print_sparsity(network, spike_info):
    if network == 'spikformer':
        with open(spike_info, 'rb') as f:
            sparse_act = pickle.load(f)
            for key, value in sparse_act.items():
                logging.info(f'{key}: {get_density(value):.4f}')
    elif network == 'spikeBERT':
        with open(spike_info, 'rb') as f:
            sparse_act = pickle.load(f)
            for key, value in sparse_act.items():
                logging.info(f'{key}: {get_density(value):.4f}')
    elif network == 'sdt':
        with open(spike_info, 'rb') as f:
            sparse_act = torch.load(f)
            for key, value in sparse_act.items():
                logging.info(f'{key}: {get_density(value):.4f}')
    elif network == 'vgg16':
        with open(spike_info, 'rb') as f:
            sparse_act = pickle.load(f)
            for key, value in sparse_act.items():
                logging.info(f'{key}: {get_density(value):.4f}')

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    ops = create_network('SDT', 'sdt_cifar10dvs_new1.pkl')
    # total_ops = compute_num_OPS(ops)
    # num_adder = 128
    # frequency = 500 * 1024 * 1024
    # accelerator_ops = num_adder * frequency
    # accelerator_time = total_ops / accelerator_ops
    # logging.info(f'Total number of operations: {total_ops}')
    # logging.info(f'Accelerator time: {accelerator_time} s')
    # print_sparsity('vgg16', 'vgg16_cifar10_new.pkl')
    # attention 32 * 64, 64 * 32, 12 32 * 32 * 2 * 12






