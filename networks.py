import logging
from collections import OrderedDict
import pickle
import torch

def spikformer_config():
    dim = 384
    batch_size = 1
    time_steps = 4
    depth = 4
    num_head = 12
    mlp_ratio = 4
    spikformer_SPS = OrderedDict([
        ('conv2d_1', [32, dim // 8, dim // 4, 3, 1, 1, batch_size, time_steps]),
        ('lif_1', [32 * 32 * dim // 4, batch_size, time_steps]),
        ('conv2d_2', [32, dim // 4, dim // 2, 3, 1, 1, batch_size, time_steps]),
        ('lif_2', [32 * 32 * dim // 2, batch_size, time_steps]),
        ('maxpool2d_2', [32, dim // 2, 3, 1, 2, batch_size, time_steps]),
        ('conv2d_3', [16, dim // 2, dim, 3, 1, 1, batch_size, time_steps]),
        ('lif_3', [16 * 16 * dim, batch_size, time_steps]),
        ('maxpool2d_3', [16, dim, 3, 1, 2, batch_size, time_steps]),
        ('conv2d_rpe', [8, dim, dim, 3, 1, 1, batch_size, time_steps]),
        ('lif_rpe', [8 * 8 * dim, batch_size, time_steps]),
    ])
    spikformer_encoder = OrderedDict([
        ('fc_q', [dim, dim, 64, batch_size, time_steps]),
        ('lif_q', [dim * 64, batch_size, time_steps]),
        ('fc_k', [dim, dim, 64, batch_size, time_steps]),
        ('lif_k', [dim * 64, batch_size, time_steps]),
        ('fc_v', [dim, dim, 64, batch_size, time_steps]),
        ('lif_v', [dim * 64, batch_size, time_steps]),
        ('attention', [dim, 64, num_head, batch_size, time_steps]),
        ('lif_attn', [dim * 64, batch_size, time_steps]),
        ('fc_o', [dim, dim, 64, batch_size, time_steps]),
        ('lif_o', [dim * 64, batch_size, time_steps]),
        ('fc_1', [dim, dim * mlp_ratio, 64, batch_size, time_steps]),
        ('lif_1', [dim * mlp_ratio * 64, batch_size, time_steps]),
        ('fc_2', [dim * mlp_ratio, dim, 64, batch_size, time_steps]),
        ('lif_2', [dim * 64, batch_size, time_steps]),
    ])

    spikformer = OrderedDict([(key + '_sps', value) for key, value in spikformer_SPS.items()])
    for i in range(depth):
        encoder_with_idx = OrderedDict([(key + '_enc_' + str(i), value) for key, value in spikformer_encoder.items()])
        spikformer.update(encoder_with_idx)

    return spikformer

class Tensor:
    def __init__(self, shape, dtype, sparse=False):
        self.is_activation = True
        self.sparse = True if dtype == 'spike' else sparse
        self.shape = shape
        self.dtype = dtype
        self.nbits = 8 if dtype == 'fp8' else 1
        self.sparse_map = None

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
        self.output_tensor = Tensor([batch_size, time_steps, num_neuron], 'spike', sparse=True)

class Attention:
    def __init__(self, name, dim, sequence_length, num_head, batch_size, time_steps, attention_type='spikformer'):
        self.name = name
        self.dim = dim
        self.sequence_length = sequence_length
        self.num_head = num_head
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.act_q_tensor = Tensor([batch_size, time_steps, num_head, sequence_length, dim // num_head], 'spike', sparse=True)
        self.act_k_tensor = Tensor([batch_size, time_steps, num_head, sequence_length, dim // num_head], 'spike', sparse=True)
        self.act_v_tensor = Tensor([batch_size, time_steps, num_head, sequence_length, dim // num_head], 'spike', sparse=True)
        self.attn_tensor = Tensor([batch_size, time_steps, num_head, sequence_length, sequence_length], 'fp8', sparse=True)
        self.output_tensor = Tensor([batch_size, time_steps, num_head, sequence_length, dim // num_head], 'fp8', sparse=False)

def create_network(name, spike_info):
    if name == 'spikformer':
        config = spikformer_config()
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
                ops.append(Attention(key, *value))
        with open(spike_info, 'rb') as f:
            sparse_act = pickle.load(f)
            for op in ops:
                if isinstance(op, Attention):
                    op.act_q_tensor.sparse_map = sparse_act[op.name + '_q'][0].contiguous()
                    op.act_k_tensor.sparse_map = sparse_act[op.name + '_k'][0].contiguous()
                    op.act_v_tensor.sparse_map = sparse_act[op.name + '_v'][0].contiguous()
                elif op.activation_tensor.sparse:
                    op.activation_tensor.sparse_map = sparse_act[op.name][0].contiguous()
        return ops
    else:
        raise ValueError('Unknown network name')
    

    
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    ops = create_network('spikformer', 'test.pkl')
    for op in ops:
        logging.info(op.__dict__)