from collections import OrderedDict

def spikeBERT_config(dataset):
    dim = 768
    batch_size = 1
    time_steps = 4
    depth = 12
    num_head = 8
    mlp_ratio = 4
    if dataset == 'sst2' or dataset == 'sst5':
        sequence_length = 128
    elif dataset == 'mr':
        sequence_length = 256
    else:
        raise ValueError('Unknown dataset')

    spikeBERT_encoder = OrderedDict([
        ('fc_q', [dim, dim * 3, sequence_length, batch_size, time_steps]), # q, k, and v are concatenated into one layer
        ('layernorm_q', [dim, batch_size * 3, sequence_length, time_steps]),
        ('lif_q', [dim * sequence_length * 3, batch_size, time_steps]),
        ('attention', [dim, sequence_length, num_head, batch_size, time_steps]),
        ('lif_attn', [dim * sequence_length, batch_size, time_steps]),
        ('fc_o', [dim, dim, sequence_length, batch_size, time_steps]),
        ('layernorm_o', [dim, batch_size, sequence_length, time_steps]),
        ('lif_o', [dim * sequence_length, batch_size, time_steps]),
        ('fc_1', [dim, dim * mlp_ratio, sequence_length, batch_size, time_steps]),
        ('layernorm_1', [dim * mlp_ratio, batch_size, sequence_length, time_steps]),
        ('lif_1', [dim * mlp_ratio * sequence_length, batch_size, time_steps]),
        ('fc_2', [dim * mlp_ratio, dim, sequence_length, batch_size, time_steps]),
        ('layernorm_2', [dim, batch_size, sequence_length, time_steps]),
        ('lif_2', [dim * sequence_length, batch_size, time_steps]),
    ])

    spikeBERT = OrderedDict()
    for i in range(depth):
        encoder_with_idx = OrderedDict([(key + '_enc_' + str(i), value) for key, value in spikeBERT_encoder.items()])
        spikeBERT.update(encoder_with_idx)

    return spikeBERT

def spikingBERT_config(dataset):
    dim = 768
    batch_size = 1
    time_steps = 16
    depth = 4
    num_head = 12
    sequence_length = 128
    mlp_ratio = 4

    spikingBERT_encoder = OrderedDict([
        ('fc_qkv', [dim, dim * 3, sequence_length, batch_size, time_steps]), # q, k, and v are concatenated into one layer
        ('lif_kv', [dim * sequence_length * 2, batch_size, time_steps]),
        ('attention', [dim, sequence_length, num_head, batch_size, time_steps]),
        ('lif_attn', [dim * sequence_length, batch_size, time_steps]),
        ('fc_o', [dim, dim, sequence_length, batch_size, time_steps]),
        ('layernorm_o', [dim, batch_size, sequence_length, time_steps]),
        ('lif_o', [dim * sequence_length, batch_size, time_steps]),
        ('fc_1', [dim, dim * mlp_ratio, sequence_length, batch_size, time_steps]),
        ('lif_1', [dim * mlp_ratio * sequence_length, batch_size, time_steps]),
        ('fc_2', [dim * mlp_ratio, dim, sequence_length, batch_size, time_steps]),
        ('layernorm_2', [dim, batch_size, sequence_length, time_steps]),
        ('lif_2', [dim * sequence_length, batch_size, time_steps]),
    ])

    spikingBERT = OrderedDict()
    for i in range(depth):
        encoder_with_idx = OrderedDict([(key + '_enc_' + str(i), value) for key, value in spikingBERT_encoder.items()])
        spikingBERT.update(encoder_with_idx)

    return spikingBERT

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
        ('fc_q', [dim, dim * 3, sequence_length, batch_size, time_steps]), # q, k, and v are concatenated into one layer
        ('lif_q', [dim * sequence_length * 3, batch_size, time_steps]),
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
        depth = 4
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
    else:
        print(dataset)
        raise ValueError('Unknown dataset')

        
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
    SDT_encoder = OrderedDict([('fc_q', [dim, dim * 3, sequence_length, batch_size, time_steps]), # q, k, and v are concatenated into one layer
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

def resnet18_config():
    batch_size = 1
    time_steps = 32
    resnet18_config = OrderedDict([
        ('conv2d_1', [8, 64, 64, 3, 1, 1, batch_size, time_steps]),
        ('lif_1', [8 * 8 * 64, batch_size, time_steps]),
        ('conv2d_2', [8, 64, 64, 3, 1, 1, batch_size, time_steps]),
        ('lif_2', [8 * 8 * 64, batch_size, time_steps]),
        ('conv2d_3', [8, 64, 64, 3, 1, 1, batch_size, time_steps]),
        ('lif_3', [8 * 8 * 128, batch_size, time_steps]),
        ('conv2d_4', [8, 64, 64, 3, 1, 1, batch_size, time_steps]),
        ('lif_4', [8 * 8 * 128, batch_size, time_steps]),
        ('conv2d_5', [8, 64, 128, 3, 1, 1, batch_size, time_steps]),
        ('lif_5', [8 * 8 * 128, batch_size, time_steps]),
        ('conv2d_6', [8, 128, 128, 3, 1, 1, batch_size, time_steps]),
        ('lif_6', [8 * 8 * 128, batch_size, time_steps]),
        ('conv2d_7', [8, 128, 128, 3, 1, 1, batch_size, time_steps]),
        ('lif_7', [8 * 8 * 128, batch_size, time_steps]),
        ('conv2d_8', [8, 128, 128, 3, 1, 1, batch_size, time_steps]),
        ('lif_8', [8 * 8 * 128, batch_size, time_steps]),
        ('conv2d_9', [8, 128, 256, 3, 1, 1, batch_size, time_steps]),
        ('lif_9', [8 * 8 * 256, batch_size, time_steps]),
        ('conv2d_10', [8, 256, 256, 3, 1, 1, batch_size, time_steps]),
        ('lif_10', [8 * 8 * 256, batch_size, time_steps]),
        ('conv2d_11', [8, 256, 256, 3, 1, 1, batch_size, time_steps]),
        ('lif_11', [8 * 8 * 256, batch_size, time_steps]),
        ('conv2d_12', [8, 256, 256, 3, 1, 1, batch_size, time_steps]),
        ('lif_12', [8 * 8 * 256, batch_size, time_steps]),
        ('conv2d_13', [8, 256, 512, 3, 1, 1, batch_size, time_steps]),
        ('lif_13', [8 * 8 * 512, batch_size, time_steps]),
        ('conv2d_14', [8, 512, 512, 3, 1, 1, batch_size, time_steps]),
        ('lif_14', [8 * 8 * 512, batch_size, time_steps]),
        ('conv2d_15', [8, 512, 512, 3, 1, 1, batch_size, time_steps]),
        ('lif_15', [8 * 8 * 512, batch_size, time_steps]),
        ('conv2d_16', [8, 512, 512, 3, 1, 1, batch_size, time_steps]),
        ('lif_16', [8 * 8 * 512, batch_size, time_steps]),
    ])
    return resnet18_config

def vgg16_config():
    batch_size = 1
    time_steps = 32
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

def vgg9_config(dataset):
    batch_size = 1
    time_steps = 4
    if dataset == 'cifar10dvs':
        init_shape = 128
    else:
        init_shape = 32
    img_size = [init_shape, init_shape // 2, init_shape // 2, init_shape // 4, init_shape // 4, init_shape // 8]
    vgg9 = OrderedDict([
        ('conv2d_1', [img_size[0], 64, 64, 3, 1, 1, batch_size, time_steps]),
        ('lif_1', [img_size[0] * init_shape * 64, batch_size, time_steps]),
        ('maxpool2d_1', [img_size[0], 64, 2, 1, 2, batch_size, time_steps]),
        ('conv2d_2', [img_size[1], 64, 128, 3, 1, 1, batch_size, time_steps]),
        ('lif_2', [img_size[1] * img_size[1] * 128, batch_size, time_steps]),
        ('conv2d_3', [img_size[2], 128, 128, 3, 1, 1, batch_size, time_steps]),
        ('lif_3', [img_size[2] * img_size[2] * 128, batch_size, time_steps]),
        ('maxpool2d_2', [img_size[2], 128, 2, 1, 2, batch_size, time_steps]),
        ('conv2d_4', [img_size[3], 128, 256, 3, 1, 1, batch_size, time_steps]),
        ('lif_4', [img_size[3] * img_size[3] * 256, batch_size, time_steps]),
        ('conv2d_5', [img_size[4], 256, 256, 3, 1, 1, batch_size, time_steps]),
        ('lif_5', [img_size[4] * img_size[4] * 256, batch_size, time_steps]),
        ('maxpool2d_3', [img_size[4], 256, 2, 1, 2, batch_size, time_steps]),
        ('fc_0', [img_size[5] * img_size[5] * 256, 1024, 1, batch_size, time_steps]),
        ('lif_6', [4096, batch_size, time_steps])
    ])
    return vgg9

def lenet5_config():
    batch_size = 1
    time_steps = 32
    lenet5 = OrderedDict([
        ('conv2d_1', [16, 6, 16, 5, 0, 1, batch_size, time_steps]),
        ('lif_2', [12 * 12 * 16, batch_size, time_steps]),
        ('fc_0', [6 * 6 * 16, 120, 1, batch_size, time_steps]),
        ('lif_3', [120, batch_size, time_steps]),
        ('fc_1', [120, 84, 1, batch_size, time_steps]),
        ('lif_4', [84, batch_size, time_steps]),
    ])
    return lenet5