import pickle

with open('data/vgg16_cifar100_train.pkl', 'rb') as f:
    sparse_act = pickle.load(f)

    for i in range(len(sparse_act['Conv2d'])):
        sparse_act[f'conv2d_{i}'] = sparse_act['Conv2d'][i]

    # 将 sparse_act['Linear'] 的元素映射到 sparse_act['fc_0'] 到 sparse_act['fc_2']
    for i in range(len(sparse_act['Linear'])):
        sparse_act[f'fc_{i}'] = sparse_act['Linear'][i]

    # 清理原始的 'Conv2d' 和 'Linear' 列表
    del sparse_act['Conv2d']
    del sparse_act['Linear']

    # 保存新的 sparse_act
    with open('data/vgg16_cifar100_train.pkl', 'wb') as f:
        pickle.dump(sparse_act, f)