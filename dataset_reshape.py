import pickle

if __name__ == '__main__':
    # Load the data
    with open('data/spikformer_cifar10dvs_train.pkl', 'rb') as f:
        data = pickle.load(f)
    
    T = 16
    B = 128
    
    # Modify the data
    for key in data:
        

        if data[key].shape[1] != B:
            x_reshaped = data[key].reshape(T, B, -1)
            x_permuted = x_reshaped.permute(1, 0, 2)
            data[key] = x_permuted.reshape(-1, *data[key].shape[1:])
            print(key, data[key].shape)
                                         
    # Save the modified data back to the file
    with open('data/spikformer_cifar10dvs_train_t.pkl', 'wb') as f:
        pickle.dump(data, f)
