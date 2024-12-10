from utils import *
import torch
import os
import pandas as pd

def gather_data(folder_name, file_prefix, row_prefix, tile_size_M_list, tile_size_K_list):
    for i, tile_size_M in enumerate(tile_size_M_list):
        for j, tile_size_K in enumerate(tile_size_K_list):
            file_name = os.path.join(folder_name, f'{file_prefix}_M{tile_size_M}_K{tile_size_K}.csv')

            # read data
            cur_df = pd.read_csv(file_name)
            # get the row with row_prefix
            cur_df = cur_df.loc[cur_df['model_name'] == row_prefix]
            # change the first column first row value to tile size
            cur_df.iloc[0, 0] = 'M'+str(tile_size_M)+'_K'+str(tile_size_K)

            if i == 0 and j == 0:
                df_dse = cur_df
            else:
                df_dse = pd.concat([df_dse, cur_df])

    df_dse = df_dse.reset_index(drop=True)

    return df_dse

def normalize_data(df_dse, df_bit_sparsity):
    # normalize each column of df_all by bit sparsity
    for i in range(1, len(df_dse.columns)):
        # get column name
        col_name = df_dse.columns[i]
        # get bit sparsity for this column
        bit_sparsity_time = df_bit_sparsity[col_name].item()
        df_dse.iloc[:, i] = df_dse.iloc[:, i] / bit_sparsity_time

    return df_dse

def add_geometric_mean(df_dse):
    # Only select numeric columns for geometric mean calculation (skip the first column)
    numeric_data = df_dse.iloc[:, 1:].astype(float)
    log_data = torch.log(torch.tensor(numeric_data.values))
    df_dse['geometric_mean'] = torch.exp(torch.mean(log_data, dim=1)).numpy()

    return df_dse

if __name__ == '__main__':
    
    folder_name = '../dse'
    tile_size_M_list = [32, 64, 128, 256, 512, 1024, 2048]
    tile_size_K = 16

    file_bit_sparsity_time = os.path.join(folder_name, 'time_bit_sparsity.csv')
    df_bit_sparsity_time = pd.read_csv(file_bit_sparsity_time)

    df_bit_sparsity_density = pd.read_csv(os.path.join(folder_name, 'density_M256_K16.csv'))
    df_bit_sparsity_density = df_bit_sparsity_density.loc[df_bit_sparsity_density['model_name'] == 'bit density']

    # gather data
    df_all_time = gather_data(folder_name, 'time', 'Prosperity', tile_size_M_list, [tile_size_K])
    df_all_time = normalize_data(df_all_time, df_bit_sparsity_time)
    df_all_time = add_geometric_mean(df_all_time)

    # save data
    df_all_time.to_csv(os.path.join(folder_name, 'normed_time_dse_M.csv'), index=False)

    df_all_density = gather_data(folder_name, 'density', 'product density', tile_size_M_list, [tile_size_K])
    # concat with bit sparsity
    df_all_density = pd.concat([df_all_density, df_bit_sparsity_density])
    df_all_density = add_geometric_mean(df_all_density)

    # concat time with density, with a empty row in between
    # construct a row with all -1.0
    delimiter = pd.DataFrame([['delimiter'] + ['-1.0'] * (len(df_all_time.columns) - 1)], columns=df_all_time.columns)
    df_all = pd.concat([df_all_time, delimiter], axis=0)
    df_all = pd.concat([df_all, df_all_density], axis=0)

    # save data
    df_all.to_csv(os.path.join(folder_name, 'M_dse.csv'), index=False)


    # save data
    df_all_density.to_csv(os.path.join(folder_name, 'density_dse_M.csv'), index=False)

    tile_size_M = 256
    tile_size_K_list = [4, 8, 16, 32, 64, 128]

    df_all_time = gather_data(folder_name, 'time', 'Prosperity', [tile_size_M], tile_size_K_list)
    df_all_time = normalize_data(df_all_time, df_bit_sparsity_time)
    df_all_time = add_geometric_mean(df_all_time)

    # save data
    df_all_time.to_csv(os.path.join(folder_name, 'normed_time_dse_K.csv'), index=False)

    df_all_density = gather_data(folder_name, 'density', 'product density', [tile_size_M], tile_size_K_list)
    # concat with bit sparsity
    df_all_density = pd.concat([df_all_density, df_bit_sparsity_density])
    df_all_density = add_geometric_mean(df_all_density)

    # save data
    df_all_density.to_csv(os.path.join(folder_name, 'density_dse_K.csv'), index=False)
    
    # concat time with density, with a empty row in between
    delimiter = pd.DataFrame([['delimiter'] + ['-1.0'] * (len(df_all_time.columns) - 1)], columns=df_all_time.columns)
    df_all = pd.concat([df_all_time, delimiter], axis=0)
    df_all = pd.concat([df_all, df_all_density], axis=0)

    # save data
    df_all.to_csv(os.path.join(folder_name, 'K_dse.csv'), index=False)




