import random
import numpy as np
import pandas as pd
np.random.seed(0)

def move_sliding_window(data, window_size, inputs_cols_indices, label_col_index):
    """
    data: numpy array including data
    window_size: size of window
    inputs_cols_indices: col indices to include
    """

    # (# instances created by movement, seq_len (timestamps), # features (input_len))
    inputs = np.zeros((len(data) - window_size, window_size, len(inputs_cols_indices)))
    labels = np.zeros(len(data) - window_size)

    for i in range(window_size, len(data)):
        inputs[i - window_size] = data[i - window_size : i, inputs_cols_indices]
        labels[i - window_size] = data[i, label_col_index]
    inputs = inputs.reshape(-1, window_size, len(inputs_cols_indices))
    labels = labels.reshape(-1, 1)
    print(inputs.shape, labels.shape)

    return inputs, labels

def swap_losses(lst, wgt):
    # Create lists of tuples (index, loss) and (index, weight)
    indexed_lst = list(enumerate(lst))
    indexed_wgt = list(enumerate(wgt))

    # Sort the list of tuples by loss values
    sorted_lst = sorted(indexed_lst, key=lambda x: x[1])
    # Create a mapping of old index to new index
    index_mapping = {sorted_lst[i][0]: sorted_lst[-(i + 1)][0] for i in range(len(sorted_lst) // 2)}
    # Swap elements
    for old_index, new_index in index_mapping.items():
        lst[old_index], lst[new_index] = lst[new_index], lst[old_index]
        wgt[old_index], wgt[new_index] = wgt[new_index], wgt[old_index]

    return wgt

def swap_weights(my_dict):
    sorted_keys = sorted(my_dict, key=lambda k: my_dict[k][0])
    # Swap the tuples of the highest and lowest values, second highest and second lowest values, and so on
    for i in range(len(sorted_keys) // 2):
        my_dict[sorted_keys[i]], my_dict[sorted_keys[-(i+1)]] = my_dict[sorted_keys[-(i+1)]], my_dict[sorted_keys[i]]
    return my_dict

def swap_rand(my_dict):
    values = list(my_dict.values())
    # Shuffle the list of values randomly
    random.shuffle(values)
    # Create a new dictionary with the shuffled values
    new_dict = {}
    for i, key in enumerate(my_dict.keys()):
        new_dict[key] = values[i]
    return new_dict

def swap_rand_wgt(wgt):
    # Shuffle the list of tuples randomly
    random.shuffle(wgt)
    return wgt

def diff_loss(my_dict): # calculate the difference btwn highest and lowest loss
    # Find the key for the highest value
    max_key = max(my_dict, key=lambda k: my_dict[k][0])
    # Find the key for the lowest value
    min_key = min(my_dict, key=lambda k: my_dict[k][0])
    return my_dict[max_key][0] - my_dict[min_key][0]

def num_params(model):
    """ """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
