import numpy as np
import torch
import pandas as pd
import dgl
import pyximport
pyximport.install(setup_args={"include_dirs": np.get_include()})

from utils import algos
from utils.utils import get_svd_dense, get_eig_dense


def generate_graph(graph_path, node_encoding, spatial_encoding, svd_dim, eig_dim, device):

    pems_relation = pd.read_csv(graph_path)

    src = torch.tensor(pems_relation['from'].values)
    dst = torch.tensor(pems_relation['to'].values)
    graph = dgl.graph((torch.cat((src, dst)), torch.cat((dst, src))))

    # positional bias
    if 'degree' in node_encoding:
        # N
        graph.ndata['in_degrees'] = graph.in_degrees()
        graph.ndata['out_degrees'] = graph.out_degrees()

    graph = dgl.add_self_loop(graph)
    adj = graph.adj().to_dense()

    if 'eig' in node_encoding:
        eigval, eigvec = get_eig_dense(adj)
        eig_idx = eigval.argsort()
        eigval, eigvec = eigval[eig_idx],eigvec[:, eig_idx]
        eig_pos_emb = eigvec[:, 1:eig_dim+1]
        # N, eig_pos_dim
        graph.ndata['eig_emb'] = eig_pos_emb

    if 'svd' in node_encoding:
        pu,pv = get_svd_dense(adj, svd_dim)
        svd_pos_emb = torch.cat([pu,pv], dim=-1)
        # N, svd_pos_emb*2
        graph.ndata['svd_emb'] = svd_pos_emb

    # attention bias
    if 'spatial' in spatial_encoding:
        shortest_path_result, path = algos.floyd_warshall(adj.bool().numpy())
        spatial_pos = torch.from_numpy((shortest_path_result)).long()
        # N, N
        graph.ndata['spatial_emb'] = spatial_pos

    graph = graph.to(device)

    return graph


def train_val_test_split(feat_path, train_batch_size, val_batch_size, device):

    for idx, (x, y) in enumerate(generate_data(feat_path)):
        y = y.squeeze(axis=-1)
        x = torch.from_numpy(x).float().to(device)
        y = torch.from_numpy(y).float().to(device)
        print(x.shape, y.shape)

        if idx == 0:
            train_x_tensor, train_y_tensor = x, y
        elif idx == 1:
            val_x_tensor, val_y_tensor = x, y
        elif idx == 2:
            test_x_tensor, test_y_tensor = x, y

    # ------- train_loader -------
    train_dataset = torch.utils.data.TensorDataset(train_x_tensor, train_y_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=train_batch_size, shuffle=False)

    # ------- val_loader -------
    val_dataset = torch.utils.data.TensorDataset(val_x_tensor, val_y_tensor)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)

    # ------- test_loader -------
    test_dataset = torch.utils.data.TensorDataset(test_x_tensor, test_y_tensor)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=val_batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, train_y_tensor, val_y_tensor, test_y_tensor


def train_val_test_split_gb(feat_path, device):

    for idx, (x, y) in enumerate(generate_data(feat_path)):
        y = y.squeeze(axis=-1)
        x = torch.from_numpy(x).float().to(device)
        # x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float().to(device)
        # y = torch.from_numpy(y).float()
        print(x.shape, y.shape)

        if idx == 0:
            train_x_tensor, train_y_tensor = x, y
        elif idx == 1:
            val_x_tensor, val_y_tensor = x, y
        elif idx == 2:
            test_x_tensor, test_y_tensor = x, y

    return train_x_tensor, val_x_tensor, test_x_tensor, train_y_tensor, val_y_tensor, test_y_tensor



def generate_from_train_val_test(data, transformer):
    mean = None
    std = None
    for key in ('train', 'val', 'test'):
        x, y = generate_seq(data[key], 12, 12)
        if transformer:
            x = transformer(x)
            y = transformer(y)
        if mean is None:
            mean = x.mean()
        if std is None:
            std = x.std()
        yield (x - mean) / std, y


def generate_from_data(data, length, transformer):
    mean = None
    std = None
    train_line, val_line = int(length * 0.6), int(length * 0.8)
    for line1, line2 in ((0, train_line),
                         (train_line, val_line),
                         (val_line, length)):
        x, y = generate_seq(data['data'][line1: line2], 12, 12)
        if transformer:
            x = transformer(x)
            y = transformer(y)
        if mean is None:
            mean = x.mean()
        if std is None:
            std = x.std()
        yield (x - mean) / std, y


def generate_data(graph_signal_matrix_filename, transformer=None):
    '''
    shape is (num_of_samples, 12, num_of_vertices, 1)
    '''
    data = np.load(graph_signal_matrix_filename)
    keys = data.keys()
    if 'train' in keys and 'val' in keys and 'test' in keys:
        for i in generate_from_train_val_test(data, transformer):
            yield i
    elif 'data' in keys:
        length = data['data'].shape[0]
        for i in generate_from_data(data, length, transformer):
            yield i
    else:
        raise KeyError("neither data nor train, val, test is in the data")


def generate_seq(data, train_length, pred_length):
    seq = np.concatenate([np.expand_dims(
        data[i: i + train_length + pred_length], 0)
        for i in range(data.shape[0] - train_length - pred_length + 1)],
        axis=0)[:, :, :, 0: 1]
    return np.split(seq, 2, axis=1)