import os
import torch
import numpy as np
import random
import logging
import argparse

from utils.data import train_val_test_split_gb, generate_graph
from model.training_STGBN import train_STGBN


# dataset
args = argparse.ArgumentParser(description='arguments')
args.add_argument('--dataset', default='PEMS04', type=str)
args.add_argument('--device', default='0', type=str)

# model paras
args.add_argument('--n_inp', default=1, type=int)
args.add_argument('--n_hid', default=128, type=int)
args.add_argument('--n_out', default=12, type=int)
args.add_argument('--n_heads', default=8, type=int)
args.add_argument('--n_temporal_layers', default=1, type=int)
args.add_argument('--n_spatial_layers', default=2, type=int)
args.add_argument('--node_encoding', default=['svd', 'degree'], nargs='*')
args.add_argument('--eig_pos_dim', default=32, type=int)
args.add_argument('--svd_pos_dim', default=32, type=int)
args.add_argument('--spatial_encoding', default=['spatial'], nargs='*')
args.add_argument('--dropout', default=0.1, type=float)

# train paras
args.add_argument('--n_epochs', default=20, type=int)
args.add_argument('--n_iter_per_layer', default=3, type=int)
args.add_argument('--shrinkage_rate', default=0.15, type=float)
args.add_argument('--train_batch_size', default=32, type=int)
args.add_argument('--val_batch_size', default=32, type=int)
args.add_argument('--lr', default=3e-4, type=float)
args.add_argument('--weight_decay', default=5e-4, type=float)
args.add_argument('--seed', default=0, type=int)

args = args.parse_args()

device = torch.device(f'cuda:{args.device}')

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

set_seed(args.seed)

input_path = f'data/{args.dataset}'
feat_path = f"{input_path}/{args.dataset}.npz"
rela_path = f"{input_path}/{args.dataset}.csv"

train_x_tensor, val_x_tensor, test_x_tensor, \
train_y_tensor, val_y_tensor, test_y_tensor = train_val_test_split_gb(feat_path,
                                                                      device)

graph = generate_graph(rela_path,
                       args.node_encoding,
                       args.spatial_encoding,
                       args.svd_pos_dim,
                       args.eig_pos_dim,
                       device)


output_path = f'output/{args.dataset}'
os.makedirs(output_path)

logging.basicConfig(filename=f"{output_path}/configurations.log",
                    format='%(asctime)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO)

logging.info(args)

train_STGBN(args,
            output_path,
            device,
            graph,
            train_x_tensor,
            train_y_tensor,
            val_x_tensor,
            val_y_tensor,
            test_x_tensor,
            test_y_tensor)
