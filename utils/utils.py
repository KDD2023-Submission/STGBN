import numpy as np
import torch

def mask_np(array, null_val):
    if np.isnan(null_val):
        return (~np.isnan(null_val)).astype('float32')
    else:
        return np.not_equal(array, null_val).astype('float32')


def masked_mape_np(y_true, y_pred, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = mask_np(y_true, null_val)
        mask /= mask.mean()
        mape = np.abs((y_pred - y_true) / y_true)
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100


def masked_mape_np_t(y_true, y_pred, null_val=np.nan):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(y_pred, y_true).astype('float32'),
                      y_true))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape) * 100


def masked_mse_np(y_true, y_pred, null_val=np.nan):
    mask = mask_np(y_true, null_val)
    mask /= mask.mean()
    mse = (y_true - y_pred) ** 2
    return np.mean(np.nan_to_num(mask * mse))


def masked_mae_np(y_true, y_pred, null_val=np.nan):
    mask = mask_np(y_true, null_val)
    mask /= mask.mean()
    mae = np.abs(y_true - y_pred)
    return np.mean(np.nan_to_num(mask * mae))


def get_svd_dense(mx,q=3):
    # mx = mx.float()
    u,s,v = torch.svd_lowrank(mx,q=q)
    s=torch.diag(s)
    pu = u@s.pow(0.5)
    pv = v@s.pow(0.5)
    return pu,pv


def get_eig_dense(adj):
    # adj = adj.float()
    rowsum = adj.sum(1)
    r_inv =rowsum.pow(-0.5)
    r_mat_inv = torch.diag(r_inv)
    nr_adj = torch.matmul(torch.matmul(r_mat_inv,adj),r_mat_inv)
    graph_laplacian = torch.eye(adj.shape[0])-nr_adj
    L,V = torch.eig(graph_laplacian,eigenvectors=True)
    return L.T[0],V
