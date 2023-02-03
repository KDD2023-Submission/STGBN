import logging
import torch
import torch.nn.functional as F
from time import time
import copy
from sklearn.metrics import mean_absolute_error

from model.stgnn import STGNN
from utils.utils import *


def train_STGBN(args,
                output_path,
                device,
                g,
                train_x_tensor,
                train_y_tensor,
                val_x_tensor,
                val_y_tensor,
                test_x_tensor,
                test_y_tensor):

    n_timestamps, n_nodes = train_y_tensor.shape[1:]

    model = STGNN(g, n_nodes,
                  args.n_inp,
                  args.n_hid,
                  args.n_out,
                  args.n_temporal_layers,
                  args.n_spatial_layers,
                  args.n_heads,
                  args.node_encoding,
                  args.spatial_encoding,
                  args.svd_pos_dim,
                  args.dropout).to(device)


    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    logging.info(model)
    logging.info(f'# params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    train_y_residual = copy.deepcopy(train_y_tensor)
    val_y_residual = copy.deepcopy(val_y_tensor)
    pred_buff = torch.zeros_like(test_y_tensor)

    # validation dataloader
    val_dataset = torch.utils.data.TensorDataset(val_x_tensor, val_y_tensor)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False)

    # testing dataloader
    test_dataset = torch.utils.data.TensorDataset(test_x_tensor, test_y_tensor)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.val_batch_size, shuffle=False)

    start_time = time()

    for layer in reversed(range(n_timestamps)):

        for iter in range(args.n_iter_per_layer):

            # training dataloader
            train_dataset = torch.utils.data.TensorDataset(train_x_tensor, train_y_residual)
            train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)

            logging.info('-' * 10 + f"Layer {n_timestamps-layer} of {n_timestamps}: iteration {iter+1} of {args.n_iter_per_layer}" + '-' * 10)

            best_val_loss = np.inf

            for epoch in range(args.n_epochs):

                train_loss = 0.0

                # (1) training
                model.train()

                # x->B, T, N, F, y->B, T, N
                for x, y in train_loader:

                    x = x[:,layer:,:,:]
                    # B, T, N
                    outputs = model(x)
                    loss = F.smooth_l1_loss(outputs, y)

                    optim.zero_grad()
                    loss.backward()
                    optim.step()

                    train_loss += loss.item() * x.size(0)

                train_loss = train_loss / len(train_loader.sampler)
                train_time = time() - start_time

                logging.info(f'Epoch: {epoch:2}, train loss: {train_loss:.2f}')

                # (2) validation
                prediction = _validation(model, layer, val_loader)
                valid_loss = F.l1_loss(prediction, val_y_residual)
                valid_time = time() - start_time

                logging.info(f'Epoch: {epoch:2}, valid loss: {valid_loss:.2f}')

                if valid_loss < best_val_loss:

                    logging.info(f'Val loss decreased ({best_val_loss:.6f} --> {valid_loss:.6f})')

                    best_val_loss = valid_loss
                    torch.save(model.state_dict(), f"{output_path}/STGBN_{layer}_{iter}.pt")

            # (3) gradient boosting

            model.load_state_dict(torch.load(f"{output_path}/STGBN_{layer}_{iter}.pt"))

            train_x_tensor, train_y_residual = _get_residual(model, layer, train_loader, args.shrinkage_rate)

            temp = _validation(model, layer, val_loader)
            val_y_residual -= args.shrinkage_rate * temp

            pred = _validation(model, layer, test_loader)
            pred_buff += args.shrinkage_rate * pred

            mae = mean_absolute_error(test_y_tensor.cpu().reshape(-1, 1), pred_buff.cpu().reshape(-1, 1))
            mape = masked_mape_np(test_y_tensor.cpu().numpy(), pred_buff.cpu().numpy(), 0)
            rmse = masked_mse_np(test_y_tensor.cpu().numpy(), pred_buff.cpu().numpy(), 0) ** 0.5

            logging.info(f'Layer: {n_timestamps-layer:2}, Iteration: {iter+1}, MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}')


def _get_residual(model, x_start, data_loader, shrinkage_rate):

    model.eval()
    with torch.no_grad():

        input, residual = [], []
        for x, y in data_loader:
            input.append(x)
            x_ = x[:,x_start:,:,:]

            outputs = model(x_).detach()

            residual.append(y - shrinkage_rate * outputs)

        input = torch.cat(input)
        residual = torch.cat(residual)

    return input, residual


def _validation(model, x_start, data_loader):

    model.eval()
    with torch.no_grad():

        prediction = []
        for x, y in data_loader:

            x = x[:,x_start:,:,:]

            outputs = model(x)
            prediction.append(outputs.detach())

        prediction = torch.cat(prediction)

    return prediction
