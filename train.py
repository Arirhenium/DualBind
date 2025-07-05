import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from DualBind import DualBind
from dataset import GraphDataset, PLIDataLoader
from config.config_dict import Config
from log.train_logger import TrainLogger
from utils import *
from sklearn.metrics import mean_squared_error

def val(model, dataloader, device):
    model.eval()

    pred_list = []
    label_list = []
    for data in dataloader:
        data['complex'] = data['complex'].to(device)
        data['pkt_tensor'] = data['pkt_tensor'].to(device)
        data['prot_trans'], data['prot_msk'] = data['prot_trans'].to(device), data['prot_msk'].to(device)
        data['smi_trans'], data['smi_msk'] = data['smi_trans'].to(device), data['smi_msk'].to(device)
        with torch.no_grad():
            pred = model(data)
            label = data['complex'].y
            pred_list.append(pred.detach().cpu().numpy())
            label_list.append(label.detach().cpu().numpy())
            
    pred = np.concatenate(pred_list, axis=0)
    label = np.concatenate(label_list, axis=0)

    coff = np.corrcoef(pred, label)[0, 1]
    rmse = np.sqrt(mean_squared_error(label, pred))

    model.train()

    return rmse, coff

# %%
if __name__ == '__main__':
    cfg = 'TrainConfig_DualBind'
    config = Config(cfg)
    args = config.get_config()
    graph_type = args.get("graph_type")
    save_model = args.get("save_model")
    batch_size = args.get("batch_size")
    data_root = args.get('data_root')
    epochs = args.get('epochs')
    repeats = args.get('repeat')
    early_stop_epoch = args.get("early_stop_epoch")

    for repeat in range(repeats):

        args['repeat'] = repeat
        train_dir = os.path.join(data_root, 'train')
        valid_dir = os.path.join(data_root, 'valid')
        test2013_dir = os.path.join(data_root, 'test2013')
        test2016_dir = os.path.join(data_root, 'test2016')
        test2020_dir = os.path.join(data_root, 'test2020')

        train_df = pd.read_csv(os.path.join(data_root, 'train.csv'))
        valid_df = pd.read_csv(os.path.join(data_root, 'valid.csv'))
        test2013_df = pd.read_csv(os.path.join(data_root, 'test2013.csv'))
        test2016_df = pd.read_csv(os.path.join(data_root, 'test2016.csv'))
        test2020_df = pd.read_csv(os.path.join(data_root, 'test2020.csv'))

        train_set = GraphDataset(train_dir, train_df, graph_type=graph_type, create=False)
        valid_set = GraphDataset(valid_dir, valid_df, graph_type=graph_type, create=False)
        test2013_set = GraphDataset(test2013_dir, test2013_df, graph_type=graph_type, create=False)
        test2016_set = GraphDataset(test2016_dir, test2016_df, graph_type=graph_type, create=False)
        test2020_set = GraphDataset(test2020_dir, test2020_df, graph_type=graph_type, create=False)

        train_loader = PLIDataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
        valid_loader = PLIDataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=4)
        test2013_loader = PLIDataLoader(test2013_set, batch_size=batch_size, shuffle=False, num_workers=4)
        test2016_loader = PLIDataLoader(test2016_set, batch_size=batch_size, shuffle=False, num_workers=4)
        test2020_loader = PLIDataLoader(test2020_set, batch_size=batch_size, shuffle=False, num_workers=4)

        logger = TrainLogger(args, cfg, create=True)
        logger.info(__file__)
        logger.info(f"train data: {len(train_set)}")
        logger.info(f"valid data: {len(valid_set)}")
        logger.info(f"test2013 data: {len(test2013_set)}")
        logger.info(f"test2016 data: {len(test2016_set)}")
        logger.info(f"test2020 data: {len(test2020_set)}")

        device = torch.device('cuda:0')
        model = DualBind(35, 256).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)
        criterion = nn.MSELoss()

        running_loss = AverageMeter()
        running_acc = AverageMeter()
        running_best_mse = BestMeter("min")
        best_model_list = []

        model.train()
        for epoch in range(epochs):
            for data in train_loader:
                data['complex'] = data['complex'].to(device)
                data['pkt_tensor'] = data['pkt_tensor'].to(device)
                data['prot_trans'], data['prot_msk'] = data['prot_trans'].to(device), data['prot_msk'].to(device)
                data['smi_trans'], data['smi_msk'] = data['smi_trans'].to(device), data['smi_msk'].to(device)
                pred = model(data)
                label = data['complex'].y

                loss = criterion(pred, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss.update(loss.item(), label.size(0))

            epoch_loss = running_loss.get_average()
            epoch_rmse = np.sqrt(epoch_loss)
            running_loss.reset()

            # start validating
            valid_rmse, valid_pr = val(model, valid_loader, device)
            msg = "epoch-%d, train_loss-%.4f, train_rmse-%.4f, valid_rmse-%.4f, valid_pr-%.4f" \
                    % (epoch, epoch_loss, epoch_rmse, valid_rmse, valid_pr)
            logger.info(msg)

            if valid_rmse < running_best_mse.get_best():
                running_best_mse.update(valid_rmse)
                if save_model:
                    msg = "epoch-%d, train_loss-%.4f, train_rmse-%.4f, valid_rmse-%.4f, valid_pr-%.4f" \
                    % (epoch, epoch_loss, epoch_rmse, valid_rmse, valid_pr)
                    model_path = os.path.join(logger.get_model_dir(), msg + '.pt')
                    best_model_list.append(model_path)
                    save_model_dict(model, logger.get_model_dir(), msg)

                    msg = "valid_rmse-%.4f, valid_pr-%.4f" \
                          % (
                          valid_rmse, valid_pr)
                    logger.info(msg)
            else:
                count = running_best_mse.counter()
                if count > early_stop_epoch:
                    best_mse = running_best_mse.get_best()
                    msg = "best_rmse: %.4f" % best_mse
                    logger.info(f"early stop in epoch {epoch}")
                    logger.info(msg)
                    break_flag = True
                    break

        # final testing
        load_model_dict(model, best_model_list[-1])
        valid_rmse, valid_pr = val(model, valid_loader, device)
        test2013_rmse, test2013_pr = val(model, test2013_loader, device)
        test2016_rmse, test2016_pr = val(model, test2016_loader, device)
        test2020_rmse, test2020_pr = val(model, test2020_loader, device)

        msg = "valid_rmse-%.4f, valid_pr-%.4f, test2013_rmse-%.4f, test2013_pr-%.4f, test2016_rmse-%.4f, test2016_pr-%.4f,test2020_rmse-%.4f, test2020_pr-%.4f/" \
                    % (valid_rmse, valid_pr, test2013_rmse, test2013_pr, test2016_rmse, test2016_pr, test2020_rmse, test2020_pr)

        logger.info(msg)
        

# %%