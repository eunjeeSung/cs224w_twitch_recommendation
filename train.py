import argparse
import sys
import os
import time
import yaml

import pandas as pd
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model import liverec
import data
import eval
import preprocess
import utils


if __name__ == '__main__':
    # Basic configuration.
    torch.manual_seed(42)

    # Minimum parser for the configuration file path.
    # All other configurations should be stored in the configuration file (.yaml).
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        help='path to the config file (.yaml)')
    args = parser.parse_args()

    with open(args.config) as file:
        cfgs = yaml.load(file, Loader=yaml.FullLoader)
    print(cfgs)

    # TensorBoard writer for logging.
    log_dir = f'{cfgs["lr"]}_dc{cfgs["l2"]}_b{cfgs["batch_size"]}_K{cfgs["K"]}_{int(time.time())}'
    writer = SummaryWriter(f'logs/{log_dir}')

    # Load data.
    preprocess.load_data(cfgs)  # To load 'ts' field into cfgs.
    train_loader, val_loader, test_loader = data.get_dataloaders(cfgs)

    model = liverec.LiveRec(cfgs).to(cfgs['device'])
    optimizer = optim.Adam(model.parameters(),
                           lr=cfgs['lr'],
                           weight_decay=cfgs['l2'])

    # Train the model.
    best_val = 0.25
    best_max = cfgs['early_stop']
    best_cnt = best_max
    print("training...")
    for epoch in range(cfgs['num_epochs']):
        loss_all = 0.0
        loss_cnt = 0
        model.train()

        for data in tqdm(train_loader):
            data = data.to(cfgs['device'])
            target_reshape_tuple = (
                cfgs['batch_size'], -1, data.liverec_data.size(1))
            liverec_data_tensor = torch.reshape(
                data.liverec_data, target_reshape_tuple)
            optimizer.zero_grad()

            loss = model.train_step(data)
            loss_all += loss.item()
            loss_cnt += (liverec_data_tensor[:,:,5] != 0).sum()
            
            loss.backward()
            optimizer.step()
            
            if torch.isnan(loss):
                print("loss is nan !")

        # Log and check validation performance.
        scores = eval.compute_recall(
            model, val_loader, cfgs, writer, epoch, maxit=500)
        print('Epoch: {:03d}, Loss: {:.5f}'.format(epoch, loss_all / loss_cnt))
        writer.add_scalar('train/loss', loss, epoch)
        utils.print_scores(scores)

        hall = scores['all']['h01']
        if hall > best_val:
            best_val = hall
            torch.save(model.state_dict(),
                       f'{cfgs["model_path"]}/{log_dir}/ckpt_epoch{epoch}_{loss}.pth')
            best_cnt = best_max
        else:
            best_cnt -= 1
            if best_cnt == 0:
                break

    # model = liverec.LiveRec().to(args.device)
    # model.load_state_dict(torch.load(MPATH))

    # scores = compute_recall(model, test_loader, args)
    # print("Final score")
    # print("="*11)
    # print('Epoch: {:03d}, Loss: {:.5f}'.format(epoch, loss_all/loss_cnt))
    # print_scores(scores)
    # save_scores(scores,args)