# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 15:13:32 2021
@author: wangxu
"""

import torch, time, os, shutil
import models, utils
import numpy as np
import pandas as pd
from torch import nn, optim
from torch.utils.data import DataLoader
from dataset import FECGDataset
from config import config
import matplotlib.pyplot as plt
from torch.optim import lr_scheduler
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm


import warnings
warnings.filterwarnings("ignore")


os.environ["CUDA_DEVICE_OEDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(41)
torch.cuda.manual_seed(41)


# 保存当前模型的权重，并且更新最佳的模型权重
def save_ckpt(state, is_best, model_save_dir):
    current_w = os.path.join(model_save_dir, config.current_w)
    best_w = os.path.join(model_save_dir, config.best_w)
    torch.save(state, current_w)
    if is_best: shutil.copyfile(current_w, best_w)
    


   
    

def train_epoch(model, optimizer, criterion, scheduler, train_dataloader, show_interval=100):
    model.train()
    
    
    
    losses = []
    total = 0
    tbar = tqdm(train_dataloader)
    for i, (inputs, target) in enumerate(tbar):      
        data = inputs.to(device)
        labelt = target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        # plt.plot(labelt[0].squeeze(0).detach().cpu().numpy(),'b')
        # plt.plot(output[0].squeeze(0).detach().cpu().numpy(),'r')
        # plt.title('train')
        # plt.show()
        
        
        loss = criterion(output,labelt.to(torch.float32))
        # loss.requires_grad_(True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())
        # print("epoch:{},lossf:{}".format(i,loss.item()))
                
    tbar.close()       
    for i in range(len(losses)):
        total = total + losses[i]
        
    total /= len(losses)
    # print("epoch:{},loss_train:{}".format(show_interval,total))
                 
    return total,0



def val_epoch(model, optimizer, criterion, scheduler, val_dataloader, show_interval=100):
    model.eval()
    losses = []
    total = 0
    tbar = tqdm(val_dataloader)
    for i, (inputs, target) in enumerate(tbar):     
        data = inputs.to(device)
        labelt = target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output,labelt.to(torch.float32))
        # loss.requires_grad_(True)
        # loss.backward()
        # optimizer.step()
        # scheduler.step()
        losses.append(loss.item())
        # print("epoch:{},lossf:{}".format(i,loss.item()))
        
        # plt.plot(output[0].squeeze(0).detach().cpu().numpy(),'r')
        # plt.plot(labelt[0].squeeze(0).detach().cpu().numpy(),'b')
        # plt.title('linear')
        # plt.show()
            
            
    for i in range(len(losses)):
        total = total + losses[i]
        
    total /= len(losses)
    # print("epoch:{},loss_val:{}".format(show_interval,total))
                 
    return total,0
    

def weights_init_normal2(m):
    if isinstance(m, nn.Conv1d):
        m.weight.data.normal_(0, 0.1)
        # m.bias.data.zero_()
    elif isinstance(m, nn.InstanceNorm2d):
        # m.weight.data.fill_(1)
        pass

def weights_init_normal(m):
    if isinstance(m, nn.Conv1d):
        tanh_gain = nn.init.calculate_gain('tanh')
        nn.init.xavier_uniform_(m.weight.data, gain=tanh_gain)



def train(args):
    model = getattr(models, config.model_name)(output_size=128)

    # args.ckpt = "./ckpt/fecg/current_w.pth"
    args.ckpt = None
    if args.ckpt and not args.resume:
        state = torch.load(args.ckpt, map_location='cpu')
        model.load_state_dict(state['state_dict'])
        print('train with pretrained weight val_f1', state['f1'])
   

    model.apply(weights_init_normal)
    # model = nn.DataParallel(model)
    model = model.to(device)
    
    # data
    train_dataset = FECGDataset(data_path=config.train_dir, train=True)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)
    
    val_dataset = FECGDataset(data_path=config.val_dir, train=False)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, num_workers=0)
    
    test_dataset = FECGDataset(data_path=config.test_dir, train=False)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, num_workers=0)
 
    optimizer = optim.Adam(model.parameters(), lr=config.lr)  
    criterion = nn.MSELoss()
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=800, gamma=0.1)
    
    # 模型保存文件夹
    model_save_dir = '%s/%s' % (config.ckpt, config.model_name)
    if args.ex: model_save_dir += args.ex
    best_f1 = -1
    lr = config.lr
    start_epoch = 1
    stage = 1
    min_loss = 100000
    
    
    # model = nn.DataParallel(model)
    for epoch in range(start_epoch, config.max_epoch + 1):
        since = time.time()
        train_loss, train_f1 = train_epoch(model, optimizer, criterion, exp_lr_scheduler, train_dataloader, show_interval=epoch)
        val_loss, val_f1 = val_epoch(model, optimizer, criterion, exp_lr_scheduler, val_dataloader, show_interval=epoch)
        print('\n')
        print('#epoch:%02d stage:%d train_loss:%.4e train_f1:%.4f  val_loss:%0.4e val_f1:%.4f time:%s\n'
              % (epoch, stage, train_loss, train_f1, val_loss, val_f1, utils.print_time_cost(since)))
        state = {"state_dict": model.state_dict(), "epoch": epoch, "loss": val_loss, 'f1': val_f1, 'lr': lr,
                  'stage': stage}
        save_ckpt(state, val_loss < min_loss, model_save_dir)
        min_loss = min(val_loss, min_loss)

 



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('command', action='store_true', default=False,help='train')
    parser.add_argument("--ckpt", type=str, help="the path of model weight file")
    parser.add_argument("--ex", type=str, help="experience name")
    parser.add_argument("--resume", action='store_true', default=False)
    args = parser.parse_args()
    train(args)
