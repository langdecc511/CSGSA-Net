# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 15:13:32 2021
@author: wangxu
"""
import os


class Config:
    train_dir = "./ADFECGDB/"
    test_dir = "./ADFECGDB/"
    val_dir = "./ADFECGDB/"
    model_name = 'fecg'
 
    batch_size = 30
    max_epoch = 256
    ckpt = 'ckpt' 
    lr =0.0001
    current_w = 'current_w.pth'
    best_w = 'best_w.pth'
    
 


config = Config()
