# -*- coding: utf-8 -*-
import torch
class Config:
    def __init__(self):
        self.train_cme_path = './datasets/train.json'  # CMeEE 训练集
        self.eval_cme_path = './datasets/eval.json'  # CMeEE 测试集
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.BATCH_SIZE = 256
        self.ENT_CLS_NUM = 9
        self.head_size = 64
        self.emb_size = 300
        self.lstm_hidden_size = 128
        self.out_size = 128