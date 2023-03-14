# -*- coding: utf-8 -*-
"""
@Description: 实体抽取.
"""
from data_loader import EntDataset, load_data
from torch.utils.data import DataLoader
import torch
from GlobalPointer import GlobalPointer, MetricsCalculator
from tqdm import tqdm
from bilstm import BiLSTM
from config import Config

# train_cme_path = './datasets/train.json'  # CMeEE 训练集
# eval_cme_path = './datasets/eval.json'  # CMeEE 测试集
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# BATCH_SIZE = 256
#
# ENT_CLS_NUM = 9

# tokenizer




def multilabel_categorical_crossentropy(y_pred, y_true):
    y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
    y_pred_neg = y_pred - y_true * 1e12  # mask the pred outputs of pos classes
    y_pred_pos = y_pred - (1 - y_true) * 1e12  # mask the pred outputs of neg classes
    zeros = torch.zeros_like(y_pred[..., :1])
    y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
    y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
    neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
    pos_loss = torch.logsumexp(y_pred_pos, dim=-1)
    return (neg_loss + pos_loss).mean()


def loss_fun(y_true, y_pred):
    """
    y_true:(batch_size, ent_type_size, seq_len, seq_len)
    y_pred:(batch_size, ent_type_size, seq_len, seq_len)
    """
    batch_size, ent_type_size = y_pred.shape[:2]
    y_true = y_true.reshape(batch_size * ent_type_size, -1)
    y_pred = y_pred.reshape(batch_size * ent_type_size, -1)
    loss = multilabel_categorical_crossentropy(y_true, y_pred)
    return loss

if __name__ == '__main__':
    config = Config()
    # train_data and val_data
    train_data, token2id = load_data(config.train_cme_path, data_type="train_data")
    evl_data, _ = load_data(config.eval_cme_path)
    ner_train = EntDataset(train_data, token2id)
    ner_loader_train = DataLoader(ner_train, batch_size=config.BATCH_SIZE, collate_fn=ner_train.collate, shuffle=True)
    ner_evl = EntDataset(evl_data, token2id)
    ner_loader_evl = DataLoader(ner_evl, batch_size=config.BATCH_SIZE, collate_fn=ner_evl.collate, shuffle=False)

    # GP MODEL
    encoder = BiLSTM(len(token2id), config.emb_size, config.lstm_hidden_size, config.out_size)
    model = GlobalPointer(encoder, config.ENT_CLS_NUM, config.head_size, config.lstm_hidden_size).to(config.device)  # 9个实体类型
    # optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    metrics = MetricsCalculator()
    max_f, max_recall = 0.0, 0.0
    for eo in range(config.epoch):
        total_loss, total_f1 = 0., 0.
        for idx, batch in enumerate(ner_loader_train):
            raw_text_list, input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(config.device), attention_mask.to(config.device), labels.to(config.device)
            logits = model(input_ids, attention_mask)
            loss = loss_fun(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.25)
            optimizer.step()
            sample_f1 = metrics.get_sample_f1(logits, labels)
            total_loss += loss.item()
            total_f1 += sample_f1.item()

            avg_loss = total_loss / (idx + 1)
            avg_f1 = total_f1 / (idx + 1)
            print("trian_loss:", avg_loss, "\t train_f1:", avg_f1)

        with torch.no_grad():
            total_f1_, total_precision_, total_recall_ = 0., 0., 0.
            model.eval()
            for batch in tqdm(ner_loader_evl, desc="Valing"):
                raw_text_list, input_ids, attention_mask, labels = batch
                input_ids, attention_mask, labels = input_ids.to(config.device), attention_mask.to(config.device), labels.to(config.device)
                logits = model(input_ids, attention_mask)
                f1, p, r = metrics.get_evaluate_fpr(logits, labels)
                total_f1_ += f1
                total_precision_ += p
                total_recall_ += r
            avg_f1 = total_f1_ / (len(ner_loader_evl))
            avg_precision = total_precision_ / (len(ner_loader_evl))
            avg_recall = total_recall_ / (len(ner_loader_evl))
            print("EPOCH：{}\tEVAL_F1:{}\tPrecision:{}\tRecall:{}\t".format(eo, avg_f1, avg_precision, avg_recall))
            # 最好EVAL_F1:0.8995531687357605	Precision:0.9139890567804219	Recall:0.8860058040693457
            if avg_f1 > max_f:
                torch.save(model.state_dict(), config.best_model_save_path)
                max_f = avg_f1
            model.train()
