# -*- coding: utf-8 -*-
"""
@Time: 2021/8/27 16:12
@Auth: Xhw
@File: entity_extract.py
@Description: 实体抽取.
"""
from data_loader import EntDataset, load_data
from transformers import BertTokenizerFast, BertModel
from torch.utils.data import DataLoader
from data_process import DataProcess
import torch
from GlobalPointer import GlobalPointer, MetricsCalculator
from tqdm import tqdm, trange
from bilstm import BiLSTM

device = torch.device("cuda:0")
BATCH_SIZE = 128

ENT_CLS_NUM = 9
dp = DataProcess()
train_word_lists, train_tag_lists, dev_word_lists, dev_tag_lists, test_word_lists, test_tag_lists = dp.prepare_data()


# GP MODEL
encoder = BiLSTM()
model = GlobalPointer(encoder, ENT_CLS_NUM, 64).to(device)  # 9个实体类型
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)


def sort_by_lengths(word_lists, tag_lists):
    pairs = list(zip(word_lists, tag_lists))
    indices = sorted(range(len(pairs)), key=lambda x: len(pairs[x][0]), reverse=True)

    pairs = [pairs[i] for i in indices]
    word_lists, tag_lists = list(zip(*pairs))
    return word_lists, tag_lists, indices

def batch_sents_to_tensorized(batch, maps):
    PAD = maps.get('<pad>')
    UNK = maps.get('<unk>')

    max_len = len(batch[0])
    batch_size = len(batch)

    batch_tensor = torch.ones(batch_size, max_len).long() * PAD
    attention_mask = torch.ones(batch_size, max_len).long()
    for i, l in enumerate(batch):
        for j, e in enumerate(l):
            batch_tensor[i][j] = maps.get(e, UNK)
            attention_mask[i][j] = 0
    lengths = [len(l) for l in batch]
    return batch_tensor, lengths, attention_mask



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

train_word_lists, train_tag_lists, _ = sort_by_lengths(train_word_lists, train_tag_lists)
dev_word_lists, dev_tag_lists, _ = sort_by_lengths(dev_word_lists, dev_tag_lists)
test_word_lists, test_tag_lists, _ = sort_by_lengths(test_word_lists, test_tag_lists)
metrics = MetricsCalculator()
max_f, max_recall = 0.0, 0.0
for eo in range(10):
    total_loss, total_f1 = 0., 0.
    for idx in trange(0, len(train_word_lists), BATCH_SIZE, desc="Iteration:"):
        batch_sents = train_word_lists[idx: idx + BATCH_SIZE]
        batch_tags = train_tag_lists[idx: idx + BATCH_SIZE]
        batch_sents_tensor, sents_lengths, attention_mask = batch_sents_to_tensorized(batch_sents, word2id)
        labels_tensor, _, _ = batch_sents_to_tensorized(batch_tags, tag2id)
        logits = model(batch_sents_tensor, sents_lengths, attention_mask)
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
            raw_text_list, input_ids, attention_mask, segment_ids, labels = batch
            input_ids, attention_mask, segment_ids, labels = input_ids.to(device), attention_mask.to(
                device), segment_ids.to(device), labels.to(device)
            logits = model(input_ids, attention_mask, segment_ids)
            f1, p, r = metrics.get_evaluate_fpr(logits, labels)
            total_f1_ += f1
            total_precision_ += p
            total_recall_ += r
        avg_f1 = total_f1_ / (len(ner_loader_evl))
        avg_precision = total_precision_ / (len(ner_loader_evl))
        avg_recall = total_recall_ / (len(ner_loader_evl))
        print("EPOCH：{}\tEVAL_F1:{}\tPrecision:{}\tRecall:{}\t".format(eo, avg_f1, avg_precision, avg_recall))

        if avg_f1 > max_f:
            torch.save(model.state_dict(), './outputs/ent_model.pth'.format(eo))
            max_f = avg_f1
        model.train()
