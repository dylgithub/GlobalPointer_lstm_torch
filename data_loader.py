# -*- coding: utf-8 -*-
"""
@Time: 2021/8/27 13:52
@Auth: Xhw
@Description: 实体识别的数据载入器
"""
import json
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

max_len = 256
# ent2id = {"bod": 0, "dis": 1, "sym": 2, "mic": 3, "pro": 4, "ite": 5, "dep": 6, "dru": 7, "equ": 8}
ent2id = {'ID': 0, 'CONT': 1, 'ORG': 2, 'RACE': 3, 'TITLE': 4, 'PRO': 5, 'NAME': 6, 'EDU': 7, 'LOC': 8}
id2ent = {}
for k, v in ent2id.items():
    id2ent[v] = k


def load_data_bak(path):
    D = []
    for d in json.load(open(path)):
        D.append([d['text']])
        for e in d['entities']:
            start, end, label = e['start_idx'], e['end_idx'], e['type']
            if start <= end:
                D[-1].append((start, end, ent2id[label]))
    return D


def load_data(path):
    D = []
    token2id = {"pad": 0}
    _id = 1
    with open(path, "r", encoding="utf-8") as fr:
        for d in json.load(fr):
            text = d['text']
            for token in text:
                if token not in token2id:
                    token2id[token] = _id
                    _id += 1
            D.append([text])
            for e in d['entities']:
                start, end, label = e['start_idx'], e['end_idx'], e['type']
                if start <= end:
                    D[-1].append((start, end, ent2id[label]))
    token2id["unk"] = _id
    # [['对儿童SARST细胞亚群的研究表明，与成人SARS相比，儿童细胞下降不明显，证明上述推测成立。', (3, 9, 0), (19, 24, 1)]]
    return D, token2id


class EntDataset(Dataset):
    def __init__(self, data, token2id, istrain=True):
        self.data = data
        self.token2id = token2id
        self.istrain = istrain

    def __len__(self):
        return len(self.data)

    def encoder(self, item):
        if self.istrain:
            text = item[0]
            input_ids = []
            for token in text:
                input_ids.append(self.token2id.get(token, self.token2id.get("unk")))
            attention_mask = [1] * len(input_ids)
            start_end_list = item[1:]
            # token2char_span_mapping = \
            #     self.tokenizer(text, return_offsets_mapping=True, max_length=max_len, truncation=True)["offset_mapping"]
            # start_mapping = {j[0]: i for i, j in enumerate(token2char_span_mapping) if j != (0, 0)}
            # end_mapping = {j[-1] - 1: i for i, j in enumerate(token2char_span_mapping) if j != (0, 0)}
            # # 将raw_text的下标 与 token的start和end下标对应
            # encoder_txt = self.tokenizer.encode_plus(text, max_length=max_len, truncation=True)
            # input_ids = encoder_txt["input_ids"]
            # token_type_ids = encoder_txt["token_type_ids"]
            # attention_mask = encoder_txt["attention_mask"]

            return text, input_ids, start_end_list, attention_mask
        else:
            # TODO 测试
            pass

    def sequence_padding(self, inputs, length=None, value=0, seq_dims=1, mode='post'):
        """Numpy函数，将序列padding到同一长度
        """
        if length is None:
            length = np.max([np.shape(x)[:seq_dims] for x in inputs], axis=0)
        elif not hasattr(length, '__getitem__'):
            length = [length]

        slices = [np.s_[:length[i]] for i in range(seq_dims)]
        slices = tuple(slices) if len(slices) > 1 else slices[0]
        pad_width = [(0, 0) for _ in np.shape(inputs[0])]

        outputs = []
        for x in inputs:
            x = x[slices]
            for i in range(seq_dims):
                if mode == 'post':
                    pad_width[i] = (0, length[i] - np.shape(x)[i])
                elif mode == 'pre':
                    pad_width[i] = (length[i] - np.shape(x)[i], 0)
                else:
                    raise ValueError('"mode" argument must be "post" or "pre".')
            x = np.pad(x, pad_width, 'constant', constant_values=value)
            outputs.append(x)

        return np.array(outputs)

    def collate(self, examples):
        raw_text_list, batch_input_ids, batch_attention_mask, batch_labels = [], [], [], []
        for item in examples:
            raw_text, input_ids, start_end_list, attention_mask = self.encoder(item)

            labels = np.zeros((len(ent2id), max_len, max_len))
            for start, end, label in item[1:]:
                labels[label, start, end] = 1
            raw_text_list.append(raw_text)
            batch_input_ids.append(input_ids)
            batch_attention_mask.append(attention_mask)
            batch_labels.append(labels[:, :len(input_ids), :len(input_ids)])
        batch_inputids = torch.tensor(self.sequence_padding(batch_input_ids)).long()
        batch_attentionmask = torch.tensor(self.sequence_padding(batch_attention_mask)).float()
        batch_labels = torch.tensor(self.sequence_padding(batch_labels, seq_dims=3)).float()

        return raw_text_list, batch_inputids, batch_attentionmask, batch_labels

    def __getitem__(self, index):
        item = self.data[index]
        return item


if __name__ == '__main__':
    path = "./datasets/train.json"
    data, token2id = load_data(path)
    print(data)
    # labels = np.zeros((5, 25))
    # labels[0 , 1, 1] = 1.
    # labels = torch.tensor(labels)
    # labels_1 = labels.unsqueeze(1)
    # labels_2 = labels.unsqueeze(1).unsqueeze(1).expand(5, 3, 3, 25)
    # print(labels.shape)
    # print(labels_1.shape)
    # print(labels_2.shape)
    # print(labels[:, :5, :5])
    # ner_train = EntDataset(data, token2id)
    # ner_loader_train = DataLoader(ner_train, batch_size=4, collate_fn=ner_train.collate, shuffle=True)
    # for idx, batch in enumerate(ner_loader_train):
    #     if idx == 0:
    #         print(len(batch))
    #         print(batch[0])
    #         print(batch[1].shape, batch[1])
    #         print(batch[2].shape, batch[2])
    #         print(batch[3].shape, batch[3])
