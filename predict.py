# -*- coding: utf-8 -*-
from GlobalPointer import GlobalPointer
from data_loader import load_data, EntDataset
from torch.utils.data import DataLoader
import pickle
from tqdm import tqdm
import torch
import numpy as np
from bilstm import BiLSTM
from config import Config


def load_pickle_obj(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


ent2id, id2ent = {"bod": 0, "dis": 1, "sym": 2, "mic": 3, "pro": 4, "ite": 5, "dep": 6, "dru": 7, "equ": 8}, {}
for k, v in ent2id.items(): id2ent[v] = k
token2id = load_pickle_obj("mapping/token2id_mapping.pkl")
config = Config()
encoder = BiLSTM(len(token2id), config.emb_size, config.lstm_hidden_size, config.out_size)
model = GlobalPointer(encoder, config.ENT_CLS_NUM, config.head_size, config.lstm_hidden_size).to(
    config.device)  # 9个实体类型
model.load_state_dict(torch.load(config.best_model_save_path))
model.eval()
test_data, _ = load_data(config.test_cme_path)
ner_test = EntDataset(test_data, token2id)
ner_loader_test = DataLoader(test_data, batch_size=config.BATCH_SIZE, collate_fn=ner_test.collate, shuffle=False)


def decode_ent(text, pred_matrix, threshold=0):
    id2ent = {id: ent for ent, id in ent2id.items()}
    pred_matrix = pred_matrix.cpu().numpy()
    ent_list = {}
    for ent_type_id, token_start_index, token_end_index in zip(*np.where(pred_matrix > threshold)):
        ent_type = id2ent[ent_type_id]
        ent_text = text[token_start_index:token_end_index + 1]
        print(ent_text)
        # ent_type_dict = ent_list.get(ent_type, {})
        # ent_text_list = ent_type_dict.get(ent_text, [])
        # ent_text_list.append(ent_char_span)
        # ent_type_dict.update({ent_text: ent_text_list})
        # ent_list.update({ent_type: ent_type_dict})
    # print(ent_list)
    return ent_list


def predict():
    predict_res = []
    for batch in tqdm(ner_loader_test, desc="testing"):
        batch_samples, input_ids, attention_mask, _ = batch
        # print(input_ids)
        input_ids, attention_mask = input_ids.to(config.device), attention_mask.to(config.device)
        with torch.no_grad():
            # model的输出是[batch_size, entity_num, batch_seq_max_length, batch_seq_max_length]
            batch_logits = model(input_ids, attention_mask)
        # print(batch_logits.shape)
        # print(len(batch_samples))
        for ind in range(len(batch_samples)):
            text = batch_samples[ind]
            pred_matrix = batch_logits[ind]
            # print(pred_matrix.shape)
            labels = decode_ent(text, pred_matrix, tokenizer)
            predict_res.append({"text": text, "label": labels})
    return predict_res


predict_res = predict()
print(predict_res)
# with torch.no_grad():
#     i = 1
#     for batch in tqdm(ner_loader_test, desc="testing"):
#         text_list, input_ids, attention_mask, _ = batch
#         print(len(text_list))
#         input_ids, attention_mask = input_ids.to(config.device), attention_mask.to(config.device)
#         # model的输出是[batch_size, entity_num, batch_seq_max_length, batch_seq_max_length]
#         scores = model(input_ids, attention_mask)[0].data.cpu().numpy()
#         print(scores.shape)
#         # scores_0 = model(input_ids, attention_mask)[0]
#         # scores_2 = scores_0[:, [0, -1]]
#         # print(scores.shape, scores_0.shape)
#         # print(scores_2.shape)
#         scores[:, [0, -1]] -= np.inf
#         # print(scores)
#         scores[:, :, [0, -1]] -= np.inf
#         # print(scores)
#         print(np.where(scores > 0))
#         # new_span, entities = [], []
#         #
#         print(i)
#         for l, start, end in zip(*np.where(scores > 0)):
#             print(l, start, end)
#         i += 1
#         #     entities.append({"start_idx": new_span[start][0], "end_idx": new_span[end][-1], "type": id2ent[l]})
#         # print(entities)
