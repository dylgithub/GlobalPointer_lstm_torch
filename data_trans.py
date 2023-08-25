# -*- coding: utf-8 -*-
import json
import pickle
"""
把BMESO五标签格式数据转换为globalpointer需用的数据格式
"""

def data_load(original_file_location):
    """
    通过\n进行句子分隔，获得文本和标签列表[[]]
    """
    word_lists = []
    tag_lists = []
    tag_set = set()
    with open(original_file_location, 'r', encoding="utf-8") as f:
        word_list = []
        tag_list = []
        for line in f:
            if line != '\n':
                word, tag = line.strip('\n').split()
                word_list.append(word)
                tag_list.append(tag)
            else:
                word_lists.append(word_list)
                tag_lists.append(tag_list)
                tag_set = tag_set | set(tag_list)
                word_list = []
                tag_list = []

    # print(word_lists, tag_lists)
    # print(tag_lists)
    # print(tag_set)
    return word_lists, tag_lists, tag_set


def data_trans(word_lists, tag_lists, tag_set, write_data_location):
    entity2id_mapping = {}
    # 获得实体标签的种类
    for tag in tag_set:
        if tag[0] == "B":
            entity = tag.split("-")[1]
            if entity not in entity2id_mapping:
                entity2id_mapping[entity] = len(entity2id_mapping)
    # with open("mapping/entity2id_mapping.pkl", "wb") as fw:
    #     pickle.dump(entity2id_mapping, fw)
    json_data = []
    for index, word_list in enumerate(word_lists):
        single = {"text": "".join(word_list), "entities": []}
        tag_list = tag_lists[index]
        temp_entity = ""
        entity_dict = {}
        for i, tag in enumerate(tag_list):
            if tag[0] == "S":
                if temp_entity != "":
                    entity_dict["end_idx"] = i - 1
                    entity_dict["entity"] = temp_entity
                    single["entities"].append(entity_dict)
                now_dict = {
                    "start_idx": i,
                    "end_idx": i,
                    "type": tag.split("-")[1],
                    "entity": word_list[i]
                }
                single["entities"].append(now_dict)
                temp_entity = ""
                entity_dict = {}
            elif tag[0] == "B":
                if temp_entity != "":
                    entity_dict["end_idx"] = i - 1
                    entity_dict["entity"] = temp_entity
                    single["entities"].append(entity_dict)
                    temp_entity = ""
                    entity_dict = {}
                temp_entity += word_list[i]
                entity_dict["start_idx"] = i
                entity_dict["type"] = tag.split("-")[1]
            elif tag[0] == "O":
                if temp_entity != "":
                    entity_dict["end_idx"] = i - 1
                    entity_dict["entity"] = temp_entity
                    single["entities"].append(entity_dict)
                temp_entity = ""
                entity_dict = {}
            else:
                temp_entity += word_list[i]
        json_data.append(single)
    json_string = json.dumps(json_data, ensure_ascii=False)
    with open(write_data_location, "w", encoding="utf-8") as fw:
        fw.write(json_string + "\n")


def read():
    with open("mapping/entity2id_mapping.pkl", "rb") as fr:
        _object = pickle.load(fr)
    print(_object)


if __name__ == '__main__':
    word_lists, tag_lists, tag_set = data_load("datasets/train.txt")
    data_trans(word_lists, tag_lists, tag_set, "datasets/train.json")
    # read()
