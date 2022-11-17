# -*- coding: utf-8 -*-
import pickle


class DataProcess:
    def __init__(self):
        self.train_data_path = "datasets/train.txt"
        self.eval_data_path = "datasets/eval.txt"
        self.test_data_path = "datasets/test.txt"
        self.word2id_path = "mapping/word2id.pkl"
        self.tag2id_path = "mapping/tag2id.pkl"
        self.id2word_path = "mapping/id2word.pkl"

    def save_as_pickle(self, obj, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

    def load_pickle_obj(self, file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def load_data(self, file_path):
        """
        # 加载数据
        word_lists: [[]]
        tag_lists: [[]]
        """
        word_lists = []
        tag_lists = []
        with open(file_path, 'r', encoding="utf-8") as f:
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
                    word_list = []
                    tag_list = []
        return word_lists, tag_lists

    def get_word2id(self, lists):
        """
        # 得到 word2id dict
        """
        maps = {}
        for list_ in lists:
            for e in list_:
                if e not in maps:
                    maps[e] = len(maps)
        return maps

    def extend_vocab(self, word2id, tag2id):
        """
        # 补充word2id
        未登陆词: <unk>
        补码: <pad>
        句子开始标志: <start>
        句子结束标志: <end>
        """
        word2id['<unk>'] = len(word2id)
        word2id['<pad>'] = len(word2id)
        tag2id['<unk>'] = len(tag2id)
        tag2id['<pad>'] = len(tag2id)
        return word2id, tag2id

    def prepare_data(self):
        # 数据加载word_lists: [[]]  tag_lists: [[]]
        self.train_word_lists, self.train_tag_lists = self.load_data(self.train_data_path)
        self.eval_word_lists, self.eval_tag_list = self.load_data(self.eval_data_path)
        self.test_word_lists, self.test_tag_list = self.load_data(self.test_data_path)

        # token像id的映射
        self.word2id = self.get_word2id(self.train_word_lists)
        self.tag2id = self.get_word2id(self.train_tag_lists)
        # 加入unk,start，end等特殊字符的id
        self.word2id, self.tag2id = self.extend_vocab(self.word2id, self.tag2id)

        self.id2word = {self.word2id[w]: w for w in self.word2id}

        self.save_as_pickle(self.word2id, self.word2id_path)
        self.save_as_pickle(self.tag2id, self.tag2id_path)
        self.save_as_pickle(self.id2word, self.id2word_path)

        return (self.train_word_lists, self.train_tag_lists,
                self.eval_word_lists, self.eval_tag_list,
                self.test_word_lists, self.test_tag_list)


if __name__ == '__main__':
    dp = DataProcess()
    train_word_lists, train_tag_lists = dp.load_data(dp.test_data_path)
    print(train_word_lists)
    print(train_tag_lists)
