#!/usr/bin/env python
import jieba
import sqlite3

class Vocab:
    def __init__(self):
        self.token2id_list = {"<UNK>":0}
        self.id2token_list = {}
        self.db_string = "danmu.db"
        self.connect = sqlite3.connect(self.db_string)
        self.cursor = self.connect.cursor()
        self.query_content_sql_string = "select content from danmu"
    
    def build_vocab(self, top_n = None, min_freq = 1):
        raw_vocab = {}
        results = self.cursor.execute(self.query_content_sql_string).fetchall()
        for result_turple in results:
            content_string = result_turple[0]
            words_list = jieba.cut(content_string)
            for word in words_list:
                raw_vocab[word] = raw_vocab.get(word, 0) + 1
        self.raw_vocab = raw_vocab
        if top_n:
            stopword = list(map(lambda x: x[0], sorted(self.raw_vocab.items(), key=lambda x: x[1])[-top_n:]))
        else:
            stopword = []
        token2id_list = self.token2id_list
        for word in self.raw_vocab.keys():
            if self.raw_vocab[word] > min_freq:
                if not word in token2id_list and not word in stopword:
                    token2id[word] = len(token2id_list)
        self.num_vocab = len(token2id_list)
    
    def set_id2token(self):
        self.id2token_list = {v:k for k, v in self.token2id_list.items()}
