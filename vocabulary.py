#!/usr/bin/env python
import os
import jieba
import sqlite3
import langconv

class Vocab:
    def __init__(self):
        self.token2id_dict = {"<UNK>":0}
        self.id2token_dict = {}
        self.db_string = "/home/renhao/github/danmu.db"
        self.danmu_table_name_string = "filtered_danmu_without_sc"
        self.connect = sqlite3.connect(self.db_string)
        self.cursor = self.connect.cursor()
        self.query_content_sql_string = "select content from %s limit 10000"%self.danmu_table_name_string
        self.vocabulary_name_string = "vocabulary.txt"
        if os.path.exists(self.vocabulary_name_string):
            self.is_created_bool = True
        else:
            self.is_created_bool = False

    def tradition2simple(self, line_string):
        line_string = langconv.Converter('zh-hans').convert(line_string)
        return line_string
   
    def build_vocab(self, top_n = None, min_freq = 1):
        if self.is_created_bool:
            with open(self.vocabulary_name_string, 'r') as vocab_file:
                for line in vocab_file.readlines():
                    temp_string = line.split()
                    key_string = temp_string[0]                      
                    value_int = int(temp_string[1])
                    self.token2id_dict[key_string] = value_int
            self.is_created_bool = True
            return
        raw_vocab = {}
        results = self.cursor.execute(self.query_content_sql_string).fetchall()
        for result_turple in results:
            content_string = self.tradition2simple(result_turple[0])
            #words_list = jieba.cut(content_string)
            #print(words_list)
            for word in content_string:
                raw_vocab[word] = raw_vocab.get(word, 0) + 1
            #for word in words_list:
            #    raw_vocab[word] = raw_vocab.get(word, 0) + 1
        self.raw_vocab = raw_vocab
        if top_n:
            stopword = list(map(lambda x: x[0], sorted(self.raw_vocab.items(), key=lambda x: x[1])[-top_n:]))
        else:
            stopword = []
        token2id_dict = self.token2id_dict
        for word in self.raw_vocab.keys():
            if self.raw_vocab[word] > min_freq:
                if not word in token2id_dict and not word in stopword:
                    token2id_dict[word] = str(len(token2id_dict))
        self.num_vocab = len(token2id_dict)
        with open(self.vocabulary_name_string, 'w') as vocab_file:
            for key_string, value_string in token2id_dict.items():
                vocab_file.write(key_string + ' ' + str(value_string) + '\n')
        self.token2id_dict = token2id_dict
        self.is_created_bool = True
    
    def set_id2token(self):
        self.id2token_dict = {v:k for k, v in self.token2id_dict.items()}

    def convert2id(self, _danmu_list, max_sequence_len_int = 150): 
        danmu_id_list = list()
        for sentence_string in _danmu_list:
            danmu_id_list.append([self.token2id_dict.get(word,0) for word in sentence_string] + [0]*(max_sequence_len_int - len(sentence_string)))
        return danmu_id_list

if __name__=="__main__":
    test_list = ['你们好','我知道']
    vocab = Vocab()
    vocab.build_vocab()
    out_list = vocab.convert2id(test_list)
    print(out_list)
