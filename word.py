import numpy as np
import time
import jieba
import pdb

class Word:
    def __init__(self):
        self.unknown_string = "<UNK>"
        self.token2id_dict = {self.unknown_string: 0}
        self.id2token_dict = {}
        self.word2embedding_dict = {}
        self.embedding_dim_int = 0
        self.word_size_int = 0

    def build_words(self):
        print("build vocab...")
        start = time.clock()
        with open("/media/dataset/renhao/wiki.zh.vec") as file:
            meta_string = file.readline()
            temp_list = meta_string.split()
            self.word_size_int = int(temp_list[0])
            self.embedding_dim_int = int(temp_list[1])
            self.embedding_matrix_array = np.zeros((self.word_size_int + 1, self.embedding_dim_int))
            for line_string in file.readlines():
                temp_list = line_string.split()
                if len(temp_list) < self.embedding_dim_int + 1:
                    continue
                id_int = len(self.token2id_dict)
                token_string = temp_list[0]
                embedding_vector_array = np.asarray(temp_list[1:], dtype='float32')
                #print(token_string,embedding_vector_array.shape)
                self.token2id_dict[token_string] = id_int
                self.word2embedding_dict[token_string] = embedding_vector_array
                self.embedding_matrix_array[id_int] = embedding_vector_array
        for key_string, value_string in self.token2id_dict.items():
            self.id2token_dict[value_string] = key_string
        elapsed = (time.clock() - start)
        print("build vocab finish, time: ",elapsed)

    def get_id2token_dict(self):
        return self.id2token_dict

    def get_embedding_matrix(self):
        return self.embedding_matrix_array

    def convert2id(self, sentence_string, sequence_size_int = 20):
        tokens_list = list(jieba.cut(sentence_string))
        if len(tokens_list) > sequence_size_int:
            before_token = tokens_list
            tokens_list = tokens_list[:sequence_size_int]
        else:
            before_token = tokens_list
            tokens_list += [self.unknown_string]*(sequence_size_int - len(tokens_list))
        return_list = list(map(lambda x: self.token2id_dict.get(x, 0), tokens_list))
        if return_list == [151, 17, 0, 3]:
            print(sentence_string, before_token, tokens_list)
        return return_list

    def get_embedding_matrix(self):
        return self.embedding_matrix_array

if __name__=="__main__":
    word = Word()
    word.build_words()
    test_string = "在眼前静静躺着尐䴸"
    result_list = word.convert2id(test_string)
    print(result_list)
