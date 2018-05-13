# This file is to provide data to model.py
import sqlite3
from vocabulary import Vocab
import numpy as np

class DataLoader:
    def __init__(self, _num_pos_int = 4, _num_neg_int = 0):
        self.num_pos_int = _num_pos_int
        self.num_neg_int = _num_neg_int
        self.db_string = "/home/renhao/github/danmu.db"
        self.connect = sqlite3.connect(self.db_string)
        self.cursor = self.connect.cursor()
        self.danmu_table_name_string = "filtered_danmu_without_sc"
        
        self.query_all_aid_string = "select distinct aid from %s"%self.danmu_table_name_string
        self.query_danmu_string = "select content, rtime from %s where aid = "%self.danmu_table_name_string
        self.query_danmu_count_string = "select count(content) from %s where aid = "%self.danmu_table_name_string
        
        self.aids_list = list()

        self.batch_size_int = 3
        self.sequence_size_int = 100
        self.aids_list = [turple[0] for turple in self.cursor.execute(self.query_all_aid_string).fetchall()]

    def get_batch_size(self):
        return self.batch_size_int

    def sort_danmu_list(self, _danmu_list):
        point_int = self.num_pos_int
        new_list = list()
        item_list = list()
        for i_int in range(int(self.num_pos_int/2), int(len(_danmu_list) - self.num_pos_int/2) - 1):
            item_list = list()
            item_list.append(_danmu_list[i_int])
            for j_int in range(1, int(self.num_pos_int/2) + 1):
                item_list.append(_danmu_list[i_int + j_int])
                item_list.append(_danmu_list[i_int - j_int])
            new_list.append(item_list) 
        return new_list

    def __iter__(self):
        batch_y_array = np.array(([1.0/self.num_pos_int]*self.num_pos_int+[0.0]*self.num_neg_int)*self.batch_size_int).reshape(self.batch_size_int, self.num_pos_int+self.num_neg_int)
        batch_list= list()
        for aid_string in self.aids_list:
            query_danmu_string = self.query_danmu_string + "'%s'"%aid_string
            query_danmu_count_string = self.query_danmu_count_string + "'%s'"%aid_string
            results = self.cursor.execute(query_danmu_count_string).fetchone()
            print(results[0])
            total_int = int(results[0])
            if total_int < 1 + self.num_pos_int + self.num_neg_int:
                continue
            results = self.cursor.execute(query_danmu_string).fetchall()
            data_list = [result[0] for result in results]
            data_list = self.sort_danmu_list(data_list)
            for item_list in data_list:
                if len(batch_list) == self.batch_size_int:
                    yield batch_list
                    batch_list.clear()
                else:
                    batch_list.append(item_list)
            data_list.clear()

if __name__=="__main__":
    dataLoader = DataLoader()
    vocab = Vocab()
    vocab.build_vocab()
    count_int = 0
    data_id_list = list()
    batch_list = list()
    for data_list in dataLoader:
        for item_list in data_list:
            print(item_list)
            batch_list.append(vocab.convert2id(item_list, 20))
        batch_array = np.array(batch_list)
        batch_list.clear()
        print(batch_array)
        if count_int == 2:
            break
        else:
            count_int += 1
