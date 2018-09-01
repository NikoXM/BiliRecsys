# This file is to provide data to model.py
import sqlite3
import numpy as np
import pdb
from word import Word
import pandas as pd
import random

class DataLoader:
    def __init__(self, _num_pos_int = 2, _num_neg_int = 0, _batch_size_int = 3, _sequence_size_int = 5):
        self.num_pos_int = _num_pos_int
        self.num_neg_int = _num_neg_int
        self.db_string = "/home/renhao/github/BiliRecsys/data/danmu.db"
        self.danmu_csv_string = "/home/renhao/github/BiliRecsys/data/danmu.csv"
        self.connect = sqlite3.connect(self.db_string)
        self.cursor = self.connect.cursor()
        self.danmu_table_name_string = "filtered_danmu_without_sc"
        self.query_all_aid_string = "select distinct aid from %s"%self.danmu_table_name_string
        self.query_danmu_string = "select content, rtime from %s where aid = "%self.danmu_table_name_string
        self.query_danmu_count_string = "select count(content) from %s where aid = "%self.danmu_table_name_string
        self.aids_list = list()

        self.batch_size_int = _batch_size_int
        self.sequence_size_int = _sequence_size_int

    def prepare_data(self):
        self.data_dataFrame = pd.read_table(self.danmu_csv_string)
        self.aids_list = [result[0] for result in self.cursor.execute(self.query_all_aid_string)]
        self.word = Word()
        self.word.build_words()

    def set_danmu_table(self, _danmu_table_name_string, _danmu_csv_string):
        self.danmu_table_name_string = _danmu_table_name_string
        self.danmu_csv_string = _danmu_csv_string

    def get_embedding_matrix(self):
        return self.word.get_embedding_matrix()

    def get_batch_size(self):
        return self.batch_size_int

    #function: we should add positive and negtive samples in this function
    def sort_danmu_list(self, _danmu_list):
        point_int = self.num_pos_int
        new_list = list()
        item_list = list()
        for i_int in range(int(self.num_pos_int/2), len(_danmu_list) - int(self.num_pos_int/2)):
            item_list = list()
            item_list.append(_danmu_list[i_int])
            for j_int in range(1, int(self.num_pos_int/2) + 1):
                item_list.append(_danmu_list[i_int + j_int])
                item_list.append(_danmu_list[i_int - j_int])
            item_list += random.sample(_danmu_list[:i_int - int(self.num_pos_int/2)] + _danmu_list[i_int + int(self.num_pos_int/2):], self.num_neg_int)
            new_list.append(item_list)
        return new_list

    def get_batch_aid_user_content(self):
        batch_x_list= list()
        for i_int in range(self.num_pos_int + self.num_neg_int + 1):
            batch_x_list.append([])
        user_batch_list = []
        count_int = 0
        for aid_string in self.aids_list:
            #temp_dataFrame = self.data_dataFrame.loc[self.data_dataFrame['aid'] == int(aid_string)]
            temp_dataFrame = self.data_dataFrame.loc[self.data_dataFrame['aid'] == int(aid_string)]
            if temp_dataFrame.shape[0] < 1 + self.num_pos_int + self.num_neg_int:
                continue
            data_list = [self.word.convert2id(content_string, self.sequence_size_int) for content_string in temp_dataFrame['content']]
            data_list = self.sort_danmu_list(data_list)
            user_list = list(temp_dataFrame['user'])
            # Here I remove the head and tail of user list on purpose to keep consistent with data_list
            user_list = user_list[int(self.num_pos_int/2):int(len(user_list) - self.num_pos_int/2)]
            #print('user list size: %d'%len(user_list))
            #print('data list size: %d'%len(data_list))
            user_int = 0
            for item_list in data_list:
                user_batch_list.append(user_list[user_int])
                for i_int in range(self.num_pos_int + self.num_neg_int + 1):
                    batch_x_list[i_int].append(item_list[i_int])
                count_int += 1
                user_int += 1
                if count_int == self.batch_size_int:
                    for i_int in range(self.num_pos_int + 1):
                        batch_x_arrays_list = [np.array(batch_x_i) for batch_x_i in batch_x_list]
                    yield (batch_x_arrays_list, aid_string, user_batch_list)
                    for i_int in range(self.num_pos_int + self.num_neg_int + 1):
                        batch_x_list[i_int].clear()
                    count_int = 0
                    user_batch_list.clear()
                    
            assert(user_int == len(user_list))
            data_list.clear()
            user_list.clear()

    def data2File(self):
        danmu_csv_file = open(self.danmu_csv_string, 'w')
        danmu_csv_file.write("aid\tcontent\trtime\tuser\n")
        query_all_danmu_string = "select aid, content, rtime, user from %s"%self.danmu_table_name_string
        results = self.cursor.execute(query_all_danmu_string)
        count = 0
        for result_tuple in results:
            print(count)
            count += 1
            danmu_csv_file.write("%s\t%s\t%s\t%s\n"%(result_tuple[0], result_tuple[1], result_tuple[2], result_tuple[3]))
        danmu_csv_file.close()

    def __iter__(self):
        batch_y_array = np.array(([1.0/self.num_pos_int]*self.num_pos_int+[0.0]*self.num_neg_int)*self.batch_size_int).reshape(self.batch_size_int, self.num_pos_int+self.num_neg_int)
        batch_x_list= list()
        for i_int in range(self.num_pos_int + self.num_neg_int + 1):
            batch_x_list.append([])
        count_int = 0
        for aid_string in self.aids_list:
            #print('aid_string :' + aid_string)
            temp_dataFrame = self.data_dataFrame.loc[self.data_dataFrame['aid'] == int(aid_string)]
            if temp_dataFrame.shape[0] < 1 + self.num_pos_int + self.num_neg_int:
                continue
            data_list = [self.word.convert2id(content_string, self.sequence_size_int) for content_string in temp_dataFrame['content']]
            data_list = self.sort_danmu_list(data_list)
            for item_list in data_list:
                for i_int in range(self.num_pos_int + self.num_neg_int + 1):
                    batch_x_list[i_int].append(item_list[i_int])
                count_int += 1
                if count_int == self.batch_size_int:
                    for i_int in range(self.num_pos_int + 1):
                        batch_x_arrays_list = [np.array(batch_x_i) for batch_x_i in batch_x_list]
                    yield (batch_x_arrays_list, batch_y_array)
                    for i_int in range(self.num_pos_int + self.num_neg_int + 1):
                        batch_x_list[i_int].clear()
                    count_int = 0
            data_list.clear()

if __name__=="__main__":
    dataLoader = DataLoader()
    dataLoader.set_danmu_table("danmu_test","danmu_small_test.csv")
    dataLoader.prepare_data()
    
    count_int = 0
    data_id_list = list()
    batch_list = list()
    for data_turple in dataLoader.get_batch_aid_user_content():
        print(data_turple)
        #if count_int == 2:
        #    break
        #else:
        #    count_int += 1
    #dataLoader.data2File()
