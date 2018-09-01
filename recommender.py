import pandas as pd
import numpy as np
from topkHeap import *
from tools import *
import time
import pdb

class Recommender:
    def __init__(self):
        self.danmu_test_file_string = "data/danmu_test_test.csv"
        self.danmu_embedding_file_string = "data/danmu_embedding_test.csv"
        self.video_feature_file_string = "data/video_feature_test.csv"
        self.user_feature_file_string = "data/user_feature_test.csv"
        self.user_recommend_file_string = "data/user_recommend_test.csv"
        self.video_embeddings_dict = {}
        self.user_embeddings_dict = {}
        self.video_feature_dict = {}
        self.user_feature_dict = {}
        self.user_video_dict = {}
        self.user_recommend_dict = {}
        self.embedding_dim_int = 10
        return

    def calculate_feature(self):
        danmu_embedding = pd.read_table(self.danmu_embedding_file_string)
        aid_group_dict = danmu_embedding.groupby(['aid'])
        video_feature_file = open(self.video_feature_file_string, 'w')
        video_feature_file.write("aid\tfeature\n")
        count = 0
        for key_string in aid_group_dict.groups.keys():
            print("aid: " + str(count))
            video_feature_array = np.zeros(self.embedding_dim_int)
            sentence_embeddings_dataFrame = aid_group_dict.get_group(key_string)['sentence_embedding']
            for embedding_string in sentence_embeddings_dataFrame:
                video_feature_array += np.array([float(i_string) for i_string in embedding_string.split(',')])
            video_feature_array /= len(video_feature_array)
            self.video_feature_dict[key_string] = video_feature_array
            video_feature_file.write("%s\t%s\n"%(key_string, ','.join([str(i_float) for i_float in video_feature_array])))
            count += 1
        video_feature_file.close()
        user_group_dict= danmu_embedding.groupby(['user'])
        user_feature_file = open(self.user_feature_file_string, 'w')
        user_feature_file.write("user\tfeature\n")
        count = 0
        for key_string in user_group_dict.groups.keys():
            print("user: " + str(count))
            user_feature_array = np.zeros(self.embedding_dim_int)
            sentence_embeddings_dataFrame = user_group_dict.get_group(key_string)['sentence_embedding']
            for embedding_string in sentence_embeddings_dataFrame:
                user_feature_array += np.array([float(i_string) for i_string in embedding_string.split(',')])
            user_feature_array /= len(user_feature_array)
            self.user_embeddings_dict[key_string] = user_feature_array
            user_feature_file.write("%s\t%s\n"%(key_string, ','.join([str(i_float) for i_float in user_feature_array])))
            count += 1
        user_feature_file.close()
        #for key_string, value_array in video_feature_dict.items():

    def cos_similarity(self, x_array, y_array):
        return np.dot(x_array, y_array)/np.linalg.norm(x_array)/np.linalg.norm(y_array)

    def recommend(self):
        if len(self.video_feature_dict) == 0:
            video_feature_dataFrame = pd.read_table(self.video_feature_file_string)
            for index, row in video_feature_dataFrame.iterrows():
                self.video_feature_dict[row['aid']] = np.array([float(i_string) for i_string in row['feature'].split(',')])
            user_feature_dataFrame = pd.read_table(self.user_feature_file_string)
            for index, row in user_feature_dataFrame.iterrows():
                self.user_feature_dict[row['user']] = np.array([float(i_string) for i_string in row['feature'].split(',')])
        count = 0
        self.user_recommend_dict = {}
        for user_string, user_feature_array in self.user_feature_dict.items():
            print("user" + str(count))
            topK = TopkHeap(10)
            for video_string, video_feature_array in self.video_feature_dict.items():
                topK.push((self.cos_similarity(user_feature_array, video_feature_array), video_string))
            self.user_recommend_dict[user_string] = [tup[1] for tup in topK.topK()]
            count += 1
        Tools.save_string2array_dict(self.user_recommend_dict, self.user_recommend_file_string, "user\tmovies\n")

    def evaluation(self):
        if len(self.user_recommend_dict) == 0:
            print("start loading user recommend dictionary...")
            start = time.clock()
            metainfo_string, self.user_recommend_dict = Tools.load_string2array_float_dict(self.user_recommend_file_string)
            #with open("data/itemknn") as file:
            #    for line in file.readlines():
            #        [user_string, video_string, rating_string] = line.split(',')
            #        if len(self.user_recommend_dict.get(user_string, [])) != 0:
            #            self.user_recommend_dict[user_string].append(video_string)
            #        else:
            #            self.user_recommend_dict[user_string] = [video_string]
            print("finish loading in %d seconds"%(time.clock() - start))
        if len(self.user_video_dict) == 0:
            print("start loading user video dictionary...")
            start = time.clock()
            user_group_dict = pd.read_table(self.danmu_test_file_string).groupby('user')
            for user_string in user_group_dict.groups.keys():
                self.user_video_dict[user_string] = [aid_string for aid_string in user_group_dict.get_group(user_string)['aid']]
            print("finish loading in %d seconds"%(time.clock() - start))
        count_int = 0
        i_int = 0
        #pdb.set_trace()
        for user_string, recommend_list in self.user_recommend_dict.items():
            print(i_int)
            i_int += 1
            for recommend_string in recommend_list:
                if recommend_string in self.user_video_dict[user_string]:
                    print("hit: user %s, video %s"%(user_string, recommend_string))
                    count_int += 1

        print("hit number: " + str(count_int))
        
if __name__=="__main__":
    recommender = Recommender()
    #recommender.calculate_feature()
    #recommender.recommend()
    recommender.evaluation()
