# # -*- coding: UTF-8 -*-
import sqlite3
import numpy as np

class Convector:
    def __init__(self):
        self.watch_file_string = "data/watch.csv"
        self.db_string = "data/danmu.db"

    def softmax(self, _ratings_list):
        ratings_array = np.array(_ratings_list)
        maximum_int = np.max(ratings_array)
        index_list = ratings_array > 5
        ratings_array[index_list] = 5
        return ratings_array.tolist()

    def convert(self):
        query_sql_string = "select distinct user, aid, count(aid) from danmu_test group by user,aid"
        connect = sqlite3.connect(self.db_string)
        cursor = connect.cursor()
        try:
            watch_file = open(self.watch_file_string,'w')
            result_list = cursor.execute(query_sql_string)
            user_point_int = 0
            video_point_int = 0
            user_mapping_dict = dict()
            video_mapping_dict = dict()
            users_list = list()
            videos_list = list()
            ratings_list = list()
            for result_turple in result_list:
                user_string = result_turple[0]
                video_string = result_turple[1]
                rating_string = result_turple[2]
                if user_string not in user_mapping_dict:
                   user_point_int += 1
                   user_mapping_dict[user_string] = user_point_int
                if video_string not in video_mapping_dict:
                    video_point_int += 1
                    video_mapping_dict[video_string] = video_point_int
                users_list.append(user_mapping_dict[user_string])
                videos_list.append(video_mapping_dict[video_string])
                ratings_list.append(rating_string)
            normalized_ratings_list = self.softmax(ratings_list)
            for i_int in range(len(users_list)):
                watch_file.write("%s\t %s\t %s\n"%(users_list[i_int], videos_list[i_int], normalized_ratings_list[i_int]))
            watch_file.close()
        except Exception,e:
            print str(e)
        connect.commit()
        connect.close()

if __name__ == "__main__":
    c_Convector = Convector()
    c_Convector.convert()
