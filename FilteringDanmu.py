# -*- coding:utf-8 -*-
import sqlite3
import re

class Filter:
    def __init__(self):
        self.db_string = "danmu.db"
        self.users_list = list()
        self.videos_list= list()
        self.danmus_list = list()
        self.user_video_dict = dict()
        self.video_user_dict = dict()
    
    def filter(self):
        self.read_database()
        self.prefiltering()
        #self.write_database()

    def prefiltering(self):
        print("filtering")
        # self.filtered_users_list = [user_string for user_string, videos_list in self.user_video_dict.items() if len(videos_list) > 1]
        # self.filtered_videos_list = list()
        # self.filtered_user_video_dict = dict()
        # self.filtered_video_user_dict = dict()
        # for user_string, videos_list in self.user_video_dict.items():
        #     if len(videos_list) > 1:
        #         self.filtered_users_list.append(user_string)
        # for video_string, users_list in self.video_user_dict.items():
        #     if len(users_list) > 100:
        #         self.filtered_videos_list.append(video_string)
        # for user_string, videos_list in self.user_video_dict.items():
        #     self.filtered_user_video_dict[user_string] = [video_string for video_string in videos_list if video_string in self.filtered_videos_list]
        # for video_string, users_list in self.video_user_dict.items():
        #     self.filtered_video_user_dict[video_string] = [user_string for user_string in users_list if user_string in self.filtered_users_list]
        multi_insert_sql_string = "insert into filtered_danmu_without_sc(aid,cid,rtime,atime,user,content) values(?,?,?,?,?,?)"
        clear_filtered_danmu_sql_string = "delete from filtered_danmu_without_sc"
        try:
            connect = sqlite3.connect(self.db_string)
            cursor = connect.cursor()
            cursor.execute(clear_filtered_danmu_sql_string)
            connect.commit()
            count_int = 1
            filtered_danmus_list = list()
            for line_turple in self.danmus_list:
                if len(filtered_danmus_list) >= 1000:
                    print(count_int)
                    count_int += 1
                    try:
                        cursor.executemany(multi_insert_sql_string, filtered_danmus_list)
                        connect.commit()
                        del filtered_danmus_list[:]
                    except Exception, e:
                        print(str(e))
                danmu_string = line_turple[5]
                #rule_string = u'[a-zA-Z0-9’!"#$%&\'()*+,-./:;<=>?@，。?★、…【】《》？“”‘’！[\\]^_`{|}~]+'
                #rule_compile = re.compile(ur"[^a-zA-Z0-9\u4e00-\u9fa5]")  #chinese, alphabet, number
                rule_compile = re.compile(u"[^\u4E00-\u9FA5]")
                try:
                    filtered_danmu_string = rule_compile.sub(r'', danmu_string)
                except Exception,e:
                    print(str(e))
                    print("error:")
                    print(danmu_string)
                    print("line turple:")
                    print(line_turple)
                #print "original content: " + danmu_string
                #print "after process: " + filtered_danmu_string
                if len(filtered_danmu_string) > 0 and len(filtered_danmu_string) < 150:
                    filtered_danmus_list.append((line_turple[0],line_turple[1],line_turple[2],line_turple[3],line_turple[4],filtered_danmu_string))
            cursor.executemany(multi_insert_sql_string, filtered_danmus_list)
            connect.commit()
            del filtered_danmus_list[:]
        except Exception,e:
            print(str(e))
        # print "Filtered users size %d"%(len(self.filtered_users_list))
        # print "Filtered videos size %d"%(len(self.filtered_videos_list))

    def read_database(self):
        print "reading database..."
        query_sql_string = "select aid, cid, rtime, atime, user, content from filtered_danmu"
        try:
            connect = sqlite3.connect(self.db_string)
            cursor = connect.cursor()
            results_list = cursor.execute(query_sql_string)
            for result_turple in results_list:
                self.danmus_list.append(result_turple)
            connect.commit()
            connect.close()
            print("Original danmus list size: %d"%(len(self.danmus_list)))
            # for result_turple in results_list:
            #     user_string = result_turple[0]
            #     video_string = result_turple[1]
            #     if user_string not in self.users_list:
            #         self.users_list.append(user_string)
            #         self.user_video_dict[user_string] = [video_string]
            #     else:
            #         self.user_video_dict[user_string].append(video_string)
            #     if video_string not in self.videos_list:
            #         self.videos_list.append(video_string)
            #         self.video_user_dict[video_string] = [user_string]
            #     else:
            #         self.video_user_dict[video_string].append(user_string)
        except Exception,e:
            print(str(e))
        print("complete read")

    def write_database(self):
        insert_filtered_users_list = [(user_string,) for user_string in self.filtered_users_list]
        connect = sqlite3.connect(self.db_string)
        cursor = connect.cursor()
        insert_user_sql_string = "insert into filtered_users(user) values (?)"
        try:
            cursor.executemany(insert_user_sql_string, insert_filtered_users_list)
        except Exception,e:
            print(str(e))
        connect.commit()
        connect.close()

if __name__ == "__main__":
    f_Filter = Filter()
    f_Filter.filter()
