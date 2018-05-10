import sqlite3

class DataLoader:
    def __init__(self):
        self.db_string = "danmu.db"
        self.connect = sqlite3.connect(self.db)
        self.cursor = self.connect.cursor()
        
        self.query_all_aid_string = "select distinc aid from danmu"
        self.query_danmu_string = "select content, rtime from danmu where aid = "
        self.query_danmu_count_string = "select count(content) from danmu where aid = "
        
        self.aids_list = list()

        self.batch_size = 32
        self.sequence_size = 100
        result_list = self.cursor.execute(self.query_all_aid_string).fectchall()
        for result in result_list:
            self.aids_list.append(result[0])
    
    
    def __iter__(self):
        for aid_string in self.aids_list:
            query_danmu_string = self.query_danmu_string + aid_string
            query_danmu_count_string = self.query_danmu_count_string + aid_string
            results = self.cursor.execute(query_danmu_count_string).fectchall()
            total_int = int(results[0])
            if total_int < 5:
                continue
            results = self.cursor.execute(query_danmu_string).fectchall()
            for i_int in range(self.batch_size):
                for turple in result_list:
                    content = turple[0]
                    rtime = turple[1]