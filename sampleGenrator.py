import pandas as pd
import random

class SampleGenerator:
    def __init__(self, origin_data_path_string="data/danmu_test.csv", train_data_path_string="data/danmu_test_train.csv", test_data_path_string="data/danmu_test_test.csv", ratio_float=0.8):
        self.origin_data_path_string = origin_data_path_string
        self.ratio_float = ratio_float
        self.train_data_path_string = train_data_path_string
        self.test_data_path_string = test_data_path_string
        #self.metainfo_string = "aid\tcontent\trtime\tuser\n"

    def generate_by_user(self):
        danmu_dataFrame = pd.read_table(self.origin_data_path_string)
        user_group_dict = danmu_dataFrame.groupby(['user']).groups
        user_list = list(user_group_dict.keys())
        random.shuffle(user_list)
        num_train = int(len(user_list)*self.ratio_float)
        train_user_list = user_list[:num_train]
        test_user_list = user_list[num_train:]
        #train_data_file = open(self.train_data_path_string, 'w')
        #test_data_file = open(self.test_data_path_string, 'w')
        #train_data_file.write(self.metainfo_string)
        #test_data_file.write(self.metainfo_string)
        #train_data_list = list()
        #test_data_list = list()
        #count = 1
        #for index, row in danmu_dataFrame.iterrows():
        #    print(count)
        #    count += 1
        #    data_string = "%s\t%s\t%s\t%s\n"%(row['aid'], row['content'], row['rtime'], row['user'])
        #    if row['user'] in train_user_list:
        #        train_data_list.append(data_string)
        #    else:
        #        test_data_list.append(data_string)
        train_dataFrame = danmu_dataFrame[danmu_dataFrame['user'].isin(train_user_list)]
        test_dataFrame = danmu_dataFrame[danmu_dataFrame['user'].isin(test_user_list)]
        train_dataFrame.to_csv(self.train_data_path_string, sep = '\t')
        test_dataFrame.to_csv(self.test_data_path_string, sep = '\t')
        #train_data_file.writelines(data_string)
        #test_data_file.writelines(data_string)
        #train_data_file.close()
        #test_data_file.close()

    def generate_by_video(self):
        return

if __name__=="__main__":
    sampleGenerator = SampleGenerator()
    sampleGenerator.generate_by_user()
