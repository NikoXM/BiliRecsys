from keras.models import load_model, Model
from dataLoader import *
from model import Attention

def generate_feature():
    batch_size_int = 30
    sequence_size_int = 5
    num_positive_int = 4
    num_negtive_int = 4
    dataLoader = DataLoader(_num_pos_int = num_positive_int, _num_neg_int = num_negtive_int, _batch_size_int = batch_size_int, _sequence_size_int = sequence_size_int)
    dataLoader.set_danmu_table("danmu_test", "data/danmu_test_test.csv")
    dataLoader.prepare_data()
    model = load_model("data/model.h5", custom_objects={'Attention' : Attention})
    sentence_embedding_model = Model(inputs=model.inputs, outputs=model.get_layer('sentence_embedding').output)
    sentence_embedding_file = open("data/danmu_embedding_test.csv", 'w')
    sentence_embedding_file.write("aid\tuser\tsentence_embedding\n")
    aid_string = ""
    write_batch_list = []
    write_batch_size = 1000
    count_int = 0
    for data_tuple in dataLoader.get_batch_aid_user_content():
        new_aid_string = data_tuple[1]
        if new_aid_string != aid_string:
            print("aid: " + new_aid_string)
            aid_string = new_aid_string
        embedding_array = sentence_embedding_model.predict(data_tuple[0])
        for i_int in range(batch_size_int):
            write_batch_list.append(("%s\t%s\t%s\n"%(data_tuple[1], data_tuple[2][i_int], ','.join([str(item_int) for item_int in list(embedding_array[i_int])]))))
        if len(write_batch_list) >= write_batch_size:
            sentence_embedding_file.writelines(write_batch_list)
            write_batch_list.clear()
    sentence_embedding_file.close() 

if __name__=="__main__":
    generate_feature()
