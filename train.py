from model import *
from word import *
from dataLoader import *
from keras.models import load_model
from kmeans import *
import pdb
import sys
import tensorflow as tf
from keras.models import Model

if __name__ == "__main__":
    batch_size_int = 300
    sequence_size_int = 5
    num_positive = 4
    num_negtive = 4
    dataLoader = DataLoader(_num_pos_int = num_positive, _num_neg_int = num_negtive, _batch_size_int = batch_size_int, _sequence_size_int = sequence_size_int)
    dataLoader.set_danmu_table("danmu_test", "data/danmu_test.csv")
    dataLoader.prepare_data()
    model = None
    if sys.argv[1] == "0":
        # embedding trainable = False
        embedding_weights = dataLoader.get_embedding_matrix()
        #print(embedding_weights.shape[0])
        #print(embedding_weights.shape[1])
        #embedding_weights = np.zeros((10,300))
        #hierarchical_model = HierarchicalAttentionNetword(maxSeq = 5, embWeights = embedding_weights, num_positive = num_positive, num_negtive = num_negtive)
        # embedding trainable = True
        hierarchical_model = HierarchicalAttentionNetword(maxSeq = 5, embeddingSize = 300, vocabSize = 332648 ,embWeights = None, num_positive = num_positive, num_negtive = num_negtive)
        hierarchical_model.fit_generator(iter(dataLoader),steps_per_epoch=1000, epochs=3)
        hierarchical_model.save("/home/renhao/github/BiliRecsys/data/model.h5")
        model = hierarchical_model.model
    elif sys.argv[1] == "1":
        model = load_model("/home/renhao/github/BiliRecsys/data/model.h5", custom_objects={'Attention':Attention})
    else:
        print("error arguments 0/1")
    #sentence_embedding_model = Model(inputs=model.inputs, outputs=model.get_layer('sentence_embedding').output)

    #
    ##sentence_embedding_model = model.get_sentence_embedding_model()
    #x_test_list= list()
    #y_test_list = list()
    #max_test = 5
    #count = 0
    #sentence_embedding_file = open("data/danmu_embedding.csv", 'w')
    #sentence_embedding_dict = {}
    #danmu_kmeans = Danmu_kmeans()
    #for test_tuple in dataLoader:
    #    if count >= max_test:
    #        break
    #    prediction = sentence_embedding_model.predict(test_tuple[0])
    #    for i_int in range(batch_size_int):
    #        sentence_embedding_file.write("%s\t%s\n"%(','.join([str(item_int) for item_int in list(test_tuple[0][0][i_int])]),','.join([str(item_int) for item_int in list(prediction[i_int])])))
    #    count += 1
    #    print(prediction)
    ##pdb.set_trace()
    #sentence_embedding_file.close()
