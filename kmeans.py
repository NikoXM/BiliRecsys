from sklearn.cluster import KMeans
import numpy as np
from word import *

class Danmu_kmeans:
    def __init__(self, n_cluster = 3):
        self.n_cluster = n_cluster
        self.estimator = KMeans(self.n_cluster)
        return
    def kmeans(self, data):
        self.estimator.fit(data)
        return self.estimator.labels
        
if __name__=="__main__":
    word = Word()
    word.build_words()
    id2token_dict = word.get_id2token_dict()
    sentence_embedding_file = open('data/danmu_embedding.csv')
    sentence_embeddings = sentence_embedding_file.readlines()
    sentence_string_list = []
    embeddings_string_list = []
    for sentence_embedding_line in sentence_embeddings:
        temp_list = sentence_embedding_line.split()
        sentence_string_list.append(temp_list[0])
        embeddings_string_list.append(temp_list[1])

    embeddings_float_list = []
    for embedding_string in embeddings_string_list:
        embeddings_float_list.append([float(s_string) for s_string in embedding_string.split(',')])
    embedding_array = np.array(embeddings_float_list)
    print(embedding_array)

    #data = np.random.rand(100,3)
    cluster_size_int = 5
    estimator = KMeans(n_clusters=cluster_size_int)
    estimator.fit(embedding_array)
    label_pred = estimator.labels_

    embedding_cluster_dict = {}
    for i_int in range(cluster_size_int):
        embedding_cluster_dict[i_int] = []
    for index_int in range(len(label_pred)):
        embedding_cluster_dict[label_pred[index_int]].append(sentence_string_list[index_int])
    for i_int in range(cluster_size_int):
        print('cluster %d size: %s'%(i_int, len(embedding_cluster_dict[i_int])))
    
    count = 0
    for cluster_list in embedding_cluster_dict.values():
        sentence_list = []
        for sentence_string in cluster_list:
            ids_list = [int(id_string) for id_string in sentence_string.split(',')]
            sentence_list.append([id2token_dict.get(id_int) for id_int in ids_list])
        print("cluster %d: "%(count))
        print(sentence_list)
        count += 1
    #centroids = estimator.cluster_centers_
    #inertia = estimator.inertia_
    #print(data)
    #print(label_pred)
    #print(centroids)
    #print(inertia)
