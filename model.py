import numpy as np
from keras.models import Model
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers import Dropout, Dense, Lambda, merge, concatenate, Activation, Input, Multiply, dot
from keras.engine.topology import Layer
import tensorflow as tf
import pdb

from keras import backend as K
from keras import initializers,regularizers,constraints

def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)
class Attention(Layer):
    """
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    How to use:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.
    Note: The layer has been tested with Keras 2.0.6
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())
        # next add a Dense layer (for classification/regression) or whatever...
    """

    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')
        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)
        self.bias = bias
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3
        #pdb.set_trace()
        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        self.u = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)
        super(Attention, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = dot_product(x, self.W)
        if self.bias:
            uit += self.b
        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)
        a = K.exp(ait)
        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())
        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]


class HierarchicalAttentionNetword:
    
    def __init__(self, maxSeq, num_positive = 2, num_negtive = 0,embWeights=None, embeddingSize = None, vocabSize = None,wordRnnSize=5, sentenceRnnSize=5,dropWordEmb = 0.2, dropWordRnnOut = 0.2, dropSentenceRnnOut = 0.5):
               #pdb.set_trace()
        #input_size = 1 + num_positive + num_negtive
        main_input = Input(shape=(maxSeq,), dtype="int32", name="main_input")
        pos_inputs = [Input(shape=(maxSeq,), dtype="int32", name="pos_input_{}".format(i)) for i in range(num_positive)]
        neg_inputs = [Input(shape=(maxSeq,), dtype="int32", name="neg_input_{}".format(i)) for i in range(num_negtive)]
        #wordInp = Input(shape=(maxSeq,),dtype='int32')

        if embWeights is None:
            embedding = Embedding(vocabSize, embeddingSize, input_length=maxSeq, trainable=True)
            x = embedding(main_input)
            x_pos = [embedding(i) for i in pos_inputs]
            x_neg = [embedding(i) for i in neg_inputs]
        else:
            embedding = Embedding(embWeights.shape[0], embWeights.shape[1], weights=[embWeights], trainable=False)
            x = embedding(main_input)
            x_pos = [embedding(i) for i in pos_inputs]
            x_neg = [embedding(i) for i in neg_inputs]
        #average = Lambda(antirectifier, output_shape=antirectifier_output_shape, name='average')
        #main_ave = average(x)
        #pos_ave = [average(i) for i in x_pos]
        #neg_ave = [average(i) for i in x_neg]
        #cos_p = [merge([main_ave, l], mode=lambda x: cossim(x), output_shape=(1,), name='p_cos_sim_{}'.format(i)) for i, l in enumerate(pos_ave)]
        #cos_n = [merge([main_ave, l], mode=lambda x: cossim(x), output_shape=(1,), name='n_cos_sim_{}'.format(i)) for i, l in enumerate(neg_ave)]
        #z = concatenate(cos_p + cos_n, axis=1, name='concatenate')
        #pred = Activation('softmax')(z)
        #model = Model(inputs=[main_input] + pos_inputs + neg_inputs, outputs=pred)
        #model.compile(optimizer='adam', loss='categorical_crossentropy')
        #self.model = model
        #wordRnn = Bidirectional(GRU(wordRnnSize,return_sequences=True))(x)
        #wordRnn_pos = [Bidirectional(GRU(wordRnnSize, return_sequences=True))(i_pos) for i_pos in x_pos]
        #pdb.set_trace()
        wordRnn = Bidirectional(GRU(wordRnnSize, return_sequences=True))(x)
        wordRnn_pos = [Bidirectional(GRU(wordRnnSize, return_sequences=True))(i_pos) for i_pos in x_pos]
        wordRnn_neg = [Bidirectional(GRU(wordRnnSize, return_sequences=True))(i_neg) for i_neg in x_neg]
        #word_dense = TimeDistributed(Dense(2*wordRnnSize))(wordRnn)
        #word_dense_pos = [TimeDistributed(Dense(2*wordRnnSize))(i_dense_pos) for i_dense_pos in wordRnn_pos]
        #word_dense_neg = [TimeDistributed(Dense(2*wordRnnSize))(i_dense_neg) for i_dense_neg in wordRnn_neg]
        word_dense = Dense(2*wordRnnSize, name='word_dense_main')(wordRnn)
        word_dense_pos = [Dense(2*wordRnnSize)(i_dense_pos) for i_dense_pos in wordRnn_pos]
        word_dense_neg = [Dense(2*wordRnnSize)(i_dense_neg) for i_dense_neg in wordRnn_neg]

        word_attention = Attention(name='word_attention')(word_dense)
        word_attention_pos = [Attention()(i_dense_pos) for i_dense_pos in word_dense_pos]
        word_attention_neg = [Attention()(i_dense_neg) for i_dense_neg in word_dense_neg]

        #sentence_embedding = K.sum(word_attention, axis=1)
        #sentence_embedding_pos = [K.sum(wordRnn_pos_i, axis=1) for wordRnn_pos_i in word_attention_pos]
        #sentence_embedding_neg = [K.sum(wordRnn_neg_i, axis=1) for wordRnn_neg_i in word_attention_neg]

        #sentence_embedding = Lambda((lambda x: K.sum(x, axis=1)), output_shape=(2*wordRnnSize,))(word_attention)
        #sentence_embedding_pos = [Lambda((lambda x: K.sum(x, axis=1)), output_shape=(2*wordRnnSize,))(wordRnn_pos_i) for wordRnn_pos_i in word_attention_pos]
        #sentence_embedding_neg = [Lambda((lambda x: K.sum(x, axis=1)), output_shape=(2*wordRnnSize,))(wordRnn_neg_i) for wordRnn_neg_i in word_attention_neg]

        sentence_embedding = Dense(wordRnnSize*2, activation='softmax', name='sentence_embedding')(word_attention)
        sentence_embedding_pos = [Dense(wordRnnSize*2, activation='softmax')(wordRnn_pos_i) for wordRnn_pos_i in word_attention_pos]
        sentence_embedding_neg = [Dense(wordRnnSize*2, activation='softmax')(wordRnn_neg_i) for wordRnn_neg_i in word_attention_neg]

        #print("sentence embedding output shape #####:")
        #print(sentence_embedding.output_shape)

        '''1'''
        #cos_pos = [merge([sentence_embedding, pos], mode=lambda x: (K.batch_dot(x[0], x[1], axes=[1,1])/(K.l2_normalize(x[0], axis=1))/(K.l2_normalize(x[1], axis=1))), output_shape=(1,)) for i,pos in enumerate(sentence_embedding_pos)]
        #cos_neg = [merge([sentence_embedding, neg], mode=lambda x: (K.batch_dot(x[0], x[1], axes=[1,1])/(K.l2_normalize(x[0], axis=1))/(K.l2_normalize(x[1], axis=1))), output_shape=(1,)) for i,neg in enumerate(sentence_embedding_neg)]
        '''2'''
        #cos_pos = [merge([sentence_embedding, pos], mode=lambda x: (K.batch_dot(x[0], x[1], axes=[1,1])), output_shape=(1,)) for i,pos in enumerate(sentence_embedding_pos)]
        #cos_neg = [merge([sentence_embedding, neg], mode=lambda x: (K.batch_dot(x[0], x[1], axes=[1,1])), output_shape=(1,)) for i,neg in enumerate(sentence_embedding_neg)]
        '''3'''
        #def cossim(x):
        #    dot_products = K.batch_dot(x[0], x[1], axes=[1,1])
        #    norm0 = tf.norm(x[0], ord=2, axis=1, keep_dims=True)
        #    norm1 = tf.norm(x[1], ord=2, axis=1, keep_dims=True)
        #    return dot_products / norm0 / norm1
        #cos_pos = [merge([sentence_embedding, pos], mode=lambda x: cossim(x), output_shape=(1,)) for i,pos in enumerate(sentence_embedding_pos)]
        #cos_neg = [merge([sentence_embedding, neg], mode=lambda x: cossim(x), output_shape=(1,)) for i,neg in enumerate(sentence_embedding_neg)]
        '''4'''
        cos_pos = [dot([sentence_embedding, pos], axes=1, normalize=True) for i,pos in enumerate(sentence_embedding_pos)]
        cos_neg = [dot([sentence_embedding, neg], axes=1, normalize=True) for i,neg in enumerate(sentence_embedding_neg)]
        '''5'''
        #cos_pos = [merge([sentence_embedding, pos], mode='cos', output_shape=(1,)) for i,pos in enumerate(sentence_embedding_pos)]
        #cos_neg = [merge([sentence_embedding, neg], mode='cos', output_shape=(1,)) for i,neg in enumerate(sentence_embedding_neg)]

        z = concatenate(cos_pos + cos_neg, axis=1)
        pred = Activation('softmax')(z)
        model = Model(inputs=[main_input] + pos_inputs + neg_inputs, outputs=pred)
        model.compile(optimizer='adam', loss='categorical_crossentropy')
        #modelSentEncoder = Model(wordInp, word_attention)
        self.model = model
        self.model.summary()
        
    def antirectifier(self, x):
        sums = tf.sum(x, axis=1, keepdims=False)
        normalisers = tf.count_nonzero(
            K.count_nonzero(x, axis=2, keep_dims=False, dtype=K.float32),
            axis=1, keep_dims=True, dtype=K.float32)
        return sums / normalisers

    def antirectifier_output_shape(self, input_shape):
        shape = list(input_shape)
        assert len(shape) == 3  # only valid for 3D tensors
        return (shape[0], shape[-1],)
       
    def fit_generator(self, generator, steps_per_epoch=None, epochs=1, verbose=1):
        """Train the model on tha data fed by `generator`
        Args:
            generator
        """
        self.model.fit_generator(generator, steps_per_epoch=steps_per_epoch, epochs=epochs, verbose=verbose)
    
    def save(self, path):
        self.model.save(path)
    
    def load(self, path):
        self.model = load_model(path)

    def evaluate(self, x, y):
        return self.model.evaluate(x,y)
    
    def predict(self, x):
        return self.model.predict(x)
              
