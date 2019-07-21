# -*- coding:utf-8 -*-
import tensorflow as tf
from tensorflow.contrib import rnn
import tensorflow.contrib.layers as layers

from tensorflow.python.ops.rnn_cell import GRUCell
from tensorflow.python.ops.rnn_cell import LSTMCell
from tensorflow.python.ops.rnn_cell import MultiRNNCell


class Settings(object):
    def __init__(self):
        self.model_name = 'lstm'
        self.doc_len =  300
        self.batch_size=256
        self.idiom_num=10
        self.hidden_size=100


class Lstm(object):

    def __init__(self,word_embedding,idiom_embedding,settings):
        self.model_name = settings.model_name
        self.doc_len = settings.doc_len
        self.batch_size = settings.batch_size
        self.idiom_num = settings.idiom_num
        self.hidden_size = settings.hidden_size

        self.global_step = tf.Variable(0, trainable=False, name='Global_Step')
        self.update_emas = list()


        # placeholders
        self.tst = tf.placeholder(tf.bool)
        self.keep_prob = tf.placeholder(tf.float32, [])
        self.word_embedding = tf.get_variable(name='word_embedding', shape=word_embedding.shape,
                                             initializer=tf.constant_initializer(word_embedding), trainable=True)
        self.idiom_embedding = tf.get_variable(name='idiom_embedding', shape=idiom_embedding.shape,
                                     initializer=tf.constant_initializer(idiom_embedding), trainable=True)
        with tf.name_scope('Inputs'):
            self.idiom_inputs = tf.placeholder(tf.int32, [None, self.idiom_num]) 
            self.doc_inputs=tf.placeholder(tf.int32, [None,self.doc_len])
            self.loc_inputs=tf.placeholder(tf.int32, [None,1])
            self.label_inputs = tf.placeholder(tf.int32, [None,self.idiom_num])          #label
            self.sequence_len_inputs=tf.placeholder(tf.int32, [None])
        with tf.name_scope('Embedding_layer'):
            self.idiom=tf.nn.embedding_lookup(self.idiom_embedding, self.idiom_inputs) #batchsize*idiom_num*embedding_size
            self.doc=tf.nn.embedding_lookup(self.word_embedding, self.doc_inputs) #batchsize*doc_len*embedding_size
            self.doc=tf.nn.dropout(self.doc, self.keep_prob)

        with tf.variable_scope('LSTM_layer'):
            self.doc=self.lstm_inference(self.doc,self.sequence_len_inputs) # [batchsize, doc_len, 2 * hidden_size]
            self.doc=tf.nn.dropout(self.doc, self.keep_prob)
            self.loc=tf.concat([tf.reshape(tf.range(tf.shape(self.loc_inputs)[0]),[-1,1]),tf.reshape(self.loc_inputs,[-1,1])],1)     # B 2
            self.doc=tf.gather_nd(self.doc,self.loc) # [batchsize, 2 * hidden_size]
            self.doc=tf.reshape(self.doc,[-1,1,2*self.hidden_size])
        with tf.variable_scope('Out_layer'):
            match_matrix = tf.matmul(self.doc, tf.transpose(self.idiom, [0, 2, 1]))  # [batchsize, 1, idiom_num]
            self.output = tf.nn.softmax(tf.reshape(match_matrix,[-1,self.idiom_num]))
        with tf.name_scope('loss'):
            self.loss = tf.reduce_mean(
               tf.nn.softmax_cross_entropy_with_logits(logits=self.output, labels=self.label_inputs))




    def weight_variable(self, shape, name):
        """Create a weight variable with appropriate initialization."""
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name=name)

    def bias_variable(self, shape, name):
        """Create a bias variable with appropriate initialization."""
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name=name)


    def vanilla_attention(self,queries, keys, keys_length):

        queries = tf.expand_dims(queries, 1) # [B, 1, H]
  # Multiplication
        outputs = tf.matmul(queries, tf.transpose(keys, [0, 2, 1])) # [B, 1, T]

  # Mask
        key_masks = tf.sequence_mask(keys_length, tf.shape(keys)[1])   # [B, T]
        key_masks = tf.expand_dims(key_masks, 1) # [B, 1, T]
        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(key_masks, outputs, paddings)  # [B, 1, T]

  # Scale
        outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)

  # Activation
        outputs = tf.nn.softmax(outputs)  # [B, 1, T]

  # Weighted sum
        outputs = tf.matmul(outputs, keys)  # [B, 1, H]
        return tf.reshape(outputs,[-1,self.item_dim])





    def lstm_inference(self, sequence_inputs, sequence_len):
        cell_fw_doc = tf.nn.rnn_cell.LSTMCell(self.hidden_size, initializer=tf.orthogonal_initializer())
        cell_bw_doc = tf.nn.rnn_cell.LSTMCell(self.hidden_size, initializer=tf.orthogonal_initializer())
        h_doc, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw_doc, cell_bw_doc, sequence_inputs, sequence_len, dtype=tf.float32,scope="bi_lstm")
        state_doc = tf.concat(h_doc, 2)  # [batch, sequence_len, 2 * hidden_size]

        return state_doc
