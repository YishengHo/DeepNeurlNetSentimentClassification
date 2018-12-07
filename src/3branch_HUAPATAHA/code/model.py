#-*- coding: utf-8 -*-
#author: Zhen Wu

import tensorflow as tf
# from tensorflow.python.ops import rnn
from tensorflow.contrib import rnn
tf.nn.rnn_cell = rnn

class huapahata(object):

    def __init__(self, max_sen_len, max_doc_len, class_num, embedding_file,
            embedding_dim, hidden_size, user_num, product_num, helpfulness_num, time_num):
        self.max_sen_len = max_sen_len
        self.max_doc_len = max_doc_len
        self.class_num = class_num
        self.embedding_file = embedding_file
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.user_num = user_num
        self.product_num = product_num
        self.helpfulness_num = helpfulness_num
        self.time_num = time_num

        with tf.name_scope('input'):
            self.userid = tf.placeholder(tf.int32, [None], name="user_id")
            self.productid = tf.placeholder(tf.int32, [None], name="product_id")
            self.helpfulnessid = tf.placeholder(tf.int32, [None], name="helpfulness_id")
            self.timeid = tf.placeholder(tf.int32, [None], name="time_id")
            self.input_x = tf.placeholder(tf.int32, [None, self.max_doc_len, self.max_sen_len], name="input_x")
            self.input_x_s = tf.placeholder(tf.int32, [None, self.max_doc_len, self.max_sen_len], name="input_x_s")
            self.input_x_sd = tf.placeholder(tf.int32, [None, self.max_doc_len, self.max_sen_len], name="input_x_sd")
            self.input_y = tf.placeholder(tf.float32, [None, self.class_num], name="input_y")
            self.sen_len = tf.placeholder(tf.int32, [None, self.max_doc_len], name="sen_len")
            self.doc_len = tf.placeholder(tf.int32, [None], name="doc_len")

        with tf.name_scope('weights'):
            self.weights = {
                'softmax': tf.Variable(tf.random_uniform([3 * 8 * self.hidden_size, self.class_num], -0.01, 0.01)),
                'd_softmax': tf.Variable(tf.random_uniform([8 * self.hidden_size, self.class_num], -0.01, 0.01)),
                's_softmax': tf.Variable(tf.random_uniform([8 * self.hidden_size, self.class_num], -0.01, 0.01)),
                'sd_softmax': tf.Variable(tf.random_uniform([8 * self.hidden_size, self.class_num], -0.01, 0.01)),

                'd_u_softmax': tf.Variable(tf.random_uniform([2 * self.hidden_size, self.class_num], -0.01, 0.01)),
                'd_u_wh_1': tf.Variable(tf.random_uniform([2 * self.hidden_size, 2 * self.hidden_size], -0.01, 0.01)),
                'd_u_v_1': tf.Variable(tf.random_uniform([2 * self.hidden_size, 1], -0.01, 0.01)),
                'd_u_wh_2': tf.Variable(tf.random_uniform([2 * self.hidden_size, 2 * self.hidden_size], -0.01, 0.01)),
                'd_u_v_2': tf.Variable(tf.random_uniform([2 * self.hidden_size, 1], -0.01, 0.01)),

                'd_p_softmax': tf.Variable(tf.random_uniform([2 * self.hidden_size, self.class_num], -0.01, 0.01)),
                'd_p_wh_1': tf.Variable(tf.random_uniform([2 * self.hidden_size, 2 * self.hidden_size], -0.01, 0.01)),
                'd_p_v_1': tf.Variable(tf.random_uniform([2 * self.hidden_size, 1], -0.01, 0.01)),
                'd_p_wh_2': tf.Variable(tf.random_uniform([2 * self.hidden_size, 2 * self.hidden_size], -0.01, 0.01)),
                'd_p_v_2': tf.Variable(tf.random_uniform([2 * self.hidden_size, 1], -0.01, 0.01)),

                'd_h_softmax': tf.Variable(tf.random_uniform([2 * self.hidden_size, self.class_num], -0.01, 0.01)),
                'd_h_wh_1': tf.Variable(tf.random_uniform([2 * self.hidden_size, 2 * self.hidden_size], -0.01, 0.01)),
                'd_h_v_1': tf.Variable(tf.random_uniform([2 * self.hidden_size, 1], -0.01, 0.01)),
                'd_h_wh_2': tf.Variable(tf.random_uniform([2 * self.hidden_size, 2 * self.hidden_size], -0.01, 0.01)),
                'd_h_v_2': tf.Variable(tf.random_uniform([2 * self.hidden_size, 1], -0.01, 0.01)),

                'd_t_softmax': tf.Variable(tf.random_uniform([2 * self.hidden_size, self.class_num], -0.01, 0.01)),
                'd_t_wh_1': tf.Variable(tf.random_uniform([2 * self.hidden_size, 2 * self.hidden_size], -0.01, 0.01)),
                'd_t_v_1': tf.Variable(tf.random_uniform([2 * self.hidden_size, 1], -0.01, 0.01)),
                'd_t_wh_2': tf.Variable(tf.random_uniform([2 * self.hidden_size, 2 * self.hidden_size], -0.01, 0.01)),
                'd_t_v_2': tf.Variable(tf.random_uniform([2 * self.hidden_size, 1], -0.01, 0.01)),

                'd_wu_1': tf.Variable(tf.random_uniform([2 * self.hidden_size, 2 * self.hidden_size], -0.01, 0.01)),
                'd_wp_1': tf.Variable(tf.random_uniform([2 * self.hidden_size, 2 * self.hidden_size], -0.01, 0.01)),
                'd_wh_1': tf.Variable(tf.random_uniform([2 * self.hidden_size, 2 * self.hidden_size], -0.01, 0.01)),
                'd_wt_1': tf.Variable(tf.random_uniform([2 * self.hidden_size, 2 * self.hidden_size], -0.01, 0.01)),
                'd_wu_2': tf.Variable(tf.random_uniform([2 * self.hidden_size, 2 * self.hidden_size], -0.01, 0.01)),
                'd_wp_2': tf.Variable(tf.random_uniform([2 * self.hidden_size, 2 * self.hidden_size], -0.01, 0.01)),
                'd_wh_2': tf.Variable(tf.random_uniform([2 * self.hidden_size, 2 * self.hidden_size], -0.01, 0.01)),
                'd_wt_2': tf.Variable(tf.random_uniform([2 * self.hidden_size, 2 * self.hidden_size], -0.01, 0.01)),

                # summary:
                's_u_softmax': tf.Variable(tf.random_uniform([2 * self.hidden_size, self.class_num], -0.01, 0.01)),
                's_u_wh_1': tf.Variable(tf.random_uniform([2 * self.hidden_size, 2 * self.hidden_size], -0.01, 0.01)),
                's_u_v_1': tf.Variable(tf.random_uniform([2 * self.hidden_size, 1], -0.01, 0.01)),
                's_u_wh_2': tf.Variable(tf.random_uniform([2 * self.hidden_size, 2 * self.hidden_size], -0.01, 0.01)),
                's_u_v_2': tf.Variable(tf.random_uniform([2 * self.hidden_size, 1], -0.01, 0.01)),

                's_p_softmax': tf.Variable(tf.random_uniform([2 * self.hidden_size, self.class_num], -0.01, 0.01)),
                's_p_wh_1': tf.Variable(tf.random_uniform([2 * self.hidden_size, 2 * self.hidden_size], -0.01, 0.01)),
                's_p_v_1': tf.Variable(tf.random_uniform([2 * self.hidden_size, 1], -0.01, 0.01)),
                's_p_wh_2': tf.Variable(tf.random_uniform([2 * self.hidden_size, 2 * self.hidden_size], -0.01, 0.01)),
                's_p_v_2': tf.Variable(tf.random_uniform([2 * self.hidden_size, 1], -0.01, 0.01)),

                's_h_softmax': tf.Variable(tf.random_uniform([2 * self.hidden_size, self.class_num], -0.01, 0.01)),
                's_h_wh_1': tf.Variable(tf.random_uniform([2 * self.hidden_size, 2 * self.hidden_size], -0.01, 0.01)),
                's_h_v_1': tf.Variable(tf.random_uniform([2 * self.hidden_size, 1], -0.01, 0.01)),
                's_h_wh_2': tf.Variable(tf.random_uniform([2 * self.hidden_size, 2 * self.hidden_size], -0.01, 0.01)),
                's_h_v_2': tf.Variable(tf.random_uniform([2 * self.hidden_size, 1], -0.01, 0.01)),

                's_t_softmax': tf.Variable(tf.random_uniform([2 * self.hidden_size, self.class_num], -0.01, 0.01)),
                's_t_wh_1': tf.Variable(tf.random_uniform([2 * self.hidden_size, 2 * self.hidden_size], -0.01, 0.01)),
                's_t_v_1': tf.Variable(tf.random_uniform([2 * self.hidden_size, 1], -0.01, 0.01)),
                's_t_wh_2': tf.Variable(tf.random_uniform([2 * self.hidden_size, 2 * self.hidden_size], -0.01, 0.01)),
                's_t_v_2': tf.Variable(tf.random_uniform([2 * self.hidden_size, 1], -0.01, 0.01)),

                's_wu_1': tf.Variable(tf.random_uniform([2 * self.hidden_size, 2 * self.hidden_size], -0.01, 0.01)),
                's_wp_1': tf.Variable(tf.random_uniform([2 * self.hidden_size, 2 * self.hidden_size], -0.01, 0.01)),
                's_wh_1': tf.Variable(tf.random_uniform([2 * self.hidden_size, 2 * self.hidden_size], -0.01, 0.01)),
                's_wt_1': tf.Variable(tf.random_uniform([2 * self.hidden_size, 2 * self.hidden_size], -0.01, 0.01)),
                's_wu_2': tf.Variable(tf.random_uniform([2 * self.hidden_size, 2 * self.hidden_size], -0.01, 0.01)),
                's_wp_2': tf.Variable(tf.random_uniform([2 * self.hidden_size, 2 * self.hidden_size], -0.01, 0.01)),
                's_wh_2': tf.Variable(tf.random_uniform([2 * self.hidden_size, 2 * self.hidden_size], -0.01, 0.01)),
                's_wt_2': tf.Variable(tf.random_uniform([2 * self.hidden_size, 2 * self.hidden_size], -0.01, 0.01)),

                # summary + text
                'sd_u_softmax': tf.Variable(tf.random_uniform([2 * self.hidden_size, self.class_num], -0.01, 0.01)),
                'sd_u_wh_1': tf.Variable(tf.random_uniform([2 * self.hidden_size, 2 * self.hidden_size], -0.01, 0.01)),
                'sd_u_v_1': tf.Variable(tf.random_uniform([2 * self.hidden_size, 1], -0.01, 0.01)),
                'sd_u_wh_2': tf.Variable(tf.random_uniform([2 * self.hidden_size, 2 * self.hidden_size], -0.01, 0.01)),
                'sd_u_v_2': tf.Variable(tf.random_uniform([2 * self.hidden_size, 1], -0.01, 0.01)),

                'sd_p_softmax': tf.Variable(tf.random_uniform([2 * self.hidden_size, self.class_num], -0.01, 0.01)),
                'sd_p_wh_1': tf.Variable(tf.random_uniform([2 * self.hidden_size, 2 * self.hidden_size], -0.01, 0.01)),
                'sd_p_v_1': tf.Variable(tf.random_uniform([2 * self.hidden_size, 1], -0.01, 0.01)),
                'sd_p_wh_2': tf.Variable(tf.random_uniform([2 * self.hidden_size, 2 * self.hidden_size], -0.01, 0.01)),
                'sd_p_v_2': tf.Variable(tf.random_uniform([2 * self.hidden_size, 1], -0.01, 0.01)),

                'sd_h_softmax': tf.Variable(tf.random_uniform([2 * self.hidden_size, self.class_num], -0.01, 0.01)),
                'sd_h_wh_1': tf.Variable(tf.random_uniform([2 * self.hidden_size, 2 * self.hidden_size], -0.01, 0.01)),
                'sd_h_v_1': tf.Variable(tf.random_uniform([2 * self.hidden_size, 1], -0.01, 0.01)),
                'sd_h_wh_2': tf.Variable(tf.random_uniform([2 * self.hidden_size, 2 * self.hidden_size], -0.01, 0.01)),
                'sd_h_v_2': tf.Variable(tf.random_uniform([2 * self.hidden_size, 1], -0.01, 0.01)),

                'sd_t_softmax': tf.Variable(tf.random_uniform([2 * self.hidden_size, self.class_num], -0.01, 0.01)),
                'sd_t_wh_1': tf.Variable(tf.random_uniform([2 * self.hidden_size, 2 * self.hidden_size], -0.01, 0.01)),
                'sd_t_v_1': tf.Variable(tf.random_uniform([2 * self.hidden_size, 1], -0.01, 0.01)),
                'sd_t_wh_2': tf.Variable(tf.random_uniform([2 * self.hidden_size, 2 * self.hidden_size], -0.01, 0.01)),
                'sd_t_v_2': tf.Variable(tf.random_uniform([2 * self.hidden_size, 1], -0.01, 0.01)),

                'sd_wu_1': tf.Variable(tf.random_uniform([2 * self.hidden_size, 2 * self.hidden_size], -0.01, 0.01)),
                'sd_wp_1': tf.Variable(tf.random_uniform([2 * self.hidden_size, 2 * self.hidden_size], -0.01, 0.01)),
                'sd_wh_1': tf.Variable(tf.random_uniform([2 * self.hidden_size, 2 * self.hidden_size], -0.01, 0.01)),
                'sd_wt_1': tf.Variable(tf.random_uniform([2 * self.hidden_size, 2 * self.hidden_size], -0.01, 0.01)),
                'sd_wu_2': tf.Variable(tf.random_uniform([2 * self.hidden_size, 2 * self.hidden_size], -0.01, 0.01)),
                'sd_wp_2': tf.Variable(tf.random_uniform([2 * self.hidden_size, 2 * self.hidden_size], -0.01, 0.01)),
                'sd_wh_2': tf.Variable(tf.random_uniform([2 * self.hidden_size, 2 * self.hidden_size], -0.01, 0.01)),
                'sd_wt_2': tf.Variable(tf.random_uniform([2 * self.hidden_size, 2 * self.hidden_size], -0.01, 0.01)),
            }

        with tf.name_scope('biases'):
            self.biases = {
                'softmax': tf.Variable(tf.random_uniform([self.class_num], -0.01, 0.01)),
                'd_softmax': tf.Variable(tf.random_uniform([self.class_num], -0.01, 0.01)),
                's_softmax': tf.Variable(tf.random_uniform([self.class_num], -0.01, 0.01)),
                'sd_softmax': tf.Variable(tf.random_uniform([self.class_num], -0.01, 0.01)),

                'd_u_softmax': tf.Variable(tf.random_uniform([self.class_num], -0.01, 0.01)),
                'd_u_wh_1': tf.Variable(tf.random_uniform([2 * self.hidden_size], -0.01, 0.01)),
                'd_u_wh_2': tf.Variable(tf.random_uniform([2 * self.hidden_size], -0.01, 0.01)),

                'd_p_softmax': tf.Variable(tf.random_uniform([self.class_num], -0.01, 0.01)),
                'd_p_wh_1': tf.Variable(tf.random_uniform([2 * self.hidden_size], -0.01, 0.01)),
                'd_p_wh_2': tf.Variable(tf.random_uniform([2 * self.hidden_size], -0.01, 0.01)),

                'd_h_softmax': tf.Variable(tf.random_uniform([self.class_num], -0.01, 0.01)),
                'd_h_wh_1': tf.Variable(tf.random_uniform([2 * self.hidden_size], -0.01, 0.01)),
                'd_h_wh_2': tf.Variable(tf.random_uniform([2 * self.hidden_size], -0.01, 0.01)),

                'd_t_softmax': tf.Variable(tf.random_uniform([self.class_num], -0.01, 0.01)),
                'd_t_wh_1': tf.Variable(tf.random_uniform([2 * self.hidden_size], -0.01, 0.01)),
                'd_t_wh_2': tf.Variable(tf.random_uniform([2 * self.hidden_size], -0.01, 0.01)),

                # summary:
                's_u_softmax': tf.Variable(tf.random_uniform([self.class_num], -0.01, 0.01)),
                's_u_wh_1': tf.Variable(tf.random_uniform([2 * self.hidden_size], -0.01, 0.01)),
                's_u_wh_2': tf.Variable(tf.random_uniform([2 * self.hidden_size], -0.01, 0.01)),

                's_p_softmax': tf.Variable(tf.random_uniform([self.class_num], -0.01, 0.01)),
                's_p_wh_1': tf.Variable(tf.random_uniform([2 * self.hidden_size], -0.01, 0.01)),
                's_p_wh_2': tf.Variable(tf.random_uniform([2 * self.hidden_size], -0.01, 0.01)),

                's_h_softmax': tf.Variable(tf.random_uniform([self.class_num], -0.01, 0.01)),
                's_h_wh_1': tf.Variable(tf.random_uniform([2 * self.hidden_size], -0.01, 0.01)),
                's_h_wh_2': tf.Variable(tf.random_uniform([2 * self.hidden_size], -0.01, 0.01)),

                's_t_softmax': tf.Variable(tf.random_uniform([self.class_num], -0.01, 0.01)),
                's_t_wh_1': tf.Variable(tf.random_uniform([2 * self.hidden_size], -0.01, 0.01)),
                's_t_wh_2': tf.Variable(tf.random_uniform([2 * self.hidden_size], -0.01, 0.01)),

                # summary + text:
                'sd_u_softmax': tf.Variable(tf.random_uniform([self.class_num], -0.01, 0.01)),
                'sd_u_wh_1': tf.Variable(tf.random_uniform([2 * self.hidden_size], -0.01, 0.01)),
                'sd_u_wh_2': tf.Variable(tf.random_uniform([2 * self.hidden_size], -0.01, 0.01)),

                'sd_p_softmax': tf.Variable(tf.random_uniform([self.class_num], -0.01, 0.01)),
                'sd_p_wh_1': tf.Variable(tf.random_uniform([2 * self.hidden_size], -0.01, 0.01)),
                'sd_p_wh_2': tf.Variable(tf.random_uniform([2 * self.hidden_size], -0.01, 0.01)),

                'sd_h_softmax': tf.Variable(tf.random_uniform([self.class_num], -0.01, 0.01)),
                'sd_h_wh_1': tf.Variable(tf.random_uniform([2 * self.hidden_size], -0.01, 0.01)),
                'sd_h_wh_2': tf.Variable(tf.random_uniform([2 * self.hidden_size], -0.01, 0.01)),

                'sd_t_softmax': tf.Variable(tf.random_uniform([self.class_num], -0.01, 0.01)),
                'sd_t_wh_1': tf.Variable(tf.random_uniform([2 * self.hidden_size], -0.01, 0.01)),
                'sd_t_wh_2': tf.Variable(tf.random_uniform([2 * self.hidden_size], -0.01, 0.01)),
            }

        with tf.name_scope('embedding'):
            self.word_embedding = tf.constant(self.embedding_file, name="word_embedding", dtype=tf.float32)
            self.x_d = tf.nn.embedding_lookup(self.word_embedding, self.input_x)
            self.x_s = tf.nn.embedding_lookup(self.word_embedding, self.input_x_s)
            self.x_sd = tf.nn.embedding_lookup(self.word_embedding, self.input_x_sd)
            self.user_embedding = tf.Variable(tf.random_uniform([self.user_num, 2*self.hidden_size], -0.01, 0.01), dtype=tf.float32)
            self.product_embedding = tf.Variable(tf.random_uniform([self.product_num, 2*self.hidden_size], -0.01, 0.01), dtype=tf.float32)
            self.helpfulness_embedding = tf.Variable(tf.random_uniform([self.helpfulness_num, 2*self.hidden_size], -0.01, 0.01), dtype=tf.float32)
            self.time_embedding = tf.Variable(tf.random_uniform([self.time_num, 2*self.hidden_size], -0.01, 0.01), dtype=tf.float32)
            self.user = tf.nn.embedding_lookup(self.user_embedding, self.userid)
            self.product = tf.nn.embedding_lookup(self.product_embedding, self.productid)
            self.helpfulness = tf.nn.embedding_lookup(self.helpfulness_embedding, self.helpfulnessid)
            self.time = tf.nn.embedding_lookup(self.time_embedding, self.timeid)


    def softmax(self, inputs, length, max_length):
        inputs = tf.cast(inputs, tf.float32)
        inputs = tf.exp(inputs)
        length = tf.reshape(length, [-1])
        mask = tf.reshape(tf.cast(tf.sequence_mask(length, max_length), tf.float32), tf.shape(inputs))
        inputs *= mask
        _sum = tf.reduce_sum(inputs, reduction_indices=2, keep_dims=True) + 1e-9
        return inputs / _sum

    # user_session
    def user_attention_d(self):
        inputs = tf.reshape(self.x_d, [-1, self.max_sen_len, self.embedding_dim])
        sen_len = tf.reshape(self.sen_len, [-1])
        with tf.name_scope('d_u_word_encode'):
            outputs, state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=tf.nn.rnn_cell.LSTMCell(self.hidden_size, forget_bias=1.0),
                cell_bw=tf.nn.rnn_cell.LSTMCell(self.hidden_size, forget_bias=1.0),
                inputs=inputs,
                sequence_length=sen_len,
                dtype=tf.float32,
                scope='d_u_word'
            )
            outputs = tf.concat(outputs, 2)

        batch_size = tf.shape(outputs)[0]
        with tf.name_scope('d_u_word_attention'):
            output = tf.reshape(outputs, [-1, 2 * self.hidden_size])
            u = tf.matmul(output, self.weights['d_u_wh_1']) + self.biases['d_u_wh_1']
            u = tf.reshape(u, [-1, self.max_doc_len*self.max_sen_len, 2*self.hidden_size])
            u += tf.matmul(self.user, self.weights['d_wu_1'])[:,None,:]
            u = tf.tanh(u)
            u = tf.reshape(u, [-1, 2 * self.hidden_size])
            alpha = tf.reshape(tf.matmul(u, self.weights['d_u_v_1']),
                               [batch_size, 1, self.max_sen_len])
            alpha = self.softmax(alpha, self.sen_len, self.max_sen_len)
            outputs = tf.matmul(alpha, outputs)


        outputs = tf.reshape(outputs, [-1, self.max_doc_len, 2 * self.hidden_size])
        with tf.name_scope('d_u_sentence_encode'):
            outputs, state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=tf.nn.rnn_cell.LSTMCell(self.hidden_size, forget_bias=1.0),
                cell_bw=tf.nn.rnn_cell.LSTMCell(self.hidden_size, forget_bias=1.0),
                inputs=outputs,
                sequence_length=self.doc_len,
                dtype=tf.float32,
                scope='d_u_sentence'
            )
            outputs = tf.concat(outputs, 2)

        batch_size = tf.shape(outputs)[0]
        with tf.name_scope('d_u_sentence_attention'):
            output = tf.reshape(outputs, [-1, 2 * self.hidden_size])
            u = tf.matmul(output, self.weights['d_u_wh_2']) + self.biases['d_u_wh_2']
            u = tf.reshape(u, [-1, self.max_doc_len, 2*self.hidden_size])
            u += tf.matmul(self.user, self.weights['d_wu_2'])[:,None,:]
            u = tf.tanh(u)
            u = tf.reshape(u, [-1, 2*self.hidden_size])
            alpha = tf.reshape(tf.matmul(u, self.weights['d_u_v_2']),
                               [batch_size, 1, self.max_doc_len])
            alpha = self.softmax(alpha, self.doc_len, self.max_doc_len)
            outputs = tf.matmul(alpha, outputs)

        with tf.name_scope('d_u_softmax'):
            self.d_u_doc = tf.reshape(outputs, [batch_size, 2 * self.hidden_size])
            self.d_u_scores = tf.matmul(self.d_u_doc, self.weights['d_u_softmax']) + self.biases['d_u_softmax']
            self.d_u_predictions = tf.argmax(self.d_u_scores, 1, name="d_u_predictions")

        with tf.name_scope("d_u_loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.d_u_scores, labels=self.input_y)
            self.d_u_loss = tf.reduce_mean(losses)

        with tf.name_scope("d_u_accuracy"):
            correct_predictions = tf.equal(self.d_u_predictions, tf.argmax(self.input_y, 1))
            self.d_u_correct_num = tf.reduce_sum(tf.cast(correct_predictions, dtype=tf.int32))
            self.d_u_accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="d_u_accuracy")

    def user_attention_s(self):
        inputs = tf.reshape(self.x_s, [-1, self.max_sen_len, self.embedding_dim])
        sen_len = tf.reshape(self.sen_len, [-1])
        with tf.name_scope('s_u_word_encode'):
            outputs, state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=tf.nn.rnn_cell.LSTMCell(self.hidden_size, forget_bias=1.0),
                cell_bw=tf.nn.rnn_cell.LSTMCell(self.hidden_size, forget_bias=1.0),
                inputs=inputs,
                sequence_length=sen_len,
                dtype=tf.float32,
                scope='s_u_word'
            )
            outputs = tf.concat(outputs, 2)

        batch_size = tf.shape(outputs)[0]
        with tf.name_scope('s_u_word_attention'):
            output = tf.reshape(outputs, [-1, 2 * self.hidden_size])
            u = tf.matmul(output, self.weights['s_u_wh_1']) + self.biases['s_u_wh_1']
            u = tf.reshape(u, [-1, self.max_doc_len*self.max_sen_len, 2*self.hidden_size])
            u += tf.matmul(self.user, self.weights['s_wu_1'])[:,None,:]
            u = tf.tanh(u)
            u = tf.reshape(u, [-1, 2 * self.hidden_size])
            alpha = tf.reshape(tf.matmul(u, self.weights['s_u_v_1']),
                               [batch_size, 1, self.max_sen_len])
            alpha = self.softmax(alpha, self.sen_len, self.max_sen_len)
            outputs = tf.matmul(alpha, outputs)


        outputs = tf.reshape(outputs, [-1, self.max_doc_len, 2 * self.hidden_size])
        with tf.name_scope('s_u_sentence_encode'):
            outputs, state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=tf.nn.rnn_cell.LSTMCell(self.hidden_size, forget_bias=1.0),
                cell_bw=tf.nn.rnn_cell.LSTMCell(self.hidden_size, forget_bias=1.0),
                inputs=outputs,
                sequence_length=self.doc_len,
                dtype=tf.float32,
                scope='s_u_sentence'
            )
            outputs = tf.concat(outputs, 2)

        batch_size = tf.shape(outputs)[0]
        with tf.name_scope('s_u_sentence_attention'):
            output = tf.reshape(outputs, [-1, 2 * self.hidden_size])
            u = tf.matmul(output, self.weights['s_u_wh_2']) + self.biases['s_u_wh_2']
            u = tf.reshape(u, [-1, self.max_doc_len, 2*self.hidden_size])
            u += tf.matmul(self.user, self.weights['s_wu_2'])[:,None,:]
            u = tf.tanh(u)
            u = tf.reshape(u, [-1, 2*self.hidden_size])
            alpha = tf.reshape(tf.matmul(u, self.weights['s_u_v_2']),
                               [batch_size, 1, self.max_doc_len])
            alpha = self.softmax(alpha, self.doc_len, self.max_doc_len)
            outputs = tf.matmul(alpha, outputs)

        with tf.name_scope('s_u_softmax'):
            self.s_u_doc = tf.reshape(outputs, [batch_size, 2 * self.hidden_size])
            self.s_u_scores = tf.matmul(self.s_u_doc, self.weights['s_u_softmax']) + self.biases['s_u_softmax']
            self.s_u_predictions = tf.argmax(self.s_u_scores, 1, name="s_u_predictions")

        with tf.name_scope("s_u_loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.s_u_scores, labels=self.input_y)
            self.s_u_loss = tf.reduce_mean(losses)

        with tf.name_scope("s_u_accuracy"):
            correct_predictions = tf.equal(self.s_u_predictions, tf.argmax(self.input_y, 1))
            self.s_u_correct_num = tf.reduce_sum(tf.cast(correct_predictions, dtype=tf.int32))
            self.s_u_accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="s_u_accuracy")


    def user_attention_sd(self):
        inputs = tf.reshape(self.x_sd, [-1, self.max_sen_len, self.embedding_dim])
        sen_len = tf.reshape(self.sen_len, [-1])
        with tf.name_scope('sd_u_word_encode'):
            outputs, state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=tf.nn.rnn_cell.LSTMCell(self.hidden_size, forget_bias=1.0),
                cell_bw=tf.nn.rnn_cell.LSTMCell(self.hidden_size, forget_bias=1.0),
                inputs=inputs,
                sequence_length=sen_len,
                dtype=tf.float32,
                scope='sd_u_word'
            )
            outputs = tf.concat(outputs, 2)

        batch_size = tf.shape(outputs)[0]
        with tf.name_scope('sd_u_word_attention'):
            output = tf.reshape(outputs, [-1, 2 * self.hidden_size])
            u = tf.matmul(output, self.weights['sd_u_wh_1']) + self.biases['sd_u_wh_1']
            u = tf.reshape(u, [-1, self.max_doc_len*self.max_sen_len, 2*self.hidden_size])
            u += tf.matmul(self.user, self.weights['sd_wu_1'])[:,None,:]
            u = tf.tanh(u)
            u = tf.reshape(u, [-1, 2 * self.hidden_size])
            alpha = tf.reshape(tf.matmul(u, self.weights['sd_u_v_1']),
                               [batch_size, 1, self.max_sen_len])
            alpha = self.softmax(alpha, self.sen_len, self.max_sen_len)
            outputs = tf.matmul(alpha, outputs)


        outputs = tf.reshape(outputs, [-1, self.max_doc_len, 2 * self.hidden_size])
        with tf.name_scope('sd_u_sentence_encode'):
            outputs, state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=tf.nn.rnn_cell.LSTMCell(self.hidden_size, forget_bias=1.0),
                cell_bw=tf.nn.rnn_cell.LSTMCell(self.hidden_size, forget_bias=1.0),
                inputs=outputs,
                sequence_length=self.doc_len,
                dtype=tf.float32,
                scope='sd_u_sentence'
            )
            outputs = tf.concat(outputs, 2)

        batch_size = tf.shape(outputs)[0]
        with tf.name_scope('sd_u_sentence_attention'):
            output = tf.reshape(outputs, [-1, 2 * self.hidden_size])
            u = tf.matmul(output, self.weights['sd_u_wh_2']) + self.biases['sd_u_wh_2']
            u = tf.reshape(u, [-1, self.max_doc_len, 2*self.hidden_size])
            u += tf.matmul(self.user, self.weights['sd_wu_2'])[:,None,:]
            u = tf.tanh(u)
            u = tf.reshape(u, [-1, 2*self.hidden_size])
            alpha = tf.reshape(tf.matmul(u, self.weights['sd_u_v_2']),
                               [batch_size, 1, self.max_doc_len])
            alpha = self.softmax(alpha, self.doc_len, self.max_doc_len)
            outputs = tf.matmul(alpha, outputs)

        with tf.name_scope('sd_u_softmax'):
            self.sd_u_doc = tf.reshape(outputs, [batch_size, 2 * self.hidden_size])
            self.sd_u_scores = tf.matmul(self.sd_u_doc, self.weights['sd_u_softmax']) + self.biases['sd_u_softmax']
            self.sd_u_predictions = tf.argmax(self.sd_u_scores, 1, name="sd_u_predictions")

        with tf.name_scope("sd_u_loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.sd_u_scores, labels=self.input_y)
            self.sd_u_loss = tf.reduce_mean(losses)

        with tf.name_scope("sd_u_accuracy"):
            correct_predictions = tf.equal(self.sd_u_predictions, tf.argmax(self.input_y, 1))
            self.sd_u_correct_num = tf.reduce_sum(tf.cast(correct_predictions, dtype=tf.int32))
            self.sd_u_accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="sd_u_accuracy")


    # product_session
    def product_attention_d(self):
        inputs = tf.reshape(self.x_d, [-1, self.max_sen_len, self.embedding_dim])
        sen_len = tf.reshape(self.sen_len, [-1])
        with tf.name_scope('d_p_word_encode'):
            outputs, state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=tf.nn.rnn_cell.LSTMCell(self.hidden_size, forget_bias=1.0),
                cell_bw=tf.nn.rnn_cell.LSTMCell(self.hidden_size, forget_bias=1.0),
                inputs=inputs,
                sequence_length=sen_len,
                dtype=tf.float32,
                scope='d_p_word'
            )
            outputs = tf.concat(outputs, 2)

        batch_size = tf.shape(outputs)[0]
        with tf.name_scope('d_p_word_attention'):
            output = tf.reshape(outputs, [-1, 2 * self.hidden_size])
            u = tf.matmul(output, self.weights['d_p_wh_1']) + self.biases['d_p_wh_1']
            u = tf.reshape(u, [-1, self.max_doc_len*self.max_sen_len, 2*self.hidden_size])
            u += tf.matmul(self.product, self.weights['d_wp_1'])[:,None,:]
            u = tf.tanh(u)
            u = tf.reshape(u, [-1, 2 * self.hidden_size])
            alpha = tf.reshape(tf.matmul(u, self.weights['d_p_v_1']),
                               [batch_size, 1, self.max_sen_len])
            alpha = self.softmax(alpha, self.sen_len, self.max_sen_len)
            outputs = tf.matmul(alpha, outputs)


        outputs = tf.reshape(outputs, [-1, self.max_doc_len, 2 * self.hidden_size])
        with tf.name_scope('d_p_sentence_encode'):
            outputs, state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=tf.nn.rnn_cell.LSTMCell(self.hidden_size, forget_bias=1.0),
                cell_bw=tf.nn.rnn_cell.LSTMCell(self.hidden_size, forget_bias=1.0),
                inputs=outputs,
                sequence_length=self.doc_len,
                dtype=tf.float32,
                scope='d_p_sentence'
            )
            outputs = tf.concat(outputs, 2)

        batch_size = tf.shape(outputs)[0]
        with tf.name_scope('d_p_sentence_attention'):
            output = tf.reshape(outputs, [-1, 2 * self.hidden_size])
            u = tf.matmul(output, self.weights['d_p_wh_2']) + self.biases['d_p_wh_2']
            u = tf.reshape(u, [-1, self.max_doc_len, 2*self.hidden_size])
            u += tf.matmul(self.product, self.weights['d_wp_2'])[:,None,:]
            u = tf.tanh(u)
            u = tf.reshape(u, [-1, 2*self.hidden_size])
            alpha = tf.reshape(tf.matmul(u, self.weights['d_p_v_2']),
                               [batch_size, 1, self.max_doc_len])
            alpha = self.softmax(alpha, self.doc_len, self.max_doc_len)
            outputs = tf.matmul(alpha, outputs)

        with tf.name_scope('d_p_softmax'):
            self.d_p_doc = tf.reshape(outputs, [batch_size, 2 * self.hidden_size])
            self.d_p_scores = tf.matmul(self.d_p_doc, self.weights['d_p_softmax']) + self.biases['d_p_softmax']
            self.d_p_predictions = tf.argmax(self.d_p_scores, 1, name="d_p_predictions")

        with tf.name_scope("d_p_loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.d_p_scores, labels=self.input_y)
            self.d_p_loss = tf.reduce_mean(losses)

        with tf.name_scope("d_p_accuracy"):
            correct_predictions = tf.equal(self.d_p_predictions, tf.argmax(self.input_y, 1))
            self.d_p_correct_num = tf.reduce_sum(tf.cast(correct_predictions, dtype=tf.int32))
            self.d_p_accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="d_p_accuracy")

    def product_attention_s(self):
        inputs = tf.reshape(self.x_s, [-1, self.max_sen_len, self.embedding_dim])
        sen_len = tf.reshape(self.sen_len, [-1])
        with tf.name_scope('s_p_word_encode'):
            outputs, state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=tf.nn.rnn_cell.LSTMCell(self.hidden_size, forget_bias=1.0),
                cell_bw=tf.nn.rnn_cell.LSTMCell(self.hidden_size, forget_bias=1.0),
                inputs=inputs,
                sequence_length=sen_len,
                dtype=tf.float32,
                scope='s_p_word'
            )
            outputs = tf.concat(outputs, 2)

        batch_size = tf.shape(outputs)[0]
        with tf.name_scope('s_p_word_attention'):
            output = tf.reshape(outputs, [-1, 2 * self.hidden_size])
            u = tf.matmul(output, self.weights['s_p_wh_1']) + self.biases['s_p_wh_1']
            u = tf.reshape(u, [-1, self.max_doc_len*self.max_sen_len, 2*self.hidden_size])
            u += tf.matmul(self.product, self.weights['s_wp_1'])[:,None,:]
            u = tf.tanh(u)
            u = tf.reshape(u, [-1, 2 * self.hidden_size])
            alpha = tf.reshape(tf.matmul(u, self.weights['s_p_v_1']),
                               [batch_size, 1, self.max_sen_len])
            alpha = self.softmax(alpha, self.sen_len, self.max_sen_len)
            outputs = tf.matmul(alpha, outputs)


        outputs = tf.reshape(outputs, [-1, self.max_doc_len, 2 * self.hidden_size])
        with tf.name_scope('s_p_sentence_encode'):
            outputs, state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=tf.nn.rnn_cell.LSTMCell(self.hidden_size, forget_bias=1.0),
                cell_bw=tf.nn.rnn_cell.LSTMCell(self.hidden_size, forget_bias=1.0),
                inputs=outputs,
                sequence_length=self.doc_len,
                dtype=tf.float32,
                scope='s_p_sentence'
            )
            outputs = tf.concat(outputs, 2)

        batch_size = tf.shape(outputs)[0]
        with tf.name_scope('s_p_sentence_attention'):
            output = tf.reshape(outputs, [-1, 2 * self.hidden_size])
            u = tf.matmul(output, self.weights['s_p_wh_2']) + self.biases['s_p_wh_2']
            u = tf.reshape(u, [-1, self.max_doc_len, 2*self.hidden_size])
            u += tf.matmul(self.product, self.weights['s_wp_2'])[:,None,:]
            u = tf.tanh(u)
            u = tf.reshape(u, [-1, 2*self.hidden_size])
            alpha = tf.reshape(tf.matmul(u, self.weights['s_p_v_2']),
                               [batch_size, 1, self.max_doc_len])
            alpha = self.softmax(alpha, self.doc_len, self.max_doc_len)
            outputs = tf.matmul(alpha, outputs)

        with tf.name_scope('s_p_softmax'):
            self.s_p_doc = tf.reshape(outputs, [batch_size, 2 * self.hidden_size])
            self.s_p_scores = tf.matmul(self.s_p_doc, self.weights['s_p_softmax']) + self.biases['s_p_softmax']
            self.s_p_predictions = tf.argmax(self.s_p_scores, 1, name="s_p_predictions")

        with tf.name_scope("s_p_loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.s_p_scores, labels=self.input_y)
            self.s_p_loss = tf.reduce_mean(losses)

        with tf.name_scope("s_p_accuracy"):
            correct_predictions = tf.equal(self.s_p_predictions, tf.argmax(self.input_y, 1))
            self.s_p_correct_num = tf.reduce_sum(tf.cast(correct_predictions, dtype=tf.int32))
            self.s_p_accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="s_p_accuracy")


    def product_attention_sd(self):
        inputs = tf.reshape(self.x_sd, [-1, self.max_sen_len, self.embedding_dim])
        sen_len = tf.reshape(self.sen_len, [-1])
        with tf.name_scope('sd_p_word_encode'):
            outputs, state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=tf.nn.rnn_cell.LSTMCell(self.hidden_size, forget_bias=1.0),
                cell_bw=tf.nn.rnn_cell.LSTMCell(self.hidden_size, forget_bias=1.0),
                inputs=inputs,
                sequence_length=sen_len,
                dtype=tf.float32,
                scope='sd_p_word'
            )
            outputs = tf.concat(outputs, 2)

        batch_size = tf.shape(outputs)[0]
        with tf.name_scope('sd_p_word_attention'):
            output = tf.reshape(outputs, [-1, 2 * self.hidden_size])
            u = tf.matmul(output, self.weights['sd_p_wh_1']) + self.biases['sd_p_wh_1']
            u = tf.reshape(u, [-1, self.max_doc_len*self.max_sen_len, 2*self.hidden_size])
            u += tf.matmul(self.product, self.weights['sd_wp_1'])[:,None,:]
            u = tf.tanh(u)
            u = tf.reshape(u, [-1, 2 * self.hidden_size])
            alpha = tf.reshape(tf.matmul(u, self.weights['sd_p_v_1']),
                               [batch_size, 1, self.max_sen_len])
            alpha = self.softmax(alpha, self.sen_len, self.max_sen_len)
            outputs = tf.matmul(alpha, outputs)


        outputs = tf.reshape(outputs, [-1, self.max_doc_len, 2 * self.hidden_size])
        with tf.name_scope('sd_p_sentence_encode'):
            outputs, state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=tf.nn.rnn_cell.LSTMCell(self.hidden_size, forget_bias=1.0),
                cell_bw=tf.nn.rnn_cell.LSTMCell(self.hidden_size, forget_bias=1.0),
                inputs=outputs,
                sequence_length=self.doc_len,
                dtype=tf.float32,
                scope='sd_p_sentence'
            )
            outputs = tf.concat(outputs, 2)

        batch_size = tf.shape(outputs)[0]
        with tf.name_scope('sd_p_sentence_attention'):
            output = tf.reshape(outputs, [-1, 2 * self.hidden_size])
            u = tf.matmul(output, self.weights['sd_p_wh_2']) + self.biases['sd_p_wh_2']
            u = tf.reshape(u, [-1, self.max_doc_len, 2*self.hidden_size])
            u += tf.matmul(self.product, self.weights['sd_wp_2'])[:,None,:]
            u = tf.tanh(u)
            u = tf.reshape(u, [-1, 2*self.hidden_size])
            alpha = tf.reshape(tf.matmul(u, self.weights['sd_p_v_2']),
                               [batch_size, 1, self.max_doc_len])
            alpha = self.softmax(alpha, self.doc_len, self.max_doc_len)
            outputs = tf.matmul(alpha, outputs)

        with tf.name_scope('sd_p_softmax'):
            self.sd_p_doc = tf.reshape(outputs, [batch_size, 2 * self.hidden_size])
            self.sd_p_scores = tf.matmul(self.sd_p_doc, self.weights['sd_p_softmax']) + self.biases['sd_p_softmax']
            self.sd_p_predictions = tf.argmax(self.sd_p_scores, 1, name="sd_p_predictions")

        with tf.name_scope("sd_p_loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.sd_p_scores, labels=self.input_y)
            self.sd_p_loss = tf.reduce_mean(losses)

        with tf.name_scope("sd_p_accuracy"):
            correct_predictions = tf.equal(self.sd_p_predictions, tf.argmax(self.input_y, 1))
            self.sd_p_correct_num = tf.reduce_sum(tf.cast(correct_predictions, dtype=tf.int32))
            self.sd_p_accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="sd_p_accuracy")


    # helpfulness_session
    def helpfulness_attention_d(self):
        inputs = tf.reshape(self.x_d, [-1, self.max_sen_len, self.embedding_dim])
        sen_len = tf.reshape(self.sen_len, [-1])
        with tf.name_scope('d_h_word_encode'):
            outputs, state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=tf.nn.rnn_cell.LSTMCell(self.hidden_size, forget_bias=1.0),
                cell_bw=tf.nn.rnn_cell.LSTMCell(self.hidden_size, forget_bias=1.0),
                inputs=inputs,
                sequence_length=sen_len,
                dtype=tf.float32,
                scope='d_h_word'
            )
            outputs = tf.concat(outputs, 2)

        batch_size = tf.shape(outputs)[0]
        with tf.name_scope('d_h_word_attention'):
            output = tf.reshape(outputs, [-1, 2 * self.hidden_size])
            u = tf.matmul(output, self.weights['d_h_wh_1']) + self.biases['d_h_wh_1']
            u = tf.reshape(u, [-1, self.max_doc_len*self.max_sen_len, 2*self.hidden_size])
            u += tf.matmul(self.helpfulness, self.weights['d_wh_1'])[:,None,:]
            u = tf.tanh(u)
            u = tf.reshape(u, [-1, 2 * self.hidden_size])
            alpha = tf.reshape(tf.matmul(u, self.weights['d_h_v_1']),
                               [batch_size, 1, self.max_sen_len])
            alpha = self.softmax(alpha, self.sen_len, self.max_sen_len)
            outputs = tf.matmul(alpha, outputs)


        outputs = tf.reshape(outputs, [-1, self.max_doc_len, 2 * self.hidden_size])
        with tf.name_scope('d_h_sentence_encode'):
            outputs, state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=tf.nn.rnn_cell.LSTMCell(self.hidden_size, forget_bias=1.0),
                cell_bw=tf.nn.rnn_cell.LSTMCell(self.hidden_size, forget_bias=1.0),
                inputs=outputs,
                sequence_length=self.doc_len,
                dtype=tf.float32,
                scope='d_h_sentence'
            )
            outputs = tf.concat(outputs, 2)

        batch_size = tf.shape(outputs)[0]
        with tf.name_scope('d_h_sentence_attention'):
            output = tf.reshape(outputs, [-1, 2 * self.hidden_size])
            u = tf.matmul(output, self.weights['d_h_wh_2']) + self.biases['d_h_wh_2']
            u = tf.reshape(u, [-1, self.max_doc_len, 2*self.hidden_size])
            u += tf.matmul(self.helpfulness, self.weights['d_wh_2'])[:,None,:]
            u = tf.tanh(u)
            u = tf.reshape(u, [-1, 2*self.hidden_size])
            alpha = tf.reshape(tf.matmul(u, self.weights['d_h_v_2']),
                               [batch_size, 1, self.max_doc_len])
            alpha = self.softmax(alpha, self.doc_len, self.max_doc_len)
            outputs = tf.matmul(alpha, outputs)

        with tf.name_scope('d_h_softmax'):
            self.d_h_doc = tf.reshape(outputs, [batch_size, 2 * self.hidden_size])
            self.d_h_scores = tf.matmul(self.d_h_doc, self.weights['d_h_softmax']) + self.biases['d_h_softmax']
            self.d_h_predictions = tf.argmax(self.d_h_scores, 1, name="d_h_predictions")

        with tf.name_scope("d_h_loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.d_h_scores, labels=self.input_y)
            self.d_h_loss = tf.reduce_mean(losses)

        with tf.name_scope("d_h_accuracy"):
            correct_predictions = tf.equal(self.d_h_predictions, tf.argmax(self.input_y, 1))
            self.d_h_correct_num = tf.reduce_sum(tf.cast(correct_predictions, dtype=tf.int32))
            self.d_h_accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="d_h_accuracy")

    def helpfulness_attention_s(self):
        inputs = tf.reshape(self.x_s, [-1, self.max_sen_len, self.embedding_dim])
        sen_len = tf.reshape(self.sen_len, [-1])
        with tf.name_scope('s_h_word_encode'):
            outputs, state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=tf.nn.rnn_cell.LSTMCell(self.hidden_size, forget_bias=1.0),
                cell_bw=tf.nn.rnn_cell.LSTMCell(self.hidden_size, forget_bias=1.0),
                inputs=inputs,
                sequence_length=sen_len,
                dtype=tf.float32,
                scope='s_h_word'
            )
            outputs = tf.concat(outputs, 2)

        batch_size = tf.shape(outputs)[0]
        with tf.name_scope('s_h_word_attention'):
            output = tf.reshape(outputs, [-1, 2 * self.hidden_size])
            u = tf.matmul(output, self.weights['s_h_wh_1']) + self.biases['s_h_wh_1']
            u = tf.reshape(u, [-1, self.max_doc_len*self.max_sen_len, 2*self.hidden_size])
            u += tf.matmul(self.helpfulness, self.weights['s_wh_1'])[:,None,:]
            u = tf.tanh(u)
            u = tf.reshape(u, [-1, 2 * self.hidden_size])
            alpha = tf.reshape(tf.matmul(u, self.weights['s_h_v_1']),
                               [batch_size, 1, self.max_sen_len])
            alpha = self.softmax(alpha, self.sen_len, self.max_sen_len)
            outputs = tf.matmul(alpha, outputs)


        outputs = tf.reshape(outputs, [-1, self.max_doc_len, 2 * self.hidden_size])
        with tf.name_scope('s_h_sentence_encode'):
            outputs, state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=tf.nn.rnn_cell.LSTMCell(self.hidden_size, forget_bias=1.0),
                cell_bw=tf.nn.rnn_cell.LSTMCell(self.hidden_size, forget_bias=1.0),
                inputs=outputs,
                sequence_length=self.doc_len,
                dtype=tf.float32,
                scope='s_h_sentence'
            )
            outputs = tf.concat(outputs, 2)

        batch_size = tf.shape(outputs)[0]
        with tf.name_scope('s_h_sentence_attention'):
            output = tf.reshape(outputs, [-1, 2 * self.hidden_size])
            u = tf.matmul(output, self.weights['s_h_wh_2']) + self.biases['s_h_wh_2']
            u = tf.reshape(u, [-1, self.max_doc_len, 2*self.hidden_size])
            u += tf.matmul(self.helpfulness, self.weights['s_wh_2'])[:,None,:]
            u = tf.tanh(u)
            u = tf.reshape(u, [-1, 2*self.hidden_size])
            alpha = tf.reshape(tf.matmul(u, self.weights['s_h_v_2']),
                               [batch_size, 1, self.max_doc_len])
            alpha = self.softmax(alpha, self.doc_len, self.max_doc_len)
            outputs = tf.matmul(alpha, outputs)

        with tf.name_scope('s_h_softmax'):
            self.s_h_doc = tf.reshape(outputs, [batch_size, 2 * self.hidden_size])
            self.s_h_scores = tf.matmul(self.s_h_doc, self.weights['s_h_softmax']) + self.biases['s_h_softmax']
            self.s_h_predictions = tf.argmax(self.s_h_scores, 1, name="s_h_predictions")

        with tf.name_scope("s_h_loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.s_h_scores, labels=self.input_y)
            self.s_h_loss = tf.reduce_mean(losses)

        with tf.name_scope("s_h_accuracy"):
            correct_predictions = tf.equal(self.s_h_predictions, tf.argmax(self.input_y, 1))
            self.s_h_correct_num = tf.reduce_sum(tf.cast(correct_predictions, dtype=tf.int32))
            self.s_h_accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="s_h_accuracy")


    def helpfulness_attention_sd(self):
        inputs = tf.reshape(self.x_sd, [-1, self.max_sen_len, self.embedding_dim])
        sen_len = tf.reshape(self.sen_len, [-1])
        with tf.name_scope('sd_h_word_encode'):
            outputs, state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=tf.nn.rnn_cell.LSTMCell(self.hidden_size, forget_bias=1.0),
                cell_bw=tf.nn.rnn_cell.LSTMCell(self.hidden_size, forget_bias=1.0),
                inputs=inputs,
                sequence_length=sen_len,
                dtype=tf.float32,
                scope='sd_h_word'
            )
            outputs = tf.concat(outputs, 2)

        batch_size = tf.shape(outputs)[0]
        with tf.name_scope('sd_h_word_attention'):
            output = tf.reshape(outputs, [-1, 2 * self.hidden_size])
            u = tf.matmul(output, self.weights['sd_h_wh_1']) + self.biases['sd_h_wh_1']
            u = tf.reshape(u, [-1, self.max_doc_len*self.max_sen_len, 2*self.hidden_size])
            u += tf.matmul(self.helpfulness, self.weights['sd_wh_1'])[:,None,:]
            u = tf.tanh(u)
            u = tf.reshape(u, [-1, 2 * self.hidden_size])
            alpha = tf.reshape(tf.matmul(u, self.weights['sd_h_v_1']),
                               [batch_size, 1, self.max_sen_len])
            alpha = self.softmax(alpha, self.sen_len, self.max_sen_len)
            outputs = tf.matmul(alpha, outputs)


        outputs = tf.reshape(outputs, [-1, self.max_doc_len, 2 * self.hidden_size])
        with tf.name_scope('sd_h_sentence_encode'):
            outputs, state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=tf.nn.rnn_cell.LSTMCell(self.hidden_size, forget_bias=1.0),
                cell_bw=tf.nn.rnn_cell.LSTMCell(self.hidden_size, forget_bias=1.0),
                inputs=outputs,
                sequence_length=self.doc_len,
                dtype=tf.float32,
                scope='sd_h_sentence'
            )
            outputs = tf.concat(outputs, 2)

        batch_size = tf.shape(outputs)[0]
        with tf.name_scope('sd_h_sentence_attention'):
            output = tf.reshape(outputs, [-1, 2 * self.hidden_size])
            u = tf.matmul(output, self.weights['sd_h_wh_2']) + self.biases['sd_h_wh_2']
            u = tf.reshape(u, [-1, self.max_doc_len, 2*self.hidden_size])
            u += tf.matmul(self.helpfulness, self.weights['sd_wh_2'])[:,None,:]
            u = tf.tanh(u)
            u = tf.reshape(u, [-1, 2*self.hidden_size])
            alpha = tf.reshape(tf.matmul(u, self.weights['sd_h_v_2']),
                               [batch_size, 1, self.max_doc_len])
            alpha = self.softmax(alpha, self.doc_len, self.max_doc_len)
            outputs = tf.matmul(alpha, outputs)

        with tf.name_scope('sd_h_softmax'):
            self.sd_h_doc = tf.reshape(outputs, [batch_size, 2 * self.hidden_size])
            self.sd_h_scores = tf.matmul(self.sd_h_doc, self.weights['sd_h_softmax']) + self.biases['sd_h_softmax']
            self.sd_h_predictions = tf.argmax(self.sd_h_scores, 1, name="sd_h_predictions")

        with tf.name_scope("sd_h_loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.sd_h_scores, labels=self.input_y)
            self.sd_h_loss = tf.reduce_mean(losses)

        with tf.name_scope("sd_h_accuracy"):
            correct_predictions = tf.equal(self.sd_h_predictions, tf.argmax(self.input_y, 1))
            self.sd_h_correct_num = tf.reduce_sum(tf.cast(correct_predictions, dtype=tf.int32))
            self.sd_h_accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="sd_h_accuracy")


    # time_session
    def time_attention_d(self):
        inputs = tf.reshape(self.x_d, [-1, self.max_sen_len, self.embedding_dim])
        sen_len = tf.reshape(self.sen_len, [-1])
        with tf.name_scope('d_t_word_encode'):
            outputs, state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=tf.nn.rnn_cell.LSTMCell(self.hidden_size, forget_bias=1.0),
                cell_bw=tf.nn.rnn_cell.LSTMCell(self.hidden_size, forget_bias=1.0),
                inputs=inputs,
                sequence_length=sen_len,
                dtype=tf.float32,
                scope='d_t_word'
            )
            outputs = tf.concat(outputs, 2)

        batch_size = tf.shape(outputs)[0]
        with tf.name_scope('d_t_word_attention'):
            output = tf.reshape(outputs, [-1, 2 * self.hidden_size])
            u = tf.matmul(output, self.weights['d_t_wh_1']) + self.biases['d_t_wh_1']
            u = tf.reshape(u, [-1, self.max_doc_len*self.max_sen_len, 2*self.hidden_size])
            u += tf.matmul(self.time, self.weights['d_wt_1'])[:,None,:]
            u = tf.tanh(u)
            u = tf.reshape(u, [-1, 2 * self.hidden_size])
            alpha = tf.reshape(tf.matmul(u, self.weights['d_t_v_1']),
                               [batch_size, 1, self.max_sen_len])
            alpha = self.softmax(alpha, self.sen_len, self.max_sen_len)
            outputs = tf.matmul(alpha, outputs)


        outputs = tf.reshape(outputs, [-1, self.max_doc_len, 2 * self.hidden_size])
        with tf.name_scope('d_t_sentence_encode'):
            outputs, state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=tf.nn.rnn_cell.LSTMCell(self.hidden_size, forget_bias=1.0),
                cell_bw=tf.nn.rnn_cell.LSTMCell(self.hidden_size, forget_bias=1.0),
                inputs=outputs,
                sequence_length=self.doc_len,
                dtype=tf.float32,
                scope='d_t_sentence'
            )
            outputs = tf.concat(outputs, 2)

        batch_size = tf.shape(outputs)[0]
        with tf.name_scope('d_t_sentence_attention'):
            output = tf.reshape(outputs, [-1, 2 * self.hidden_size])
            u = tf.matmul(output, self.weights['d_t_wh_2']) + self.biases['d_t_wh_2']
            u = tf.reshape(u, [-1, self.max_doc_len, 2*self.hidden_size])
            u += tf.matmul(self.time, self.weights['d_wt_2'])[:,None,:]
            u = tf.tanh(u)
            u = tf.reshape(u, [-1, 2*self.hidden_size])
            alpha = tf.reshape(tf.matmul(u, self.weights['d_t_v_2']),
                               [batch_size, 1, self.max_doc_len])
            alpha = self.softmax(alpha, self.doc_len, self.max_doc_len)
            outputs = tf.matmul(alpha, outputs)

        with tf.name_scope('d_t_softmax'):
            self.d_t_doc = tf.reshape(outputs, [batch_size, 2 * self.hidden_size])
            self.d_t_scores = tf.matmul(self.d_t_doc, self.weights['d_t_softmax']) + self.biases['d_t_softmax']
            self.d_t_predictions = tf.argmax(self.d_t_scores, 1, name="d_t_predictions")

        with tf.name_scope("d_t_loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.d_t_scores, labels=self.input_y)
            self.d_t_loss = tf.reduce_mean(losses)

        with tf.name_scope("d_t_accuracy"):
            correct_predictions = tf.equal(self.d_t_predictions, tf.argmax(self.input_y, 1))
            self.d_t_correct_num = tf.reduce_sum(tf.cast(correct_predictions, dtype=tf.int32))
            self.d_t_accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="d_t_accuracy")

    def time_attention_s(self):
        inputs = tf.reshape(self.x_s, [-1, self.max_sen_len, self.embedding_dim])
        sen_len = tf.reshape(self.sen_len, [-1])
        with tf.name_scope('s_t_word_encode'):
            outputs, state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=tf.nn.rnn_cell.LSTMCell(self.hidden_size, forget_bias=1.0),
                cell_bw=tf.nn.rnn_cell.LSTMCell(self.hidden_size, forget_bias=1.0),
                inputs=inputs,
                sequence_length=sen_len,
                dtype=tf.float32,
                scope='s_t_word'
            )
            outputs = tf.concat(outputs, 2)

        batch_size = tf.shape(outputs)[0]
        with tf.name_scope('s_t_word_attention'):
            output = tf.reshape(outputs, [-1, 2 * self.hidden_size])
            u = tf.matmul(output, self.weights['s_t_wh_1']) + self.biases['s_t_wh_1']
            u = tf.reshape(u, [-1, self.max_doc_len*self.max_sen_len, 2*self.hidden_size])
            u += tf.matmul(self.time, self.weights['s_wt_1'])[:,None,:]
            u = tf.tanh(u)
            u = tf.reshape(u, [-1, 2 * self.hidden_size])
            alpha = tf.reshape(tf.matmul(u, self.weights['s_t_v_1']),
                               [batch_size, 1, self.max_sen_len])
            alpha = self.softmax(alpha, self.sen_len, self.max_sen_len)
            outputs = tf.matmul(alpha, outputs)


        outputs = tf.reshape(outputs, [-1, self.max_doc_len, 2 * self.hidden_size])
        with tf.name_scope('s_t_sentence_encode'):
            outputs, state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=tf.nn.rnn_cell.LSTMCell(self.hidden_size, forget_bias=1.0),
                cell_bw=tf.nn.rnn_cell.LSTMCell(self.hidden_size, forget_bias=1.0),
                inputs=outputs,
                sequence_length=self.doc_len,
                dtype=tf.float32,
                scope='s_t_sentence'
            )
            outputs = tf.concat(outputs, 2)

        batch_size = tf.shape(outputs)[0]
        with tf.name_scope('s_t_sentence_attention'):
            output = tf.reshape(outputs, [-1, 2 * self.hidden_size])
            u = tf.matmul(output, self.weights['s_t_wh_2']) + self.biases['s_t_wh_2']
            u = tf.reshape(u, [-1, self.max_doc_len, 2*self.hidden_size])
            u += tf.matmul(self.time, self.weights['s_wt_2'])[:,None,:]
            u = tf.tanh(u)
            u = tf.reshape(u, [-1, 2*self.hidden_size])
            alpha = tf.reshape(tf.matmul(u, self.weights['s_t_v_2']),
                               [batch_size, 1, self.max_doc_len])
            alpha = self.softmax(alpha, self.doc_len, self.max_doc_len)
            outputs = tf.matmul(alpha, outputs)

        with tf.name_scope('s_t_softmax'):
            self.s_t_doc = tf.reshape(outputs, [batch_size, 2 * self.hidden_size])
            self.s_t_scores = tf.matmul(self.s_t_doc, self.weights['s_t_softmax']) + self.biases['s_t_softmax']
            self.s_t_predictions = tf.argmax(self.s_t_scores, 1, name="s_t_predictions")

        with tf.name_scope("s_t_loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.s_t_scores, labels=self.input_y)
            self.s_t_loss = tf.reduce_mean(losses)

        with tf.name_scope("s_t_accuracy"):
            correct_predictions = tf.equal(self.s_t_predictions, tf.argmax(self.input_y, 1))
            self.s_t_correct_num = tf.reduce_sum(tf.cast(correct_predictions, dtype=tf.int32))
            self.s_t_accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="s_t_accuracy")


    def time_attention_sd(self):
        inputs = tf.reshape(self.x_sd, [-1, self.max_sen_len, self.embedding_dim])
        sen_len = tf.reshape(self.sen_len, [-1])
        with tf.name_scope('sd_t_word_encode'):
            outputs, state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=tf.nn.rnn_cell.LSTMCell(self.hidden_size, forget_bias=1.0),
                cell_bw=tf.nn.rnn_cell.LSTMCell(self.hidden_size, forget_bias=1.0),
                inputs=inputs,
                sequence_length=sen_len,
                dtype=tf.float32,
                scope='sd_t_word'
            )
            outputs = tf.concat(outputs, 2)

        batch_size = tf.shape(outputs)[0]
        with tf.name_scope('sd_t_word_attention'):
            output = tf.reshape(outputs, [-1, 2 * self.hidden_size])
            u = tf.matmul(output, self.weights['sd_t_wh_1']) + self.biases['sd_t_wh_1']
            u = tf.reshape(u, [-1, self.max_doc_len*self.max_sen_len, 2*self.hidden_size])
            u += tf.matmul(self.time, self.weights['sd_wt_1'])[:,None,:]
            u = tf.tanh(u)
            u = tf.reshape(u, [-1, 2 * self.hidden_size])
            alpha = tf.reshape(tf.matmul(u, self.weights['sd_t_v_1']),
                               [batch_size, 1, self.max_sen_len])
            alpha = self.softmax(alpha, self.sen_len, self.max_sen_len)
            outputs = tf.matmul(alpha, outputs)


        outputs = tf.reshape(outputs, [-1, self.max_doc_len, 2 * self.hidden_size])
        with tf.name_scope('sd_t_sentence_encode'):
            outputs, state = tf.nn.bidirectional_dynamic_rnn(
                cell_fw=tf.nn.rnn_cell.LSTMCell(self.hidden_size, forget_bias=1.0),
                cell_bw=tf.nn.rnn_cell.LSTMCell(self.hidden_size, forget_bias=1.0),
                inputs=outputs,
                sequence_length=self.doc_len,
                dtype=tf.float32,
                scope='sd_t_sentence'
            )
            outputs = tf.concat(outputs, 2)

        batch_size = tf.shape(outputs)[0]
        with tf.name_scope('sd_t_sentence_attention'):
            output = tf.reshape(outputs, [-1, 2 * self.hidden_size])
            u = tf.matmul(output, self.weights['sd_t_wh_2']) + self.biases['sd_t_wh_2']
            u = tf.reshape(u, [-1, self.max_doc_len, 2*self.hidden_size])
            u += tf.matmul(self.time, self.weights['sd_wt_2'])[:,None,:]
            u = tf.tanh(u)
            u = tf.reshape(u, [-1, 2*self.hidden_size])
            alpha = tf.reshape(tf.matmul(u, self.weights['sd_t_v_2']),
                               [batch_size, 1, self.max_doc_len])
            alpha = self.softmax(alpha, self.doc_len, self.max_doc_len)
            outputs = tf.matmul(alpha, outputs)

        with tf.name_scope('sd_t_softmax'):
            self.sd_t_doc = tf.reshape(outputs, [batch_size, 2 * self.hidden_size])
            self.sd_t_scores = tf.matmul(self.sd_t_doc, self.weights['sd_t_softmax']) + self.biases['sd_t_softmax']
            self.sd_t_predictions = tf.argmax(self.sd_t_scores, 1, name="sd_t_predictions")

        with tf.name_scope("sd_t_loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.sd_t_scores, labels=self.input_y)
            self.sd_t_loss = tf.reduce_mean(losses)

        with tf.name_scope("sd_t_accuracy"):
            correct_predictions = tf.equal(self.sd_t_predictions, tf.argmax(self.input_y, 1))
            self.sd_t_correct_num = tf.reduce_sum(tf.cast(correct_predictions, dtype=tf.int32))
            self.sd_t_accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="sd_t_accuracy")



    def build_model(self):
        self.user_attention_d()
        self.user_attention_s()
        self.user_attention_sd()
        self.product_attention_d()
        self.product_attention_s()
        self.product_attention_sd()
        self.helpfulness_attention_d()
        self.helpfulness_attention_s()
        self.helpfulness_attention_sd()
        self.time_attention_d()
        self.time_attention_s()
        self.time_attention_sd()

        with tf.name_scope('softmax'):
            outputs_d = tf.concat(
                [
                    self.d_u_doc, self.d_p_doc, self.d_h_doc, self.d_t_doc,
                ], 1)
            outputs_s = tf.concat(
                [
                    self.s_u_doc, self.s_p_doc, self.s_h_doc, self.s_t_doc,
                ], 1)
            outputs_sd = tf.concat(
                [
                    self.sd_u_doc, self.sd_p_doc, self.sd_h_doc, self.sd_t_doc,
                ], 1)
            self.scores_d = tf.matmul(outputs_d, self.weights['d_softmax']) + self.biases['d_softmax']
            self.predictions_d = tf.argmax(self.scores_d, 1, name="d_predictions")
            self.scores_s = tf.matmul(outputs_s, self.weights['s_softmax']) + self.biases['s_softmax']
            self.predictions_s = tf.argmax(self.scores_s, 1, name="s_predictions")
            self.scores_sd = tf.matmul(outputs_sd, self.weights['sd_softmax']) + self.biases['sd_softmax']
            self.predictions_sd = tf.argmax(self.scores_sd, 1, name="sd_predictions")

            outputs = tf.concat(
                [
                    self.d_u_doc, self.d_p_doc, self.d_h_doc, self.d_t_doc,
                    self.s_u_doc, self.s_p_doc, self.s_h_doc, self.s_t_doc,
                    self.sd_u_doc, self.sd_p_doc, self.sd_h_doc, self.sd_t_doc
                ], 1)
            self.scores = tf.matmul(outputs, self.weights['softmax']) + self.biases['softmax']
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            losses_d = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores_d, labels=self.input_y)
            losses_s = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores_s, labels=self.input_y)
            losses_sd = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores_sd, labels=self.input_y)
            self.loss = 0.4 * tf.reduce_mean(losses) \
                        + 0.2 * (0.4*tf.reduce_mean(losses_d) + 0.6*(0.25*self.d_u_loss  + 0.25*self.d_p_loss + 0.25*self.d_h_loss + 0.25*self.d_t_loss)) \
                        + 0.1 * (0.4*tf.reduce_mean(losses_s) + 0.6*(0.25*self.s_u_loss  + 0.25*self.s_p_loss + 0.25*self.s_h_loss + 0.25*self.s_t_loss)) \
                        + 0.3 * (0.4*tf.reduce_mean(losses_sd) + 0.6*(0.25*self.sd_u_loss  + 0.25*self.sd_p_loss + 0.25*self.sd_h_loss + 0.25*self.sd_t_loss))


        with tf.name_scope("metrics"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.mse = tf.reduce_sum(tf.square(self.predictions - tf.argmax(self.input_y, 1)), name="mse")
            self.correct_num = tf.reduce_sum(tf.cast(correct_predictions, dtype=tf.int32), name="correct_num")
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
