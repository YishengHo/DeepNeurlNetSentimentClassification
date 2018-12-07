#!/usr/bin/env mdl
import tensorflow as tf
import numpy as np

t1 = tf.placeholder(tf.float32, [None, 400])
t2 = tf.placeholder(tf.float32, [None, 1176])
t3 = tf.concat([t1, t2], axis = 1)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    t3_val = sess.run(t3, feed_dict = {t1: np.ones((300, 400)), t2: np.ones((300, 1176))})

    print(t3_val.shape)
# vim: ts=4 sw=4 sts=4 expandtab
