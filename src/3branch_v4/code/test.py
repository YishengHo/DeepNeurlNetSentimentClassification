#-*- coding: utf-8 -*-
#author: aaronzark

import datetime, os, time, pickle
import numpy as np
import tensorflow as tf

from data_helpers import Dataset
import data_helpers
from model import huapahata


# Data loading params
tf.flags.DEFINE_integer("n_class", 5, "Numbers of class")
tf.flags.DEFINE_string("dataset", 'yelp13', "The dataset")
tf.flags.DEFINE_integer('max_sen_len', 50, 'max number of tokens per sentence')
tf.flags.DEFINE_integer('max_doc_len', 40, 'max number of tokens per sentence')

tf.flags.DEFINE_integer("batch_size", 100, "Batch Size")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# Load data
checkpoint_file = tf.train.latest_checkpoint("../checkpoints/"+FLAGS.dataset+"/"+FLAGS.checkpoint_dir+"/")
testset = Dataset('../../data/'+FLAGS.dataset+'/test.ss')
with open("../checkpoints/"+FLAGS.dataset+"/"+FLAGS.checkpoint_dir+"/wordsdict.txt", 'rb') as f:
    wordsdict = pickle.load(f)
with open("../checkpoints/"+FLAGS.dataset+"/"+FLAGS.checkpoint_dir+"/usrdict.txt", 'rb') as f:
    usrdict = pickle.load(f)
with open("../checkpoints/"+FLAGS.dataset+"/"+FLAGS.checkpoint_dir+"/prddict.txt", 'rb') as f:
    prddict = pickle.load(f)
with open("../checkpoints/"+FLAGS.dataset+"/"+FLAGS.checkpoint_dir+"/hlpdict.txt", 'rb') as f:
    hlpdict = pickle.load(f)
with open("../checkpoints/"+FLAGS.dataset+'/'+FLAGS.checkpoint_dir+"/tmedict.txt", 'rb') as f:
    tmedict = pickle.load(f)

testset.genBatch(usrdict, prddict, hlpdict, tmedict, wordsdict, FLAGS.batch_size,
                  FLAGS.max_sen_len, FLAGS.max_doc_len, FLAGS.n_class)

graph = tf.Graph()
with graph.as_default():
    session_config = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement
    )
    session_config.gpu_options.allow_growth = True
    sess = tf.Session(config=session_config)
    with sess.as_default():
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        huapahata_userid = graph.get_operation_by_name("input/user_id").outputs[0]
        huapahata_productid = graph.get_operation_by_name("input/product_id").outputs[0]
        huapahata_helpfulnessid = graph.get_operation_by_name("input/helpfulness_id").outputs[0]
        huapahata_timeid = graph.get_operation_by_name("input/time_id").outputs[0]
        huapahata_input_x = graph.get_operation_by_name("input/input_x").outputs[0]
        huapahata_input_y = graph.get_operation_by_name("input/input_y").outputs[0]
        huapahata_sen_len = graph.get_operation_by_name("input/sen_len").outputs[0]
        huapahata_doc_len = graph.get_operation_by_name("input/doc_len").outputs[0]

        huapahata_accuracy = graph.get_operation_by_name("metrics/accuracy").outputs[0]
        huapahata_correct_num = graph.get_operation_by_name("metrics/correct_num").outputs[0]
        huapahata_mse = graph.get_operation_by_name("metrics/mse").outputs[0]

        def predict_step(u, p, h, t, x, y, sen_len, doc_len, name=None):
            feed_dict = {
                huapahata_userid: u,
                huapahata_productid: p,
                huapahata_helpfulnessid: h,
                huapahata_timeid: t,
                huapahata_input_x: x,
                huapahata_input_y: y,
                huapahata_sen_len: sen_len,
                huapahata_doc_len: doc_len
            }
            accuracy, correct_num, mse = sess.run(
                [huapahata_accuracy, huapahata_correct_num, huapahata_mse],
                feed_dict)
            return correct_num, accuracy, mse

        def predict(dataset, name=None):
            acc = 0
            rmse = 0.
            for i in range(dataset.epoch):
                correct_num, _, mse = predict_step(dataset.usr[i], dataset.prd[i], dataset.hlp[i], dataset.tme[i], dataset.docs[i],
                                                   dataset.label[i], dataset.sen_len[i], dataset.doc_len[i], name)
                acc += correct_num
                rmse += mse
            acc = acc * 1.0 / dataset.data_size
            rmse = np.sqrt(rmse / dataset.data_size)
            return acc, rmse

        test_acc, test_rmse = predict(testset, name="test")
        print("\ntest_acc: %.4f    test_RMSE: %.4f\n" % (test_acc, test_rmse))

