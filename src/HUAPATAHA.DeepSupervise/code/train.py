#-*- coding: utf-8 -*-
#author: Zhen Wu

import os, time, pickle, datetime
import numpy as np
import tensorflow as tf

from data_helpers import Dataset
import data_helpers
from model import huapahata
from argparse import ArgumentParser
import math

parser = ArgumentParser();

# os.environ['CUDA_VISIBLE_DEVICES']='0'


# Data loading params

parser.add_argument("--n_class", default=5, type=int, help="Numbers of class")
parser.add_argument("--dataset", default='yelp13', help="The dataset")

# Model Hyperparameters
parser.add_argument("--embedding_dim", default=200, type=int, help="Dimensionality of character embedding")
parser.add_argument("--hidden_size", default=100, type=int, help="hidden_size of rnn")
parser.add_argument('--max_sen_len', default=50, type=int, help='max number of tokens per sentence')
parser.add_argument('--max_doc_len', default=40, type=int, help='max number of tokens per sentence')
parser.add_argument("--lr", default=0.005, type=float, help="Learning rate")

# Training parameters
parser.add_argument("--batch_size", default=100, type=int, help="Batch Size")
parser.add_argument("--num_epochs", default=1000, type=int, help="Number of training epochs")
parser.add_argument("--evaluate_every", default=400, type=int,  help="Evaluate model on dev set after this many steps")

# Misc Parameters
parser.add_argument("--allow_soft_placement", default=True, type=bool, help="Allow device soft device placement")
parser.add_argument("--log_device_placement", default=False, type=bool, help="Log placement of ops on devices")

FLAGS = parser.parse_args();
# FLAGS = tf.flags.FLAGS
# FLAGS._parse_flags()
# FLAGS(sys.argv)

print("\nParameters:")
print(FLAGS)
# for attr, value in sorted(FLAGS.items()):
#    print("{}={}".format(attr.upper(), value))
print("")


# Load data
print("Loading data...")
trainset = Dataset('../../data/'+FLAGS.dataset+'/train.ss')
devset = Dataset('../../data/'+FLAGS.dataset+'/dev.ss')
testset = Dataset('../../data/'+FLAGS.dataset+'/test.ss')

alldata = np.concatenate([trainset.t_docs, devset.t_docs, testset.t_docs], axis=0)
embeddingpath = '../../data/'+FLAGS.dataset+'/embedding.txt'
embeddingfile, wordsdict = data_helpers.load_embedding(embeddingpath, alldata, FLAGS.embedding_dim)
del alldata
print("Loading data finished...")

usrdict, prddict, hlpdict, tmedict = trainset.get_usr_prd_hlp_tme_dict()
trainbatches = trainset.batch_iter(usrdict, prddict, hlpdict, tmedict, wordsdict, FLAGS.n_class, FLAGS.batch_size,
                                 FLAGS.num_epochs, FLAGS.max_sen_len, FLAGS.max_doc_len)
devset.genBatch(usrdict, prddict, hlpdict, tmedict, wordsdict, FLAGS.batch_size,
                  FLAGS.max_sen_len, FLAGS.max_doc_len, FLAGS.n_class)
testset.genBatch(usrdict, prddict, hlpdict, tmedict, wordsdict, FLAGS.batch_size,
                  FLAGS.max_sen_len, FLAGS.max_doc_len, FLAGS.n_class)


with tf.Graph().as_default():
    session_config = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement
    )
    session_config.gpu_options.allow_growth = True
    sess = tf.Session(config=session_config)
    with sess.as_default():
        huapahata = huapahata(
            max_sen_len = FLAGS.max_sen_len,
            max_doc_len = FLAGS.max_doc_len,
            class_num = FLAGS.n_class,
            embedding_file = embeddingfile,
            embedding_dim = FLAGS.embedding_dim,
            hidden_size = FLAGS.hidden_size,
            user_num = len(usrdict),
            product_num = len(prddict),
            helpfulness_num = len(hlpdict),
            time_num = len(tmedict)
        )
        huapahata.build_model()
        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(FLAGS.lr)
        grads_and_vars = optimizer.compute_gradients(huapahata.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Save dict
        timestamp = str(int(time.time()))
        checkpoint_dir = os.path.abspath("../checkpoints/"+FLAGS.dataset+"/"+timestamp)
        checkpoint_prefix = os.path.join(checkpoint_dir, "model")
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)
        with open(checkpoint_dir + "/wordsdict.txt", 'wb') as f:
            pickle.dump(wordsdict, f)
        with open(checkpoint_dir + "/usrdict.txt", 'wb') as f:
            pickle.dump(usrdict, f)
        with open(checkpoint_dir + "/prddict.txt", 'wb') as f:
            pickle.dump(prddict, f)
        with open(checkpoint_dir + "/hlpdict.txt", 'wb') as f:
            pickle.dump(hlpdict, f)
        with open(checkpoint_dir + "/tmedict.txt", 'wb') as f:
            pickle.dump(tmedict, f)

        sess.run(tf.global_variables_initializer())

        def train_step(batch):
            u, p, h, t, x, y, sen_len, doc_len = zip(*batch)
            feed_dict = {
                huapahata.userid: u,
                huapahata.productid: p,
                huapahata.helpfulnessid: h,
                huapahata.timeid: t,
                huapahata.input_x: x,
                huapahata.input_y: y,
                huapahata.sen_len: sen_len,
                huapahata.doc_len: doc_len
            }
            _, step, loss, accuracy = sess.run(
                [train_op, global_step, huapahata.loss, huapahata.accuracy],
                feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

        def predict_step(u, p, h, t, x, y, sen_len, doc_len, name=None):
            feed_dict = {
                huapahata.userid: u,
                huapahata.productid: p,
                huapahata.helpfulnessid: h,
                huapahata.timeid: t,
                huapahata.input_x: x,
                huapahata.input_y: y,
                huapahata.sen_len: sen_len,
                huapahata.doc_len: doc_len
            }
            step, loss, accuracy, correct_num, mse = sess.run(
                [global_step, huapahata.loss, huapahata.accuracy, huapahata.correct_num, huapahata.mse],
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

        topacc = 0.
        toprmse = 0.
        better_dev_acc = 0.
        better_test_acc = 0.
        predict_round = 0

        # Training loop. For each batch...
        for tr_batch in trainbatches:
            train_step(tr_batch)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % FLAGS.evaluate_every == 0:
                predict_round += 1
                print("\nEvaluation round %d:" % (predict_round))

                dev_acc, dev_rmse = predict(devset, name="dev")
                print("dev_acc: %.4f    dev_RMSE: %.4f" % (dev_acc, dev_rmse))
                test_acc, test_rmse = predict(testset, name="test")
                print("test_acc: %.4f    test_RMSE: %.4f" % (test_acc, test_rmse))

                # print topacc with best dev acc
                if dev_acc >= better_test_acc:
                    # better_dev_acc = dev_acc
                    better_test_acc = test_acc
                    topacc = test_acc
                    toprmse = test_rmse
                    path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                    print("Saved model checkpoint to {}\n".format(path))
                print("topacc: %.4f   RMSE: %.4f" % (topacc, toprmse))
                with open("topacc.txt", 'a') as of:
                    print("topacc: %.4f   RMSE: %.4f" % (topacc, toprmse), file=of)
