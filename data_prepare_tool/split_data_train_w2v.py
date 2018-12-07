#!/usr/bin/env mdl
from tqdm import tqdm
import os
import gzip
import json
import pandas as pd
from tqdm import tqdm
from collections import namedtuple
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import gensim
from gensim.models import word2vec
import numpy as np
import pickle
from argparse import ArgumentParser
import sys


parser = ArgumentParser()
parser.add_argument('--input', help='input gzip file name')
parser.add_argument('--folder-name', help='output dataset name')
parser.add_argument('--summary', default=0, type=int)
parser.add_argument('--rsummary', default=0.0, type=float)
parser.add_argument('--word-min-count', default=1, type=int)
parser.add_argument('--max_vocab_size', default=50000, type=int)
parser.add_argument('--embed_size', default=256, type=int)
parser.add_argument('--only-word2vec', default=False, action='store_true')
parser.add_argument('--complete-dataset', default=False, action='store_true')
args = parser.parse_args()

folder_name = '../' + args.folder_name
SUMMARY = args.summary
RSUMMARY = args.rsummary
id_to_sent_dict = {}
id_to_sum_sent_dict = {}
num_plen_dict = {}
num_sentence_dict = {}


def myiter(d, cols=None):
    if cols is None:
        if 'Id' not in d.columns.values.tolist():
            idxs = [i for i in range(len(d.index))]
            d.insert(0,'Id',idxs)
        v = d.values.tolist()
        cols = d.columns.values.tolist()
        print(cols)
    else:
        j = [d.columns.get_loc(c) for c in cols]
        v = d.values[:, j].tolist()
    print(cols)

    n = namedtuple('MyTuple', cols)

    for line in iter(v):
        yield n(*line)


def proc_para(text):
    sens = sent_tokenize(text)
    sens_words = []
    plen = len(sens)
    for s in sens:
        ws = word_tokenize(s)
        ws = [w.lower() for w in ws]
        ws.append('<sssss>')
        sens_words += ws
    return sens_words, plen


def get_all_sentence_pd_usr(l_all, subname, filename='data_all_sentences.txt'):
    cur_folder = folder_name + "_" + subname
    if not os.path.exists(cur_folder):
        os.mkdir(cur_folder)
    of_sen = open(cur_folder+'/'+filename, 'w')
    of_prd = open(cur_folder+'/'+'prdlist.txt', 'w')
    of_usr = open(cur_folder+'/'+'usrlist.txt', 'w')
    of_tm = open(cur_folder+'/'+'timelist.txt', 'w')
    of_pn = open(cur_folder+'/'+'profilenamelist.txt', 'w')
    of_helpfull = open(cur_folder+'/'+'helplist.txt', 'w')
    l_usr = []
    l_prd = []
    l_tm = []
    l_pn = []
    for t in tqdm(l_all):
        l_usr.append(t.reviewerID)
        l_prd.append(t.asin)
        l_tm.append(t.reviewTime)
        l_pn.append(t.reviewerName)
        sens, plent = proc_para(t.reviewText)
        sens_sum, plens = None, 0
        summary = '';
        if t.summary is None or len(str(t.summary)) == 0:
            summary = '.'
        if t.summary is not None:
            if (len(summary) == 0):
                summary = str(t.summary)
            if str(summary[len(summary)-1]).isalpha():
                summary = summary + '.'
            sens_sum, plens = proc_para(summary)
        if sens_sum is not None:
            id_to_sum_sent_dict[t.Id] = sens_sum
            if SUMMARY == 1:
                sens_sum1 = sens_sum
                plens1 = plens
                if RSUMMARY > 0.:
                    while len(sens_sum) < RSUMMARY * len(sens):
                        sens_sum += sens_sum1
                        plens += plens1
                    sens += sens_sum
            elif SUMMARY == 2:
                sens = sens_sum + sens
            elif SUMMARY == 3:
                sens = sens + sens_sum
            elif SUMMARY == 4:
                sens = sens_sum + sens + sens_sum
                plens *= 2
            elif SUMMARY == 5:
                sens = sens_sum
                plent = 0

        id_to_sent_dict[t.Id] = sens
        num_plen_dict[len(sens)] += 1
        num_sentence_dict[plent+plens] += 1

        if SUMMARY == 0:
            sens = sens_sum + sens
        for idx, w in enumerate(sens, start=1):
            if idx != len(sens):
                print(w, end=' ', file=of_sen)
            else:
                print(w, end='\n', file=of_sen)
    of_sen.close()

    with open(cur_folder+'/'+'num_plen_dict.json', 'w') as of_plen:
        json.dump(num_plen_dict, of_plen)
    with open(cur_folder+'/'+'num_sentence_dict.json', 'w') as of_sd:
        json.dump(num_sentence_dict, of_sd)

    l_usr = set(l_usr)
    for u in l_usr:
        print(u, end='\n', file=of_usr)
    of_usr.close()

    l_prd = set(l_prd)
    for p in l_prd:
        print(p, end='\n', file=of_prd)
    of_prd.close()

    l_tm = set(l_tm)
    for t in l_tm:
        print(t, end='\n', file=of_tm)
    of_tm.close()

    l_pn = set(l_pn)
    for p in l_pn:
        print(p, end='\n', file=of_pn)
    of_pn.close()

    for h in range(5000):
        print(h, end='\n', file=of_helpfull)
    of_helpfull.close()


def gen_sub_dataset(l_data, folder_sub_name, file_name):
    cur_folder = folder_name + "_" + folder_sub_name
    if not os.path.exists(cur_folder):
        os.mkdir(cur_folder)
    of = open(cur_folder+'/'+file_name, 'w')
    for l in tqdm(l_data):
        print(l.reviewerID, end='\t\t', file=of)
        print(l.asin, end='\t\t', file=of)

        print(l.overall, end='\t\t', file=of)
        sens = id_to_sent_dict[l.Id]
        for idx, w in enumerate(sens, start=1):
            if idx != len(sens):
                print(w, end=' ', file=of)
            else:
                print(w, end='\t\t', file=of)
        print(l.reviewTime, end='\t\t', file=of)
        print(l.reviewerName, end='\t\t', file=of)
        print(l.helpful[0], end='\t\t', file=of)
        print(l.helpful[1], end='\t\t', file=of)
        sens_sum = id_to_sum_sent_dict.get(l.Id, '.')
        for idx, w in enumerate(sens_sum, start=1):
            if idx != len(sens_sum):
                print(w, end=' ', file=of)
            else:
                print(w, end='\n', file=of)


def gen_dataset(d, subname):
    if SUMMARY == 1:
        subname = 'rsummary{}_'.format(RSUMMARY) + subname

    for i in range(50000):
        num_plen_dict[i], num_sentence_dict[i] = 0, 0

    rdev, rtest = .1, .1

    l_all = list(myiter(d))

    # from IPython import embed
    # embed()

    get_all_sentence_pd_usr(l_all, subname)

    tot = float(d.shape[0])
    rtest = rtest * tot / (tot * (1 - rdev))
    dev_d = d.sample(frac=rdev, replace=True, random_state=757)
    d = d.append(dev_d)
    d = d.drop_duplicates(subset=['Id'], keep=False)

    test_d = d.sample(frac=rtest, replace=True, random_state=11119)
    d = d.append(test_d)
    d = d.drop_duplicates(subset=['Id'], keep=False)

    print(subname+':', tot, dev_d.shape, test_d.shape, d.shape)

    l_test = list(myiter(test_d))
    gen_sub_dataset(l_test, subname, 'test.ss')

    l_dev = list(myiter(dev_d))
    gen_sub_dataset(l_dev, subname, 'dev.ss')

    l_train = list(myiter(d))
    gen_sub_dataset(l_train, subname, 'train.ss')

    print(subname+":", len(l_test), len(l_dev), len(l_train))


def gen_small_dataset(d):
    rsmall = .2
    d_small = d.sample(frac=rsmall, random_state=16903)
    gen_dataset(d_small, "small")


def not_empty(s):
    return s and s.strip()


def get_sentences(path):
    docs = map(lambda x: x.strip().split('<sssss>'), open(path, 'r').readlines())
    docs = map(lambda doc: map(lambda sent: sent.split(' '), doc), docs)
    idx = 0
    for i in docs:
        for j in docs:
            for k in j:
                lk = list(filter(not_empty, list(k)))
                if idx < 20:
                    print(lk)
                idx += 1
                yield(lk)


def train_word2vec(subname):
    cur_folder = folder_name.strip('/') + "_" + subname
    model_path = cur_folder+'/'+'word2vec_model_vocsize{}_ebdsize{}'.format(
        args.max_vocab_size,
        args.embed_size
    )
    if os.path.exists(model_path):
        model = gensim.models.Word2Vec.load(model_path)
    else:
        sentences = list(get_sentences(cur_folder+'/'+'data_all_sentences.txt'))
        model = word2vec.Word2Vec(
            sentences,
            min_count=args.word_min_count,
            max_vocab_size=args.max_vocab_size,
            size=args.embed_size,
            workers=10
        )
        model.save(model_path)

    vocab = model.wv.vocab
    all_wordvec = []
    word_emb_dict = []
    of_word = open(cur_folder+'/'+'wordlist_wmc{}.txt'.format(args.word_min_count), 'w')
    of_emb = open(cur_folder+'/'+'embinit_wmc{}.save'.format(args.word_min_count), 'wb')
    of_embedding = open(cur_folder+'/'+'embedding_ebdsize{}.txt'.format(args.embed_size), 'wb')
    for word in tqdm(vocab):
        print(word, file=of_word)
        all_wordvec.append(model[word])
        word_emb_dict.append((word, model[word]))
    all_wordvec = np.array(all_wordvec)
    pickle.dump(all_wordvec, of_emb, protocol=pickle.HIGHEST_PROTOCOL)
    pickle.dump(word_emb_dict, of_embedding, protocol=pickle.HIGHEST_PROTOCOL)
    of_word.close()
    of_emb.close()
    of_embedding.close()


def parse(path):
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)


def getDF(path):
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')


def read_tset():
    docs = []
    with open(folder_name+'/'+'test.ss', 'r') as f:
        for line in f:
            line = line.strip().split('\t\t')
            docs.append(line)
    return docs


def complete_dataset():
    tset = read_tset()
    d = getDF(args.input)
    f_sp = open(folder_name+'/test_add_sp.ss', 'w')
    f_sb = open(folder_name+'/test_add_sb.ss', 'w')
    f_spb = open(folder_name+'/test_add_spb.ss', 'w')
    for l in tqdm(tset):
        """
        idx = d.index(d['reviewerID']==l[0] and \
                      d['asin']==l[1] and \
                      d['reviewTime']==l[4])
        """
        idx = d[d['reviewTime'].str.match(l[4])]
        idx = idx[idx['reviewerID'].str.match(l[0])]
        idx = idx[idx['asin'].str.match(l[1])]
        dsum = idx['summary'].values
        sum_str = ''
        if dsum is None or len(dsum) == 0:
            dsum = '.'
        else:
            dsum = dsum[0]
        if dsum is not None:
            summary = str(dsum)
            if str(summary[len(summary)-1]).isalpha():
                summary = summary + '.'
            sens_sum, plens = proc_para(summary)
            for idx, w in enumerate(sens_sum, start=1):
                if idx != len(sens_sum):
                    sum_str += w + ' '
                else:
                    sum_str += w
        for idx, item in enumerate(l):
            if idx == len(l) - 1:
                print(item, file=f_sp)
                print(item, file=f_sb)
                print(item, file=f_spb)
            elif idx != 3:
                print(item, end='\t\t', file=f_sp)
                print(item, end='\t\t', file=f_sb)
                print(item, end='\t\t', file=f_spb)
            else:
                print(sum_str+' '+item, end='\t\t', file=f_sp)
                print(item+' '+sum_str, end='\t\t', file=f_sb)
                print(sum_str+' '+item+' '+sum_str, end='\t\t', file=f_spb)
    f_sp.close()
    f_sb.close()
    f_spb.close()



def main():
    if args.complete_dataset:
        complete_dataset()
        sys.exit(0)
    if not args.only_word2vec:
        # d = pd.read_csv('food_score_data.csv')
        d = getDF(args.input)
        # gen_small_dataset(d)
        gen_dataset(d, "full")
    if SUMMARY == 1:
        subname = 'rsummary{}_'.format(RSUMMARY) + "full"
    else:
        subname = 'full'

    train_word2vec(subname)
    # train_word2vec(subname.replace('full', 'small'))


if __name__ == "__main__":
    main()
# vim: ts=4 sw=4 sts=4 expandtab
