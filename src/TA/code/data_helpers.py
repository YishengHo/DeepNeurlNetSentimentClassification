#-*- coding: utf-8 -*-
#author: Zhen Wu

import numpy as np
import pickle
import tqdm

def load_embedding(embedding_file_path, corpus, embedding_dim):
    wordset = set();
    for line in corpus:
        line = line.strip().split()
        for w in line:
            wordset.add(w.lower())
    words_dict = dict(); word_embedding = []; index = 1
    words_dict['$EOF$'] = 0  #add EOF
    word_embedding.append(np.zeros(embedding_dim))
    with open(embedding_file_path, 'rb') as f:
        word_emb_dict = pickle.loads(f.read())
        print(len(word_emb_dict))
        for index, line in enumerate(word_emb_dict,start=1):
            embedding = line[1]
            word_embedding.append(embedding)
            words_dict[line[0]] = index

    """
    with open(embedding_file_path, 'r') as f:
        for line in f:
            check = line.strip().split()
            if len(check) == 2: continue
            line = line.strip().split()
            if line[0] not in wordset: continue
            embedding = [float(s) for s in line[1:]]
            word_embedding.append(embedding)
            words_dict[line[0]] = index
            index +=1
    """
    return np.asarray(word_embedding), words_dict


def fit_transform(x_text, words_dict, max_sen_len, max_doc_len):
    x, sen_len, doc_len = [], [], []
    for index, doc in enumerate(x_text):
        t_sen_len = [0] * max_doc_len
        t_x = np.zeros((max_doc_len, max_sen_len), dtype=int)
        sentences = doc.split('<sssss>')
        i = 0
        for sen in sentences:
            j = 0
            for word in sen.strip().split():
                if j >= max_sen_len:
                    break
                if word not in words_dict: continue
                t_x[i, j] = words_dict[word]
                j += 1
            t_sen_len[i] = j
            i += 1
            if i >= max_doc_len:
                break
        doc_len.append(i)
        sen_len.append(t_sen_len)
        x.append(t_x)
    return np.asarray(x), np.asarray(sen_len), np.asarray(doc_len)

class Dataset(object):
    def __init__(self, data_file):
        self.t_usr = []
        self.t_prd = []
        self.t_hlp = []
        self.t_tme = []
        self.t_label = []
        self.t_docs = []
        self.t_sums = []
        with open(data_file, 'r') as f:
            idx = 0
            for line in f:
                # line = line.strip().decode('utf8', 'ignore').split('\t\t')
                line = line.strip().split('\t\t')
                if idx == 0 :
                    print("one input data line: ", line)
                    idx = 2
                    print("length:", len(line))
                if len(line) < 8:
                    continue
                self.t_usr.append(line[0].strip())
                self.t_prd.append(line[1].strip())
                self.t_label.append(int(float(line[2].strip()))-1)
                self.t_docs.append(line[3].strip().lower())
                self.t_tme.append(line[4].strip().lower())
                self.t_hlp.append(line[6].strip().lower())
                # self.t_sums.append(line[8].strip().lower())
        self.data_size = len(self.t_docs)
        self.sum_size = len(self.t_sums)

    def get_usr_prd_hlp_tme_dict(self):
        usrdict, prddict, hlpdict, tmedict = dict(), dict(), dict(), dict()
        usridx, prdidx, hlpidx, tmeidx  = 0, 0, 0, 0
        for u in self.t_usr:
            if u not in usrdict:
                usrdict[u] = usridx
                usridx += 1
        for p in self.t_prd:
            if p not in prddict:
                prddict[p] = prdidx
                prdidx += 1
        for h in self.t_hlp:
            if h not in hlpdict:
                hlpdict[h] = hlpidx
                hlpidx += 1
        for t in self.t_tme:
            if t not in tmedict:
                tmedict[t] = tmeidx
                tmeidx += 1
        return usrdict, prddict, hlpdict, tmedict

    def genBatch(self, usrdict, prddict, hlpdict, tmedict, wordsdict, batch_size, max_sen_len, max_doc_len, n_class):
        self.epoch = int(len(self.t_docs) / batch_size)
        if len(self.t_docs) % batch_size != 0:
            self.epoch += 1
        self.usr = []
        self.prd = []
        self.hlp = []
        self.tme = []
        self.label = []
        self.docs = []
        self.sen_len = []
        self.doc_len = []
        self.sums = []
        self.ssen_len = []
        self.sdoc_len = []

        for i in range(self.epoch):
            self.usr.append(np.asarray(list(map(lambda x: usrdict.get(x, len(usrdict)), self.t_usr[i*batch_size:(i+1)*batch_size])), dtype=np.int32))
            self.prd.append(np.asarray(list(map(lambda x: prddict.get(x, len(prddict)), self.t_prd[i*batch_size:(i+1)*batch_size])), dtype=np.int32))
            self.hlp.append(np.asarray(list(map(lambda x: hlpdict.get(x, len(hlpdict)), self.t_hlp[i*batch_size:(i+1)*batch_size])), dtype=np.int32))
            self.tme.append(np.asarray(list(map(lambda x: tmedict.get(x, len(tmedict)), self.t_tme[i*batch_size:(i+1)*batch_size])), dtype=np.int32))
            self.label.append(np.eye(n_class, dtype=np.float32)[self.t_label[i*batch_size:(i+1)*batch_size]])
            b_docs, b_sen_len, b_doc_len = fit_transform(self.t_docs[i*batch_size:(i+1)*batch_size],
                                                         wordsdict, max_sen_len, max_doc_len)
            # b_sums, b_ssen_len, b_sdoc_len = fit_transform(self.t_sums[i*batch_size:(i+1)*batch_size],
            #                                              wordsdict, max_ssen_len, max_sdoc_len)
            self.docs.append(b_docs)
            self.sen_len.append(b_sen_len)
            self.doc_len.append(b_doc_len)

            """
            self.sums.append(b_sums)
            self.ssen_len.append(b_ssen_len)
            self.sdoc_len.append(b_sdoc_len)
            """

    def batch_iter(self, usrdict, prddict, hlpdict, tmedict, wordsdict, n_class, batch_size, num_epochs, max_sen_len, max_doc_len, shuffle=True):
        data_size = len(self.t_docs)
        num_batches_per_epoch = int(data_size / batch_size) + \
                                (1 if data_size % batch_size else 0)
        self.t_usr = np.asarray(self.t_usr)
        self.t_prd = np.asarray(self.t_prd)
        self.t_hlp = np.asarray(self.t_hlp)
        self.t_tme = np.asarray(self.t_tme)
        self.t_label = np.asarray(self.t_label)
        self.t_docs = np.asarray(self.t_docs)

        for epoch in range(num_epochs):
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                self.t_usr = self.t_usr[shuffle_indices]
                self.t_prd = self.t_prd[shuffle_indices]
                self.t_hlp = self.t_hlp[shuffle_indices]
                self.t_tme  = self.t_tme[shuffle_indices]
                self.t_label = self.t_label[shuffle_indices]
                self.t_docs = self.t_docs[shuffle_indices]

            for batch_num in range(num_batches_per_epoch):
                start = batch_num * batch_size
                end = min((batch_num + 1) * batch_size, data_size)
                usr = map(lambda x: usrdict[x], self.t_usr[start:end])
                prd = map(lambda x: prddict[x], self.t_prd[start:end])
                hlp = map(lambda x: hlpdict[x], self.t_hlp[start:end])
                tme = map(lambda x: tmedict[x], self.t_tme[start:end])
                label = np.eye(n_class, dtype=np.float32)[self.t_label[start:end]]
                docs, sen_len, doc_len = fit_transform(self.t_docs[start:end], wordsdict, max_sen_len, max_doc_len)
                batch_data = zip(usr, prd, hlp, tme, docs, label, sen_len, doc_len)
                yield batch_data


