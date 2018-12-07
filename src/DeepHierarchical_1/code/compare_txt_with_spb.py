#!/usr/bin/env mdl
import pickle
import numpy as np
from tqdm import tqdm

dp1 = 'GGF_spb_full_attention_value.pkl'
dp2 = 'GGF_full_attention_value.pkl'

def tqdm_enumerate(iter, start=0):
    idx = start
    for item1, item2 in tqdm(iter):
        yield idx, item1, item2
        idx += 1


def main():
    data1 = pickle.load(open(dp1, 'rb'))
    data2 = pickle.load(open(dp2, 'rb'))
    idx_doc = 0
    f1 = open("spq_true_txt_false.txt", 'w')
    f2 = open("txt_true_spb_false.txt", 'w')

    # from IPython import embed
    # embed()

    for idx, d1, d2 in tqdm_enumerate(zip(data1, data2)):
        dlens1 = d1['dlen']
        equal1 = d1['equal']
        dlens2 = d2['dlen']
        equal2 = d2['equal']
        label = d1['label']
        for i in range(len(equal1)):
            if equal1[i] and not equal2[i]:
                print(idx_doc, int(np.where(label[i]==1)[0]), dlens2[i], file=f1)
            elif equal2[i] and not equal1[i]:
                print(idx_doc, int(np.where(label[i]==1)[0]), dlens2[i], file=f2)
            idx_doc += 1
    f1.close()
    f2.close()


if __name__ == '__main__':
    main()

# vim: ts=4 sw=4 sts=4 expandtab
