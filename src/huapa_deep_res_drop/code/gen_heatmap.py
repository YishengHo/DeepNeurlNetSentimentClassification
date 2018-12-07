#!/usr/bin/env mdl
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from argparse import ArgumentParser
import time
import concurrent.futures

parse = ArgumentParser()
parse.add_argument("--dataset", default="")
parse.add_argument("--full", default=False, action='store_true')
parse.add_argument("--nopart", default=False, action='store_true')
parse.add_argument("--output", default='')
args = parse.parse_args()
inf = args.dataset + '_attention_value.pkl'
if 'spb' in args.dataset:
    data_path = '../../data/' + args.dataset + '/test_add_spb.ss'
else:
    data_path = '../../data/' + args.dataset + '/test.ss'
img_path = './' + args.dataset + '_' + args.output
if not os.path.exists(img_path):
    os.mkdir(img_path)


def read_docs():
    docs = []
    with open(data_path, 'r') as f:
        for line in f:
            line = line.strip().split('\t\t')
            doc = line[3].strip().lower()
            sens = doc.split('<sssss>')
            sens = [sen.strip().split() for sen in sens]
            docs.append(sens)
    return docs


def draw_save_heatmap(label_x, label_y, uptha, fname):
    sns.set()

    fontsize_pt = plt.rcParams['xtick.labelsize']
    fontsize_pt_h = plt.rcParams['ytick.labelsize']
    dpi = 7.27
    matrix_width_pt = fontsize_pt * len(label_x)
    matrix_width_in = matrix_width_pt / dpi
    matrix_height_pt = fontsize_pt_h * 5
    matrix_height_in = matrix_height_pt / dpi

    top_margin = 0.2
    bottom_margin = 0.1
    left_margin = 0.2
    right_margin = 0.1
    figure_width = matrix_width_in / (1 - left_margin - right_margin)
    figure_height = matrix_height_in / (1 - top_margin - bottom_margin)

    fig, ax = plt.subplots(
        figsize = (figure_height, figure_width),
        gridspec_kw=dict(top=1-top_margin, bottom=bottom_margin,
                         left=left_margin, right=1-right_margin
                        )
    )

    ax = sns.heatmap(uptha,
                     xticklabels=label_x,
                     yticklabels=label_y,
                     vmin=0, vmax=1,
                     cbar=False,
                     ax=ax,
                     square=True,
                     cmap="Blues")
    ax.xaxis.tick_top()
    plt.xticks(rotation=45)
    plt.savefig(img_path+'/'+fname)
    plt.close()


def tqdm_enumerate(iter):
    i = 0
    for y in tqdm(iter):
        yield i, y
        i += 1


def gen_sa_heatmap(data, docs, mask = None):
    idx_doc = 0
    if mask:
        to_draw = set([int(i[0]) for i in mask])
    else:
        to_draw = None
    for idx, d in tqdm_enumerate(data):
        dlens = d['dlen']
        slens = d['slen']

        uas = np.resize(d['usa'], (100, 40))
        pas = np.resize(d['psa'], (100, 40))
        tas = np.resize(d['tsa'], (100, 40))
        has = np.resize(d['hsa'], (100, 40))
        equal = d['equal']

        label_y = ['UA', 'PA', 'TA', 'HA']

        for i in range(len(dlens)):
            dlen, slen, ua, pa, ta, ha = dlens[i], slens[i], uas[i], pas[i], tas[i], has[i]
            sz = 0
            if to_draw is not None and idx_doc not in to_draw:
                idx_doc += 1
                continue
            if 'spb' in args.dataset and not equal[i]:
                idx_doc += 1
                continue
            ua, pa, ta, ha = np.asarray(ua)[None, : dlen], np.asarray(pa)[None, : dlen], \
                             np.asarray(ta)[None, : dlen], np.asarray(ha)[None, : dlen]
            part_uptha = np.concatenate((ua, pa, ta, ha))
            fname = '{}_sa.png'.format(idx_doc)
            label_x = [i[0] for i in docs[idx_doc] if len(i) > 0]
            if not args.nopart:
                draw_save_heatmap(label_x, label_y, part_uptha, fname)

            idx_doc += 1


def gen_heatmap(data, docs, mask = None):
    idx_doc = 0
    if mask:
        to_draw = set([int(i[0]) for i in mask])
    else:
        to_draw = None
    for idx, d in tqdm_enumerate(data):
        dlens = d['dlen']
        slens = d['slen']
        uas = np.resize(d['uwa'], (100, 40, 50))
        pas = np.resize(d['pwa'], (100, 40, 50))
        tas = np.resize(d['twa'], (100, 40, 50))
        has = np.resize(d['hwa'], (100, 40, 50))
        equal = d['equal']

        label_y = ['UA', 'PA', 'TA', 'HA']

        for i in range(len(dlens)):
            dlen, slen, ua, pa, ta, ha = dlens[i], slens[i], uas[i], pas[i], tas[i], has[i]
            dua, dpa, dta, dha = [], [], [], []
            sz = 0

            if to_draw is not None and idx_doc not in to_draw:
                idx_doc += 1
                continue
            if 'spb' in args.dataset and not equal[i]:
                idx_doc += 1
                continue
            for j in range(dlen-1):
                l = slen[j]
                sz += l
                ua, pa, ta, ha = uas[i][j][:l], pas[i][j][:l], tas[i][j][:l], has[i][j][:l]
                if args.full:
                    dua.extend(ua)
                    dpa.extend(pa)
                    dta.extend(ta)
                    dha.extend(ha)
                ua, pa, ta, ha = np.asarray(ua)[None, :], np.asarray(pa)[None, :], \
                                 np.asarray(ta)[None, :], np.asarray(ha)[None, :]
                part_uptha = np.concatenate((ua, pa, ta, ha))
                fname = '.png'
                if dlen-1 == 1:
                    fname = str(idx_doc) + fname
                else:
                    fname = '{}_{}'.format(idx_doc, j) + fname
                label_x = docs[idx_doc][j]
                if not args.nopart:
                    draw_save_heatmap(label_x, label_y, part_uptha, fname)
            if args.full:
                dua, dpa, dta, dha = np.asarray(dua)[None,:], np.asarray(dpa)[None, :], \
                                     np.asarray(dta)[None, :], np.asarray(dha)[None, :]
                uptha = np.concatenate((dua,dpa,dta,dha))
                doc = docs[idx_doc]
                label_x = []
                for s in doc:
                    label_x.extend(s)
                draw_save_heatmap(label_x, label_y, uptha, '{}_full.png'.format(idx_doc))
            idx_doc += 1


def main():
    data = pickle.load(open(inf, 'rb'))
    docs = read_docs()
    mask = [i.split(' ') for i in open('./spq_true_txt_false.txt', 'r').readlines()]
    gen_sa_heatmap(data, docs, mask)
    gen_heatmap(data, docs, mask)


if __name__ == '__main__':
    main()
# vim: ts=4 sw=4 sts=4 expandtab
