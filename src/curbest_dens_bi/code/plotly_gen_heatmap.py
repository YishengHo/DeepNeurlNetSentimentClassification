#!/usr/bin/env mdl
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import plotly.plotly as py
import plotly as pyo
pyo.tools.set_credentials_file(username='ethanhe01', api_key='6KdPbMzANVsJ37IlO2Mm')
import plotly.graph_objs as go
from argparse import ArgumentParser

parse = ArgumentParser()
parse.add_argument("--dataset", default="")
args = parse.parse_args()
inf = args.dataset + '_attention_value.pkl'
data_path = '../../data/' + args.dataset + '/test.ss'

def read_docs():
    docs = []
    with open(data_path, 'r') as f:
        idx = 0
        for line in f:
            line = line.strip().split('\t\t')
            doc = line[3].strip().lower()
            sens = doc.split('<sssss>')
            sens = [sen.strip().split() for sen in sens]
            docs.append(sens)
    return docs


def main():
    data = pickle.load(open(inf, 'rb'))
    docs = read_docs()
    idx_doc = 0
    for idx, d in enumerate(data):
        if idx >= 1:
            continue
        dlens = d['dlen']
        slens = d['slen']
        uas = np.resize(d['uwa'], (100, 40, 50))
        pas = np.resize(d['pwa'], (100, 40, 50))
        tas = np.resize(d['twa'], (100, 40, 50))
        has = np.resize(d['hwa'], (100, 40, 50))
        img_path = './' + args.dataset
        if not os.path.exists(img_path):
            os.mkdir(img_path)

        for i in range(100):
            dlen, slen, ua, pa, ta, ha = dlens[i], slens[i], uas[i], pas[i], tas[i], has[i]
            dua, dpa, dta, dha = [], [], [], []
            sz = 0
            for j in range(dlen-1):
                l = slen[j]
                sz += l
                ua, pa, ta, ha = uas[i][j][:l], pas[i][j][:l], tas[i][j][:l], has[i][j][:l]
                dua.extend(ua)
                dpa.extend(pa)
                dta.extend(ta)
                dha.extend(ha)
                uptha = np.concatenate((ua,pa,ta,ha))
            dua, dpa, dta, dha = ua[None,:], pa[None, :], ta[None, :], ha[None, :]
            uptha = np.concatenate((dua,dpa,dta,dha))
            print(np.shape(uptha))
            label_y = ['UA', 'PA', 'TA', 'HA']
            doc = docs[idx_doc]
            label_x = []
            for s in doc:
                label_x.extend(s)
            print(len(label_x), label_x)
            sns.set()

            """
            fontsize_pt = plt.rcParams['xtick.labelsize']
            dpi = 12.27
            matrix_width_pt = fontsize_pt * len(doc)
            matrix_width_in = matrix_width_pt / dpi
            top_margin = 0.04
            bottom_margin = 0.04
            figure_width = matrix_width_in / (1 - top_margin)

            fig, ax = plt.subplots(
                figsize = (100, figure_width),
                gridspec_kw=dict(right=1-top_margin, left= bottom_margin)
            )
            """
            """
            ax = sns.heatmap(uptha,
                             annot=False,
                             xticklabels=label_x,
                             yticklabels=label_y,
                             vmin=1, vmax=0,
                             square=True,
                             cmap="Blues")
            ax.xaxis.tick_top()
            plt.xticks(rotation=90)
            plt.savefig(img_path+'/'+'{}.png'.format(idx_doc))
            """
            trace = go.Heatmap(z=uptha,
                               x=label_x,
                               y=label_y)
            data = [trace]
            py.iplot(data,
                     colorscale=[
                         # Let first 10% (0.1) of the values have color rgb(0, 0, 0)
                        [0, 'rgb(0, 0, 0)'],
                        [0.1, 'rgb(0, 0, 0)'],

                        # Let values between 10-20% of the min and max of z
                        # have color rgb(20, 20, 20)
                        [0.1, 'rgb(20, 20, 20)'],
                        [0.2, 'rgb(20, 20, 20)'],

                        # Values between 20-30% of the min and max of z
                        # have color rgb(40, 40, 40)
                        [0.2, 'rgb(40, 40, 40)'],
                        [0.3, 'rgb(40, 40, 40)'],

                        [0.3, 'rgb(60, 60, 60)'],
                        [0.4, 'rgb(60, 60, 60)'],

                        [0.4, 'rgb(80, 80, 80)'],
                        [0.5, 'rgb(80, 80, 80)'],

                        [0.5, 'rgb(100, 100, 100)'],
                        [0.6, 'rgb(100, 100, 100)'],

                        [0.6, 'rgb(120, 120, 120)'],
                        [0.7, 'rgb(120, 120, 120)'],

                        [0.7, 'rgb(140, 140, 140)'],
                        [0.8, 'rgb(140, 140, 140)'],

                        [0.8, 'rgb(160, 160, 160)'],
                        [0.9, 'rgb(160, 160, 160)'],

                        [0.9, 'rgb(180, 180, 180)'],
                        [1.0, 'rgb(180, 180, 180)']
                     ],
                     filename='{}.png'.format(idx_doc))
            idx_doc += 1



if __name__ == '__main__':
    main()
# vim: ts=4 sw=4 sts=4 expandtab
