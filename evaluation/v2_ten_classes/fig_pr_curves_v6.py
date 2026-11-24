"""Precision-recall curves of the ten-class model on the validation set (a) and on the
independent test set (b), from the arrays returned by Ultralytics val (px, prec_values)."""
import json, sys
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
val = json.load(open(sys.argv[1])); test = json.load(open(sys.argv[2]))
pretty = {'mask': 'Mask', 'helmet': 'Helmet', 'belt': 'Harness', 'vest': 'Vest', 'glasses': 'Glasses',
          'welding_helmet': 'Welding helmet', 'ear_protection': 'Ear protection', 'toolbox': 'Toolbox',
          'safety_gloves': 'Safety gloves', 'protective_suit': 'Protective suit'}
colors = dict(zip(pretty, ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                           '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']))
plt.rcParams.update({'font.family': 'TeX Gyre Termes', 'font.size': 8, 'axes.linewidth': 0.6})
fig, axes = plt.subplots(1, 2, figsize=(150 / 25.4, 66 / 25.4))
for ax, d, title in ((axes[0], val, '(a) Validation set'), (axes[1], test, '(b) Independent test set')):
    px = np.array(d['px']); prec = np.array(d['prec'])
    ap = np.array(d['ap50'])
    for k, c in enumerate(d['cls']):
        ax.plot(px, prec[k], lw=0.9, color=colors[c], label=f'{pretty[c]} {ap[k]:.3f}')
    ax.plot(px, prec.mean(0), lw=1.8, color='k', label=f'All classes {ap.mean():.3f}')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02); ax.set_xlabel('Recall'); ax.set_ylabel('Precision')
    ax.set_title(title, fontsize=8); ax.tick_params(width=0.6)
    ax.legend(loc='lower left', frameon=True, framealpha=0.9, edgecolor='0.8', fontsize=8,
              handlelength=1.4, borderpad=0.3, labelspacing=0.2)
fig.tight_layout(w_pad=1.5)
fig.savefig('figures/fig_pr_curves_v6.png', dpi=600)
