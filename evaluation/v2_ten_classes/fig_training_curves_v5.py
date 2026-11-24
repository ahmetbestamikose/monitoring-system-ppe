"""Training curves of the ten-class model from the Ultralytics results.csv (training report)."""
import csv, sys
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
src = sys.argv[1] if len(sys.argv) > 1 else 'results.csv'
rows = [{k.strip(): float(v) for k, v in r.items()} for r in csv.DictReader(open(src))]
ep = [r['epoch'] for r in rows]
plt.rcParams.update({'font.family': 'TeX Gyre Termes', 'font.size': 8, 'axes.linewidth': 0.6})
fig, (a, b) = plt.subplots(1, 2, figsize=(150 / 25.4, 58 / 25.4))
cols = {'box': '#1f4e79', 'cls': '#c55a11', 'dfl': '#548235'}
lab = {'box': 'Box', 'cls': 'Classification', 'dfl': 'DFL'}
for k in ('box', 'cls', 'dfl'):
    a.plot(ep, [r[f'train/{k}_loss'] for r in rows], color=cols[k], lw=1.0)
    a.plot(ep, [r[f'val/{k}_loss'] for r in rows], color=cols[k], lw=1.0, ls='--')
a.set_xlabel('Epoch'); a.set_ylabel('Loss'); a.set_title('(a) Losses', fontsize=8)
from matplotlib.lines import Line2D
handles = [Line2D([], [], color=cols[k], lw=1.0) for k in ('box', 'cls', 'dfl')] + \
          [Line2D([], [], color='k', lw=1.0), Line2D([], [], color='k', lw=1.0, ls='--')]
a.legend(handles, ['Box', 'Classification', 'DFL', 'Training', 'Validation'], frameon=False, ncol=2,
         loc='upper right', handlelength=1.8, columnspacing=1.0)
for key, name, c in (('metrics/precision(B)', 'Precision', '#7f7f7f'), ('metrics/recall(B)', 'Recall', '#2e75b6'),
                     ('metrics/mAP50(B)', 'mAP@0.5', '#1f4e79'), ('metrics/mAP50-95(B)', 'mAP@0.5:0.95', '#c00000')):
    b.plot(ep, [r[key] for r in rows], lw=1.0, color=c, label=name)
b.set_xlabel('Epoch'); b.set_ylabel('Score'); b.set_ylim(0, 1); b.set_title('(b) Validation metrics', fontsize=8)
b.legend(frameon=False, loc='lower right')
for ax in (a, b):
    ax.axvline(90, color='0.5', lw=0.6, ls=':'); ax.set_xlim(0, 100); ax.tick_params(width=0.6)
fig.tight_layout(w_pad=1.5)
fig.savefig('figures/fig_training_curves_v5.png', dpi=600)
