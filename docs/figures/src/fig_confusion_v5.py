"""Re-draw the normalised confusion matrix of the ten-class model with print-size fonts (>= 8 pt).

The object counts are reconstructed exactly from the validation report: for every class the
number of validation objects (Table 6), the objects missed (predicted as background) and the
detections without a matching annotation are known; one vest was predicted as a harness.
Columns: ground truth, rows: prediction; each column is normalised to 1 as in the original plot.
"""
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

names = ['Mask', 'Helmet', 'Harness', 'Vest', 'Glasses', 'Welding helmet', 'Ear protection',
         'Toolbox', 'Safety gloves', 'Protective suit', 'Background']
inst   = [252, 315, 103, 170, 21, 59, 44, 120, 82, 70]       # validation objects per class
missed = [18, 32, 3, 11, 2, 1, 0, 0, 0, 3]                   # predicted as background
spur   = [16, 18, 4, 7, 2, 3, 0, 7, 0, 2]                     # detections without annotation
n = len(names)
C = np.zeros((n, n))
for k in range(10):
    C[k, k] = inst[k] - missed[k]
    C[10, k] = missed[k]
    C[k, 10] = spur[k]
C[3, 3] -= 1; C[2, 3] += 1                                    # one vest predicted as harness
assert all(C[:, k].sum() == inst[k] for k in range(10))
N = C / C.sum(axis=0, keepdims=True)

plt.rcParams.update({'font.family': 'TeX Gyre Termes', 'font.size': 8})
fig, ax = plt.subplots(figsize=(130 / 25.4, 96 / 25.4))
masked = np.ma.masked_where(N == 0, N)
im = ax.imshow(masked, cmap='Blues', vmin=0, vmax=1)
for i in range(n):
    for j in range(n):
        if N[i, j] > 0:
            ax.text(j, i, f'{N[i, j]:.2f}', ha='center', va='center', fontsize=8,
                    color='white' if N[i, j] > 0.5 else 'black')
ax.set_xticks(range(n)); ax.set_yticks(range(n))
ax.set_xticklabels(names, rotation=45, ha='right', rotation_mode='anchor', fontsize=8)
ax.set_yticklabels(names, fontsize=8)
ax.set_xlabel('Ground truth', fontsize=8); ax.set_ylabel('Prediction', fontsize=8)
ax.set_xticks(np.arange(-.5, n), minor=True); ax.set_yticks(np.arange(-.5, n), minor=True)
ax.grid(which='minor', color='0.85', linewidth=0.4); ax.tick_params(which='minor', length=0)
cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.ax.tick_params(labelsize=8)
fig.tight_layout()
fig.savefig('confusion_matrix_10class.png', dpi=600)
print(C.astype(int))
