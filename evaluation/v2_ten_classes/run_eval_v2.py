"""Evaluate the ten-class model on the full validation set, its clean / overlapping subsets
and the independent test set (first-version validation images never used for training)."""
import json, sys, numpy as np
from ultralytics import YOLO
model = YOLO('model_v2.pt')
out = {}
for name in sys.argv[1:]:
    m = model.val(data=f'{name}.yaml', split='val', imgsz=640, batch=16, conf=0.001, iou=0.7,
                  device='cpu', plots=False, verbose=False, project='runs', name=name, exist_ok=True)
    b = m.box
    res = dict(all=dict(P=float(b.mp), R=float(b.mr), mAP50=float(b.map50), mAP5095=float(b.map)), cls={})
    for i, c in enumerate(b.ap_class_index):
        res['cls'][m.names[int(c)]] = dict(P=float(b.p[i]), R=float(b.r[i]), mAP50=float(b.ap50[i]), mAP5095=float(b.ap[i]))
    # curves: PR (x=recall grid, y per class) and F1-confidence
    curves = {}
    for (x, y, xl, yl), key in zip(b.curves_results, b.curves):
        curves[key] = dict(x=np.asarray(x).tolist(), y=np.asarray(y).tolist())
    res['ap_class_index'] = [int(c) for c in b.ap_class_index]
    res['nt_per_class'] = [int(v) for v in m.nt_per_class] if hasattr(m, 'nt_per_class') else None
    res['speed'] = m.speed
    out[name] = res
    json.dump(res, open(f'res_{name}.json', 'w'))
    json.dump(curves, open(f'curves_{name}.json', 'w'))
    print(name, res['all'], flush=True)
