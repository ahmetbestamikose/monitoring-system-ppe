"""Re-evaluate a trained YOLOv8 model on the full, clean and overlapping validation subsets.

usage: python evaluate_subsets.py <weights.pt> <subset_dir>
<subset_dir> must contain all.txt, clean.txt and overlap.txt (one image path per line),
for example produced by overlap_check.py.
"""
import json, os, sys
from ultralytics import YOLO

weights, subset_dir = sys.argv[1], sys.argv[2]
model = YOLO(weights)
names = model.names
out = {}
for subset in ('all', 'clean', 'overlap'):
    yaml_path = os.path.join(subset_dir, f'{subset}.yaml')
    with open(yaml_path, 'w') as fh:
        fh.write(f"path: {os.path.abspath(subset_dir)}\ntrain: {subset}.txt\nval: {subset}.txt\nnames:\n")
        for i in sorted(names):
            fh.write(f"  {i}: {names[i]}\n")
    r = model.val(data=yaml_path, imgsz=640, batch=8, device='cpu', plots=False, verbose=True, workers=0)
    res = {'all': [float(x) for x in r.box.mean_results()]}
    for i, c in enumerate(r.box.ap_class_index):
        res[names[int(c)]] = [float(x) for x in r.box.class_result(i)]
    out[subset] = res
with open('subset_results.json', 'w') as fh:
    json.dump(out, fh, indent=1)
print(json.dumps(out, indent=1))
