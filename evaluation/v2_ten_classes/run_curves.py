import json, sys, numpy as np
from ultralytics import YOLO
model = YOLO('model_v2.pt')
for name in sys.argv[1:]:
    m = model.val(data=f'{name}.yaml', split='val', imgsz=640, batch=16, conf=0.001, iou=0.7,
                  device='cpu', plots=False, verbose=False, project='runs', name=name + '_c', exist_ok=True)
    b = m.box
    out = dict(px=np.asarray(b.px).tolist(), prec=np.asarray(b.prec_values).tolist(),
               f1=np.asarray(b.f1_curve).tolist(), p=np.asarray(b.p_curve).tolist(), r=np.asarray(b.r_curve).tolist(),
               cls=[m.names[int(c)] for c in b.ap_class_index], ap50=np.asarray(b.ap50).tolist())
    json.dump(out, open(f'curvesdata_{name}.json', 'w'))
    print(name, 'done', flush=True)
