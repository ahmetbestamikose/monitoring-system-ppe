"""Estimate the per-frame processing time of the two detection servers on a CPU.

Camera capture and network transfer are excluded. Frames are taken from a folder of
images and resized to 640x480, as in the servers. Three configurations are timed:
  inference : one YOLOv8m prediction at the default input size (640)
  mobile    : resize + prediction (conf 0.15) + drawing + JPEG encoding (quality 50)
  desktop   : resize + prediction (conf 0.10) + two extra predictions at 960 px
              (whole frame and upper 55 %, as in the ear-protection passes) + drawing + JPEG encoding
usage: python benchmark_cpu.py <weights.pt> <image_dir> [n_frames] [threads]
"""
import glob, json, os, platform, statistics, sys, time
import cv2, torch
from ultralytics import YOLO

weights, image_dir = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 50
threads = int(sys.argv[4]) if len(sys.argv) > 4 else os.cpu_count()
torch.set_num_threads(threads)
model = YOLO(weights)
files = sorted(glob.glob(os.path.join(image_dir, '*')))[:n]
frames = [cv2.resize(cv2.imread(f), (640, 480)) for f in files]

def draw_and_encode(frame, res, quality):
    out = frame.copy()
    for b in res[0].boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.rectangle(out, (5, 15), (400, 45), (0, 0, 255), -1)
    params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    return cv2.imencode('.jpg', out, params)[1]

def run(cfg, frame):
    t = time.perf_counter()
    if cfg == 'inference':
        model.predict(frame, verbose=False)
    elif cfg == 'mobile':
        f = cv2.resize(frame, (640, 480))
        r = model.predict(f, conf=0.15, verbose=False)
        draw_and_encode(f, r, 50)
    elif cfg == 'desktop':
        f = cv2.resize(frame, (640, 480))
        r = model.predict(f, conf=0.10, verbose=False)
        model.predict(f, conf=0.03, imgsz=960, verbose=False)
        model.predict(f[:int(480 * 0.55)], conf=0.03, imgsz=960, verbose=False)
        draw_and_encode(f, r, 95)
    return (time.perf_counter() - t) * 1000

for f in frames[:5]:                       # warm-up
    run('desktop', f)
result = {'cpu': platform.processor() or platform.machine(), 'threads': threads, 'frames': len(frames)}
for cfg in ('inference', 'mobile', 'desktop'):
    times = [run(cfg, f) for f in frames]
    result[cfg] = {'mean_ms': statistics.mean(times), 'median_ms': statistics.median(times),
                   'p95_ms': sorted(times)[int(0.95 * len(times)) - 1]}
print(json.dumps(result, indent=1))
with open('benchmark_cpu_result.json', 'w') as fh:
    json.dump(result, fh, indent=1)
