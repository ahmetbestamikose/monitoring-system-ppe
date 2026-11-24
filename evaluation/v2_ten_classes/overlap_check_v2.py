"""Overlap between the training and validation sets of the ten-class dataset (ISG_Final_Dataset2)
and selection of the independent test set from the validation folder of the five-class dataset.

A validation image is counted as a copy of a training image if a training image has the same
original file name (the part before the "_jpg.rf." suffix of Roboflow exports) or the same MD5
checksum. MD5 sums are computed only for files whose size occurs more than once.
usage: python overlap_check_v2.py <ISG_Final_Dataset2> <ISG_Final_Dataset>   -> subsets_v2.json
"""
import collections, hashlib, json, os, re, sys
D2, D1 = sys.argv[1], sys.argv[2]
NAMES = ['mask', 'helmet', 'belt', 'vest', 'glasses', 'welding_helmet', 'ear_protection', 'toolbox',
         'safety_gloves', 'protective_suit']

def stem(fn):
    return re.split(r'_(jpg|jpeg|png|JPG|JPEG|PNG|webp)\.rf\.', os.path.splitext(fn)[0])[0]

def scan(root, split):
    out = {}
    for e in os.scandir(f'{root}/{split}/images'):
        lab = f'{root}/{split}/labels/{os.path.splitext(e.name)[0]}.txt'
        cls = [int(l.split()[0]) for l in open(lab) if l.strip()] if os.path.exists(lab) else []
        out[e.name] = dict(stem=stem(e.name), size=e.stat().st_size, cls=cls, path=e.path)
    return out

tr2, va2, va1 = scan(D2, 'train'), scan(D2, 'valid'), scan(D1, 'valid')
by_size = collections.Counter(d['size'] for s in (tr2, va2, va1) for d in s.values())
md5 = lambda d: hashlib.md5(open(d['path'], 'rb').read()).hexdigest() if by_size[d['size']] > 1 else None
for s in (tr2, va2, va1):
    for d in s.values():
        d['md5'] = md5(d)
tr_stems = {d['stem'] for d in tr2.values()}
tr_md5 = {d['md5'] for d in tr2.values() if d['md5']}
va2_stems = {d['stem'] for d in va2.values()}
is_copy = lambda d: d['stem'] in tr_stems or (d['md5'] in tr_md5 if d['md5'] else False)
clean = [f for f, d in va2.items() if not is_copy(d)]
over = [f for f, d in va2.items() if is_copy(d)]
indep = [f for f, d in va1.items() if f not in va2 and not is_copy(d) and d['stem'] not in va2_stems]
per = {n: [0, 0] for n in NAMES}
for f, d in va2.items():
    for c in set(d['cls']):
        per[NAMES[c]][0] += 1; per[NAMES[c]][1] += is_copy(d)
print('validation images', len(va2), 'copies', len(over), 'clean', len(clean), 'independent test images', len(indep))
for n, (a, b) in per.items():
    print(f'{n:16s} validation images {a:3d}  copies of training images {b:3d}')
json.dump(dict(clean=clean, over=over, indep=indep), open('subsets_v2.json', 'w'))
