"""Find training/validation overlap in a YOLO dataset built from Roboflow exports.

Roboflow names augmented copies of a source image '<original>_<ext>.rf.<hash>.<ext>'.
Two files with the same part before '.rf.' therefore show the same photograph.
The script reports, per class prefix, how many validation images have such a copy
in the training set, and lists exact (MD5) duplicates across the two splits.
It also writes the file lists of the 'clean' and 'overlap' validation subsets.

usage: python overlap_check.py <dataset_root>   (root contains train/images and valid/images)
"""
import collections, hashlib, os, sys

def stem(name):
    return os.path.splitext(name)[0].split('.rf.')[0]

def class_prefix(name):
    if name.startswith('maksssksksss'):
        return 'mask'
    if name.startswith('hard_hat'):
        return 'helmet'
    return name.split('_')[0]

def md5(path):
    with open(path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

root = sys.argv[1]
train = os.listdir(os.path.join(root, 'train', 'images'))
valid = sorted(os.listdir(os.path.join(root, 'valid', 'images')))
train_stems = {stem(f) for f in train}

hashes = collections.defaultdict(list)
for split, files in (('train', train), ('valid', valid)):
    for f in files:
        hashes[md5(os.path.join(root, split, 'images', f))].append((split, f))
exact = [v for v in hashes.values() if len({s for s, _ in v}) > 1]
exact_valid = {f for group in exact for s, f in group if s == 'valid'}

overlap = [f for f in valid if stem(f) in train_stems or f in exact_valid]
clean = [f for f in valid if f not in set(overlap)]

total = collections.Counter(class_prefix(f) for f in valid)
over = collections.Counter(class_prefix(f) for f in overlap)
print(f'validation images: {len(valid)}, with a copy in training: {len(overlap)}, clean: {len(clean)}')
for c in sorted(total):
    print(f'  {c:10s} {over[c]:4d} / {total[c]:4d}')
print('exact duplicates across splits:', exact)

for name, files in (('clean', clean), ('overlap', overlap)):
    with open(f'valid_{name}.txt', 'w') as fh:
        fh.write('\n'.join(os.path.abspath(os.path.join(root, 'valid', 'images', f)) for f in files) + '\n')
