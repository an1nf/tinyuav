"""Person datasets for the detector.

VisDrone2019-DET (Zhu et al., images from drones; annotation per box: x, y, w, h, score, category,
truncation, occlusion) → YOLO format with one class, *person* = categories 1 (pedestrian) and 2 (people).
Category 0 marks *ignored regions* (crowds too small to annotate); they are painted with the image mean so
that the detector is not trained to call real people "background".

The zips are on the ultralytics GitHub releases (reachable from the development workspace too):
  https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-{train,val,test-dev}.zip

COCO person (ground-level views, closer to a drone flying at 2 m behind someone) can be added with the
ultralytics `coco.yaml` download and `classes=[0]`; see docs/research/R19_seguimiento_personas.md.
"""
import glob
import os
import shutil
import zipfile

import cv2
import numpy as np

URL = 'https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-{split}.zip'
PERSON = {1, 2}


def download(dest, splits=('train', 'val')):
    import urllib.request
    os.makedirs(dest, exist_ok=True)
    for sp in splits:
        z = os.path.join(dest, f'VisDrone2019-DET-{sp}.zip')
        if not os.path.exists(z):
            print(f'downloading {URL.format(split=sp)} …', flush=True)
            urllib.request.urlretrieve(URL.format(split=sp), z + '.part')
            os.replace(z + '.part', z)
        if not os.path.isdir(os.path.join(dest, f'VisDrone2019-DET-{sp}')):
            with zipfile.ZipFile(z) as f:
                f.extractall(dest)


def convert_split(src, out, split, mask_ignored=True, min_h=4):
    """src = …/VisDrone2019-DET-<split> (images/, annotations/) → out/images/<split>, out/labels/<split>."""
    im_out, lb_out = os.path.join(out, 'images', split), os.path.join(out, 'labels', split)
    os.makedirs(im_out, exist_ok=True)
    os.makedirs(lb_out, exist_ok=True)
    n_img = n_box = 0
    for ann in sorted(glob.glob(os.path.join(src, 'annotations', '*.txt'))):
        name = os.path.splitext(os.path.basename(ann))[0]
        img_p = os.path.join(src, 'images', name + '.jpg')
        if not os.path.exists(img_p):
            continue
        rows = np.loadtxt(ann, delimiter=',', ndmin=2, usecols=range(8)) if os.path.getsize(ann) else np.zeros((0, 8))
        img = cv2.imread(img_p)
        H, W = img.shape[:2]
        lines = []
        for x, y, w, h, score, cat, _, _ in rows:
            if cat == 0 and mask_ignored:
                x0, y0 = int(max(x, 0)), int(max(y, 0))
                img[y0:int(y + h), x0:int(x + w)] = img.reshape(-1, 3).mean(0)
                continue
            if int(cat) in PERSON and score != 0 and h >= min_h:
                lines.append(f'0 {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}')
        if mask_ignored and (rows[:, 5] == 0).any():
            cv2.imwrite(os.path.join(im_out, name + '.jpg'), img)
        else:
            shutil.copy(img_p, os.path.join(im_out, name + '.jpg'))
        with open(os.path.join(lb_out, name + '.txt'), 'w') as f:
            f.write('\n'.join(lines))
        n_img += 1
        n_box += len(lines)
    return n_img, n_box


def write_yaml(out, splits):
    """Dataset file for ultralytics. With a single split (e.g. a quick test on `val`), it is used for both."""
    p = os.path.join(out, 'person.yaml')
    tr = 'train' if 'train' in splits else splits[0]
    va = 'val' if 'val' in splits else tr
    with open(p, 'w') as f:
        f.write(f'path: {os.path.abspath(out)}\ntrain: images/{tr}\nval: images/{va}\n')
        if 'test-dev' in splits:
            f.write('test: images/test-dev\n')
        f.write('names:\n  0: person\n')
    return p


def prepare(raw, out, splits=('train', 'val'), fetch=False):
    if fetch:
        download(raw, splits)
    stats = {}
    for sp in splits:
        src = os.path.join(raw, f'VisDrone2019-DET-{sp}')
        if not os.path.isdir(src):
            raise FileNotFoundError(f'{src} (download it or pass --fetch)')
        stats[sp] = convert_split(src, out, sp)
        print(f'[visdrone] {sp}: {stats[sp][0]} images, {stats[sp][1]} people', flush=True)
    return write_yaml(out, splits), stats


def recall_by_height(model, data_dir, split='val', conf=0.15, imgsz=640, bins=(0, 8, 12, 16, 24, 32, 48, 96, 1e9),
                     max_images=None, iou_thr=0.5):
    """Detector recall vs. person box height (pixels at the original resolution) and false positives per
    image: what `SimDetector` needs (its recall curve is a logistic in the box height)."""
    from nanouav.follow.detector import YoloDetector
    from nanouav.follow.tracker import iou
    det = model if hasattr(model, 'detect') else YoloDetector(model, conf=conf, imgsz=imgsz)
    ims = sorted(glob.glob(os.path.join(data_dir, 'images', split, '*.jpg')))[:max_images]
    hits = np.zeros(len(bins) - 1)
    tot = np.zeros(len(bins) - 1)
    fp = 0
    for p in ims:
        img = cv2.imread(p)[..., ::-1]
        H, W = img.shape[:2]
        lb = os.path.join(data_dir, 'labels', split, os.path.splitext(os.path.basename(p))[0] + '.txt')
        g = np.loadtxt(lb, ndmin=2) if os.path.getsize(lb) else np.zeros((0, 5))
        gt = np.c_[(g[:, 1] - g[:, 3] / 2) * W, (g[:, 2] - g[:, 4] / 2) * H, (g[:, 1] + g[:, 3] / 2) * W,
                   (g[:, 2] + g[:, 4] / 2) * H] if len(g) else np.zeros((0, 4))
        D = det.detect(img)
        pb = np.array([d['box'] for d in D]).reshape(-1, 4)
        matched = np.zeros(len(pb), bool)
        if len(gt) and len(pb):
            I = iou(gt, pb)
            for i in np.argsort(-I.max(1)):
                j = int(np.argmax(np.where(matched, -1, I[i])))
                h = gt[i, 3] - gt[i, 1]
                b = np.searchsorted(bins, h, side='right') - 1
                tot[b] += 1
                if I[i, j] >= iou_thr and not matched[j]:
                    matched[j] = True
                    hits[b] += 1
        elif len(gt):
            for h in gt[:, 3] - gt[:, 1]:
                tot[np.searchsorted(bins, h, side='right') - 1] += 1
        fp += int((~matched).sum())
    rows = [{'h_min': bins[k], 'h_max': bins[k + 1], 'n': int(tot[k]),
             'recall': (float(hits[k] / tot[k]) if tot[k] else None)} for k in range(len(tot))]
    return {'bins': rows, 'fp_per_image': fp / max(len(ims), 1), 'images': len(ims)}


def fit_logistic(rows):
    """Recall(h) ≈ p_max · σ((h − h50)/s), least squares on the height bins → (h50, s, p_max)."""
    pts = [((r['h_min'] + min(r['h_max'], 2 * max(r['h_min'], 8))) / 2, r['recall'], r['n']) for r in rows
           if r['recall'] is not None and r['n'] >= 5]
    if len(pts) < 3:
        return None
    h, y, w = map(np.array, zip(*pts))
    best = None
    for h50 in np.arange(2, 200, 1.0):
        for s in np.arange(1.0, 40, 1.0):
            sig = 1 / (1 + np.exp(-(h - h50) / s))
            pmax = float(np.clip((w * sig * y).sum() / max((w * sig * sig).sum(), 1e-9), 0, 1))
            e = float((w * (pmax * sig - y) ** 2).sum())
            if best is None or e < best[0]:
                best = (e, h50, s, pmax)
    return {'h50': best[1], 'h_scale': best[2], 'p_max': best[3]}
