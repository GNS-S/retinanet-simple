import os
import tqdm
import csv
import numpy as np
import imagesize
import json

SPLIT = 'train' # 'train'

BBOXFILE = ''
if SPLIT == 'train':
  BBOXFILE = 'oidv6-train-annotations-bbox.csv'
elif SPLIT == 'test':
  BBOXFILE = 'test-annotations-bbox.csv'

LIMIT = 1000
classes_l = ['Bee', 'Fruit', 'Seafood']
classes_to_intid = {
  cl: i+1 for i, cl in enumerate(classes_l)
}
classes = {}

progress_bar = tqdm.tqdm(total=len(classes_to_intid)*LIMIT, desc='Making annotations...', leave=True)

with open('./annotations_meta/class-descriptions-boxable.csv') as f:
  rows = csv.reader(f, delimiter=',')
  for i, row in enumerate(rows):
    cid, label = row[0], row[1]
    if (label in classes_to_intid):
      classes[cid] = {
        'label': label,
        'intid': classes_to_intid[label]
      }

counts = {
  cl:0 for cl in classes
}
annotations = {}
dims = {}
with open(f'./annotations_meta/{BBOXFILE}') as f:
  rows = csv.reader(f, delimiter=',')
  for row in rows:
    imgid, ann_type, imgclass = row[:3]
    if (imgclass in classes):
      if (imgid not in annotations):
        if counts[imgclass] >= LIMIT:
          if all([counts[cl] >= LIMIT for cl in counts]):
            break
          else:
            continue
        else:
          counts[imgclass] += 1
          annotations[imgid] = []
          progress_bar.update(1)

      annotations[imgid].append({
        'intid': classes[row[2]]['intid'],
        'class_id': row[2],
        'x1': float(row[4]),
        'x2': float(row[5]),
        'y1': float(row[6]),
        'y2': float(row[7])
      })

progress_bar.close()

with open(f"imgids{'-test' if SPLIT == 'test' else ''}.txt", "w") as f:
  for imgid in annotations:
    f.write(f'{SPLIT}/{imgid}\n')

with open('./annotations/classes-nz.csv', mode='w',  newline='') as f:
    writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for cl in classes:
      writer.writerow([classes[cl]['label'],classes[cl]['intid'],cl])

with open(f"annotations/annotations-nz{'-test' if SPLIT == 'test' else ''}.json", "w") as outfile: 
  json.dump(annotations, outfile)

print({classes[cl]['label']:counts[cl] for cl in counts})