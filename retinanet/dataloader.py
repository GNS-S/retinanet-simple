# Modified from original repo - added JSON Dataset, removed the others

from __future__ import print_function, division
import sys
import os
import torch
import numpy as np
import random
import csv
import json

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torch.utils.data.sampler import Sampler

import skimage
import skimage.io
import skimage.transform
import skimage.color
from future.utils import raise_from

from PIL import Image

class JSONDataset(Dataset):
    def __init__(self, train_file, class_file, img_path, transform=None):
        """
        Args:
            train_file (string): JSON file with the following format:
                {
                    [id]: {
                        'intid': int, #class ids are usually strings, this is the numeric representation used in trainign
                        'class_id': string,
                        'x1': float = x1min/img_width
                        'x2': float = xymin/img_height
                        'y1': float
                        'y2': float
                    }
                }
            class_file (string): CSV file with the following format:
                class_name,class_id
            img_path (string): path to where the images can be found, expect: {img_path}/{img_id}.jpg to yield images
        """
        self.train_file = train_file
        self.class_file = class_file
        self.transform = transform
        self.img_path = img_path

        # parse class file
        try:
            with open(self.class_file, 'r', newline='') as file:
                self.classes = self.load_classes(csv.reader(file, delimiter=','))
        except ValueError as e:
            raise ValueError(f'invalid CSV class file: {self.class_file}: {e}')

        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

        # parse json annotations
        try:
            with open(self.train_file) as f:
                self.image_data = json.load(f)
        except ValueError as e:
            raise ValueError(f'invalid JSON annotations file: {self.train_file}: {e}')
        self.image_ids = list(self.image_data.keys())

    def load_classes(self, csv_reader):
        result = {}

        for line, row in enumerate(csv_reader):
            line += 1

            try:
                class_name, class_id = row[:2]
            except ValueError:
                raise(ValueError('line {}: format should be \'class_name,class_id\''.format(line)))
            class_id = int(class_id)

            if class_name in result:
                raise ValueError(f'line {line}: duplicate class name: {class_name}')
            result[class_name] = class_id
        return result

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):

        img = self.load_image(idx)
        height, width = img.shape[:2]

        annot = self.load_annotations(idx, width, height)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_image(self, image_index):
        img = skimage.io.imread(f"{self.img_path}/{self.image_ids[image_index]}.jpg")

        if len(img.shape) == 2:
            img = skimage.color.gray2rgb(img)

        return img.astype(np.float32)/255.0

    def load_annotations(self, image_index, width=None, height=None):
        # get ground truth annotations
        annotation_list = self.image_data[self.image_ids[image_index]]
        annotations = np.zeros((0, 5))

        if width == None and height == None:
            img = self.load_image(image_index)
            height, width = img.shape[:2]

        # parse annotations
        for idx, a in enumerate(annotation_list):
            # some annotations have basically no width / height, skip them
            x1 = a['x1'] * width
            x2 = a['x2'] * width
            y1 = a['y1'] * height
            y2 = a['y2'] * height

            if (x2-x1) < 1 or (y2-y1) < 1:
                continue

            annotation = np.zeros((1, 5))
            
            annotation[0, 0] = int(round(x1))
            annotation[0, 1] = int(round(y1))
            annotation[0, 2] = int(round(x2))
            annotation[0, 3] = int(round(y2))

            annotation[0, 4] = int(a['intid'])
            annotations = np.append(annotations, annotation, axis=0)

        return annotations

    def name_to_label(self, name):
        return self.classes[name]

    def label_to_name(self, label):
        return self.labels[label]

    def get_classlist(self):
        return list(self.labels.keys())

    def has_label(self, label):
        return label in self.labels

    def num_classes(self):
        return max(self.classes.values()) + 1

    def image_aspect_ratio(self, image_index):
        image = Image.open(f"{self.img_path}/{self.image_ids[image_index]}.jpg")
        return float(image.width) / float(image.height)


def collater(data):

    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]
        
    widths = [int(s.shape[0]) for s in imgs]
    heights = [int(s.shape[1]) for s in imgs]
    batch_size = len(imgs)

    max_width = np.array(widths).max()
    max_height = np.array(heights).max()

    padded_imgs = torch.zeros(batch_size, max_width, max_height, 3)

    for i in range(batch_size):
        img = imgs[i]
        padded_imgs[i, :int(img.shape[0]), :int(img.shape[1]), :] = img

    max_num_annots = max(annot.shape[0] for annot in annots)
    
    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        if max_num_annots > 0:
            for idx, annot in enumerate(annots):
                #print(annot.shape)
                if annot.shape[0] > 0:
                    annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1


    padded_imgs = padded_imgs.permute(0, 3, 1, 2)

    return {'img': padded_imgs, 'annot': annot_padded, 'scale': scales}

class Resizer(object):
    """Resize image"""

    def __call__(self, sample, min_side=608, max_side=1024):
        image, annots = sample['img'], sample['annot']

        rows, cols, cns = image.shape

        smallest_side = min(rows, cols)

        # rescale the image so the smallest side is min_side
        scale = min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)

        if largest_side * scale > max_side:
            scale = max_side / largest_side

        # resize the image with the computed scale
        image = skimage.transform.resize(image, (int(round(rows*scale)), int(round((cols*scale)))))
        rows, cols, cns = image.shape

        pad_w = 32 - rows%32
        pad_h = 32 - cols%32

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)

        annots[:, :4] *= scale

        return {'img': torch.from_numpy(new_image), 'annot': torch.from_numpy(annots), 'scale': scale}


class Augmenter(object):
    def __call__(self, sample, flip_x=0.5):

        if np.random.rand() < flip_x:
            image, annots = sample['img'], sample['annot']
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()
            
            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp

            sample = {'img': image, 'annot': annots}

        return sample


class Normalizer(object):
    def __init__(self):
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])

    def __call__(self, sample):

        image, annots = sample['img'], sample['annot']

        return {'img':((image.astype(np.float32)-self.mean)/self.std), 'annot': annots}

class UnNormalizer(object):
    def __init__(self, mean=None, std=None):
        if mean == None:
            self.mean = [0.485, 0.456, 0.406]
        else:
            self.mean = mean
        if std == None:
            self.std = [0.229, 0.224, 0.225]
        else:
            self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class AspectRatioBasedSampler(Sampler):
    def __init__(self, data_source, batch_size, drop_last):
        self.data_source = data_source
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.groups = self.group_images()

    def __iter__(self):
        random.shuffle(self.groups)
        for group in self.groups:
            yield group

    def __len__(self):
        if self.drop_last:
            return len(self.data_source) // self.batch_size
        else:
            return (len(self.data_source) + self.batch_size - 1) // self.batch_size

    def group_images(self):
        # determine the order of the images
        order = list(range(len(self.data_source)))
        order.sort(key=lambda x: self.data_source.image_aspect_ratio(x))

        # divide into groups, one group = one batch
        return [[order[x % len(order)] for x in range(i, i + self.batch_size)] for i in range(0, len(order), self.batch_size)]
