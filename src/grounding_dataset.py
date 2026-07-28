# ------------------------------------------------------------------------
# BEAUTY DETR
# Copyright (c) 2022 Ayush Jain & Nikolaos Gkanatsios
# Licensed under CC-BY-NC [see LICENSE for details]
# All Rights Reserved
# ------------------------------------------------------------------------
"""Dataset and data loader for ReferIt3D."""

import csv
from collections import defaultdict
import h5py
import json
import multiprocessing as mp
import os
import random
from six.moves import cPickle

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import RobertaTokenizerFast
from data.model_util_scannet import ScannetDatasetConfig
from data.scannet_utils import read_label_mapping
from src.visual_data_handlers import Scan
from .scannet_classes import REL_ALIASES, VIEW_DEP_RELS

import pickle
import re

NUM_CLASSES = 485
DC = ScannetDatasetConfig(NUM_CLASSES)
DC18 = ScannetDatasetConfig(18)
MAX_NUM_OBJ = 132


class GroundingDataset(Dataset):
    """Dataset utilities for ReferIt3D."""

    def __init__(self, dataset_dict={'sr3d': 1},
                 test_dataset='sr3d',
                 split='val', overfit=False,
                 data_path='./',
                 inference_scans_dir='',
                 inference_axis_alignment='auto',
                 use_color=False, use_height=False, use_multiview=False,
                 detect_intermediate=False,
                 extra_dataset_cfg=dict(), text_encoder_name_or_path="roberta-base"):
        """Initialize dataset (here for ReferIt3D utterances)."""
        self.dataset_dict = dataset_dict
        self.test_dataset = test_dataset
        self.split = split
        self.use_color = use_color
        self.use_height = use_height
        self.overfit = overfit
        self.detect_intermediate = detect_intermediate
        self.use_multiview = use_multiview
        self.data_path = data_path
        self.inference_scans_dir = inference_scans_dir
        self.inference_axis_alignment = inference_axis_alignment
        self.inference_scans = {}

        self.mean_rgb = np.array([109.8, 97.2, 83.8]) / 256
        self.label_map = read_label_mapping(
            'data/meta_data/scannetv2-labels.combined.tsv',
            label_from='raw_category',
            label_to='id'
        )
        self.label_map18 = read_label_mapping(
            'data/meta_data/scannetv2-labels.combined.tsv',
            label_from='raw_category',
            label_to='nyu40id'
        )
        self.label_mapclass = read_label_mapping(
            'data/meta_data/scannetv2-labels.combined.tsv',
            label_from='raw_category',
            label_to='nyu40class'
        )
        self.multiview_path = os.path.join(
            f'{self.data_path}/scanrefer_2d_feats',
            "enet_feats_maxpool.hdf5"
        )
        self.multiview_data = {}
        while True:
            try:
                self.tokenizer = RobertaTokenizerFast.from_pretrained(text_encoder_name_or_path)
                break
            except:
                continue
        self.text_noun_span_file = extra_dataset_cfg.get('text_noun_span_file', '')

        self.scanrefer_fix_missing_target_text = extra_dataset_cfg.get('scanrefer_fix_missing_target_text', False)
        
        # load
        self.anno_filter_rule = extra_dataset_cfg.get('anno_filter_rule', '')
        print('Loading %s files, take a breath!' % split)
        if not os.path.exists(f'{self.data_path}/{split}_v3scans.pkl'):
            save_data(f'{data_path}/{split}_v3scans.pkl', split, data_path)
        self.scans = unpickle_data(f'{self.data_path}/{split}_v3scans.pkl')
        self.scans = list(self.scans)[0]
        if self.split != 'train':
            self.annos = self.load_annos(test_dataset)
        else:
            self.annos = []
            for dset, cnt in dataset_dict.items():
                if cnt > 0:
                    _annos = self.load_annos(dset)
                    self.annos += (_annos * cnt)

        self.text_endswith_not_mentioned = extra_dataset_cfg.get('text_endswith_not_mentioned', True)
        self.scan2text = dict()
        self.scan2tid = dict()
        annos_with_text = list(self.annos)
        for a in annos_with_text:
            scan_id = a['scan_id']
            if scan_id not in self.scan2text:
                self.scan2text[scan_id] = [(a['utterance'], a['target_id'])]
                self.scan2tid[scan_id] = [a['target_id']]
            else:
                self.scan2text[scan_id].append((a['utterance'], a['target_id']))
                self.scan2tid[scan_id].append(a['target_id'])
        self.scan_ids = list(self.scan2text.keys())

        self.get_rand_utterance = extra_dataset_cfg.get('get_rand_utterance', False)
        self.get_noun_span_texts = extra_dataset_cfg.get('get_noun_span_texts', False)
        self.noun_span_texts_join_by = extra_dataset_cfg.get('noun_span_texts_join_by', '||')
        self.extract_noun_w_rand_space = extra_dataset_cfg.get('extract_noun_w_rand_space', False)
        self.extract_noun_wo_space = extra_dataset_cfg.get('extract_noun_wo_space', False)
        self.sample_noun_texts = extra_dataset_cfg.get('sample_noun_texts', False)

    def load_annos(self, dset):
        """Load annotations of given dataset."""
        loaders = {
            'nr3d': self.load_nr3d_annos,
            'sr3d': self.load_sr3d_annos,
            'sr3d+': self.load_sr3dplus_annos,
            'scanrefer': self.load_scanrefer_annos,
        }
        annos = loaders[dset]()
        if self.overfit:
            annos = annos[:128]
        return annos

    def load_sr3dplus_annos(self):
        """Load annotations of sr3d/sr3d+."""
        return self.load_sr3d_annos(dset='sr3d+')

    def load_sr3d_annos(self, dset='sr3d'):
        """Load annotations of sr3d/sr3d+."""
        split = self.split
        if split == 'val':
            split = 'test'
        with open('data/meta_data/sr3d_%s_scans.txt' % split) as f:
            scan_ids = set(eval(f.read()))
        with open(os.path.join(self.data_path, 'refer_it_3d/%s.csv' % dset)) as f:
            csv_reader = csv.reader(f)
            headers = next(csv_reader)
            headers = {header: h for h, header in enumerate(headers)}
            annos = [
                {
                    'scan_id': line[headers['scan_id']],
                    'target_id': int(line[headers['target_id']]),
                    'distractor_ids': eval(line[headers['distractor_ids']]),
                    'utterance': line[headers['utterance']],
                    'target': line[headers['instance_type']],
                    'anchors': eval(line[headers['anchors_types']]),
                    'anchor_ids': eval(line[headers['anchor_ids']]),
                    'dataset': dset
                }
                for line in csv_reader
                if line[headers['scan_id']] in scan_ids
                and
                str(line[headers['mentions_target_class']]).lower() == 'true'
            ]
        text_noun_span_file = self.text_noun_span_file
        if text_noun_span_file:
            with open(os.path.join('data/new_meta_data/', text_noun_span_file) % split, 'rb') as f:
                text_span_info = pickle.load(f)
            assert len(text_span_info) == len(annos)
            for anno, text_span in zip(annos, text_span_info):
                anno.update(text_span)
        return annos

    def load_nr3d_annos(self):
        """Load annotations of nr3d."""
        split = self.split
        if split == 'val':
            split = 'test'
        with open('data/meta_data/nr3d_%s_scans.txt' % split) as f:
            scan_ids = set(eval(f.read()))

        if self.anno_filter_rule == 'referit3d':
            from ast import literal_eval
            with open(self.data_path + 'refer_it_3d/nr3d.csv') as f:
                csv_reader = csv.reader(f)
                headers = next(csv_reader)
                headers = {header: h for h, header in enumerate(headers)}
                annos = [
                    {
                        'scan_id': line[headers['scan_id']],
                        'target_id': int(line[headers['target_id']]),
                        'target': line[headers['instance_type']],
                        'utterance': line[headers['utterance']],
                        'anchor_ids': [],
                        'anchors': [],
                        'dataset': 'nr3d'
                    }
                    for line in csv_reader
                    if line[headers['scan_id']] in scan_ids
                    and str(line[headers['mentions_target_class']]).lower() == 'true'
                    and len(literal_eval(line[headers['tokens']])) <= 24
                ]
            # Add distractor info
            for anno in annos:
                anno['distractor_ids'] = [
                    ind
                    for ind in
                    range(len(self.scans[anno['scan_id']].three_d_objects))
                    if self.scans[anno['scan_id']].get_object_instance_label(ind)
                    == anno['target']
                    and ind != anno['target_id']
                ]

        else:
            with open(self.data_path + 'refer_it_3d/nr3d.csv') as f:
                csv_reader = csv.reader(f)
                headers = next(csv_reader)
                headers = {header: h for h, header in enumerate(headers)}
                annos = [
                    {
                        'scan_id': line[headers['scan_id']],
                        'target_id': int(line[headers['target_id']]),
                        'target': line[headers['instance_type']],
                        'utterance': line[headers['utterance']],
                        'anchor_ids': [],
                        'anchors': [],
                        'dataset': 'nr3d'
                    }
                    for line in csv_reader
                    if line[headers['scan_id']] in scan_ids
                    and
                    str(line[headers['mentions_target_class']]).lower() == 'true'
                    and
                    (
                        str(line[headers['correct_guess']]).lower() == 'true'
                        or split != 'test'
                    )
                ]
            # Add distractor info
            for anno in annos:
                anno['distractor_ids'] = [
                    ind
                    for ind in
                    range(len(self.scans[anno['scan_id']].three_d_objects))
                    if self.scans[anno['scan_id']].get_object_instance_label(ind)
                    == anno['target']
                    and ind != anno['target_id']
                ]
            # Filter out sentences that do not explicitly mention the target class
            annos = [anno for anno in annos if anno['target'] in anno['utterance']]

        text_noun_span_file = self.text_noun_span_file
        if not text_noun_span_file:
            text_noun_span_file = 'nr3d_%s_text_span.info'
        with open(os.path.join('data/new_meta_data/', text_noun_span_file) % split, 'rb') as f:
            text_span_info = pickle.load(f)
        assert len(text_span_info) == len(annos)
        for anno, text_span in zip(annos, text_span_info):
            anno.update(text_span)
        return annos

    def load_scanrefer_annos(self):
        """Load annotations of ScanRefer."""
        _path = self.data_path + 'scanrefer/ScanRefer_filtered'
        split = self.split
        if split in ('val', 'test'):
            split = 'val'
        with open(_path + '_%s.txt' % split) as f:
            scan_ids = [line.rstrip().strip('\n') for line in f.readlines()]
        with open(_path + '_%s.json' % split) as f:
            reader = json.load(f)
        annos = [
            {
                'scan_id': anno['scene_id'],
                'target_id': int(anno['object_id']),
                'distractor_ids': [],
                'utterance': ' '.join(anno['token']),
                'target': ' '.join(str(anno['object_name']).split('_')),
                'anchors': [],
                'anchor_ids': [],
                'dataset': 'scanrefer'
            }
            for anno in reader
            if anno['scene_id'] in scan_ids
        ]
        if self.scanrefer_fix_missing_target_text:
            # Fix missing target reference
            for anno in annos:
                if anno['target'] not in anno['utterance']:
                    anno['utterance'] = (
                        ' '.join(anno['utterance'].split(' . ')[0].split()[:-1])
                        + ' ' + anno['target'] + ' . '
                        + ' . '.join(anno['utterance'].split(' . ')[1:])
                    )
        # Add distractor info
        scene2obj = defaultdict(list)
        sceneobj2used = defaultdict(list)
        for anno in annos:
            nyu_labels = [
                self.label_mapclass[
                    self.scans[anno['scan_id']].get_object_instance_label(ind)
                ]
                for ind in
                range(len(self.scans[anno['scan_id']].three_d_objects))
            ]
            labels = [DC18.type2class.get(lbl, 17) for lbl in nyu_labels]
            anno['distractor_ids'] = [
                ind
                for ind in
                range(len(self.scans[anno['scan_id']].three_d_objects))
                if labels[ind] == labels[anno['target_id']]
                and ind != anno['target_id']
            ][:32]
            if anno['target_id'] not in sceneobj2used[anno['scan_id']]:
                sceneobj2used[anno['scan_id']].append(anno['target_id'])
                scene2obj[anno['scan_id']].append(labels[anno['target_id']])
        # Add unique-multi
        for anno in annos:
            nyu_labels = [
                self.label_mapclass[
                    self.scans[anno['scan_id']].get_object_instance_label(ind)
                ]
                for ind in
                range(len(self.scans[anno['scan_id']].three_d_objects))
            ]
            labels = [DC18.type2class.get(lbl, 17) for lbl in nyu_labels]
            anno['unique'] = (
                np.array(scene2obj[anno['scan_id']])
                == labels[anno['target_id']]
            ).sum() == 1

        text_noun_span_file = self.text_noun_span_file
        if not text_noun_span_file:
            raise
        with open(os.path.join('data/new_meta_data/', text_noun_span_file) % split, 'rb') as f:
            text_span_info = pickle.load(f)
        assert len(text_span_info) == len(annos)
        for anno, text_span in zip(annos, text_span_info):
            anno.update(text_span)
        return annos

    def load_scannet_annos(self):
        """Load annotations of scannet."""
        split = 'train' if self.split == 'train' else 'val'
        with open('data/meta_data/scannetv2_%s.txt' % split) as f:
            scan_ids = [line.rstrip() for line in f]
        annos = []
        for scan_id in scan_ids:
            scan = self.scans[scan_id]
            # Ignore scans that have no object in our vocabulary
            keep = np.array([
                self.label_map[
                    scan.get_object_instance_label(ind)
                ] in DC.nyu40id2class
                for ind in range(len(scan.three_d_objects))
            ])
            if keep.any():
                # this will get populated randomly each time
                annos.append({
                    'scan_id': scan_id,
                    'target_id': [],
                    'distractor_ids': [],
                    'utterance': '',
                    'target': [],
                    'anchors': [],
                    'anchor_ids': [],
                    'dataset': 'scannet'
                })
        if self.split == 'train':
            annos = [
                anno for a, anno in enumerate(annos)
                if a not in {965, 977}
            ]
        return annos

    def _sample_classes(self, scan_id):
        """Sample classes for the scannet detection sentences."""
        scan = self.scans[scan_id]
        sampled_classes = set([
            self.label_map[scan.get_object_instance_label(ind)]
            for ind in range(len(scan.three_d_objects))
        ])
        sampled_classes = list(sampled_classes & set(DC.nyu40id2class))
        # sample 10 classes
        if self.split == 'train' and self.random_utt:  # random utterance
            if len(sampled_classes) > 10:
                sampled_classes = random.sample(sampled_classes, 10)
            ret = [DC.class2type[DC.nyu40id2class[i]] for i in sampled_classes]
            random.shuffle(ret)
        else:  # fixed detection sentence
            ret = [
                'cabinet', 'bed', 'chair', 'couch', 'table', 'door',
                'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain',
                'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub',
                'other furniture'
            ]
        return ret

    def _create_scannet_utterance(self, sampled_classes):
        if self.split == 'train' and self.random_utt:
            neg_names = []
            while len(neg_names) < 10:
                _ind = np.random.randint(0, len(DC.class2type))
                if DC.class2type[_ind] not in neg_names + sampled_classes:
                    neg_names.append(DC.class2type[_ind])
            mixed_names = sorted(list(set(sampled_classes + neg_names)))
            random.shuffle(mixed_names)
        else:
            mixed_names = sampled_classes
        if self.scannet_utterance_join_by:
            return self.scannet_utterance_join_by.join(mixed_names)
        utterance = ' . '.join(mixed_names)
        return utterance

    def _load_multiview(self, scan_id):
        """Load multi-view data of given scan-id."""
        pid = mp.current_process().pid
        if pid not in self.multiview_data:
            self.multiview_data[pid] = h5py.File(
                self.multiview_path, "r", libver="latest"
            )
        return self.multiview_data[pid][scan_id]

    def _augment(self, pc, color, rotate, anno=None):
        augmentations = {}

        # Rotate/flip only if we don't have a view_dep sentence
        if rotate:
            theta_z = 90*np.random.randint(0, 4) + (2*np.random.rand() - 1) * 5
            if self.select_flip:
                _utterance = anno['utterance'].lower()
                has_left_right = ('left' in _utterance) or ('right' in _utterance)
                if not has_left_right:
                    # Flipping along the YZ plane
                    augmentations['yz_flip'] = np.random.random() > 0.5
                    if augmentations['yz_flip']:
                        pc[:, 0] = -pc[:, 0]
                    # Flipping along the XZ plane
                    augmentations['xz_flip'] = np.random.random() > 0.5
                    if augmentations['xz_flip']:
                        pc[:, 1] = -pc[:, 1]
            else:
                # Flipping along the YZ plane
                augmentations['yz_flip'] = np.random.random() > 0.5
                if augmentations['yz_flip']:
                    pc[:, 0] = -pc[:, 0]
                # Flipping along the XZ plane
                augmentations['xz_flip'] = np.random.random() > 0.5
                if augmentations['xz_flip']:
                    pc[:, 1] = -pc[:, 1]

            if self.select_rotate:
                _utterance = anno['utterance'].lower()
                has_left_right = ('left' in _utterance) or ('right' in _utterance)
                if has_left_right:
                    theta_z = (2*np.random.rand() - 1) * 5
        else:
            theta_z = (2*np.random.rand() - 1) * 5
        augmentations['theta_z'] = theta_z
        pc[:, :3] = rot_z(pc[:, :3], theta_z)
        # Rotate around x
        theta_x = (2*np.random.rand() - 1) * 2.5
        augmentations['theta_x'] = theta_x
        pc[:, :3] = rot_x(pc[:, :3], theta_x)
        # Rotate around y
        theta_y = (2*np.random.rand() - 1) * 2.5
        augmentations['theta_y'] = theta_y
        pc[:, :3] = rot_y(pc[:, :3], theta_y)

        # Add noise
        noise = np.random.rand(len(pc), 3) * 5e-3
        augmentations['noise'] = noise
        pc[:, :3] = pc[:, :3] + noise

        # Translate/shift
        augmentations['shift'] = np.random.random((3,))[None, :] - 0.5
        pc[:, :3] += augmentations['shift']

        # Scale
        augmentations['scale'] = 0.98 + 0.04*np.random.random()
        pc[:, :3] *= augmentations['scale']

        # Color
        if color is not None:
            color += self.mean_rgb
            color *= 0.98 + 0.04*np.random.random((len(color), 3))
            color -= self.mean_rgb
        return pc, color, augmentations

    def _get_pc(self, anno, scan):
        """Return a point cloud representation of current scene."""
        scan_id = anno['scan_id']
        if self.inference_scans_dir:
            if scan_id not in self.inference_scans:
                self.inference_scans[scan_id] = Scan(
                    scan_id,
                    self.inference_scans_dir,
                    load_objects=False,
                    align_to_axes=self.inference_axis_alignment != 'none',
                )
            scan = self.inference_scans[scan_id]
        rel_name = "none"
        if anno['dataset'].startswith('sr3d'):
            rel_name = self._find_rel(anno['utterance'])

        # a. Color
        color = None
        if self.use_color:
            color = scan.color - self.mean_rgb

        # b .Height
        height = None
        if self.use_height:
            floor_height = np.percentile(scan.pc[:, 2], 0.99)
            height = np.expand_dims(scan.pc[:, 2] - floor_height, 1)

        # c. Multi-view 2d features
        multiview_data = None
        if self.use_multiview:
            multiview_data = self._load_multiview(scan_id)

        # e. Concatenate representations
        point_cloud = scan.pc
        if color is not None:
            point_cloud = np.concatenate((point_cloud, color), 1)
        if height is not None:
            point_cloud = np.concatenate([point_cloud, height], 1)
        if multiview_data is not None:
            point_cloud = np.concatenate([point_cloud, multiview_data], 1)

        return point_cloud

    # 得到 target box, anchor box 对应句子tokens 的 text_span
    def _get_token_positive_map(self, anno):
        """Return correspondence of boxes to tokens."""
        # Token start-end span in characters
        caption = ' '.join(anno['utterance'].replace(',', ' ,').split())
        caption = ' ' + caption + ' '
        tokens_positive = np.zeros((MAX_NUM_OBJ, 2))
        if isinstance(anno['target'], list):
            cat_names = anno['target']
        else:
            cat_names = [anno['target']]
        if self.detect_intermediate:
            cat_names += anno['anchors']
        for c, cat_name in enumerate(cat_names):
            start_span = caption.find(' ' + cat_name + ' ')
            len_ = len(cat_name)
            if start_span < 0:
                start_span = caption.find(' ' + cat_name)
                len_ = len(caption[start_span+1:].split()[0])
            if start_span < 0:
                start_span = caption.find(cat_name)
                orig_start_span = start_span
                while caption[start_span - 1] != ' ':
                    start_span -= 1
                len_ = len(cat_name) + orig_start_span - start_span
                while caption[len_ + start_span] != ' ':
                    len_ += 1
            end_span = start_span + len_
            assert start_span > -1, caption
            assert end_span > 0, caption
            tokens_positive[c][0] = start_span
            tokens_positive[c][1] = end_span

        # Positive map (for soft token prediction)
        tokenized = self.tokenizer.batch_encode_plus(
            [' '.join(anno['utterance'].replace(',', ' ,').split())],
            padding="longest", return_tensors="pt"
        )
        positive_map = np.zeros((MAX_NUM_OBJ, 256))
        gt_map = get_positive_map(tokenized, tokens_positive[:len(cat_names)])
        positive_map[:len(cat_names)] = gt_map
        return tokens_positive, positive_map

    def _get_target_boxes(self, anno, scan):
        """Return gt boxes to detect."""
        bboxes = np.zeros((MAX_NUM_OBJ, 6))
        if isinstance(anno['target_id'], list):  # scannet
            tids = anno['target_id']
        else:  # referit dataset
            tids = [anno['target_id']]
            if self.detect_intermediate:
                tids += anno.get('anchor_ids', [])
        point_instance_label = -np.ones(len(scan.pc))
        for t, tid in enumerate(tids):
            point_instance_label[scan.three_d_objects[tid]['points']] = t

        bboxes[:len(tids)] = np.stack([
            scan.get_object_bbox(tid).reshape(-1) for tid in tids
        ])
        bboxes = np.concatenate((
            (bboxes[:, :3] + bboxes[:, 3:]) * 0.5,
            bboxes[:, 3:] - bboxes[:, :3]
        ), 1)
        bboxes[len(tids):, :3] = 1000
        box_label_mask = np.zeros(MAX_NUM_OBJ)
        box_label_mask[:len(tids)] = 1
        return bboxes, box_label_mask, point_instance_label

    def __getitem__(self, index):
        """Get current batch for input index."""
        # Read annotation
        anno = self.annos[index]
        scan_id = anno['scan_id']
        scan = self.scans[scan_id]
        scan.pc = np.copy(scan.orig_pc)

        # Point cloud representation
        point_cloud = self._get_pc(anno, scan)

        # "Target" boxes: append anchors if they're to be detected
        gt_bboxes, box_label_mask, point_instance_label = \
            self._get_target_boxes(anno, scan) # [max_#obj, 6], [max_#obj,], [#point,]

        # Positive map for soft-token and contrastive losses
        # tokens_positive, positive_map = self._get_token_positive_map(anno) # [max_#object, 2], [max_#object, 256]

        # Return
        _labels = np.zeros(MAX_NUM_OBJ)
        if not isinstance(anno['target_id'], int):
            _labels[:len(anno['target_id'])] = np.array([
                DC18.nyu40id2class[self.label_map18[
                    scan.get_object_instance_label(ind)
                ]]
                for ind in anno['target_id']
            ])
        ret_dict = {
            'box_label_mask': box_label_mask.astype(np.float32), # target/anchor box's mask，[max_#obj,]
            'center_label': gt_bboxes[:, :3].astype(np.float32), # target box's cxcycz
            'sem_cls_label': _labels.astype(np.int64),
            'size_gts': gt_bboxes[:, 3:].astype(np.float32), # target box's whd
        }
        if self.text_endswith_not_mentioned:
            # Append "not mentioned" to the text, as the text span of other objects
            utterance = ' '.join(anno['utterance'].replace(',', ' ,').split())\
                         + ' . not mentioned'
        else:
            utterance = ' '.join(anno['utterance'].replace(',', ' ,').split())

        if self.get_rand_utterance:
            cnt = 0
            while True:
                cnt += 1
                _rand_scan_id = random.choice(self.scan_ids)
                if _rand_scan_id != scan_id:
                    break
                if cnt >= 100:
                    print('dead loop', self.scan_ids)
            rand_utterance, _ = random.choice(self.scan2text[_rand_scan_id])
            rand_utterance = ' '.join(rand_utterance.replace(',', ' ,').split())
            ret_dict['rand_utterances'] = (rand_utterance)

        if self.get_noun_span_texts:
            # include the Space before the noun phrase if it exists
            span_texts = [utterance[s-1:e+1] if s>0 and utterance[s-1]==' ' else utterance[s:e+1] for s, e in anno['span_indices']]
            span_texts = self.noun_span_texts_join_by.join(span_texts)
            ret_dict['span_texts'] = span_texts

        if self.sample_noun_texts:
            if self.extract_noun_w_rand_space: # randomly keep the Space before the noun phrase if it exists
                _flag = np.random.random() > 0.5
                if _flag:
                    noun_texts = [utterance[s-1:e+1] if s>0 and utterance[s-1]==' ' else utterance[s:e+1] for s, e in anno['span_indices']]
                else:
                    noun_texts = [utterance[s:e+1] for s, e in anno['span_indices']]
            elif self.extract_noun_wo_space: # exclude the space 
                noun_texts = [utterance[s:e+1] for s, e in anno['span_indices']]
            else:
                # include the Space before the noun phrase if it exists
                noun_texts = [utterance[s-1:e+1] if s>0 and utterance[s-1]==' ' else utterance[s:e+1] for s, e in anno['span_indices']]
            noun_tgt_indices = anno['tgt_indices']
            noun_texts = self.noun_span_texts_join_by.join(noun_texts)
            ret_dict.update({
                'noun_texts': noun_texts,
                'noun_indices': anno['span_indices'],
                'noun_tgt_indices': [noun_tgt_indices + [-1] * (168 - len(noun_tgt_indices))],
            })

            
        if 'tgt_indices' in anno:
            tgt_indices = anno['tgt_indices']
            anno['tgt_indices'] = [tgt_indices + [-1] * (168 - len(tgt_indices))]
        ret_dict.update({
            "scan_ids": anno['scan_id'],
            "point_clouds": point_cloud.astype(np.float32),
            "utterances": (utterance),
            "is_view_dep": self._is_view_dep(anno['utterance']),
            "is_hard": len(anno['distractor_ids']) > 1,
            "is_unique": len(anno['distractor_ids']) == 0,
            'span_indices': anno.get('span_indices', [[-1, -1],]),
            'expand_indices': anno.get('expand_indices', [[-1, -1],]),
            'tgt_indices': anno.get('tgt_indices', [[-1] * 168]),
            'is_vgdata': np.array(True),
        })
        return ret_dict

    @staticmethod
    def _is_view_dep(utterance):
        """Check whether to augment based on nr3d utterance."""
        rels = [
            'front', 'behind', 'back', 'left', 'right', 'facing',
            'leftmost', 'rightmost', 'looking', 'across'
        ]
        words = set(utterance.split())
        return any(rel in words for rel in rels)

    @staticmethod
    def _find_rel(utterance):
        utterance = ' ' + utterance.replace(',', ' ,') + ' '
        relation = "none"
        sorted_rel_list = sorted(REL_ALIASES, key=len, reverse=True)
        for rel in sorted_rel_list:
            if ' ' + rel + ' ' in utterance:
                relation = REL_ALIASES[rel]
                break
        return relation

    @staticmethod
    def _augment_nr3d(utterance):
        """Check whether to augment based on nr3d utterance."""
        rels = [
            'front', 'behind', 'back', 'left', 'right', 'facing',
            'leftmost', 'rightmost', 'looking', 'across'
        ]
        augment = True
        for rel in rels:
            if ' ' + rel + ' ' in (utterance + ' '):
                augment = False
        return augment

    def _visualize_scene(self, anno, point_cloud, og_color, all_bboxes):
        del anno, point_cloud, og_color, all_bboxes
        return None

    def __len__(self):
        """Return number of utterances."""
        return len(self.annos)


def get_positive_map(tokenized, tokens_positive):
    """Construct a map of box-token associations."""
    positive_map = torch.zeros((len(tokens_positive), 256), dtype=torch.float)
    for j, tok_list in enumerate(tokens_positive):
        (beg, end) = tok_list
        beg = int(beg)
        end = int(end)
        beg_pos = tokenized.char_to_token(beg)
        end_pos = tokenized.char_to_token(end - 1)
        if beg_pos is None:
            try:
                beg_pos = tokenized.char_to_token(beg + 1)
                if beg_pos is None:
                    beg_pos = tokenized.char_to_token(beg + 2)
            except:
                beg_pos = None
        if end_pos is None:
            try:
                end_pos = tokenized.char_to_token(end - 2)
                if end_pos is None:
                    end_pos = tokenized.char_to_token(end - 3)
            except:
                end_pos = None
        if beg_pos is None or end_pos is None:
            continue
        positive_map[j, beg_pos:end_pos + 1].fill_(1)

    positive_map = positive_map / (positive_map.sum(-1)[:, None] + 1e-12)
    return positive_map.numpy()


def get_span_map(tokenized, tokens_positive):
    """Construct a map of box-token associations."""
    span_map = torch.zeros((len(tokens_positive), 256), dtype=torch.float)
    for j, tok_list in enumerate(tokens_positive):
        (beg, end) = tok_list
        beg = int(beg)
        end = int(end)
        beg_pos = tokenized.char_to_token(beg)
        end_pos = tokenized.char_to_token(end)
        if beg_pos is None:
            try:
                beg_pos = tokenized.char_to_token(beg + 1)
                if beg_pos is None:
                    beg_pos = tokenized.char_to_token(beg + 2)
            except:
                beg_pos = None
        if end_pos is None:
            try:
                end_pos = tokenized.char_to_token(end - 1)
                if end_pos is None:
                    end_pos = tokenized.char_to_token(end - 2)
            except:
                end_pos = None
        if beg_pos is None or end_pos is None:
            continue
        span_map[j, beg_pos:end_pos + 1].fill_(1)

    span_map = span_map / (span_map.sum(-1)[:, None] + 1e-12)
    return span_map.numpy()


def rot_x(pc, theta):
    """Rotate along x-axis."""
    theta = theta * np.pi / 180
    return np.matmul(
        np.array([
            [1.0, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)]
        ]),
        pc.T
    ).T


def rot_y(pc, theta):
    """Rotate along y-axis."""
    theta = theta * np.pi / 180
    return np.matmul(
        np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1.0, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ]),
        pc.T
    ).T


def rot_z(pc, theta):
    """Rotate along z-axis."""
    theta = theta * np.pi / 180
    return np.matmul(
        np.array([
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1.0]
        ]),
        pc.T
    ).T


def box2points(box):
    """Convert box center/hwd coordinates to vertices (8x3)."""
    x_min, y_min, z_min = (box[:, :3] - (box[:, 3:] / 2)).transpose(1, 0)
    x_max, y_max, z_max = (box[:, :3] + (box[:, 3:] / 2)).transpose(1, 0)
    return np.stack((
        np.concatenate((x_min[:, None], y_min[:, None], z_min[:, None]), 1),
        np.concatenate((x_min[:, None], y_max[:, None], z_min[:, None]), 1),
        np.concatenate((x_max[:, None], y_min[:, None], z_min[:, None]), 1),
        np.concatenate((x_max[:, None], y_max[:, None], z_min[:, None]), 1),
        np.concatenate((x_min[:, None], y_min[:, None], z_max[:, None]), 1),
        np.concatenate((x_min[:, None], y_max[:, None], z_max[:, None]), 1),
        np.concatenate((x_max[:, None], y_min[:, None], z_max[:, None]), 1),
        np.concatenate((x_max[:, None], y_max[:, None], z_max[:, None]), 1)
    ), axis=1)


def points2box(box):
    """Convert vertices (Nx8x3) to box center/hwd coordinates (Nx6)."""
    return np.concatenate((
        (box.min(1) + box.max(1)) / 2,
        box.max(1) - box.min(1)
    ), axis=1)


def scannet_loader(iter_obj):
    """Load the scans in memory, helper function."""
    scan_id, scan_path = iter_obj
    print(scan_id)
    return Scan(scan_id, scan_path, True)


def save_data(filename, split, data_path):
    """Save all scans to pickle."""
    import multiprocessing as mp

    # Read all scan files
    # scan_path = data_path + 'scans/'
    scan_path = os.path.join(data_path, 'scans/')
    with open('data/meta_data/scannetv2_%s.txt' % split) as f:
        scan_ids = [line.rstrip() for line in f]
    print('{} scans found.'.format(len(scan_ids)))

    # Load data
    n_items = len(scan_ids)
    n_processes = 8  # min(mp.cpu_count(), n_items)
    pool = mp.Pool(n_processes)
    chunks = int(n_items / n_processes)
    all_scans = dict()
    iter_obj = [
        (scan_id, scan_path)
        for scan_id in scan_ids
    ]
    for i, data in enumerate(
            pool.imap(scannet_loader, iter_obj, chunksize=chunks)
    ):
        all_scans[scan_ids[i]] = data
    pool.close()
    pool.join()

    # Save data
    print('pickle time')
    pickle_data(filename, all_scans)


def pickle_data(file_name, *args):
    """Use (c)Pickle to save multiple objects in a single file."""
    out_file = open(file_name, 'wb')
    cPickle.dump(len(args), out_file, protocol=2)
    for item in args:
        cPickle.dump(item, out_file, protocol=2)
    out_file.close()


def unpickle_data(file_name, python2_to_3=False):
    """Restore data previously saved with pickle_data()."""
    in_file = open(file_name, 'rb')
    if python2_to_3:
        size = cPickle.load(in_file, encoding='latin1')
    else:
        size = cPickle.load(in_file)

    for _ in range(size):
        if python2_to_3:
            yield cPickle.load(in_file, encoding='latin1')
        else:
            yield cPickle.load(in_file)
    in_file.close()
