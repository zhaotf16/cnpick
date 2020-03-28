from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pycocotools.coco as coco
from pycocotools.cocoeval import COCOeval
import numpy as np
import json
import os
import matplotlib.pyplot as plt

import torch.utils.data as data
class TRPV1_1024(data.Dataset):
  num_classes = 1
  default_resolution = [1024, 1024]
  mean = np.array([0.505974, 0.505974, 0.505974],
                   dtype=np.float32).reshape(1, 1, 3)
  std  = np.array([0.288657, 0.288657, 0.288657],
                   dtype=np.float32).reshape(1, 1, 3)

  def __init__(self, opt, split):
    super(TRPV1_1024, self).__init__()
    self.data_dir = os.path.join(opt.data_dir, 'TrpV1_1024')
    self.img_dir = os.path.join(self.data_dir, 'images')
    if split == 'val':
      self.annot_path = os.path.join(
          self.data_dir, 'annotations', 
          'val.json')
    else:
      if opt.task == 'exdet':
        self.annot_path = os.path.join(
          self.data_dir, 'annotations', 
          'train.json')
      if split == 'test':
        self.annot_path = os.path.join(
          self.data_dir, 'annotations',
          'test.json'
        )
      else:
        self.annot_path = os.path.join(
          self.data_dir, 'annotations', 
          'train.json')
    self.max_objs = 2500
    self.class_name = [
      '__background__', 'TrpV1']
    self._valid_ids = [
      1]
    self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}
    self.voc_color = [(v // 32 * 64 + 64, (v // 8) % 4 * 64, v % 8 * 32) \
                      for v in range(1, self.num_classes + 1)]
    self._data_rng = np.random.RandomState(123)
    self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                             dtype=np.float32)
    self._eig_vec = np.array([
        [-0.58752847, -0.69563484, 0.41340352],
        [-0.5832747, 0.00994535, -0.81221408],
        [-0.56089297, 0.71832671, 0.41158938]
    ], dtype=np.float32)
    # self.mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
    # self.std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)

    self.split = split
    self.opt = opt

    print('==> initializing TrpV1 {} data.'.format(split))
    self.coco = coco.COCO(self.annot_path)
    self.images = self.coco.getImgIds()
    self.num_samples = len(self.images)

    print('Loaded {} {} samples'.format(split, self.num_samples))

  def _to_float(self, x):
    return float("{:.2f}".format(x))

  def convert_eval_format(self, all_bboxes):
    # import pdb; pdb.set_trace()
    detections = []
    for image_id in all_bboxes:
      for cls_ind in all_bboxes[image_id]:
        category_id = self._valid_ids[cls_ind - 1]
        for bbox in all_bboxes[image_id][cls_ind]:
          bbox[2] -= bbox[0]
          bbox[3] -= bbox[1]
          score = bbox[4]
          bbox_out  = list(map(self._to_float, bbox[0:4]))

          detection = {
              "image_id": int(image_id),
              "category_id": int(category_id),
              "bbox": bbox_out,
              "score": float("{:.2f}".format(score))
          }
          if len(bbox) > 5:
              extreme_points = list(map(self._to_float, bbox[5:13]))
              detection["extreme_points"] = extreme_points
          detections.append(detection)
    return detections

  def __len__(self):
    return self.num_samples

  def save_results(self, results, save_dir):
    json.dump(self.convert_eval_format(results), 
                open('{}/results.json'.format(save_dir), 'w'))

  def bbox_valid(self, bbox, imsize=512, dis=10):
    if bbox[0] < dis or bbox[0] > imsize - dis:
      return False
    elif bbox[1] < dis or bbox[1] > imsize - dis:
      return False
    else:
      return True 

  def run_eval(self, results, save_dir):
    # result_json = os.path.join(save_dir, "results.json")
    # detections  = self.convert_eval_format(results)
    # json.dump(detections, open(result_json, "w"))
    self.save_results(results, save_dir)
   # remove edge boxes
    a = json.load(open('{}/results.json'.format(save_dir)))
    for i in range(len(a)-1,-1,-1):
      if not self.bbox_valid(a[i]['bbox'], 1024, 13):
        del(a[i])
    json.dump(a, open('{}/processed_results.json'.format(save_dir), 'w'))
    coco_dets = self.coco.loadRes('{}/processed_results.json'.format(save_dir))
    #coco_dets = self.coco.loadRes('{}/results.json'.format(save_dir))
    coco_eval = COCOeval(self.coco, coco_dets, "bbox")
    coco_eval.params.maxDets=[500,1000  ,1500]
    
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    pr1 = coco_eval.eval['precision'][0, :, 0, :, 2]
    pr2 = coco_eval.eval['precision'][2, :, 0, :, 2]
    pr3 = coco_eval.eval['precision'][4, :, 0, :, 2]
    x = np.arange(0.0, 1.01, 0.01)
    
    plt.switch_backend('agg')
    plt.xlabel('recall')
    plt.ylabel('precision')
    plt.xlim(0, 1.0)
    plt.ylim(0, 1.01)
    plt.grid(True)

    plt.plot(x, pr1, 'b-', label='IoU=0.5')
    plt.plot(x, pr2, 'c-', label='IoU=0.6')
    plt.plot(x, pr3, 'y-', label='IoU=0.7')
    plt.legend((u'Iou=0.5', u'IoU=0.6', u'IoU=0.7'))
    plt.savefig('/data00/UserHome/zwang/pr.png')