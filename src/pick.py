from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import cv2
import torch
import numpy as np
from opts import opts
from PIL import Image
from detectors.detector_factory import detector_factory
from mrc_utils.mrc import parse, downsample_with_size, save_image, quantize
torch.backends.cudnn.enabled = False

#image_ext = ['jpg', 'jpeg', 'png', 'webp', 'mrc']
image_ext = ['mrc', 'png']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']

def pick(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.debug = max(opt.debug, 1)
  Detector = detector_factory[opt.task]
  detector = Detector(opt)

  if os.path.isdir(opt.data):
    image_names = []
    ls = os.listdir(opt.data)
    for file_name in sorted(ls):
        ext = file_name[file_name.rfind('.') + 1:].lower()
        if ext in image_ext:
            image_names.append(os.path.join(opt.data, file_name))
  else:
    image_names = [opt.data]
  #visual_path = os.path.join(opt.demo, 'visual_1024/')
  #if not os.path.exists(visual_path):
  #  os.makedirs(visual_path)
  if opt.data_type == 'mrc':
    mrc_thi = []
    for (image_name) in image_names:
      with open(image_name, "rb") as f:
          content = f.read()
      data, header, _ = parse(content=content)
      print('downsampling',image_name,'...')
      data = downsample_with_size(data, 1024, data.shape[1]/data.shape[0]*1024)
      data = quantize(data)
      data = cv2.equalizeHist(data)
      data = cv2.merge([data, data, data])
      name = image_name.split('/')[-1].replace('.mrc','')
      thi_name = image_name.split('/')[-1].replace('.mrc','.thi')
      mrc_thi.append((image_name[-1], thi_name))

      ret = detector.run(data, header, name)
      time_str = ''
      for stat in time_stats:
        time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
      print(time_str)
    with open('merge_results.thi', "w") as f:
      f.write('[Micrograph Particle coordinate:\n #0:MICROGRAPH_PATH    STRING\n #1:PARTICLE_PATH    STRING]\n')
      for item in mrc_thi:
        f.write('@%s\t@%s\n' % (item[0], item[1]))
  elif opt.data_type == 'png':
    for (image_name) in image_names:
      ret = detector.run(image_name, None, image_name)
      time_str = ''
      for stat in time_stats:
        time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
      print(time_str)
      
if __name__ == '__main__':
  opt = opts().init()
  pick(opt)
