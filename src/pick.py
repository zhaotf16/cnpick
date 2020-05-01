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
from mrc_utils.mrc import parse, quantize, downsample_with_size
from mrc_utils.mrc2png import save_image
torch.backends.cudnn.enabled = False

image_ext = ['jpg', 'jpeg', 'png', 'webp']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']

def pick(opt):
  os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
  opt.debug = max(opt.debug, 1)
  Detector = detector_factory[opt.task]
  detector = Detector(opt)

  if os.path.isdir(opt.demo):
    image_names = []
    ls = os.listdir(opt.demo)
    for file_name in sorted(ls):
        ext = file_name[file_name.rfind('.') + 1:].lower()
        if ext in image_ext:
            image_names.append(os.path.join(opt.demo, file_name))
  else:
    image_names = [opt.demo]
  os.mkdir(os.path.join(opt.demo, 'visual_1024'))
  for (image_name) in image_names:
    with open(file_name, "rb") as f:
        content = f.read()
    data, header, _ = parse(content=content)
    data = downsample_with_size(data, 1024, 1024)
    png_name = 'visual_1024/'+image_name.split()[-1].replace('.mrc','')
    save_image(data, png_name, f='png', verbose=True)
    ret = detector.run(png_name, header)
    time_str = ''
    for stat in time_stats:
      time_str = time_str + '{} {:.3f}s |'.format(stat, ret[stat])
    print(time_str)
if __name__ == '__main__':
  opt = opts().init()
  pick(opt)
