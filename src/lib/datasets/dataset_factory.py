from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .sample.ddd import DddDataset
from .sample.exdet import EXDetDataset
from .sample.ctdet import CTDetDataset
from .sample.multi_pose import MultiPoseDataset

from .dataset.coco import COCO
from .dataset.pascal import PascalVOC
from .dataset.kitti import KITTI
from .dataset.coco_hp import COCOHP
from .dataset.proteasome import PROTEASOME
from .dataset.proteasome_512 import PROTEASOME_512
from .dataset.GspDvc_512 import GSPDVC_512
from .dataset.TrpV1 import TRPV1_512
from .dataset.GspDvc_1024 import GSPDVC_1024
from .dataset.TrpV1_1024 import TRPV1_1024
from .dataset.particle import PARTICLE
from .dataset.Pand17and89 import Pand17and89 
dataset_factory = {
  'coco': COCO,
  'pascal': PascalVOC,
  'kitti': KITTI,
  'coco_hp': COCOHP,
  'proteasome': PROTEASOME,
  'proteasome_512': PROTEASOME_512,
  'GspDvc_512': GSPDVC_512,
  'TrpV1_512': TRPV1_512,
  'GspDvc_1024': GSPDVC_1024,
  'TrpV1_1024': TRPV1_1024,
  'particle': PARTICLE,
  'Pand17and89': Pand17and89
}

_sample_factory = {
  'exdet': EXDetDataset,
  'ctdet': CTDetDataset,
  'ddd': DddDataset,
  'multi_pose': MultiPoseDataset
}


def get_dataset(dataset, task):
  class Dataset(dataset_factory[dataset], _sample_factory[task]):
    pass
  return Dataset
  
