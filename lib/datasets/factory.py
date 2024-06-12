# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__sets = {}
from datasets.kaist import kaist
from datasets.kaist_ir import kaist_ir
from datasets.flir import flir
from datasets.flir_ir import flir_ir

import numpy as np
num_shot = 10

# set up kaist <split>
for split in ['train', 'test']:
    name = 'kaist_{}'.format(split)
    __sets[name] = (lambda split=split: kaist(split))
    tgt_name = 'kaist_{}_tgt'.format(split)
    __sets[tgt_name] = (lambda split=split, num_shot=num_shot: kaist(split, num_shot))

# set up thermal kaist <split>
for split in ['train', 'test']:
    name = 'kaist_ir_{}'.format(split)
    __sets[name] = (lambda split=split: kaist_ir(split))
    tgt_name = 'kaist_ir_{}_tgt'.format(split)
    __sets[tgt_name] = (lambda split=split, num_shot=num_shot: kaist_ir(split, num_shot))

# set up flir <split>
for split in ['train', 'val']:
    name = 'flir_{}'.format(split)
    __sets[name] = (lambda split=split: flir(split))
    tgt_name = 'flir_{}_tgt'.format(split)
    __sets[tgt_name] = (lambda split=split, num_shot=num_shot: flir(split, num_shot))

# set up flir <split>
for split in ['train', 'val']:
    name = 'flir_ir_{}'.format(split)
    __sets[name] = (lambda split=split: flir_ir(split))
    tgt_name = 'flir_ir_{}_tgt'.format(split)
    __sets[tgt_name] = (lambda split=split, num_shot=num_shot: flir_ir(split, num_shot))


def get_imdb(name):
  """Get an imdb (image database) by name."""
  if name not in __sets:
    raise KeyError('Unknown dataset: {}'.format(name))

  return __sets[name]()


def list_imdbs():
  """List all registered imdbs."""
  return list(__sets.keys())
