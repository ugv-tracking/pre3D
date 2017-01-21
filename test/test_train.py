import argparse
import logging
import os
import pprint
import mxnet as mx
import numpy as np

from rcnn.dataset import *
from rcnn.core import callback, metric
from rcnn.core.loader import AnchorLoader
from rcnn.core.module import MutableModule
from rcnn.utils.load_model import load_param
from rcnn.symbol.symbol_vgg import *
from rcnn.config import config
# set up logger
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# setup config
config.TRAIN.BATCH_IMAGES = 1
config.TRAIN.BATCH_ROIS = 128
config.TRAIN.END2END = True
config.TRAIN.BBOX_3D = True
config.TRAIN.BBOX_NORMALIZATION_PRECOMPUTED = True
config.TRAIN.BG_THRESH_LO = 0.0

# load symbol
sym = eval('get_vgg_train')()
feat_sym = sym.get_internals()['rpn_score_output']

ctx=[mx.gpu(4)]
batch_size = len(ctx)
input_batch_size = config.TRAIN.BATCH_IMAGES * batch_size
arg_shape, out_shape, _= feat_sym.infer_shape(data=(1,3,752,2491), im_info=(1,3), gt_boxes=(1,1,5), gt_dims=(1,1,3), 
                    gt_angles=(1, 1), gt_confs=(1,1,1))

arg_name = feat_sym.list_arguments()
out_name = feat_sym.list_outputs()

print {'input' : dict(zip(arg_name, arg_shape))}

print "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
print {'output' : dict(zip(out_name, out_shape))}
