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
sym = eval('get_vgg_3dbox_train')()
feat_sym = sym.get_internals()['rpn_cls_score_output']

ctx=[mx.gpu(4)]
batch_size = len(ctx)
input_batch_size = config.TRAIN.BATCH_IMAGES * batch_size
# pprint.pprint(config)
imdb = eval('Kitti')('val', 'data', 'data/kitti')
roidb=imdb.gt_roidb()

train_data = AnchorLoader(feat_sym, roidb, batch_size=input_batch_size, shuffle=True,
                              ctx=ctx, work_load_list=None)
# infer max shape
max_data_shape = [('data', (input_batch_size, 3, 800, 2500))]
max_data_shape, max_label_shape = train_data.infer_shape(max_data_shape)
max_data_shape.append(('gt_boxes',  (input_batch_size, 100, 5)))
max_data_shape.append(('gt_dims',   (input_batch_size, 100, 3)))
max_data_shape.append(('gt_angles', (input_batch_size, 100, 1)))
max_data_shape.append(('gt_confs', (input_batch_size, 100, 1)))


arg_params, aux_params = load_param('model/3dbox/3dbox', 6, convert=True)
# infer shape
data_shape_dict = dict(train_data.provide_data + train_data.provide_label)
print data_shape_dict
arg_shape, out_shape, aux_shape = sym.infer_shape(**data_shape_dict)
arg_name = sym.list_arguments()
out_name = sym.list_outputs()
print {'input' : dict(zip(arg_name, arg_shape))}
print {'output' : dict(zip(out_name, out_shape))}


'''
ctx=[mx.gpu(4)]
batch_size = len(ctx)
input_batch_size = config.TRAIN.BATCH_IMAGES * batch_size
#arg_shape, out_shape, _= feat_sym.infer_shape(bbox_target=(1,36,47,155), im_info=(1,3), gt_dims=(1,2,3), label=(1,65565), gt_boxes=(1,2,5), bbox_weight=(1,36,47,155), data=(1,3,752,2491), gt_angles=(1, 2))
arg_shape, out_shape, _= sym.infer_shape(bbox_target=(1,36,47,155), im_info=(1,3), gt_dims=(1,2,3), label=(1,65565), gt_boxes=(1,2,5), bbox_weight=(1,36,47,155), data=(1,3,752,2491), gt_angles=(1, 2))

arg_name = sym.list_arguments()
out_name = sym.list_outputs()

print {'input' : dict(zip(arg_name, arg_shape))}

print "+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++"
print {'output' : dict(zip(out_name, out_shape))}
'''
