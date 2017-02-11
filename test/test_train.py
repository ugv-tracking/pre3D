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
config.TRAIN.BBOX_3D = False
config.TRAIN.BBOX_NORMALIZATION_PRECOMPUTED = True
config.TRAIN.BG_THRESH_LO = 0.0

# load symbol
sym = eval('get_vgg_train')(num_classes=4)
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


arg_params, aux_params = load_param('/data01/hustxly/model/faster_rcnn/kitti_ry_cls_input_up_2/ry_alpha_car_only_reg', 18, convert=True)
# infer shape
data_shape_dict = dict(train_data.provide_data + train_data.provide_label)
sym.list_outputs()
arg_shape, out_shape, aux_shape = sym.infer_shape(**data_shape_dict)
arg_shape, out_shape, aux_shape = sym.infer_shape(**data_shape_dict)
arg_shape_dict = dict(zip(sym.list_arguments(), arg_shape))
out_shape_dict = dict(zip(sym.list_outputs(), out_shape))
aux_shape_dict = dict(zip(sym.list_auxiliary_states(), aux_shape))


arg_params['fc6_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['fc6_weight'])
arg_params['fc6_bias'] = mx.nd.zeros(shape=arg_shape_dict['fc6_bias'])
arg_params['fc7_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['fc7_weight'])
arg_params['fc7_bias'] = mx.nd.zeros(shape=arg_shape_dict['fc7_bias'])
arg_params['rpn_conv_3x3_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['rpn_conv_3x3_weight'])
arg_params['rpn_conv_3x3_bias'] = mx.nd.zeros(shape=arg_shape_dict['rpn_conv_3x3_bias'])
arg_params['rpn_cls_score_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['rpn_cls_score_weight'])
arg_params['rpn_cls_score_bias'] = mx.nd.zeros(shape=arg_shape_dict['rpn_cls_score_bias'])
arg_params['rpn_bbox_pred_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['rpn_bbox_pred_weight'])
arg_params['rpn_bbox_pred_bias'] = mx.nd.zeros(shape=arg_shape_dict['rpn_bbox_pred_bias'])
arg_params['cls_score_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['cls_score_weight'])
arg_params['cls_score_bias'] = mx.nd.zeros(shape=arg_shape_dict['cls_score_bias'])
arg_params['bbox_pred_weight'] = mx.random.normal(0, 0.001, shape=arg_shape_dict['bbox_pred_weight'])
arg_params['bbox_pred_bias'] = mx.nd.zeros(shape=arg_shape_dict['bbox_pred_bias'])

pprint.pprint(out_shape_dict)

fixed_param_prefix = ['conv1', 'conv2']
data_names = [k[0] for k in train_data.provide_data]
label_names = [k[0] for k in train_data.provide_label]
mod = MutableModule(sym, data_names=data_names, label_names=label_names,
                    logger=logger, context=ctx, work_load_list=None,
                    max_data_shapes=max_data_shape, max_label_shapes=max_label_shape,
                    fixed_param_prefix=fixed_param_prefix)

