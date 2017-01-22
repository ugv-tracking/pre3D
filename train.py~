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

# initial 3D BBOX estimation
if config.TRAIN.BBOX_3D: 
    arg_params['fc6_dim_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['fc6_dim_weight'])
    arg_params['fc6_dim_bias'] = mx.nd.zeros(shape=arg_shape_dict['fc6_dim_bias'])
    arg_params['fc6_angle_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['fc6_angle_weight'])
    arg_params['fc6_angle_bias'] = mx.nd.zeros(shape=arg_shape_dict['fc6_angle_bias'])
    arg_params['fc6_conf_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['fc6_conf_weight'])
    arg_params['fc6_conf_bias'] = mx.nd.zeros(shape=arg_shape_dict['fc6_conf_bias'])

    arg_params['fc7_dim_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['fc7_dim_weight'])
    arg_params['fc7_dim_bias'] = mx.nd.zeros(shape=arg_shape_dict['fc7_dim_bias'])
    arg_params['fc7_angle_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['fc7_angle_weight'])
    arg_params['fc7_angle_bias'] = mx.nd.zeros(shape=arg_shape_dict['fc7_angle_bias'])
    arg_params['fc7_conf_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['fc7_conf_weight'])
    arg_params['fc7_conf_bias'] = mx.nd.zeros(shape=arg_shape_dict['fc7_conf_bias'])

    arg_params['fc8_dim_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['fc8_dim_weight'])
    arg_params['fc8_dim_bias'] = mx.nd.zeros(shape=arg_shape_dict['fc8_dim_bias'])
    arg_params['fc8_angle_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['fc8_angle_weight'])
    arg_params['fc8_angle_bias'] = mx.nd.zeros(shape=arg_shape_dict['fc8_angle_bias'])
    arg_params['fc8_conf_weight'] = mx.random.normal(0, 0.01, shape=arg_shape_dict['fc8_conf_weight'])
    arg_params['fc8_conf_bias'] = mx.nd.zeros(shape=arg_shape_dict['fc8_conf_bias'])

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

print '-----------------------------------------------Initialization Result----------------------------------------------------------'
print 'providing maximum shape', max_data_shape, max_label_shape
print 'output shape'
pprint.pprint(out_shape_dict)
print '=============================================================================================================================='

# check parameter shapes
for k in sym.list_arguments():
    if k in data_shape_dict:
        continue
    assert k in arg_params, k + ' not initialized'
    assert arg_params[k].shape == arg_shape_dict[k], \
        'shape inconsistent for ' + k + ' inferred ' + str(arg_shape_dict[k]) + ' provided ' + str(arg_params[k].shape)
for k in sym.list_auxiliary_states():
    assert k in aux_params, k + ' not initialized'
    assert aux_params[k].shape == aux_shape_dict[k], \
        'shape inconsistent for ' + k + ' inferred ' + str(aux_shape_dict[k]) + ' provided ' + str(aux_params[k].shape)

# create solver
fixed_param_prefix = ['conv1', 'conv2']
data_names = [k[0] for k in train_data.provide_data]
label_names = [k[0] for k in train_data.provide_label]
mod = MutableModule(sym, data_names=data_names, label_names=label_names,
                    logger=logger, context=ctx, work_load_list=None,
                    max_data_shapes=max_data_shape, max_label_shapes=max_label_shape,
                    fixed_param_prefix=fixed_param_prefix)

# decide training params
# metric
rpn_eval_metric = metric.RPNAccMetric()
rpn_cls_metric = metric.RPNLogLossMetric()
rpn_bbox_metric = metric.RPNL1LossMetric()

eval_metric = metric.RCNNAccMetric()
cls_metric = metric.RCNNLogLossMetric()
bbox_metric = metric.RCNNL1LossMetric()
eval_metrics = mx.metric.CompositeEvalMetric()

if config.TRAIN.BBOX_3D: 
    conf_metric = metric.RCNNConfLossMetric()
    dim_metric = metric.RCNNDimLossMetric()
    angle_metric = metric.RCNNAngleLossMetric()

    for child_metric in [rpn_eval_metric, rpn_cls_metric, rpn_bbox_metric, eval_metric, cls_metric, bbox_metric, conf_metric, dim_metric, angle_metric]:
        eval_metrics.add(child_metric)
else:
    for child_metric in [rpn_eval_metric, rpn_cls_metric, rpn_bbox_metric, eval_metric, cls_metric, bbox_metric]:
        eval_metrics.add(child_metric)

# callback
batch_end_callback = callback.Speedometer(train_data.batch_size, frequent=20)
means = np.tile(np.array(config.TRAIN.BBOX_MEANS), imdb.num_classes)
stds = np.tile(np.array(config.TRAIN.BBOX_STDS), imdb.num_classes)
epoch_end_callback = callback.do_checkpoint('model/basic', means, stds)
# optimizer
optimizer_params = {'momentum': 0.9,
                    'wd': 0.0005,
                    'learning_rate': 0.00001,
                    'lr_scheduler': mx.lr_scheduler.FactorScheduler(30000, 0.1),
                    'rescale_grad': (1.0 / batch_size)}

print 'Start Trainning'
print '+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'

mod.fit(train_data, eval_metric=eval_metrics, epoch_end_callback=epoch_end_callback,
        batch_end_callback=batch_end_callback, kvstore='device',
        optimizer='sgd', optimizer_params=optimizer_params,
        arg_params=arg_params, aux_params=aux_params, begin_epoch=1, num_epoch=20)
