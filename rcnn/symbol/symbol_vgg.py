import mxnet as mx
import math
import proposal
import proposal_target
import proposal_3dbox_target
import numpy as np
from rcnn.config import config


class AngleOutput(mx.operator.CustomOp):
    def forward(self, is_train, req, in_data, out_data, aux):
        
        self.assign(out_data[0], req[0], in_data[0])
        
    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        
        num_bin = config.NUM_BIN
        overlap = 2*math.pi/num_bin
        
        # The out is with the same shape as data (128, 21, 8*2)
        # The label has shape (128, 21, 8*2) too
        out     = np.zeros_like(in_data[0].asnumpy())
        label   = in_data[1].asnumpy()

        bins_angle = np.linspace(0, 2*math.pi, num = num_bin, endpoint = False)
        bins_angle = np.vstack((bins_angle,bins_angle)).reshape(-1, order='F')

        print 'angle is ', label

        dist = np.abs(label - bins_angle)

        cover_bins = np.zeros_like(dist)
        cover_bins[dist < overlap] = 1
        
        for i in xrange(2*num_bin):
            if i%2 == 0:
                out[:,:,i] = -np.cos(label[:,:,i] - bins_angle[i])
            else:
                out[:,:,i] = -np.sin(label[:,:,i] - bins_angle[i])

        #print out[:, 10:100, :]
        out *= cover_bins/cover_bins.sum(axis=2)[:, :, np.newaxis]
        self.assign(in_grad[0], req[0], mx.nd.array(out))

        
@mx.operator.register("angle")
class AngleProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(AngleProp, self).__init__(need_top_grad=False)
    
    def list_arguments(self):
        return ['data', 'label'] 

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        label_shape = in_shape[1]
        #label_shape = (in_shape[0][0], congif.NUM_CLASSES, config.NUM_BIN * 2)
        output_shape = in_shape[0]
        return [data_shape, label_shape], [output_shape], []

    def create_operator(self, ctx, shapes, dtypes):
        return AngleOutput()

def get_vgg_conv(data):
    """
    shared convolutional layers
    :param data: Symbol
    :return: Symbol
    """
    # group 1
    conv1_1 = mx.symbol.Convolution(
        data=data, kernel=(3, 3), pad=(1, 1), num_filter=64, name="conv1_1")
    relu1_1 = mx.symbol.Activation(data=conv1_1, act_type="relu", name="relu1_1")
    conv1_2 = mx.symbol.Convolution(
        data=relu1_1, kernel=(3, 3), pad=(1, 1), num_filter=64, name="conv1_2")
    relu1_2 = mx.symbol.Activation(data=conv1_2, act_type="relu", name="relu1_2")
    pool1 = mx.symbol.Pooling(
        data=relu1_2, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool1")
    # group 2
    conv2_1 = mx.symbol.Convolution(
        data=pool1, kernel=(3, 3), pad=(1, 1), num_filter=128, name="conv2_1")
    relu2_1 = mx.symbol.Activation(data=conv2_1, act_type="relu", name="relu2_1")
    conv2_2 = mx.symbol.Convolution(
        data=relu2_1, kernel=(3, 3), pad=(1, 1), num_filter=128, name="conv2_2")
    relu2_2 = mx.symbol.Activation(data=conv2_2, act_type="relu", name="relu2_2")
    pool2 = mx.symbol.Pooling(
        data=relu2_2, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool2")
    # group 3
    conv3_1 = mx.symbol.Convolution(
        data=pool2, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_1")
    relu3_1 = mx.symbol.Activation(data=conv3_1, act_type="relu", name="relu3_1")
    conv3_2 = mx.symbol.Convolution(
        data=relu3_1, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_2")
    relu3_2 = mx.symbol.Activation(data=conv3_2, act_type="relu", name="relu3_2")
    conv3_3 = mx.symbol.Convolution(
        data=relu3_2, kernel=(3, 3), pad=(1, 1), num_filter=256, name="conv3_3")
    relu3_3 = mx.symbol.Activation(data=conv3_3, act_type="relu", name="relu3_3")
    pool3 = mx.symbol.Pooling(
        data=relu3_3, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool3")
    # group 4
    conv4_1 = mx.symbol.Convolution(
        data=pool3, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_1")
    relu4_1 = mx.symbol.Activation(data=conv4_1, act_type="relu", name="relu4_1")
    conv4_2 = mx.symbol.Convolution(
        data=relu4_1, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_2")
    relu4_2 = mx.symbol.Activation(data=conv4_2, act_type="relu", name="relu4_2")
    conv4_3 = mx.symbol.Convolution(
        data=relu4_2, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv4_3")
    relu4_3 = mx.symbol.Activation(data=conv4_3, act_type="relu", name="relu4_3")
    pool4 = mx.symbol.Pooling(
        data=relu4_3, pool_type="max", kernel=(2, 2), stride=(2, 2), name="pool4")
    # group 5
    conv5_1 = mx.symbol.Convolution(
        data=pool4, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_1")
    relu5_1 = mx.symbol.Activation(data=conv5_1, act_type="relu", name="relu5_1")
    conv5_2 = mx.symbol.Convolution(
        data=relu5_1, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_2")
    relu5_2 = mx.symbol.Activation(data=conv5_2, act_type="relu", name="relu5_2")
    conv5_3 = mx.symbol.Convolution(
        data=relu5_2, kernel=(3, 3), pad=(1, 1), num_filter=512, name="conv5_3")
    relu5_3 = mx.symbol.Activation(data=conv5_3, act_type="relu", name="relu5_3")

    return relu5_3

def get_vgg_train(num_classes=21, num_anchors=9):
    """
    Faster R-CNN end-to-end with VGG 16 conv layers
    :param num_classes: used to determine output size
    :param num_anchors: used to determine output size
    :return: Symbol
    """
    data = mx.symbol.Variable(name="data")
    im_info = mx.symbol.Variable(name="im_info")
    gt_boxes = mx.symbol.Variable(name="gt_boxes")
    rpn_label = mx.symbol.Variable(name='label')
    rpn_bbox_target = mx.symbol.Variable(name='bbox_target')
    rpn_bbox_weight = mx.symbol.Variable(name='bbox_weight')

    # shared convolutional layers
    relu5_3 = get_vgg_conv(data)

    # RPN layers
    rpn_conv = mx.symbol.Convolution(
        data=relu5_3, kernel=(3, 3), pad=(1, 1), num_filter=512, name="rpn_conv_3x3")
    rpn_relu = mx.symbol.Activation(data=rpn_conv, act_type="relu", name="rpn_relu")
    rpn_cls_score = mx.symbol.Convolution(
        data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=2 * num_anchors, name="rpn_cls_score")
    rpn_bbox_pred = mx.symbol.Convolution(
        data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=4 * num_anchors, name="rpn_bbox_pred")

    # prepare rpn data
    rpn_cls_score_reshape = mx.symbol.Reshape(
        data=rpn_cls_score, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape")

    # classification
    rpn_cls_prob = mx.symbol.SoftmaxOutput(data=rpn_cls_score_reshape, label=rpn_label, multi_output=True,
                                           normalization='valid', use_ignore=True, ignore_label=-1, name="rpn_cls_prob")
    # bounding box regression
    rpn_bbox_loss_ = rpn_bbox_weight * mx.symbol.smooth_l1(name='rpn_bbox_loss_', scalar=3.0, data=(rpn_bbox_pred - rpn_bbox_target))
    rpn_bbox_loss = mx.sym.MakeLoss(name='rpn_bbox_loss', data=rpn_bbox_loss_, grad_scale=1.0 / config.TRAIN.RPN_BATCH_SIZE)

    # ROI proposal
    rpn_cls_act = mx.symbol.SoftmaxActivation(
        data=rpn_cls_score_reshape, mode="channel", name="rpn_cls_act")
    rpn_cls_act_reshape = mx.symbol.Reshape(
        data=rpn_cls_act, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_act_reshape')
    rois = mx.symbol.Custom(
        cls_prob=rpn_cls_act_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
        op_type='proposal', feat_stride=16, scales=(8, 16, 32), ratios=(0.5, 1, 2),
        rpn_pre_nms_top_n=config.TRAIN.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=config.TRAIN.RPN_POST_NMS_TOP_N,
        threshold=config.TRAIN.RPN_NMS_THRESH, rpn_min_size=config.TRAIN.RPN_MIN_SIZE)

    # ROI proposal target
    gt_boxes_reshape = mx.symbol.Reshape(data=gt_boxes, shape=(-1, 5), name='gt_boxes_reshape')
    group = mx.symbol.Custom(rois=rois, gt_boxes=gt_boxes_reshape, op_type='proposal_target',
                             num_classes=num_classes, batch_images=config.TRAIN.BATCH_IMAGES,
                             batch_rois=config.TRAIN.BATCH_ROIS, fg_fraction=config.TRAIN.FG_FRACTION)
    rois = group[0]
    label = group[1]
    bbox_target = group[2]
    bbox_weight = group[3]

    # Fast R-CNN
    pool5 = mx.symbol.ROIPooling(
        name='roi_pool5', data=relu5_3, rois=rois, pooled_size=(7, 7), spatial_scale=0.0625)
    # group 6
    flatten = mx.symbol.Flatten(data=pool5, name="flatten")
    fc6 = mx.symbol.FullyConnected(data=flatten, num_hidden=4096, name="fc6")
    relu6 = mx.symbol.Activation(data=fc6, act_type="relu", name="relu6")
    drop6 = mx.symbol.Dropout(data=relu6, p=0.5, name="drop6")
    # group 7
    fc7 = mx.symbol.FullyConnected(data=drop6, num_hidden=4096, name="fc7")
    relu7 = mx.symbol.Activation(data=fc7, act_type="relu", name="relu7")
    drop7 = mx.symbol.Dropout(data=relu7, p=0.5, name="drop7")
    # classification
    cls_score = mx.symbol.FullyConnected(name='cls_score', data=drop7, num_hidden=num_classes)
    cls_prob = mx.symbol.SoftmaxOutput(name='cls_prob', data=cls_score, label=label, normalization='batch')
    # bounding box regression
    bbox_pred = mx.symbol.FullyConnected(name='bbox_pred', data=drop7, num_hidden=num_classes * 4)
    bbox_loss_ = bbox_weight * mx.symbol.smooth_l1(name='bbox_loss_', scalar=1.0, data=(bbox_pred - bbox_target))
    bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_, grad_scale=1.0 / config.TRAIN.BATCH_ROIS)

    # reshape output
    label = mx.symbol.Reshape(data=label, shape=(config.TRAIN.BATCH_IMAGES, -1), name='label_reshape')
    cls_prob = mx.symbol.Reshape(data=cls_prob, shape=(config.TRAIN.BATCH_IMAGES, -1, num_classes), name='cls_prob_reshape')
    bbox_loss = mx.symbol.Reshape(data=bbox_loss, shape=(config.TRAIN.BATCH_IMAGES, -1, 4 * num_classes), name='bbox_loss_reshape')

    group = mx.symbol.Group([rpn_cls_prob, rpn_bbox_loss, cls_prob, bbox_loss, mx.symbol.BlockGrad(label)])
    return group

def get_vgg_test(num_classes=21, num_anchors=9):
    """
    Faster R-CNN test with VGG 16 conv layers
    :param num_classes: used to determine output size
    :param num_anchors: used to determine output size
    :return: Symbol
    """
    data = mx.symbol.Variable(name="data")
    im_info = mx.symbol.Variable(name="im_info")

    # shared convolutional layers
    relu5_3 = get_vgg_conv(data)

    # RPN
    rpn_conv = mx.symbol.Convolution(
        data=relu5_3, kernel=(3, 3), pad=(1, 1), num_filter=512, name="rpn_conv_3x3")
    rpn_relu = mx.symbol.Activation(data=rpn_conv, act_type="relu", name="rpn_relu")
    rpn_cls_score = mx.symbol.Convolution(
        data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=2 * num_anchors, name="rpn_cls_score")
    rpn_bbox_pred = mx.symbol.Convolution(
        data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=4 * num_anchors, name="rpn_bbox_pred")

    # ROI Proposal
    rpn_cls_score_reshape = mx.symbol.Reshape(
        data=rpn_cls_score, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape")
    rpn_cls_prob = mx.symbol.SoftmaxActivation(
        data=rpn_cls_score_reshape, mode="channel", name="rpn_cls_prob")
    rpn_cls_prob_reshape = mx.symbol.Reshape(
        data=rpn_cls_prob, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_prob_reshape')

    '''
    rois = mx.symbol.Custom(
        cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
        op_type='proposal', feat_stride=16, scales=(4, 8, 16, 24), ratios=(0.5, 1, 2),
        rpn_pre_nms_top_n=config.TEST.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=config.TEST.RPN_POST_NMS_TOP_N,
        threshold=config.TEST.RPN_NMS_THRESH, rpn_min_size=config.TEST.RPN_MIN_SIZE)
    '''
    
    rois = mx.symbol.Custom(
        cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
        op_type='proposal', feat_stride=16, scales=(8, 16, 32), ratios=(0.5, 1, 2),
        rpn_pre_nms_top_n=config.TEST.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=config.TEST.RPN_POST_NMS_TOP_N,
        threshold=config.TEST.RPN_NMS_THRESH, rpn_min_size=config.TEST.RPN_MIN_SIZE)
    

    # Fast R-CNN
    pool5 = mx.symbol.ROIPooling(
        name='roi_pool5', data=relu5_3, rois=rois, pooled_size=(7, 7), spatial_scale=0.0625)
    # group 6
    flatten = mx.symbol.Flatten(data=pool5, name="flatten")
    fc6 = mx.symbol.FullyConnected(data=flatten, num_hidden=4096, name="fc6")
    relu6 = mx.symbol.Activation(data=fc6, act_type="relu", name="relu6")
    drop6 = mx.symbol.Dropout(data=relu6, p=0.5, name="drop6")
    # group 7
    fc7 = mx.symbol.FullyConnected(data=drop6, num_hidden=4096, name="fc7")
    relu7 = mx.symbol.Activation(data=fc7, act_type="relu", name="relu7")
    drop7 = mx.symbol.Dropout(data=relu7, p=0.5, name="drop7")
    # classification
    cls_score = mx.symbol.FullyConnected(name='cls_score', data=drop7, num_hidden=num_classes)
    cls_prob = mx.symbol.SoftmaxOutput(name='cls_prob', data=cls_score)
    # bounding box regression
    bbox_pred = mx.symbol.FullyConnected(name='bbox_pred', data=drop7, num_hidden=num_classes * 4)

    # reshape output
    cls_prob = mx.symbol.Reshape(data=cls_prob, shape=(config.TEST.BATCH_IMAGES, -1, num_classes), name='cls_prob_reshape')
    bbox_pred = mx.symbol.Reshape(data=bbox_pred, shape=(config.TEST.BATCH_IMAGES, -1, 4 * num_classes), name='bbox_pred_reshape')

    # group output
    group = mx.symbol.Group([rois, cls_prob, bbox_pred])
    return group


def get_vgg_3dbox_train(num_classes=21, num_anchors=9):
    """
    Faster R-CNN end-to-end with VGG 16 conv layers
    :param num_classes: used to determine output size
    :param num_anchors: used to determine output size
    :return: Symbol
    """
    data      = mx.symbol.Variable(name="data"     )
    im_info   = mx.symbol.Variable(name="im_info"  )
    gt_boxes  = mx.symbol.Variable(name="gt_boxes" )
    gt_dims   = mx.symbol.Variable(name="gt_dims"  )
    gt_angles = mx.symbol.Variable(name="gt_angles")


    rpn_label = mx.symbol.Variable(name='label')
    rpn_bbox_target = mx.symbol.Variable(name='bbox_target')
    rpn_bbox_weight = mx.symbol.Variable(name='bbox_weight')

    # shared convolutional layers
    relu5_3  = get_vgg_conv(data)

    # RPN layers
    rpn_conv = mx.symbol.Convolution(
        data=relu5_3, kernel=(3, 3), pad=(1, 1), num_filter=512, name="rpn_conv_3x3")
    rpn_relu = mx.symbol.Activation(data=rpn_conv, act_type="relu", name="rpn_relu")
    rpn_cls_score = mx.symbol.Convolution(
        data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=2 * num_anchors, name="rpn_cls_score")
    rpn_bbox_pred = mx.symbol.Convolution(
        data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=4 * num_anchors, name="rpn_bbox_pred")

    # prepare rpn data
    rpn_cls_score_reshape = mx.symbol.Reshape(
        data=rpn_cls_score, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape")

    # classification
    rpn_cls_prob = mx.symbol.SoftmaxOutput(data=rpn_cls_score_reshape, label=rpn_label, multi_output=True,
                                           normalization='valid', use_ignore=True, ignore_label=-1, name="rpn_cls_prob")
    # bounding box regression
    rpn_bbox_loss_ = rpn_bbox_weight * mx.symbol.smooth_l1(name='rpn_bbox_loss_', scalar=3.0, data=(rpn_bbox_pred - rpn_bbox_target))
    rpn_bbox_loss = mx.sym.MakeLoss(name='rpn_bbox_loss', data=rpn_bbox_loss_, grad_scale=1.0 / config.TRAIN.RPN_BATCH_SIZE)

    # ROI proposal
    rpn_cls_act = mx.symbol.SoftmaxActivation(
        data=rpn_cls_score_reshape, mode="channel", name="rpn_cls_act")
    rpn_cls_act_reshape = mx.symbol.Reshape(
        data=rpn_cls_act, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_act_reshape')
    rois = mx.symbol.Custom(
        cls_prob=rpn_cls_act_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
        op_type='proposal', feat_stride=16, scales=(8, 16, 32), ratios=(0.5, 1, 2),
        rpn_pre_nms_top_n=config.TRAIN.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=config.TRAIN.RPN_POST_NMS_TOP_N,
        threshold=config.TRAIN.RPN_NMS_THRESH, rpn_min_size=config.TRAIN.RPN_MIN_SIZE)

    # ROI proposal target
    gt_boxes_reshape  = mx.symbol.Reshape(data=gt_boxes,  shape=(-1, 5), name='gt_boxes_reshape')
    gt_dims_reshape   = mx.symbol.Reshape(data=gt_dims,   shape=(-1, 3), name='gt_dims_reshape')
    gt_angles_reshape = mx.symbol.Reshape(data=gt_angles, shape=(-1, 1), name='gt_angles_reshape')

    group             = mx.symbol.Custom(rois=rois, gt_boxes=gt_boxes_reshape, gt_dims=gt_dims_reshape, gt_angles=gt_angles_reshape, im_info=im_info,\
                            op_type='proposal_3dbox_target', name='roi_target', \
                            num_classes=num_classes, batch_images=config.TRAIN.BATCH_IMAGES, \
                            batch_rois=config.TRAIN.BATCH_ROIS, fg_fraction=config.TRAIN.FG_FRACTION)

    rois1       = group[0]
    label       = group[1]
    bbox_target = group[2]
    bbox_weight = group[3]
    dim_label   = group[4]
    angle_label = group[5]
    conf_label  = group[6]

    #TODO for test
    angle_label1 = mx.symbol.Dropout(data=angle_label, p=0.5, name="drop_angle")

    # Fast R-CNN
    pool5 = mx.symbol.ROIPooling(
        name='roi_pool5', data=relu5_3, rois=rois1, pooled_size=(7, 7), spatial_scale=0.0625)
    # group 6
    flatten = mx.symbol.Flatten(data=pool5, name="flatten")
    fc6 = mx.symbol.FullyConnected(data=flatten, num_hidden=4096, name="fc6")
    relu6 = mx.symbol.Activation(data=fc6, act_type="relu", name="relu6")
    drop6 = mx.symbol.Dropout(data=relu6, p=0.5, name="drop6")
    # group 7
    fc7 = mx.symbol.FullyConnected(data=drop6, num_hidden=4096, name="fc7")
    relu7 = mx.symbol.Activation(data=fc7, act_type="relu", name="relu7")
    drop7 = mx.symbol.Dropout(data=relu7, p=0.5, name="drop7")
    # classification
    cls_score = mx.symbol.FullyConnected(name='cls_score', data=drop7, num_hidden=num_classes)
    cls_prob = mx.symbol.SoftmaxOutput(name='cls_prob', data=cls_score, label=label, normalization='batch')
    # bounding box regression
    bbox_pred = mx.symbol.FullyConnected(name='bbox_pred', data=drop7, num_hidden=num_classes * 4)
    bbox_loss_ = bbox_weight * mx.symbol.smooth_l1(name='bbox_loss_', scalar=1.0, data=(bbox_pred - bbox_target))
    bbox_loss = mx.sym.MakeLoss(name='bbox_loss', data=bbox_loss_, grad_scale=1.0 / config.TRAIN.BATCH_ROIS)

    #dim branch
    fc8_dim = mx.symbol.FullyConnected(data=drop7, num_hidden=512, name="fc8_dim")
    relu8_dim = mx.symbol.Activation(data=fc8_dim, act_type="relu", name="relu8_dim")
    drop8_dim = mx.symbol.Dropout(data=relu8_dim, p=0.5, name="drop8_dim")
    
    fc9_dim = mx.symbol.FullyConnected(data=drop8_dim, num_hidden=512, name="fc9_dim")
    relu9_dim = mx.symbol.Activation(data=fc9_dim, act_type="relu", name="relu9_dim")
    drop9_dim = mx.symbol.Dropout(data=relu9_dim, p=0.5, name="drop9_dim")
    
    fc10_dim = mx.symbol.FullyConnected(data=drop9_dim, num_hidden=num_classes * config.NUM_BIN * 3, name="fc10_dim")
    fc10_dim_reshape = mx.symbol.Reshape(data=fc10_dim, shape=(-1, num_classes, config.NUM_BIN * 3), name='fc10_dim_reshape')
    dim_loss_ = mx.symbol.smooth_l1(name='dim_loss_', scalar=1.0, data=(fc10_dim_reshape - dim_label))
    dim_loss = mx.sym.MakeLoss(name='dim_loss', data=dim_loss_, grad_scale=1.0 / config.TRAIN.BATCH_ROIS)

    #angle branch
    fc8_angle = mx.symbol.FullyConnected(data=drop7, num_hidden=256, name="fc8_angle")
    relu8_angle = mx.symbol.Activation(data=fc8_angle, act_type="relu", name="relu8_angle")
    drop8_angle = mx.symbol.Dropout(data=relu8_angle, p=0.5, name="drop8_angle")

    fc9_angle = mx.symbol.FullyConnected(data=drop8_angle, num_hidden=256, name="fc9_angle")
    relu9_angle = mx.symbol.Activation(data=fc9_angle, act_type="relu", name="relu9_angle")
    drop9_angle = mx.symbol.Dropout(data=relu9_angle, p=0.5, name="drop9_angle")

    fc10_angle = mx.symbol.FullyConnected(data=drop9_angle, num_hidden=num_classes * config.NUM_BIN * 2, name="fc10_angle")
    
    fc10_angle_reshape = mx.symbol.Reshape(data=fc10_angle, shape=(-1, num_classes, config.NUM_BIN, 2), name='fc10_angle_reshape')
    L2_norm = mx.symbol.L2Normalization(data=fc10_angle_reshape, mode='spatial', name='L2_norm')
    angle_flatten = mx.symbol.Reshape(data=L2_norm, shape=(-1, num_classes, config.NUM_BIN * 2), name='angle_flatten')
    angle_loss = mx.symbol.Custom(data=angle_flatten, label=angle_label, name='angle_loss', op_type='angle')

    #conf branch
    fc8_conf = mx.symbol.FullyConnected(data=drop7, num_hidden=256, name="fc8_conf")
    relu8_conf = mx.symbol.Activation(data=fc8_conf, act_type="relu", name="relu8_conf")
    drop8_conf = mx.symbol.Dropout(data=relu8_conf, p=0.5, name="drop8_conf")

    fc9_conf = mx.symbol.FullyConnected(data=drop8_conf, num_hidden=256, name="fc9_conf")
    relu9_conf = mx.symbol.Activation(data=fc9_conf, act_type="relu", name="relu9_conf")
    drop9_conf = mx.symbol.Dropout(data=relu9_conf, p=0.5, name="drop9_conf")

    conf_score = mx.symbol.FullyConnected(name='conf_score', data=drop9_conf, num_hidden=num_classes * config.NUM_BIN)
    conf_score = mx.symbol.Reshape(data=conf_score, shape=(-1, num_classes, config.NUM_BIN), name="conf_score_reshape")
    conf_prob = mx.symbol.SoftmaxOutput(name='conf_prob', data=conf_score, label=conf_label, normalization='batch')

    # reshape output
    label       = mx.symbol.Reshape	(data=label,      shape=(config.TRAIN.BATCH_IMAGES, -1),                  					name='label_reshape'     )
    cls_prob    = mx.symbol.Reshape	(data=cls_prob,   shape=(config.TRAIN.BATCH_IMAGES, -1, num_classes),     					name='cls_prob_reshape'  )
    bbox_loss   = mx.symbol.Reshape	(data=bbox_loss,  shape=(config.TRAIN.BATCH_IMAGES, -1, num_classes * 4), 					name='bbox_loss_reshape' )
    dim_loss    = mx.symbol.Reshape	(data=dim_loss,   shape=(config.TRAIN.BATCH_IMAGES, -1, num_classes, config.NUM_BIN * 3),   name='dim_loss_reshape'  )
    angle_loss  = mx.symbol.Reshape	(data=angle_loss, shape=(config.TRAIN.BATCH_IMAGES, -1, num_classes, config.NUM_BIN * 2),   name='angle_loss_reshape')
    conf_prob   = mx.symbol.Reshape	(data=conf_prob,  shape=(config.TRAIN.BATCH_IMAGES, -1, num_classes, config.NUM_BIN * 1),   name='conf_prob_reshape')

    #group = mx.symbol.Group([rpn_cls_prob, rpn_bbox_loss, cls_prob, bbox_loss, mx.symbol.BlockGrad(label)])
    #group = mx.symbol.Group([rpn_cls_prob, rpn_bbox_loss, cls_prob, bbox_loss, mx.symbol.BlockGrad(label), dim_loss, conf_prob, mx.symbol.BlockGrad(dim_label), mx.symbol.BlockGrad(conf_label)])
    group = mx.symbol.Group([rpn_cls_prob, rpn_bbox_loss, cls_prob, bbox_loss, mx.symbol.BlockGrad(label), dim_loss, conf_prob, 
                             mx.symbol.BlockGrad(dim_label), mx.symbol.BlockGrad(conf_label), angle_loss, mx.symbol.BlockGrad(angle_label)])
    return group

def get_vgg_3dbox_test(num_classes=21, num_anchors=9):
    """
    Faster R-CNN end-to-end with VGG 16 conv layers
    :param num_classes: used to determine output size
    :param num_anchors: used to determine output size
    :return: Symbol
    """
    data = mx.symbol.Variable(name="data")
    im_info = mx.symbol.Variable(name="im_info")

    # shared convolutional layers
    relu5_3 = get_vgg_conv(data)

    # RPN layers
    rpn_conv = mx.symbol.Convolution(
        data=relu5_3, kernel=(3, 3), pad=(1, 1), num_filter=512, name="rpn_conv_3x3")
    rpn_relu = mx.symbol.Activation(data=rpn_conv, act_type="relu", name="rpn_relu")
    rpn_cls_score = mx.symbol.Convolution(
        data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=2 * num_anchors, name="rpn_cls_score")
    rpn_bbox_pred = mx.symbol.Convolution(
        data=rpn_relu, kernel=(1, 1), pad=(0, 0), num_filter=4 * num_anchors, name="rpn_bbox_pred")

    # ROI Proposal
    rpn_cls_score_reshape = mx.symbol.Reshape(
        data=rpn_cls_score, shape=(0, 2, -1, 0), name="rpn_cls_score_reshape")
    rpn_cls_prob = mx.symbol.SoftmaxActivation(
        data=rpn_cls_score_reshape, mode="channel", name="rpn_cls_prob")
    rpn_cls_prob_reshape = mx.symbol.Reshape(
        data=rpn_cls_prob, shape=(0, 2 * num_anchors, -1, 0), name='rpn_cls_prob_reshape')
    rois = mx.symbol.Custom(
        cls_prob=rpn_cls_prob_reshape, bbox_pred=rpn_bbox_pred, im_info=im_info, name='rois',
        op_type='proposal', feat_stride=16, scales=(8, 16, 32), ratios=(0.5, 1, 2),
        rpn_pre_nms_top_n=config.TEST.RPN_PRE_NMS_TOP_N, rpn_post_nms_top_n=config.TEST.RPN_POST_NMS_TOP_N,
        threshold=config.TEST.RPN_NMS_THRESH, rpn_min_size=config.TEST.RPN_MIN_SIZE)

    # Fast R-CNN
    pool5 = mx.symbol.ROIPooling(
        name='roi_pool5', data=relu5_3, rois=rois, pooled_size=(7, 7), spatial_scale=0.0625)
    # group 6
    flatten = mx.symbol.Flatten(data=pool5, name="flatten")
    fc6 = mx.symbol.FullyConnected(data=flatten, num_hidden=4096, name="fc6")
    relu6 = mx.symbol.Activation(data=fc6, act_type="relu", name="relu6")
    drop6 = mx.symbol.Dropout(data=relu6, p=0.5, name="drop6")
    # group 7
    fc7 = mx.symbol.FullyConnected(data=drop6, num_hidden=4096, name="fc7")
    relu7 = mx.symbol.Activation(data=fc7, act_type="relu", name="relu7")
    drop7 = mx.symbol.Dropout(data=relu7, p=0.5, name="drop7")
    # classification
    cls_score = mx.symbol.FullyConnected(name='cls_score', data=drop7, num_hidden=num_classes)
    cls_prob = mx.symbol.SoftmaxOutput(name='cls_prob', data=cls_score)
    # bounding box regression
    bbox_pred = mx.symbol.FullyConnected(name='bbox_pred', data=drop7, num_hidden=num_classes * 4)


    ###########################################################################################
    # prediction for dims, angle and confidence
    #dim branch
    fc8_dim = mx.symbol.FullyConnected(data=drop7, num_hidden=512, name="fc8_dim")
    relu8_dim = mx.symbol.Activation(data=fc8_dim, act_type="relu", name="relu8_dim")
    drop8_dim = mx.symbol.Dropout(data=relu8_dim, p=0.5, name="drop8_dim")
    
    fc9_dim = mx.symbol.FullyConnected(data=drop8_dim, num_hidden=512, name="fc9_dim")
    relu9_dim = mx.symbol.Activation(data=fc9_dim, act_type="relu", name="relu9_dim")
    drop9_dim = mx.symbol.Dropout(data=relu9_dim, p=0.5, name="drop9_dim")
    
    fc10_dim = mx.symbol.FullyConnected(data=drop9_dim, num_hidden= num_classes * 3, name="fc10_dim")
    dim_pred = mx.symbol.LinearRegressionOutput(data = fc10_dim, name='dim_pred')

    #angle branch
    num_bin = config.NUM_BIN
    fc8_angle = mx.symbol.FullyConnected(data=drop7, num_hidden=256, name="fc8_angle")
    relu8_angle = mx.symbol.Activation(data=fc8_angle, act_type="relu", name="relu8_angle")
    drop8_angle = mx.symbol.Dropout(data=relu8_angle, p=0.5, name="drop8_angle")

    fc9_angle = mx.symbol.FullyConnected(data=drop8_angle, num_hidden=256, name="fc9_angle")
    relu9_angle = mx.symbol.Activation(data=fc9_angle, act_type="relu", name="relu9_angle")
    drop9_angle = mx.symbol.Dropout(data=relu9_angle, p=0.5, name="drop9_angle")

    fc10_angle = mx.symbol.FullyConnected(data=drop9_angle, num_hidden=num_bin*2, name="fc10_angle")
    
    fc10_angle_reshape = mx.symbol.Reshape(data=fc10_angle, shape=(-1, num_bin, 2), name='fc10_angle_reshape')
    L2_norm = mx.symbol.L2Normalization(data=fc10_angle_reshape, mode='spatial', name='L2_norm')
    angle_flatten = mx.symbol.Reshape(data=L2_norm, shape=(-1, num_bin*2), name='angle_flatten')

    #angle_loss_ = mx.symbol.smooth_l1(name='angle_loss_', scalar=1.0, data=(angle_flatten - angle_label))
    #angle_loss = mx.sym.MakeLoss(name='angle_loss', data=angle_loss_, grad_scale=1.0 / config.TRAIN.BATCH_ROIS)
    #angle_loss = mx.symbol.Custom(data=angle_flatten, name='angle_loss', op_type='angle')


    # reshape output
    cls_prob    = mx.symbol.Reshape	(data=cls_prob,   shape=(config.TEST.BATCH_IMAGES, -1, num_classes),     name='cls_prob_reshape'  )
    bbox_pred   = mx.symbol.Reshape	(data=bbox_pred,  shape=(config.TEST.BATCH_IMAGES, -1, 4 * num_classes), name='bbox_pred_reshape' )
    dim_pred    = mx.symbol.Reshape	(data=dim_pred,   shape=(config.TEST.BATCH_IMAGES, -1, 3 * num_classes), name='dim_pred_reshape'  )
    #angle_pred  = mx.symbol.Reshape	(data=angle_pred, shape=(config.TEST.BATCH_IMAGES, -1, num_bin*2),       name='angle_pred_reshape')

    group = mx.symbol.Group([rois, cls_prob, bbox_pred, dim_pred])
    #group = mx.symbol.Group([rois, cls_prob, bbox_pred, dim_pred, angle_pred])
    return group
