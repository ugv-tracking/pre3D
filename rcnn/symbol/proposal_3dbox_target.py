"""
Proposal Target Operator selects foreground and background roi and assigns label, bbox_transform to them.
"""

import mxnet as mx
import numpy as np

from rcnn.config import config
from rcnn.core.minibatch import sample_rois

DEBUG = False


class Proposal3DboxTargetOperator(mx.operator.CustomOp):
    def __init__(self, num_classes, batch_images, batch_rois, fg_fraction):
        super(Proposal3DboxTargetOperator, self).__init__()
        self._num_classes = num_classes
        self._batch_images = batch_images
        self._batch_rois = batch_rois
        self._fg_fraction = fg_fraction

        if DEBUG:
            self._count = 0
            self._fg_num = 0
            self._bg_num = 0

    def forward(self, is_train, req, in_data, out_data, aux):
        assert self._batch_rois % self._batch_images == 0, \
            'BATCHIMAGES {} must devide BATCH_ROIS {}'.format(self._batch_images, self._batch_rois)
        rois_per_image = self._batch_rois / self._batch_images
        fg_rois_per_image = np.round(self._fg_fraction * rois_per_image).astype(int)

        all_rois  = in_data[0].asnumpy()
        gt_boxes  = in_data[1].asnumpy()
        gt_dims   = in_data[2].asnumpy()
        gt_angles = in_data[3].asnumpy()
        im_info   = in_data[4].asnumpy()

        # Include ground-truth boxes in the set of candidate rois
        zeros = np.zeros((gt_boxes.shape[0], 1), dtype=gt_boxes.dtype)
        all_rois = np.vstack((all_rois, np.hstack((zeros, gt_boxes[:, :-1]))))
        # Sanity check: single batch only
        assert np.all(all_rois[:, 0] == 0), 'Only single item batches are supported'

        rois, labels, bbox_targets, bbox_weights, dim_label, angle_label = \
            sample_rois(all_rois, fg_rois_per_image, rois_per_image, self._num_classes, gt_boxes=gt_boxes, gt_dims=gt_dims, gt_angles=gt_angles)

        if DEBUG:
            print "labels=", labels
            print 'num fg: {}'.format((labels > 0).sum())
            print 'num bg: {}'.format((labels == 0).sum())
            self._count += 1
            self._fg_num += (labels > 0).sum()
            self._bg_num += (labels == 0).sum()
            print "self._count=", self._count
            print 'num fg avg: {}'.format(self._fg_num / self._count)
            print 'num bg avg: {}'.format(self._bg_num / self._count)
            print 'ratio: {:.3f}'.format(float(self._fg_num) / float(self._bg_num))


        if config.TRAIN.BBOX_3D:
            output = [rois, labels, bbox_targets, bbox_weights, dim_label, angle_label]
        else:
            output = [rois, labels, bbox_targets, bbox_weights]            

        for ind, val in enumerate(output):
            self.assign(out_data[ind], req[ind], val)

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)
        self.assign(in_grad[1], req[1], 0)
        if config.TRAIN.BBOX_3D:
            self.assign(in_grad[2], req[2], 0)
            self.assign(in_grad[3], req[3], 0)
            self.assign(in_grad[4], req[4], 0)            


@mx.operator.register('proposal_3dbox_target')
class Proposal3DboxTargetProp(mx.operator.CustomOpProp):
    def __init__(self, num_classes, batch_images, batch_rois, fg_fraction):
        super(Proposal3DboxTargetProp, self).__init__(need_top_grad=False)
        self._num_classes = int(num_classes)
        self._batch_images = int(batch_images)
        self._batch_rois = int(batch_rois)
        self._fg_fraction = float(fg_fraction)

    def list_arguments(self):
        return ['rois', 'gt_boxes', 'gt_dims', 'gt_angles', 'im_info']
    
    def list_outputs(self):
        return ['rois_output', 'label', 'bbox_target', 'bbox_weight', 'dim_label', 'angle_label']

    def infer_shape(self, in_shape):
        rpn_rois_shape = in_shape[0]
        gt_boxes_shape = in_shape[1]
        if config.TRAIN.BBOX_3D:
            gt_dims_shape   = in_shape[2]
            gt_angles_shape = in_shape[3]
            im_info_shape   = in_shape[4]

        output_rois_shape = (self._batch_rois, 5)
        label_shape = (self._batch_rois, )
        bbox_target_shape = (self._batch_rois, self._num_classes * 4)
        bbox_weight_shape = (self._batch_rois, self._num_classes * 4)
        if config.TRAIN.BBOX_3D:
            dim_label_shape   = (self._batch_rois, 3)
            angle_label_shape = (self._batch_rois, 1)

        if config.TRAIN.BBOX_3D:
            return [rpn_rois_shape, gt_boxes_shape, gt_dims_shape, gt_angles_shape, im_info_shape], \
                   [output_rois_shape, label_shape, bbox_target_shape, bbox_weight_shape, dim_label_shape, angle_label_shape]
        else:
            return [rpn_rois_shape, gt_boxes_shape], \
                   [output_rois_shape, label_shape, bbox_target_shape, bbox_weight_shape]

    def create_operator(self, ctx, shapes, dtypes):
        return Proposal3DboxTargetOperator(self._num_classes, self._batch_images, self._batch_rois, self._fg_fraction)

    def declare_backward_dependency(self, out_grad, in_data, out_data):
        return []
