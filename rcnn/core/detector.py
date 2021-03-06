import mxnet as mx

from rcnn.config import config
from rcnn.processing.bbox_transform import bbox_pred, clip_boxes


class Detector(object):
    def __init__(self, symbol, ctx=None,
                 arg_params=None, aux_params=None):
        self.symbol = symbol
        self.ctx = ctx
        if self.ctx is None:
            self.ctx = mx.cpu()
        self.arg_params = arg_params
        self.aux_params = aux_params
        self.executor = None

    def im_detect(self, im_array, im_info=None, roi_array=None):
        """
        perform detection of designated im, box, must follow minibatch.get_testbatch format
        :param im_array: numpy.ndarray [b c h w]
        :param im_info: numpy.ndarray [b 3]
        :param roi_array: numpy.ndarray [roi_num 5]
        :return: scores, pred_boxes
        """
        # fill in data
        if config.TEST.HAS_RPN:
            self.arg_params['data'] = mx.nd.array(im_array, self.ctx)
            self.arg_params['im_info'] = mx.nd.array(im_info, self.ctx)
            arg_shapes, out_shapes, aux_shapes = \
                self.symbol.infer_shape(data=self.arg_params['data'].shape, im_info=self.arg_params['im_info'].shape)
        else:
            self.arg_params['data'] = mx.nd.array(im_array, self.ctx)
            self.arg_params['rois'] = mx.nd.array(roi_array, self.ctx)
            arg_shapes, out_shapes, aux_shapes = \
                self.symbol.infer_shape(data=self.arg_params['data'].shape, rois=self.arg_params['rois'].shape)

        # fill in label
        arg_shapes_dict = {name: shape for name, shape in zip(self.symbol.list_arguments(), arg_shapes)}
        self.arg_params['cls_prob_label']  = mx.nd.zeros(arg_shapes_dict['cls_prob_label'],  self.ctx)
        if config.TEST.BBOX_3D:
            self.arg_params['dim_pred_label']  = mx.nd.zeros(arg_shapes_dict['dim_pred_label'],  self.ctx)

        print 'to executor'
        # execute
        self.executor = self.symbol.bind(self.ctx, self.arg_params, args_grad=None,
                                         grad_req='null', aux_states=self.aux_params)
        output_dict = {name: nd for name, nd in zip(self.symbol.list_outputs(), self.executor.outputs)}
        self.executor.forward(is_train=False)

        print 'forward done'

        # save output
        scores        = output_dict['cls_prob_reshape_output'    ].asnumpy()[0]
        bbox_deltas   = output_dict['bbox_pred_reshape_output'   ].asnumpy()[0]
        if config.TEST.BBOX_3D:
            dims_deltas   = output_dict['dim_pred_reshape_output'    ].asnumpy()[0]
            #angle_deltas  = output_dict['angle_pred_reshape_output'  ].asnumpy()[0]

        if config.TEST.HAS_RPN:
            rois = output_dict['rois_output'].asnumpy()[:, 1:]
        else:
            rois = roi_array[:, 1:]

        print 'save output done'

        # post processing
        #TODO need to update the 3dbox prediction funciton
        #pred_boxes = 3dbox_pred(rois, bbox_deltas, dims_deltas, angle_deltas)
        pred_boxes = bbox_pred(rois, bbox_deltas)
        pred_boxes = clip_boxes(pred_boxes, im_array[0].shape[-2:])

        if config.TEST.BBOX_3D:
            return scores, pred_boxes, dims_deltas
        else:
            return scores, pred_boxes
