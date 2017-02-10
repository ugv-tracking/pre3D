import mxnet as mx
import math
import numpy as np

class DropExtract(mx.operator.CustomOp):
    def forward(self, is_train, req, in_data, out_data, aux):
        drop_in     = in_data[0].asnumpy()
        drop_cls    = (drop_in[:,0]>0.5).reshape(-1,1)
        drop_cls    = np.broadcast_to(drop_cls, (drop_cls.shape[0], 512))
        drop_out    = drop_in * drop_cls
        self.assign(out_data[0], req[0], mx.nd.array(drop_out))
    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        self.assign(in_grad[0], req[0], 0)
@mx.operator.register("dropExtract")
class DropExtractProp(mx.operator.CustomOpProp):
    def __init__(self):
        super(DropExtractProp, self).__init__(need_top_grad=False)

    def list_arguments(self):
        return ['data']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        output_shape = in_shape[0]
        return [data_shape], [output_shape], []

    def create_operator(self, ctx, shapes, dtypes):
        return DropExtract()
