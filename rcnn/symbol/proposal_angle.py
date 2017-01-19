from __future__ import division
from collections import namedtuple
import os
# MXNET_CPU_WORKER_NTHREADS must be greater than 1 for custom op to work on CPU
os.environ["MXNET_CPU_WORKER_NTHREADS"] = "4"

import mxnet as mx
import numpy as np
import logging
import math
logging.getLogger().setLevel(logging.INFO)

class AngleOutput(mx.operator.CustomOp):
    def forward(self, is_train, req, in_data, out_data, aux):
        
        self.assign(out_data[0], req[0], in_data[0])
        
    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        num_bin = 4
        overlap = 2*math.pi/num_bin
        #logging.info('backward
        
        
        lbl = in_data[1].asnumpy()
        y = np.zeros_like(in_data[0].asnumpy())
        
        #for i in xrange(num_bin):
            #y[:,i] = np.arctan2(y[:,i*2+1],y[:,i*2])
        
        #y = y[:,:num_bin] + math.pi
        bins_angle = np.linspace(0, 2*math.pi, num = num_bin, endpoint = False)
        bins_angle = np.vstack((bins_angle,bins_angle)).reshape(-1, order='F')
        
        #y = y.reshape(-1)[:, np.newaxis] - bins_angle

        dist = np.abs(lbl[:, np.newaxis] - bins_angle)
        
        cover_bins = np.zeros_like(dist)
        cover_bins[dist < overlap] = 1
        
        for i in xrange(2*num_bin):
            if i%2 == 0:
                y[:,i] = -np.cos(lbl - bins_angle[i])
            else:
                y[:,i] = -np.sin(lbl - bins_angle[i])

        print "=====ok1"
        print y
        print "====="
                
                
        y *= cover_bins/cover_bins.sum(axis=1)[:, np.newaxis]
        
        print cover_bins.sum(axis=1)
        print "=====ok2"
        print y
        print "====="
        
        self.assign(in_grad[0], req[0], mx.nd.array(y))

        
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
        label_shape = (in_shape[0][0],)
        output_shape = in_shape[0]
        return [data_shape, label_shape], [output_shape], []

    def create_operator(self, ctx, shapes, dtypes):
        return AngleOutput()
    
class SimpleBatch(object):
    def __init__(self, data, label, pad=0):
        self.data = data
        self.label = label
        self.pad = pad
class SimpleIter:
    def __init__(self, 
                 data_names,  data_shapes,  data_gen,
                 label_names, label_shapes, label_gen, 
                 num_batches=10):
        self._provide_data = zip(data_names, data_shapes)
        self._provide_label = zip(label_names, label_shapes)
        self.num_batches = num_batches
        self.data_gen = data_gen
        self.label_gen = label_gen
        self.cur_batch = 0

    def __iter__(self):
        return self

    def reset(self):
        self.cur_batch = 0        

    def __next__(self):
        return self.next()

    @property
    def provide_data(self):
        return self._provide_data

    @property
    def provide_label(self):
        return self._provide_label

    def next(self):
        if self.cur_batch < self.num_batches:
            self.cur_batch += 1
            data = [mx.nd.array(g(d[1])) for d,g in zip(self._provide_data, self.data_gen)]
            assert len(data) > 0, "Empty batch data."
            label = [mx.nd.array(g(d[1])) for d,g in zip(self._provide_label, self.label_gen)]
            assert len(label) > 0, "Empty batch label."
            return SimpleBatch(data, label)
        else:
            raise StopIteration
            
net = mx.sym.Variable('data')
net = mx.sym.FullyConnected(data=net, name='fc1', num_hidden=8)
net = mx.sym.Activation(data=net, name='relu1', act_type="relu")
net = mx.symbol.Custom(data=net, name='angle', op_type='angle')


n = 2
num_classes = 8
data= SimpleIter(['data'], [(n, 100)], 
                  [lambda s: np.random.uniform(-1, 1, s)],
                  ['angle_label'], [(n,)], 
                  [lambda s: np.random.randint(0, num_classes, s)])

print '==========='
print net.list_arguments()
print net.list_outputs()
print net.infer_shape(data=(2,8), angle_label=(2,))
print '==========='

mod = mx.mod.Module(symbol=net, 
                    context=mx.cpu(),
                    data_names=['data'], 
                    label_names=['angle_label'])

#mod = mx.mod.Module(net, initializer = mx.init.uniform(0.5))
#ex = net.bind(ctx=mx.cpu(), args={'data':mx.nd.ones([2,8]),'angle_label':mx.nd.array([math.pi/6, math.pi/6*5])})

print '============='
#print mod.data_shapes
#print mod.label_shapes
#print mod.output_names
print '============='
#Batch = namedtuple('Batch',['data','label'])
#b=Batch([mx.nd.ones([2,8])], [mx.nd.array([math.pi/6, math.pi/6*5])])
#print 'batch', b.data[0].asnumpy(), b.label[0].asnumpy()
#mod.init_params(initializer=mx.init.Xavier(magnitude=2.))
#print mod.get_params()
#mod.forward(b,is_train=True)
#ex.forward(is_train = True)

#print ex.outputs[0].asnumpy()
#mod.backward()
#print mod.get_outputs()[0].asnumpy()

mod.fit(data, num_epoch = 5, batch_end_callback = mx.callback.Speedometer(32, 1))

