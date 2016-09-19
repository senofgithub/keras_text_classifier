# coding:utf-8
"""
If you want to load pre-trained weights that include convolutions (layers Convolution2D or Convolution1D),
be mindful of this: Theano and TensorFlow implement convolution in different ways (TensorFlow actually implements correlation, much like Caffe),
and thus, convolution kernels trained with Theano (resp. TensorFlow) need to be converted before being with TensorFlow (resp. Theano).
"""
from keras import backend as K
from keras.utils.np_utils import convert_kernel
from text_classifier import keras_text_classifier
import sys

def th2tf( model):
    import tensorflow as tf
    ops = []
    for layer in model.layers:
        if layer.__class__.__name__ in ['Convolution1D', 'Convolution2D']:
            original_w = K.get_value(layer.W)
            converted_w = convert_kernel(original_w)
            ops.append(tf.assign(layer.W, converted_w).op)
    K.get_session().run(ops)
    return model

def tf2th(model):
    for layer in model.layers:
        if layer.__class__.__name__ in ['Convolution1D', 'Convolution2D']:
            original_w = K.get_value(layer.W)
            converted_w = convert_kernel(original_w)
            K.set_value(layer.W, converted_w)
    return model

def conv_layer_converted(tf_weights, th_weights, m = 0):
    """
    :param tf_weights:
    :param th_weights:
    :param m: 0-tf2th, 1-th2tf
    :return:
    """
    if m == 0: # tf2th
        tc = keras_text_classifier(weights_path=tf_weights)
        model = tc.loadmodel()
        model = tf2th(model)
        model.save_weights(th_weights)
    elif m == 1: # th2tf
        tc = keras_text_classifier(weights_path=th_weights)
        model = tc.loadmodel()
        model = th2tf(model)
        model.save_weights(tf_weights)
    else:
        print("0-tf2th, 1-th2tf")
        return
if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("python tf_weights th_weights <0|1>\n0-tensorflow to theano\n1-theano to tensorflow")
        sys.exit(0)
    tf_weights = sys.argv[1]
    th_weights = sys.argv[2]
    m = int(sys.argv[3])
    conv_layer_converted(tf_weights, th_weights, m)