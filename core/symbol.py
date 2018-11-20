import mxnet as mx
import negativemining
from config import config


def P_Net(mode='train'):
    """
    #Proposal Network
    #input shape 3 x 12 x 12
    """
    data = mx.symbol.Variable(name="data")
    bbox_target = mx.symbol.Variable(name="bbox_target")
    label = mx.symbol.Variable(name="label")

    #conv1 = mx.symbol.Convolution(data=data, kernel=(3, 3), num_filter=10, name="conv1")
    conv1 = mx.symbol.Convolution(data=data, kernel=(3, 3), num_filter=8, name="conv1", no_bias=True)
    bn1 = mx.sym.BatchNorm(data=conv1, name='bn1', fix_gamma=False,momentum=0.9)

    #prelu1 = mx.symbol.LeakyReLU(data=conv1, act_type="prelu", name="prelu1")
    prelu1 = mx.symbol.LeakyReLU(data=bn1, act_type="prelu", name="prelu1")
    pool1 = mx.symbol.Pooling(data=prelu1, pool_type="max",   pooling_convention="full", kernel=(2, 2), stride=(2, 2), name="pool1")
    
    #conv2 = mx.symbol.Convolution(data=pool1, kernel=(3, 3), num_filter=16, name="conv2")
    conv2_dw = mx.symbol.Convolution(data=pool1, kernel=(3, 3), num_filter=8, num_group=8, name="conv2_dw", no_bias=True)
    bn2_dw = mx.sym.BatchNorm(data=conv2_dw, name='bn2_dw', fix_gamma=False,momentum=0.9)
    prelu2_dw = mx.symbol.LeakyReLU(data=bn2_dw, act_type="prelu", name="prelu2_dw")
    conv2_sep = mx.symbol.Convolution(data=prelu2_dw, kernel=(1, 1), num_filter=32, name="conv2_sep", no_bias=True)
    bn2_sep = mx.sym.BatchNorm(data=conv2_sep, name='bn2_sep', fix_gamma=False,momentum=0.9)
	
    #prelu2 = mx.symbol.LeakyReLU(data=conv2, act_type="prelu", name="prelu2")
    prelu2 = mx.symbol.LeakyReLU(data=bn2_sep, act_type="prelu", name="prelu2")

    #conv3 = mx.symbol.Convolution(data=prelu2, kernel=(3, 3), num_filter=32, name="conv3")
    conv3_dw = mx.symbol.Convolution(data=prelu2, kernel=(3, 3), num_filter=32, num_group=32, name="conv3_dw", no_bias=True)
    bn3_dw = mx.sym.BatchNorm(data=conv3_dw, name='bn3_dw', fix_gamma=False,momentum=0.9)
    prelu3_dw = mx.symbol.LeakyReLU(data=bn3_dw, act_type="prelu", name="prelu3_dw")
    conv3_sep = mx.symbol.Convolution(data=prelu3_dw, kernel=(1, 1), num_filter=64, name="conv3_sep", no_bias=True)
    bn3_sep = mx.sym.BatchNorm(data=conv3_sep, name='bn3_sep', fix_gamma=False,momentum=0.9)
	
    #prelu3 = mx.symbol.LeakyReLU(data=conv3, act_type="prelu", name="prelu3")
    prelu3 = mx.symbol.LeakyReLU(data=bn3_sep, act_type="prelu", name="prelu3")

    conv4_1 = mx.symbol.Convolution(data=prelu3, kernel=(1, 1), num_filter=2, name="conv4_1")
	
    conv4_2 = mx.symbol.Convolution(data=prelu3, kernel=(1, 1), num_filter=4, name="conv4_2")

    if mode == 'test':
        cls_prob = mx.symbol.SoftmaxActivation(data=conv4_1, mode="channel", name="cls_prob")
        bbox_pred = conv4_2
        group = mx.symbol.Group([cls_prob, bbox_pred])

    else:
        cls_prob = mx.symbol.SoftmaxOutput(data=conv4_1, label=label,
                                           multi_output=True, use_ignore=True,
                                           #out_grad=True, name="cls_prob")
                                           name="cls_prob")
        conv4_2_reshape = mx.symbol.Reshape(data = conv4_2, shape=(-1, 4), name="conv4_2_reshape")
        bbox_pred = mx.symbol.LinearRegressionOutput(data=conv4_2_reshape, label=bbox_target,
                                                     #grad_scale=1, out_grad=True, name="bbox_pred")
                                                     grad_scale=1, name="bbox_pred")

        out = mx.symbol.Custom(cls_prob=cls_prob, bbox_pred=bbox_pred,
                               label=label, bbox_target=bbox_target,
                               op_type='negativemining', name="negative_mining")

        group = mx.symbol.Group([out])
    return group




def P_Net20(mode='train'):
    """
    #Proposal Network
    #input shape 3 x 20 x 20
    """
    data = mx.symbol.Variable(name="data")
    bbox_target = mx.symbol.Variable(name="bbox_target")
    #label = mx.symbol.Variable(name="softmax_label")
    label = mx.symbol.Variable(name="label")
    
    conv1 = mx.symbol.Convolution(data=data, kernel=(3, 3), stride=(2,2), num_filter=8, name="conv1", no_bias=True)
    bn1 = mx.sym.BatchNorm(data=conv1, name='bn1', fix_gamma=False,momentum=0.9)
    prelu1 = mx.symbol.LeakyReLU(data=bn1, act_type="prelu", name="prelu1")
    #prelu1 = mx.symbol.LeakyReLU(data=conv1, act_type="prelu", name="prelu1")
    #cur size: 9x9

    conv2_dw = mx.symbol.Convolution(data=prelu1, kernel=(3, 3), num_filter=8, num_group=8, name="conv2_dw", no_bias=True)
    bn2_dw = mx.sym.BatchNorm(data=conv2_dw, name='bn2_dw', fix_gamma=False,momentum=0.9)
    prelu2_dw = mx.symbol.LeakyReLU(data=bn2_dw, act_type="prelu", name="prelu2_dw")
    #prelu2_dw = mx.symbol.LeakyReLU(data=conv2_dw, act_type="prelu", name="prelu2_dw")
    conv2_sep = mx.symbol.Convolution(data=prelu2_dw, kernel=(1, 1), num_filter=16, name="conv2_sep", no_bias=True)
    bn2_sep = mx.sym.BatchNorm(data=conv2_sep, name='bn2_sep', fix_gamma=False,momentum=0.9)
    prelu2 = mx.symbol.LeakyReLU(data=bn2_sep, act_type="prelu", name="prelu2")
    #prelu2 = mx.symbol.LeakyReLU(data=conv2_sep, act_type="prelu", name="prelu2")	
    #cur size: 7x7
    
    conv3_dw = mx.symbol.Convolution(data=prelu2, kernel=(3, 3),stride=(2,2), num_filter=16, num_group=16, name="conv3_dw", no_bias=True)
    bn3_dw = mx.sym.BatchNorm(data=conv3_dw, name='bn3_dw', fix_gamma=False,momentum=0.9)
    prelu3_dw = mx.symbol.LeakyReLU(data=bn3_dw, act_type="prelu", name="prelu3_dw")
    #prelu3_dw = mx.symbol.LeakyReLU(data=conv3_dw, act_type="prelu", name="prelu3_dw")
    conv3_sep = mx.symbol.Convolution(data=prelu3_dw, kernel=(1, 1), num_filter=24, name="conv3_sep", no_bias=True)
    bn3_sep = mx.sym.BatchNorm(data=conv3_sep, name='bn3_sep', fix_gamma=False,momentum=0.9)
    prelu3 = mx.symbol.LeakyReLU(data=bn3_sep, act_type="prelu", name="prelu3")
    #prelu3 = mx.symbol.LeakyReLU(data=conv3_sep, act_type="prelu", name="prelu3")
	#cur size: 3x3

    conv4_dw = mx.symbol.Convolution(data=prelu3, kernel=(3, 3), num_filter=24, num_group=24, name="conv4_dw", no_bias=True)
    bn4_dw = mx.sym.BatchNorm(data=conv4_dw, name='bn4_dw', fix_gamma=False,momentum=0.9)
    prelu4_dw = mx.symbol.LeakyReLU(data=bn4_dw, act_type="prelu", name="prelu4_dw")
    #prelu4_dw = mx.symbol.LeakyReLU(data=conv4_dw, act_type="prelu", name="prelu4_dw")
	#cur size: 1x1

    conv4_1 = mx.symbol.Convolution(data=prelu4_dw, kernel=(1, 1), num_filter=2, name="conv4_1")
    conv4_2 = mx.symbol.Convolution(data=prelu4_dw, kernel=(1, 1), num_filter=4, name="conv4_2")

    if mode == 'test':
        cls_prob = mx.symbol.SoftmaxActivation(data=conv4_1, mode="channel", name="cls_prob")
        bbox_pred = conv4_2
        group = mx.symbol.Group([cls_prob, bbox_pred])
        #group = mx.symbol.Group([cls_prob])

    else:
        conv4_1_reshape = mx.symbol.Reshape(data = conv4_1, shape=(-1, 2), name="conv4_1_reshape")
        cls_prob = mx.symbol.SoftmaxOutput(data=conv4_1_reshape, label=label,
                                           multi_output=True, use_ignore=True,
                                           #out_grad=True, name="cls_prob")
                                           name="cls_prob")
        conv4_2_reshape = mx.symbol.Reshape(data = conv4_2, shape=(-1, 4), name="conv4_2_reshape")
        bbox_pred = mx.symbol.LinearRegressionOutput(data=conv4_2_reshape, label=bbox_target,
                                                     #grad_scale=1, out_grad=True, name="bbox_pred")
                                                     grad_scale=1, name="bbox_pred")

        out = mx.symbol.Custom(cls_prob=cls_prob, bbox_pred=bbox_pred,
                               label=label, bbox_target=bbox_target,
                               op_type='negativemining', name="negative_mining")
        
        group = mx.symbol.Group([out])
    return group

def R_Net(mode='train'):
    """
    Refine Network
    input shape 3 x 24 x 24
    """
    data = mx.symbol.Variable(name="data")
    bbox_target = mx.symbol.Variable(name="bbox_target")
    label = mx.symbol.Variable(name="label")

    """
    conv1 = mx.symbol.Convolution(data=data, kernel=(3, 3), num_filter=28, name="conv1")
    prelu1 = mx.symbol.LeakyReLU(data=conv1, act_type="prelu", name="prelu1")
    pool1 = mx.symbol.Pooling(data=prelu1, pool_type="max", pooling_convention="full", kernel=(3, 3), stride=(2, 2), name="pool1")

    conv2 = mx.symbol.Convolution(data=pool1, kernel=(3, 3), num_filter=48, name="conv2")
    prelu2 = mx.symbol.LeakyReLU(data=conv2, act_type="prelu", name="prelu2")
    pool2 = mx.symbol.Pooling(data=prelu2, pool_type="max", pooling_convention="full", kernel=(3, 3), stride=(2, 2), name="pool2")

    conv3 = mx.symbol.Convolution(data=pool2, kernel=(2, 2), num_filter=64, name="conv3")
    prelu3 = mx.symbol.LeakyReLU(data=conv3, act_type="prelu", name="prelu3")

    fc1 = mx.symbol.FullyConnected(data=prelu3, num_hidden=128, name="fc1")
    prelu4 = mx.symbol.LeakyReLU(data=fc1, act_type="prelu", name="prelu4")

    fc2 = mx.symbol.FullyConnected(data=prelu4, num_hidden=2, name="fc2")
    fc3 = mx.symbol.FullyConnected(data=prelu4, num_hidden=4, name="fc3")
	"""

    conv1 = mx.symbol.Convolution(data=data, kernel=(3, 3), pad=(1,1), num_filter=16, name="conv1", no_bias=True)
    bn1 = mx.sym.BatchNorm(data=conv1, name='bn1', fix_gamma=False,momentum=0.9)
    prelu1 = mx.symbol.LeakyReLU(data=bn1, act_type="prelu", name="prelu1")
    
    conv2_dw = mx.symbol.Convolution(data=prelu1, kernel=(3, 3), pad=(1,1), stride=(2, 2), num_filter=16, num_group=16, name="conv2_dw", no_bias=True)
    bn2_dw = mx.sym.BatchNorm(data=conv2_dw, name='bn2_dw', fix_gamma=False,momentum=0.9)
    prelu2_dw = mx.symbol.LeakyReLU(data=bn2_dw, act_type="prelu", name="prelu2_dw")
    conv2_sep = mx.symbol.Convolution(data=prelu2_dw, kernel=(1, 1), num_filter=32, name="conv2_sep", no_bias=True)
    bn2_sep = mx.sym.BatchNorm(data=conv2_sep, name='bn2_sep', fix_gamma=False,momentum=0.9)
    prelu2 = mx.symbol.LeakyReLU(data=bn2_sep, act_type="prelu", name="prelu2")

    conv3_dw = mx.symbol.Convolution(data=prelu2, kernel=(3, 3), pad=(1,1), stride=(2, 2), num_filter=32, num_group=32, name="conv3_dw", no_bias=True)
    bn3_dw = mx.sym.BatchNorm(data=conv3_dw, name='bn3_dw', fix_gamma=False,momentum=0.9)
    prelu3_dw = mx.symbol.LeakyReLU(data=bn3_dw, act_type="prelu", name="prelu3_dw")
    conv3_sep = mx.symbol.Convolution(data=prelu3_dw, kernel=(1, 1), num_filter=64, name="conv3_sep", no_bias=True)
    bn3_sep = mx.sym.BatchNorm(data=conv3_sep, name='bn3_sep', fix_gamma=False,momentum=0.9)
    prelu3 = mx.symbol.LeakyReLU(data=bn3_sep, act_type="prelu", name="prelu3")

    conv4_dw = mx.symbol.Convolution(data=prelu3, kernel=(3, 3), pad=(1,1), stride=(2, 2), num_filter=64, num_group=64, name="conv4_dw", no_bias=True)
    bn4_dw = mx.sym.BatchNorm(data=conv4_dw, name='bn4_dw', fix_gamma=False,momentum=0.9)
    prelu4_dw = mx.symbol.LeakyReLU(data=bn4_dw, act_type="prelu", name="prelu4_dw")
    conv4_sep = mx.symbol.Convolution(data=prelu4_dw, kernel=(1, 1), num_filter=128, name="conv4_sep", no_bias=True)
    bn4_sep = mx.sym.BatchNorm(data=conv4_sep, name='bn4_sep', fix_gamma=False,momentum=0.9)
    prelu4 = mx.symbol.LeakyReLU(data=bn4_sep, act_type="prelu", name="prelu4")

    conv5_dw = mx.symbol.Convolution(data=prelu4, kernel=(3, 3), num_filter=128, num_group=128, name="conv5_dw", no_bias=True)
    bn5_dw = mx.sym.BatchNorm(data=conv5_dw, name='bn5_dw', fix_gamma=False,momentum=0.9)
    prelu5_dw = mx.symbol.LeakyReLU(data=bn5_dw, act_type="prelu", name="prelu5_dw")
	
    conv5_1 = mx.symbol.FullyConnected(data=prelu5_dw, num_hidden=2, name="conv5_1")
    conv5_2 = mx.symbol.FullyConnected(data=prelu5_dw, num_hidden=4, name="conv5_2")

    cls_prob = mx.symbol.SoftmaxOutput(data=conv5_1, label=label, use_ignore=True,
                                       #out_grad=True, name="cls_prob")
                                       name="cls_prob")
    if mode == 'test':
        cls_prob = mx.symbol.SoftmaxOutput(data=conv5_1, label=label, use_ignore=True, name="cls_prob")
        bbox_pred = conv5_2
        group = mx.symbol.Group([cls_prob, bbox_pred])
    else:
        bbox_pred = mx.symbol.LinearRegressionOutput(data=conv5_2, label=bbox_target,
                                                     #out_grad=True, grad_scale=1, name="bbox_pred")
                                                     grad_scale=1, name="bbox_pred")

        out = mx.symbol.Custom(cls_prob=cls_prob, bbox_pred=bbox_pred, label=label,
                               bbox_target=bbox_target, op_type='negativemining', name="negative_mining")

        group = mx.symbol.Group([out])
    return group


def O_Net(mode="train"):
    """
    Refine Network
    input shape 3 x 48 x 48
    """
    data = mx.symbol.Variable(name="data")
    bbox_target = mx.symbol.Variable(name="bbox_target")
    label = mx.symbol.Variable(name="label")

    conv1 = mx.symbol.Convolution(data=data, kernel=(3, 3), num_filter=32, name="conv1")
    prelu1 = mx.symbol.LeakyReLU(data=conv1, act_type="prelu", name="prelu1")
    pool1 = mx.symbol.Pooling(data=prelu1, pool_type="max", pooling_convention="full", kernel=(3, 3), stride=(2, 2), name="pool1")

    conv2 = mx.symbol.Convolution(data=pool1, kernel=(3, 3), num_filter=64, name="conv2")
    prelu2 = mx.symbol.LeakyReLU(data=conv2, act_type="prelu", name="prelu2")
    pool2 = mx.symbol.Pooling(data=prelu2, pool_type="max", pooling_convention="full", kernel=(3, 3), stride=(2, 2), name="pool2")

    conv3 = mx.symbol.Convolution(data=pool2, kernel=(3, 3), num_filter=64, name="conv3")
    prelu3 = mx.symbol.LeakyReLU(data=conv3, act_type="prelu", name="prelu3")
    pool3 = mx.symbol.Pooling(data=prelu3, pool_type="max", pooling_convention="full", kernel=(2, 2), stride=(2, 2), name="pool3")

    conv4 = mx.symbol.Convolution(data=pool3, kernel=(2, 2), num_filter=128, name="conv4")
    prelu4 = mx.symbol.LeakyReLU(data=conv4, act_type="prelu", name="prelu4")

    fc1 = mx.symbol.FullyConnected(data=prelu4, num_hidden=256, name="fc1")
    prelu5 = mx.symbol.LeakyReLU(data=fc1, act_type="prelu", name="prelu5")

    fc2 = mx.symbol.FullyConnected(data=prelu5, num_hidden=2, name="fc2")
    fc3 = mx.symbol.FullyConnected(data=prelu5, num_hidden=4, name="fc3")

    #cls_prob = mx.symbol.SoftmaxOutput(data=fc2, label=label, use_ignore=True, out_grad=True, name="cls_prob")
    cls_prob = mx.symbol.SoftmaxOutput(data=fc2, label=label, use_ignore=True, name="cls_prob")
    if mode == "test":
        bbox_pred = fc3
        group = mx.symbol.Group([cls_prob, bbox_pred])
    else:
        bbox_pred = mx.symbol.LinearRegressionOutput(data=fc3, label=bbox_target,
                                                     #grad_scale=1, out_grad=True, name="bbox_pred")
                                                     grad_scale=1, name="bbox_pred")
        out = mx.symbol.Custom(cls_prob=cls_prob, bbox_pred=bbox_pred, label=label,
                               bbox_target=bbox_target, op_type='negativemining', name="negative_mining")
        group = mx.symbol.Group([out])
    return group
