# -*- coding:utf-8 -*-
"""
Utlize finetuning learning on Neural Style. Define the network.
"""
import mxnet as mx
from mxnet.gluon import nn
from mxnet.gluon.model_zoo import vision as models

def get_net(pretrained_net, content_layers, style_layers):
    net = nn.Sequential()
    for i in range(max(content_layers+style_layers)+1):
        # max(content_layers+style_layers) is 28.
        # Utlize pretrained_net.features[i] access ith layer of the trained model.
        net.add(pretrained_net.features[i])
        # add VGG19 1-29 layer.
        # pretrained_net.features[i] represent a layer in VGG19 model, e.g. Conv2D(...), is class object.
        # pretrained_net.features[i] is a layer. pretrained_net.features[i].weight could access the weight of layer.
    return net

if __name__ == '__main__':
    # trained model. Here, utlize VGG19 network.
    # pretrained_net = models.vgg19(pretrained=True, ctx=mx.cpu(), root="/home/ly/mxnet/models")
    '''
    See mxnet/gluon/model_zoo/vision/vgg.py for detail. 
    Utlize class VGG + layers(11, 13, 16, 19. The number of conv layer) + 
        filters(num_kernels, e.g. [64, 128, 256, 512, 512]) to define a net. 
    Utlize get_model_file(name, root) to download the trained model. Every time you run the code, it will redownload.

    So I run pretrained_net = models.vgg19(pretrained=True, ctx=mx.cpu(), root="/home/ly/mxnet/models") only once, 
    in the transforming the content images, change this code like when use trained model:
    pretrained_net = models.vgg19(pretrained=False)
    pretrained_net.load_params(filename="/home/ly/mxnet/models/vgg19-f7134366.params", ctx=ctx)

    The arguments of vgg16(), vgg19() are:
    pretrained: Whether to load the pretrained weights for model. 
        Here, is True. First, it will download the trained model, then load the pretrained model.
    ctx: Context, default CPU.
        ctx=mx.cpu() / mx.gpu(0)
    root: str, default '~/.mxnet/models'. Location of trained model.

    return net.
    '''
    
    # neural style network
    pretrained_net = models.vgg19(pretrained=False)
    # Utlize class VGG + layers(11, 13, 16, 19. The number of conv layer) + 
    #     filters(num_kernels, e.g. [64, 128, 256, 512, 512]) to define a net. 
    pretrained_net.load_params(filename="/home/ly/mxnet/models/vgg19-f7134366.params", ctx=mx.cpu())

    style_layers = [0,5,10,19,28]
    content_layers = [25]
    # content_layers+style_layers is: [25,0,5,10,19,28]. max(content_layers+style_layers) is 28.
    net = get_net(pretrained_net, content_layers, style_layers)
    print(net)
