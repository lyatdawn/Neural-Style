# -*- coding:utf-8 -*-
"""
This code contain extract fetures, define loss and train process.
"""
import logging
from time import time
import mxnet as mx
from mxnet import nd
from mxnet import image # utlize mxnet/image module to read images and transform images.
from mxnet import autograd

def try_gpu():
    """If GPU is available, return mx.gpu(0); else return mx.cpu()"""
    try:
        ctx = mx.gpu()
        _ = nd.array([0], ctx=ctx)
    except:
        ctx = mx.cpu()
    return ctx

# global variables
rgb_mean = nd.array([0.485, 0.456, 0.406])
rgb_std = nd.array([0.229, 0.224, 0.225])

def preprocess(img, image_shape):
    # img: str, required.
    # image_shape: required.
    img = image.imresize(img, *image_shape)
    img = (img.astype('float32')/255 - rgb_mean) / rgb_std
    #norm
    return img.transpose((2,0,1)).expand_dims(axis=0)
    # shape is: (1, 3, H, W)

def postprocess(img):
    # img: str, required.
    img = img[0].as_in_context(rgb_std.context)
    return (img.transpose((1,2,0))*rgb_std + rgb_mean).clip(0,1)
    # transform image to BGR.

def extract_features(net, x, content_layers, style_layers):
    # net: network, required.
    # x: input.
    # style_layers: [0,5,10,19,28].
    # content_layers: [25].
    contents = []
    styles = []
    for i in range(len(net)):
        # utlize index to access every layer of net, w.t. net[i], then use net[i](x) to get the output of ith layer.
        x = net[i](x)
        if i in style_layers:
            styles.append(x)
        if i in content_layers:
            contents.append(x)
    return contents, styles

def gram(x):
    # e.g. x is 1 * 256 * 8 * 8, NDArray.
    c = x.shape[1] # channel
    n = x.size / x.shape[1] # n is NDArray's H * W.
    y = x.reshape((c, int(n)))
    return nd.dot(y, y.T) / n
    # compute gram matrix.

# generate content image and content imag features.
def get_contents(net, ctx, content_img, image_shape, content_layers, style_layers):
    content_x = preprocess(content_img, image_shape).copyto(ctx)
    content_y, _ = extract_features(net, content_x, content_layers, style_layers)
    return content_x, content_y
    # content_x is the content image, content_y is the features of content image.
    # Our need is content_y. Features.

# generate style image and style imag features.
def get_styles(net, ctx, style_img, image_shape, content_layers, style_layers):
    style_x = preprocess(style_img, image_shape).copyto(ctx)
    _, style_y = extract_features(net, style_x, content_layers, style_layers)
    style_y = [gram(y) for y in style_y]
    return style_x, style_y
    # style_x is the style image, style_y is the features of style image.
    # Our need is style_y. Features.

def content_loss(yhat, y):
    return (yhat - y).square().mean()

def style_loss(yhat, gram_y):
    return (gram(yhat) - gram_y).square().mean()

def tv_loss(yhat):
    return 0.5*((yhat[:,:,1:,:] - yhat[:,:,:-1,:]).abs().mean() +
                (yhat[:,:,:,1:] - yhat[:,:,:,:-1]).abs().mean())

def sum_loss(loss, preds, truths, weights):
    # loss: a function. e.g. content_loss.
    return nd.add_n(*[w * loss(yhat, y) for w, yhat, y in zip(
        weights, preds, truths)])
    '''
    1. content
    content_layers is [25]. content_py, content_y, args.content_weights len(list)=1.
    use loss w.t. content_loss to compute loss(yhat, y). content_loss is MSE between yhat(predict image feature) and 
    y(true content image feature).
    2. style
    style_layers is [0,5,10,19,28], style_py, style_y, style_weights len(list)=5.
    use loss w.t. style_loss compute loss(yhat, y). content_loss is MSE between yhat(predict image feature) and 
    y(true content image feature).

    use nd.add_n() add some losses.
    '''

def train(net, x, content_y, style_y, args):
    '''
    net: network.
    x: input image, content image.
    content_y: the features of content image. 
    style_y: the features of style image.
    args contain:
        content_layers, style_layers, max_epochs, lr, lr_decay_epoch.
    '''
    tic = time()
    for i in range(args.max_epochs):
        with autograd.record():
            # the process like this:
            # input x --> in with autograd.record(), compute loss --> loss.backward(), compute gradient -->
            # x.grad(), generate x's gradient.
            # features
            content_py, style_py = extract_features(
                net, x, args.content_layers, args.style_layers)
            # argument is: net, x, content_layers, style_layers
            # x is content_x, the normed content image.
            # content_y is the true content image's content feature.
            # style_y is the true content image's style feature.
            # content_py is the predict image's content feature.
            # style_py is the predict image's style feature.

            # content loss
            content_L  = sum_loss(
                content_loss, content_py, content_y, args.content_weights)

            # style loss
            channels = [net[l].weight.shape[0] for l in args.style_layers]
            style_weights = [1e4 / n**2 for n in channels]

            style_L = sum_loss(
                style_loss, style_py, style_y, style_weights)

            # tv loss
            tv_L = args.tv_weight * tv_loss(x)

            loss = style_L + content_L + tv_L
            # neural-style total loss.

        loss.backward()
        x.grad[:] /= x.grad.abs().mean() + 1e-8
        x[:] -= args.lr * x.grad
        # add sync to avoid large mem usage
        nd.waitall()

        if i and i % 20 == 0:
            '''
            print('batch %3d, content %.2f, style %.2f, '
                  'TV %.2f, time %.1f sec' % (
                i, content_L.asscalar(), style_L.asscalar(),
                tv_L.asscalar(), time()-tic))
            '''
            logging.info('batch %3d, content loss %.4f, style loss %.4f, TV loss %.4f, time %.3f sec' % (
                i, content_L.asscalar(), style_L.asscalar(), tv_L.asscalar(), time()-tic))
            tic = time() # update time!

        if i and i % args.lr_decay_epoch == 0:
            args.lr *= 0.1
            # print('change lr to ', args.lr)
            logging.info('change lr to {}'.format(args.lr))
    # in main framework, save result.
    return x